#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end shape-policy verification on real NPU.

These tests launch simple element-wise add kernels on A5 hardware to confirm that each
shape-policy combination (DYNAMIC / STATIC / mixed / ellipsis) produces numerically correct
results across multiple runtime shapes, and that the compile cache behaves as specified:

- DYNAMIC axes never trigger recompilation.
- STATIC axes trigger one new compilation per distinct value.
- A final ellipsis expands to per-axis STATIC dims (one compilation per distinct tail).

Each test asserts BOTH correctness (vs torch golden) and cache size (vs expected variant
count), so a regression that produces right numbers but wrong caching — or vice versa —
is caught.

Requires an Ascend 950 (A5) device; skips otherwise.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 — registers npu backend

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_M = 128
TILE_N = 128


def _reset_kernel_cache(kernel):
    getattr(kernel, "_compiled_by_signature").clear()
    getattr(kernel, "_kernel_def_by_static_signature").clear()


def _compiled_variant_count(kernel):
    return len(getattr(kernel, "_compiled_by_signature"))


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _run_add(kernel, shapes, *, reset_cache=True):
    """Launch a kernel on several shapes; assert correctness and report cache size.

    Returns the number of compiled variants after all launches.
    """
    if reset_cache:
        _reset_kernel_cache(kernel)

    device = ST_DEVICE
    dtype = torch.float16

    for m, n in shapes:
        torch.manual_seed(0)
        x = torch.rand([m, n], device=device, dtype=dtype)
        y = torch.rand([m, n], device=device, dtype=dtype)
        z = torch.empty([m, n], device=device, dtype=dtype)

        num_cores = max(1, m // TILE_M)
        kernel[None, num_cores](x, y, z)
        torch.npu.synchronize()

        z_ref = x + y
        torch.testing.assert_close(z, z_ref)
        logging.info("  [%d, %d] OK", m, n)

    return _compiled_variant_count(kernel)


# =============================================================================
# 1. Fully DYNAMIC — one variant serves all shapes
# =============================================================================
@pl.jit(auto_mutex=True)
def add_dynamic(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_one_variant_for_all_shapes():
    """DYNAMIC-only kernel: 3 distinct shapes must compile exactly once."""
    _check_npu()
    shapes = [(256, 256), (512, 256), (256, 512)]
    n_variants = _run_add(add_dynamic, shapes)
    assert n_variants == 1, f"DYNAMIC-only kernel compiled {n_variants} variants for 3 shapes; expected 1"


# =============================================================================
# 2. Fully STATIC — one variant per distinct shape
# =============================================================================
@pl.jit(auto_mutex=True)
def add_static(
    x: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    y: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    z: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_static_one_variant_per_distinct_shape():
    """Fully-STATIC kernel: 3 distinct shapes must compile 3 variants; a repeat reuses."""
    _check_npu()
    shapes = [(256, 256), (512, 256), (256, 512), (256, 256)]
    n_variants = _run_add(add_static, shapes)
    assert n_variants == 3, f"STATIC kernel compiled {n_variants} variants for 3 distinct + 1 repeat shape; expected 3"


# =============================================================================
# 3. Mixed [DYNAMIC, STATIC] — only the STATIC axis keys the cache
# =============================================================================
@pl.jit(auto_mutex=True)
def add_mixed(
    x: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mixed_dynamic_axis_change_reuses_static_axis_change_recompiles():
    """[DYNAMIC, STATIC]: N changes reuse; M changes (STATIC) recompile.

    Sequence: (256,256) -> (512,256) -> (256,512)
              ^ variant A       ^ reuse A (M same)   ^ variant B (N changed)
    """
    _check_npu()
    shapes = [(256, 256), (512, 256), (256, 512)]
    n_variants = _run_add(add_mixed, shapes)
    assert n_variants == 2, (
        f"[DYNAMIC, STATIC] kernel compiled {n_variants} variants; expected 2 "
        f"(M-axis DYNAMIC changes must reuse, N-axis STATIC change must recompile)"
    )


# =============================================================================
# 4. Ellipsis [DYNAMIC, ...] — tail expands to STATIC dims
# =============================================================================
@pl.jit(auto_mutex=True)
def add_ellipsis(
    x: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ellipsis_tail_expands_to_static_dims():
    """[DYNAMIC, ...]: first axis is DYNAMIC (reused), tail axes are STATIC (recompile).

    Sequence: (256,256) -> (512,256) -> (256,512)
              ^ variant A       ^ reuse A (M dynamic)   ^ variant B (tail 256->512)
    """
    _check_npu()
    shapes = [(256, 256), (512, 256), (256, 512)]
    n_variants = _run_add(add_ellipsis, shapes)
    assert n_variants == 2, (
        f"[DYNAMIC, ...] kernel compiled {n_variants} variants; expected 2 "
        f"(first axis DYNAMIC must reuse, tail STATIC expansion must recompile on change)"
    )


@pl.jit(auto_mutex=True)
def add_ellipsis_3d(
    x: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        batch_num = x.shape[0]
        m_tile_num = x.shape[1] // TILE_M
        n_tile_num = x.shape[2] // TILE_N

        for b in pl.range(0, batch_num, 1):
            for i in pl.range(core_id, m_tile_num, num_cores):
                for j in pl.range(0, n_tile_num, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [b, i, j], order=[1, 2])
                    pl.load_tile(tile_b, y, [b, i, j], order=[1, 2])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [b, i, j], order=[1, 2])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ellipsis_accepts_higher_rank():
    """[DYNAMIC, ...] must accept a 3-D tensor whose tail expands to two STATIC dims."""
    _check_npu()
    _reset_kernel_cache(add_ellipsis_3d)

    device = ST_DEVICE
    dtype = torch.float16
    torch.manual_seed(0)

    x = torch.rand([2, 128, 128], device=device, dtype=dtype)
    y = torch.rand([2, 128, 128], device=device, dtype=dtype)
    z = torch.empty([2, 128, 128], device=device, dtype=dtype)

    add_ellipsis_3d[None, 1](x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y)
    logging.info("  [2, 128, 128] 3-D ellipsis OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="", force=True)
    test_dynamic_one_variant_for_all_shapes()
    test_static_one_variant_per_distinct_shape()
    test_mixed_dynamic_axis_change_reuses_static_axis_change_recompiles()
    test_ellipsis_tail_expands_to_static_dims()
    test_ellipsis_accepts_higher_rank()
    logging.info("\nAll shape-policy e2e tests passed!")
