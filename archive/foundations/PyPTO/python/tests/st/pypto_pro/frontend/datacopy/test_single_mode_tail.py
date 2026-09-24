# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SingleModeVec0 tail-block test with non-aligned valid size.

SingleModeVec0 moves the entire Acc to a single Vec (no split), so there is no
alignment constraint on M or N. The tail block can be any size (odd or even).

Matrix: C[M, N] = A[M, K] @ B[K, N]
  - B is identity (K==N), so C = A
  - Loop over M-axis in TILE-sized steps
  - Last iteration is the tail block with valid_m < TILE (e.g. 33, non-aligned)

Structure (mirrors test_dual_mode_tail.py):
  - section_cube: loop, load + move + matmul + SingleModeVec0 move to Vec
  - section_vector: loop, store from Vec to GM
  - cube and vector are siblings (not nested)
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO)

TILE = 128
K = 128
N = 128


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


def _inputs(device, shape, dtype=torch.float16):
    torch.manual_seed(0)
    return torch.randn(shape, device=device, dtype=dtype)


@pl.jit(auto_mutex=True)
def call_kernel_single_vec0(
    a: pl.Tensor[[pl.DYNAMIC, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, N], pl.DT_FP32],
    m_total: pl.DT_INT32,
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat), addrs=0x20000, mutex_ids=[1]
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x00000,
        mutex_ids=[5],
    )

    with pl.section_cube():
        cur_b = b_l1.current()
        br = b_l0b.current()
        pl.load(cur_b, b, [0, 0])
        pl.move(br, cur_b)
        for mi in pl.range(0, m_total, TILE):
            cur_a = a_l1.next()
            al = a_l0a.next()
            ac = c_l0c.next()
            vec = vec_grp.next()
            valid_m = pl.min(m_total - mi, TILE)
            pl.set_validshape(cur_a, [valid_m, K])
            pl.set_validshape(al, [valid_m, K])
            pl.load(cur_a, a, [mi, 0])
            pl.move(al, cur_a)
            pl.set_validshape(ac, [valid_m, N])
            pl.matmul(ac, al, br)
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=1)
            pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)
        for mi in pl.range(0, m_total, TILE):
            valid_m = pl.min(m_total - mi, TILE)
            vec = vec_grp.next()
            sub_id = pl.get_subblock_idx()
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            if sub_id == 0:
                pl.set_validshape(vec, [valid_m, N])
                pl.store(out, vec, [mi, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)


@pytest.fixture(scope="module")
def _device():
    dev = ST_DEVICE
    _require_a5(dev)
    return dev


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_single_vec0_odd_tail(_device):
    m_total = 289
    a = _inputs(_device, [m_total, K])
    b = torch.eye(K, device=_device, dtype=torch.float16)
    out = torch.zeros([m_total, N], device=_device, dtype=torch.float32)
    call_kernel_single_vec0(a, b, out, m_total)
    torch.npu.synchronize()
    ref = a.float()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    logging.info("test_single_vec0_odd_tail passed: M_total=%d (tail=33 odd)", m_total)


if __name__ == "__main__":
    dev = ST_DEVICE
    _require_a5(dev)
    test_single_vec0_odd_tail(dev)
