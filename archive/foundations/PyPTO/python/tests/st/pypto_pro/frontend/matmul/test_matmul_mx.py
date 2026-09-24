#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
"""MX matmul coverage with datatype parametrization and M/N/K multi-round tiling.

The data path is

    GM mantissa + E8M0 -> MAT -> LEFT/RIGHT + SCALE_LEFT/SCALE_RIGHT
                       -> matmul_mx -> ACC(FP32) -> GM

M and N may have tail blocks.  K is deliberately kept a multiple of 64: each
64-element K tile consumes two complete 32-element MX scale groups.

All GM tensor dimensions are dynamic and M/N/K are read from runtime tensor shapes.
Test artifact names include M/N/K so parallel pytest workers do not compile different
cases into the same directory; L1/L0 tile capacities remain compile-time constants
required by the cube instructions.
"""

import functools
import logging
import os

import numpy as np
import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO, format="%(message)s")

TILE = 64
GROUP_SIZE = 32
SCALE_K = TILE // GROUP_SIZE
E8M0_UB_ALIGNED_COLS = 32
BATCH = 2
HEADS = 2

M_TOTAL = 128
K_TOTAL = 192
N_TOTAL = 128

# M/N exercise runtime valid_shape, while K remains MX-aligned.
M_TAIL_TOTAL = 200
N_TAIL_TOTAL = 168

_EXACT_MKN = (M_TOTAL, K_TOTAL, N_TOTAL)
_MN_TAIL_MKN = (M_TAIL_TOTAL, K_TOTAL, N_TAIL_TOTAL)
_MKN_VARIANTS = [("exact", _EXACT_MKN), ("mn_tail", _MN_TAIL_MKN)]
_FEATURE_MKN_VARIANTS = [("mn_tail", _MN_TAIL_MKN)]
_TT_SHAPES = [
    ("nonsquare", 128, 192, 256),
    ("nonsquare_mn_tail", 136, 192, 200),
]


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


_MX_DTYPES = {
    "mxfp8_e4m3": (pl.DT_FP8E4M3FN, "e4m3", False),
    "mxfp8_e5m2": (pl.DT_FP8E5M2, "e5m2", False),
    "mxfp4_e2m1": (pl.DT_FP4E2M1, "e2m1", True),
    "mxfp4_e1m2": (pl.DT_FP4E1M2, "e1m2", True),
}

_MX_COMBOS = [
    ("mxfp8_e4m3", "mxfp8_e4m3"),
    ("mxfp8_e4m3", "mxfp8_e5m2"),
    ("mxfp8_e5m2", "mxfp8_e4m3"),
    ("mxfp8_e5m2", "mxfp8_e5m2"),
    ("mxfp4_e2m1", "mxfp4_e2m1"),
    ("mxfp4_e2m1", "mxfp4_e1m2"),
    ("mxfp4_e1m2", "mxfp4_e2m1"),
    ("mxfp4_e1m2", "mxfp4_e1m2"),
]

# Full dtype-direction coverage stays in t01. The remaining tests focus on
# feature paths with one mixed FP8 and one mixed FP4 representative pair.
_MX_REP_COMBOS = [
    ("mxfp8_e4m3", "mxfp8_e5m2"),
    ("mxfp4_e2m1", "mxfp4_e1m2"),
]

# case_id, A dtype, B dtype, addressing mode, transpose A
_MX_ADDRESSING_CASES = [
    ("tile_fp8", "mxfp8_e4m3", "mxfp8_e5m2", "tile", False),
    ("tile_fp4", "mxfp4_e2m1", "mxfp4_e1m2", "tile", False),
    ("tile_transpose_fp8", "mxfp8_e4m3", "mxfp8_e5m2", "tile", True),
    ("tile_transpose_fp4", "mxfp4_e2m1", "mxfp4_e1m2", "tile", True),
    ("mixed_fp8", "mxfp8_e4m3", "mxfp8_e5m2", "mixed", False),
]

_MX_INSERT_DTYPES = list(_MX_DTYPES)
_INSERT_MKN_VARIANTS = [
    ("m_tail", (47, TILE, TILE)),
]

# Non-unit binary scale verifies the store multiplier and keeps most INT8
# results away from saturation.
_SCALAR_SCALE = 0.0625

_E8M0_BOUNDARY_PAIRS = [
    (0, 254),
    (1, 254),
    (127, 127),
    (130, 127),
    (253, 0),
    (254, 0),
    (120, 134),
    (134, 120),
]

_FP8_BOUNDARY_CODES = {
    "e4m3": np.array(
        [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x7E, 0xFE],
        dtype=np.uint8,
    ),
    "e5m2": np.array(
        [0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84, 0x7B, 0xFB],
        dtype=np.uint8,
    ),
}

_T01_CASES = [
    (a_dtype, b_dtype, "exact", _EXACT_MKN)
    for a_dtype, b_dtype in _MX_COMBOS
] + [
    (a_dtype, b_dtype, "mn_tail", _MN_TAIL_MKN)
    for a_dtype, b_dtype in _MX_REP_COMBOS
]

_OUT_DTYPES = {
    "fp32": (pl.DT_FP32, torch.float32),
    "fp16": (pl.DT_FP16, torch.float16),
    "int8": (pl.DT_INT8, torch.int8),
}
_QUANT_OUTS = ["fp32", "fp16", "int8"]


def _combo_id(combo):
    return f"a_{combo[0]}__b_{combo[1]}"


def _assert_single_dynamic_variant(kernel):
    assert len(getattr(kernel, "_compiled_by_signature")) == 1, (
        f"dynamic MX kernel compiled more than one shape variant: {kernel.func_name}"
    )


def _kernel_name(tag, a_dtype_label, b_dtype_label, m, k, n):
    return f"{tag}_a_{a_dtype_label}_b_{b_dtype_label}_{m}x{k}x{n}"


@functools.lru_cache(maxsize=None)
def _make_mx_kernel(a_dtype_label, b_dtype_label, m, k, n, *, use_phase,
                    addressing="offset", transpose_a=False, direct_b_kn=False,
                    transpose_scale_a=False, multicore_atomic=False, scalar_scale=False,
                    out_dtype_label="fp32", case_tag=None):
    """Build one MX specialization; dtype values are fixed when the closure is parsed.

    Case parameters:
      - ``use_phase``: emit Partial/Final phases; otherwise use plain matmul/matmul_acc.
      - ``addressing``: select element-offset, tile, or mixed load addressing.
      - ``transpose_a``: store A as [K, M] and transpose it to [M, K] during load.
      - ``direct_b_kn``: when True, the GM B input is already [K, N] and loads directly.
        When False, the GM B input is [N, K], so the derived ``transpose_b_load=True``
        transposes it into the [K, N] Right tile during load.
      - ``transpose_scale_a``: when True, the GM A-scale input is physical
        [G/2, M, 2] and ``order=[1, 0]`` loads it as the [M, G] Mat scale view;
        when False, the GM input is [M, G/2, 2] and loads directly (G = K/32).
      - ``multicore_atomic``: split K across AI cores and atomically accumulate in GM.
      - ``scalar_scale``: apply ``_SCALAR_SCALE`` while storing the result.
      - ``out_dtype_label``: select the FP32, FP16, or INT8 output/store path.
      - ``case_tag``: identify a helper kernel that belongs to a distinct test path.
    """
    a_dtype, _, a_is_fp4 = _MX_DTYPES[a_dtype_label]
    b_dtype, _, b_is_fp4 = _MX_DTYPES[b_dtype_label]
    assert a_is_fp4 == b_is_fp4
    is_fp4 = a_is_fp4
    out_dtype, _ = _OUT_DTYPES[out_dtype_label]
    use_offset_addressing = addressing == "offset"
    use_tile_addressing = addressing == "tile"
    a_l1_layout = pl.ZN if transpose_a else pl.NZ
    transpose_b_load = not direct_b_kn
    b_l1_layout = pl.ZN if transpose_b_load else pl.NZ
    assert not transpose_scale_a or addressing == "offset"
    tag = "mx_multicore_atomic" if multicore_atomic else ("mx_phase" if use_phase else "mx_no_phase")
    b_storage_tag = "kn" if direct_b_kn else "nk"
    scale_a_shape_tag = "gm" if transpose_scale_a else "mg"
    store_tag = "qscalar" if scalar_scale else "plain"
    tag += (
        f"_{addressing}_sa{scale_a_shape_tag}_{'t' if transpose_a else 'n'}_"
        f"b{b_storage_tag}_{out_dtype_label}_{store_tag}"
    )
    name = _kernel_name(case_tag or tag, a_dtype_label, b_dtype_label, m, k, n)

    @pl.jit(auto_mutex=True, name=name)
    def matmul_mx_mnk(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], a_dtype],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], b_dtype],
        scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=a_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=a_l1_layout,
                             valid_shape=[-1, -1]),
            addrs=0x00000, mutex_ids=[0, 1])
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=b_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=b_l1_layout,
                             valid_shape=[-1, -1]),
            addrs=0x10000, mutex_ids=[2, 3])
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZZ, valid_shape=[-1, -1]),
            addrs=0x20000, mutex_ids=[4, 5])
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NN, valid_shape=[-1, -1]),
            addrs=0x21000, mutex_ids=[6, 7])
        a_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=a_dtype, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[8, 9])
        b_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=b_dtype, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[10, 11])
        sa_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                             valid_shape=[-1, -1], compact=1),
            # MX scale address must match the paired data address >> 4.
            addrs=[0x0, 0x100], mutex_ids=[12, 13])
        sb_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x0, 0x100], mutex_ids=[14, 15])
        c_l0c = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[16])

        core_id = 0
        core_count = 1
        if multicore_atomic:
            core_id = pl.get_block_idx() // pl.get_subblock_num()
            core_count = pl.get_block_num()

        with pl.section_cube():
            mt = out.shape[0]
            nt = out.shape[1]
            kt = scale_a.shape[1] * 2 * GROUP_SIZE
            if transpose_scale_a:
                kt = scale_a.shape[0] * 2 * GROUP_SIZE
            if is_fp4:
                # FP4 tensor annotations expose packed bytes; PTO strides use logical FP4 elements.
                if transpose_a:
                    pl.set_stride(a, [1, mt])
                else:
                    pl.set_stride(a, [kt, 1])
                if transpose_b_load:
                    pl.set_stride(b, [1, kt])
                else:
                    pl.set_stride(b, [nt, 1])
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(core_id * TILE, kt, core_count * TILE):
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        cur_sa = sa_l1.next()
                        cur_sb = sb_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        sal = sa_l0a.next()
                        sbl = sb_l0b.next()

                        pl.set_validshape(cur_a, [m_valid, TILE])
                        pl.set_validshape(cur_b, [TILE, n_valid])
                        pl.set_validshape(cur_sa, [m_valid, SCALE_K])
                        pl.set_validshape(cur_sb, [SCALE_K, n_valid])
                        pl.set_validshape(al, [m_valid, TILE])
                        pl.set_validshape(br, [TILE, n_valid])
                        pl.set_validshape(sal, [m_valid, SCALE_K])
                        pl.set_validshape(sbl, [SCALE_K, n_valid])

                        if use_offset_addressing:
                            if transpose_a and is_fp4:
                                pl.load(cur_a, a, [ki, mi // 2], order=[1, 0])
                            elif transpose_a:
                                pl.load(cur_a, a, [ki, mi], order=[1, 0])
                            elif is_fp4:
                                pl.load(cur_a, a, [mi, ki // 2])
                            else:
                                pl.load(cur_a, a, [mi, ki])
                            if transpose_b_load and is_fp4:
                                pl.load(cur_b, b, [ni, ki // 2], order=[1, 0])
                            elif transpose_b_load:
                                pl.load(cur_b, b, [ni, ki], order=[1, 0])
                            elif is_fp4:
                                pl.load(cur_b, b, [ki, ni // 2])
                            else:
                                pl.load(cur_b, b, [ki, ni])
                            if transpose_scale_a:
                                pl.load(cur_sa, scale_a, [ki // (GROUP_SIZE * 2), mi, 0], order=[1, 0])
                            else:
                                pl.load(cur_sa, scale_a, [mi, ki // (GROUP_SIZE * 2), 0])
                            if transpose_b_load:
                                pl.load(cur_sb, scale_b, [ni, ki // (GROUP_SIZE * 2), 0], order=[1, 0])
                            else:
                                pl.load(cur_sb, scale_b, [ki // (GROUP_SIZE * 2), ni, 0])
                        elif use_tile_addressing:
                            if transpose_a and is_fp4:
                                # FP4 tensor annotations use packed-byte columns. load_tile
                                # scales by the logical tile width, so use an explicit packed
                                # offset for the mantissa while retaining tile loads elsewhere.
                                pl.load(cur_a, a, [ki, mi // 2], order=[1, 0])
                            elif transpose_a:
                                pl.load_tile(cur_a, a, [ki // TILE, mi // TILE], order=[1, 0])
                            elif is_fp4:
                                pl.load(cur_a, a, [mi, ki // 2])
                            else:
                                pl.load_tile(cur_a, a, [mi // TILE, ki // TILE])
                            if transpose_b_load and is_fp4:
                                pl.load(cur_b, b, [ni, ki // 2], order=[1, 0])
                            elif transpose_b_load:
                                pl.load_tile(cur_b, b, [ni // TILE, ki // TILE], order=[1, 0])
                            elif is_fp4:
                                pl.load(cur_b, b, [ki, ni // 2])
                            else:
                                pl.load_tile(cur_b, b, [ki // TILE, ni // TILE])
                            pl.load(cur_sa, scale_a, [mi, ki // (GROUP_SIZE * 2), 0])
                            if transpose_b_load:
                                pl.load(cur_sb, scale_b, [ni, ki // (GROUP_SIZE * 2), 0], order=[1, 0])
                            else:
                                pl.load(cur_sb, scale_b, [ki // (GROUP_SIZE * 2), ni, 0])
                        else:
                            if is_fp4:
                                pl.load(cur_a, a, [mi, ki // 2])
                                if transpose_b_load:
                                    pl.load(cur_b, b, [ni, ki // 2], order=[1, 0])
                                else:
                                    pl.load(cur_b, b, [ki, ni // 2])
                            else:
                                pl.load_tile(cur_a, a, [mi // TILE, ki // TILE])
                                if transpose_b_load:
                                    pl.load(cur_b, b, [ni, ki], order=[1, 0])
                                else:
                                    pl.load(cur_b, b, [ki, ni])
                            pl.load(cur_sa, scale_a, [mi, ki // (GROUP_SIZE * 2), 0])
                            if transpose_b_load:
                                pl.load(cur_sb, scale_b, [ni, ki // (GROUP_SIZE * 2), 0], order=[1, 0])
                            else:
                                pl.load(cur_sb, scale_b, [ki // (GROUP_SIZE * 2), ni, 0])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        pl.move(sal, cur_sa)
                        pl.move(sbl, cur_sb)

                        if multicore_atomic:
                            pl.matmul_mx(ac, al, br, sal, sbl)
                            pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd)
                        elif use_phase:
                            if ki == 0 and ki + TILE >= kt:
                                pl.matmul_mx(ac, al, br, sal, sbl, phase=pl.AccPhase.Final)
                            elif ki == 0:
                                pl.matmul_mx(ac, al, br, sal, sbl, phase=pl.AccPhase.Partial)
                            elif ki + TILE < kt:
                                pl.matmul_mx_acc(ac, ac, al, br, sal, sbl, phase=pl.AccPhase.Partial)
                            else:
                                pl.matmul_mx_acc(ac, ac, al, br, sal, sbl, phase=pl.AccPhase.Final)
                        elif ki == 0:
                            pl.matmul_mx(ac, al, br, sal, sbl)
                        else:
                            pl.matmul_mx_acc(ac, ac, al, br, sal, sbl)
                    if not multicore_atomic:
                        if use_tile_addressing:
                            if scalar_scale:
                                pl.store_tile(out, ac, [mi // TILE, ni // TILE],
                                              scale=_SCALAR_SCALE, phase=pl.STPhase.Final)
                            elif use_phase:
                                pl.store_tile(out, ac, [mi // TILE, ni // TILE], phase=pl.STPhase.Final)
                            else:
                                pl.store_tile(out, ac, [mi // TILE, ni // TILE])
                        elif scalar_scale:
                            pl.store(out, ac, [mi, ni], scale=_SCALAR_SCALE,
                                     phase=pl.STPhase.Final)
                        elif use_phase:
                            pl.store(out, ac, [mi, ni], phase=pl.STPhase.Final)
                        else:
                            pl.store(out, ac, [mi, ni])

    return matmul_mx_mnk


@functools.lru_cache(maxsize=None)
def _make_mx_scale_tile_kernel(a_dtype_label, b_dtype_label, out_dtype_label, m, k, n):
    """Build t07: MX matmul followed by per-column Tile scaling."""
    a_dtype, _, a_is_fp4 = _MX_DTYPES[a_dtype_label]
    b_dtype, _, b_is_fp4 = _MX_DTYPES[b_dtype_label]
    assert a_is_fp4 == b_is_fp4
    is_fp4 = a_is_fp4
    out_dtype, _ = _OUT_DTYPES[out_dtype_label]
    name = _kernel_name(f"mx_scale_tile_{out_dtype_label}", a_dtype_label, b_dtype_label, m, k, n)

    @pl.jit(auto_mutex=True, name=name)
    def matmul_mx_scale_tile(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], a_dtype],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], b_dtype],
        scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        scale_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=a_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ, valid_shape=[-1, -1]),
            addrs=0x00000, mutex_ids=[0, 1])
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=b_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN, valid_shape=[-1, -1]),
            addrs=0x10000, mutex_ids=[2, 3])
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=0x20000, mutex_ids=[4, 5])
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=0x21000, mutex_ids=[6, 7])
        scale_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ND, valid_shape=[-1, -1], compact=1),
            addrs=0x60000, mutex_ids=[17])
        a_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=a_dtype, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[8, 9])
        b_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=b_dtype, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[10, 11])
        sa_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                             valid_shape=[-1, -1], compact=1),
            # MX scale address must match the paired data address >> 4.
            addrs=[0x0, 0x100], mutex_ids=[12, 13])
        sb_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x0, 0x100], mutex_ids=[14, 15])
        scale_tiles = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_INT64,
                             target_memory=pl.MemorySpace.Scaling,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[18])
        c_l0c = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[16])

        with pl.section_cube():
            mt = out.shape[0]
            nt = out.shape[1]
            kt = scale_a.shape[1] * 2 * GROUP_SIZE
            if is_fp4:
                pl.set_stride(a, [kt, 1])
                pl.set_stride(b, [1, kt])
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, kt, TILE):
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        cur_sa = sa_l1.next()
                        cur_sb = sb_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        sal = sa_l0a.next()
                        sbl = sb_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, TILE])
                        pl.set_validshape(cur_b, [TILE, n_valid])
                        pl.set_validshape(cur_sa, [m_valid, SCALE_K])
                        pl.set_validshape(cur_sb, [SCALE_K, n_valid])
                        pl.set_validshape(al, [m_valid, TILE])
                        pl.set_validshape(br, [TILE, n_valid])
                        pl.set_validshape(sal, [m_valid, SCALE_K])
                        pl.set_validshape(sbl, [SCALE_K, n_valid])
                        if is_fp4:
                            pl.load(cur_a, a, [mi, ki // 2])
                            pl.load(cur_b, b, [ni, ki // 2], order=[1, 0])
                        else:
                            pl.load(cur_a, a, [mi, ki])
                            pl.load(cur_b, b, [ni, ki], order=[1, 0])
                        pl.load(cur_sa, scale_a, [mi, ki // (GROUP_SIZE * 2), 0])
                        pl.load(cur_sb, scale_b, [ni, ki // (GROUP_SIZE * 2), 0], order=[1, 0])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        pl.move(sal, cur_sa)
                        pl.move(sbl, cur_sb)
                        # Per-channel scale Tile stores cannot carry STPhase.Final, so keep this chain phase-free.
                        if ki == 0:
                            pl.matmul_mx(ac, al, br, sal, sbl)
                        else:
                            pl.matmul_mx_acc(ac, ac, al, br, sal, sbl)
                    cur_scale = scale_mat.current()
                    scale_tile = scale_tiles.current()
                    pl.set_validshape(cur_scale, [1, n_valid])
                    pl.set_validshape(scale_tile, [1, n_valid])
                    pl.load(cur_scale, scale_params, [0, ni])
                    pl.move(scale_tile, cur_scale)
                    pl.store(out, ac, [mi, ni], scale=scale_tile)

    return matmul_mx_scale_tile


@functools.lru_cache(maxsize=None)
def _make_mx_batched_kernel(a_dtype_label, b_dtype_label, m, k, n):
    """Build t09 with high-dimensional data and scale tensors."""
    a_dtype, _, a_is_fp4 = _MX_DTYPES[a_dtype_label]
    b_dtype, _, b_is_fp4 = _MX_DTYPES[b_dtype_label]
    assert a_is_fp4 == b_is_fp4
    is_fp4 = a_is_fp4
    name = _kernel_name("mx_batched_tile_order", a_dtype_label, b_dtype_label, m, k, n)

    @pl.jit(auto_mutex=True, name=name)
    def matmul_mx_batched(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], a_dtype],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], b_dtype],
        scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=a_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ, valid_shape=[-1, -1]),
            addrs=0x00000, mutex_ids=[0, 1])
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=b_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN, valid_shape=[-1, -1]),
            addrs=0x10000, mutex_ids=[2, 3])
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=0x20000, mutex_ids=[4, 5])
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=0x21000, mutex_ids=[6, 7])
        a_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=a_dtype, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[8, 9])
        b_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=b_dtype, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[10, 11])
        sa_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                             valid_shape=[-1, -1], compact=1),
            # MX scale address must match the paired data address >> 4.
            addrs=[0x0, 0x100], mutex_ids=[12, 13])
        sb_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x0, 0x100], mutex_ids=[14, 15])
        c_l0c = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x0, mutex_ids=[16])

        with pl.section_cube():
            batch = out.shape[0]
            heads = out.shape[1]
            mt = out.shape[2]
            nt = out.shape[3]
            kt = scale_a.shape[3] * 2 * GROUP_SIZE
            m_blocks = (mt + TILE - 1) // TILE
            k_blocks = kt // TILE
            n_blocks = (nt + TILE - 1) // TILE
            if is_fp4:
                pl.set_stride(a, [heads * kt, 1])
                pl.set_stride(b, [1, heads * kt])
            for bi in pl.range(0, batch, 1):
                for hi in pl.range(0, heads, 1):
                    for mb in pl.range(0, m_blocks, 1):
                        m_valid = pl.min(TILE, mt - mb * TILE)
                        for nb in pl.range(0, n_blocks, 1):
                            n_valid = pl.min(TILE, nt - nb * TILE)
                            ac = c_l0c.current()
                            pl.set_validshape(ac, [m_valid, n_valid])
                            for kb in pl.range(0, k_blocks, 1):
                                cur_a = a_l1.next()
                                cur_b = b_l1.next()
                                cur_sa = sa_l1.next()
                                cur_sb = sb_l1.next()
                                al = a_l0a.next()
                                br = b_l0b.next()
                                sal = sa_l0a.next()
                                sbl = sb_l0b.next()
                                pl.set_validshape(cur_a, [m_valid, TILE])
                                pl.set_validshape(cur_b, [TILE, n_valid])
                                pl.set_validshape(cur_sa, [m_valid, SCALE_K])
                                pl.set_validshape(cur_sb, [SCALE_K, n_valid])
                                pl.set_validshape(al, [m_valid, TILE])
                                pl.set_validshape(br, [TILE, n_valid])
                                pl.set_validshape(sal, [m_valid, SCALE_K])
                                pl.set_validshape(sbl, [SCALE_K, n_valid])
                                if is_fp4:
                                    pl.load(
                                        cur_a, a,
                                        [bi, mb * TILE, hi, kb * TILE // 2], order=[1, 3]
                                    )
                                    pl.load(
                                        cur_b, b,
                                        [bi, nb * TILE, hi, kb * TILE // 2], order=[3, 1]
                                    )
                                else:
                                    pl.load_tile(cur_a, a, [bi, mb, hi, kb], order=[1, 3])
                                    pl.load_tile(cur_b, b, [bi, nb, hi, kb], order=[3, 1])
                                pl.load(
                                    cur_sa, scale_a,
                                    [bi, mb * TILE, hi, kb * (SCALE_K // 2), 0], order=[1, 3]
                                )
                                pl.load(
                                    cur_sb, scale_b,
                                    [bi, nb * TILE, hi, kb * (SCALE_K // 2), 0], order=[3, 1]
                                )
                                pl.move(al, cur_a)
                                pl.move(br, cur_b)
                                pl.move(sal, cur_sa)
                                pl.move(sbl, cur_sb)
                                if kb == 0 and kb + 1 >= k_blocks:
                                    pl.matmul_mx(ac, al, br, sal, sbl, phase=pl.AccPhase.Final)
                                elif kb == 0:
                                    pl.matmul_mx(ac, al, br, sal, sbl, phase=pl.AccPhase.Partial)
                                elif kb + 1 < k_blocks:
                                    pl.matmul_mx_acc(
                                        ac, ac, al, br, sal, sbl, phase=pl.AccPhase.Partial
                                    )
                                else:
                                    pl.matmul_mx_acc(
                                        ac, ac, al, br, sal, sbl, phase=pl.AccPhase.Final
                                    )
                            pl.store_tile(
                                out, ac, [bi, hi, mb, nb], order=[2, 3], phase=pl.STPhase.Final
                            )

    return matmul_mx_batched


@functools.lru_cache(maxsize=None)
def _make_mx_insert_kernel(dtype_label, m, k, n):
    """Build t10: GM -> L1 -> L0 -> ACC -> UB -> cast -> L1 -> L0 -> ACC -> GM."""
    mx_dtype, _, is_fp4 = _MX_DTYPES[dtype_label]
    name = _kernel_name("mx_insert", dtype_label, dtype_label, m, k, n)

    @pl.jit(auto_mutex=True, name=name)
    def matmul_mx_insert(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], mx_dtype],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], mx_dtype],
        scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        roundtrip_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ):
        # First L1 operand is loaded from GM; the second L1 tile receives the
        # casted ACC result from the two vector subcores.
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x00000, mutex_ids=[0])
        b_l1_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids=[1])
        sa_l1_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
            addrs=0x20000, mutex_ids=[2])
        sb_l1_group = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN),
            addrs=0x21000, mutex_ids=[3])
        result_l1_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x30000, mutex_ids=[4])
        al_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1),
            addrs=0x0, mutex_ids=[5])
        br_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN, compact=1),
            addrs=0x0, mutex_ids=[6])
        sal_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ, compact=1),
            addrs=0x0, mutex_ids=[7])
        sbl_group = pl.make_tile_group(
            type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN, compact=1),
            addrs=0x0, mutex_ids=[8])
        ac_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
                             compact=1),
            addrs=0x0, mutex_ids=[9])
        acc_vec_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE // 2, TILE], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x0000, mutex_ids=[10])
        bf16_vec_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE // 2, TILE], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                             valid_shape=[-1, -1]),
            addrs=0x2000, mutex_ids=[11])
        mx_vec_nd_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE // 2, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                             valid_shape=[-1, -1]),
            addrs=0x3000, mutex_ids=[12])
        mx_bytes_nd_group = pl.make_tile_group(
            # FP4 x2 Tile shapes use logical columns, but the generic Vec
            # ND->NZ move operates on packed bytes. This aliases mx_vec_nd.
            type=pl.TileType(shape=[TILE // 2, TILE // 2], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                             valid_shape=[-1, -1]),
            addrs=0x3000, mutex_ids=[12])
        mx_bytes_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[TILE // 2, TILE // 2], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x4000, mutex_ids=[13])
        mx_vec_nz_group = pl.make_tile_group(
            # Normal compact storage rounds a tail's valid rows up to 16;
            # 32 physical rows cover both the full half and every M tail.
            type=pl.TileType(shape=[TILE // 2, TILE], dtype=mx_dtype,
                             target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x4000, mutex_ids=[13])

        with pl.section_cube():
            mt = out.shape[0]
            nt = out.shape[1]
            a_l1_tile = a_l1.current()
            b_l1 = b_l1_group.current()
            sa_l1 = sa_l1_group.current()
            sb_l1 = sb_l1_group.current()
            al = al_group.current()
            br = br_group.current()
            sal = sal_group.current()
            sbl = sbl_group.current()
            ac = ac_group.current()
            acc_vec = acc_vec_group.current()
            inserted_l1 = result_l1_group.current()

            pl.set_validshape(a_l1_tile, [mt, TILE])
            pl.set_validshape(b_l1, [TILE, nt])
            pl.set_validshape(sa_l1, [mt, SCALE_K])
            pl.set_validshape(sb_l1, [SCALE_K, nt])
            pl.set_validshape(al, [mt, TILE])
            pl.set_validshape(br, [TILE, nt])
            pl.set_validshape(sal, [mt, SCALE_K])
            pl.set_validshape(sbl, [SCALE_K, nt])
            pl.set_validshape(ac, [mt, nt])

            if is_fp4:
                pl.set_stride(a, [TILE, 1])
                pl.set_stride(b, [nt, 1])
            pl.load(a_l1_tile, a, [0, 0])
            pl.load(b_l1, b, [0, 0])
            pl.load(sa_l1, scale_a, [0, 0, 0])
            pl.load(sb_l1, scale_b, [0, 0, 0])
            pl.move(al, a_l1_tile)
            pl.move(br, b_l1)
            pl.move(sal, sa_l1)
            pl.move(sbl, sb_l1)
            # ACC is consumed by TMOV rather than an STPhase.Final store, so
            # this single-K chain must remain phase-free (same as a per-channel scale Tile store).
            pl.matmul_mx(ac, al, br, sal, sbl)
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=1)
            pl.move(acc_vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

            # Both vector subcores insert their half into result_l1 and signal
            # event 2. Read that L1 tile through the supported L1 -> L0 path,
            # then use a second MX matmul to make its contents observable in GM.
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2)
            pl.set_validshape(inserted_l1, [mt, TILE])
            pl.set_validshape(al, [mt, TILE])
            pl.set_validshape(br, [TILE, nt])
            pl.set_validshape(sal, [mt, SCALE_K])
            pl.set_validshape(sbl, [SCALE_K, nt])
            pl.set_validshape(ac, [mt, nt])
            pl.move(al, inserted_l1)
            pl.move(br, b_l1)
            pl.move(sal, sa_l1)
            pl.move(sbl, sb_l1)
            pl.matmul_mx(ac, al, br, sal, sbl)
            pl.store(roundtrip_out, ac, [0, 0])

        with pl.section_vector():
            mt = out.shape[0]
            nt = out.shape[1]
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)
            sub_id = pl.get_subblock_idx()
            split_rows = (mt + 1) // 2
            row_offset = sub_id * split_rows
            valid_rows = pl.min(split_rows, mt - row_offset)
            acc_vec = acc_vec_group.current()
            bf16_vec = bf16_vec_group.current()
            mx_vec_nd = mx_vec_nd_group.current()
            mx_bytes_nd = mx_bytes_nd_group.current()
            mx_bytes_nz = mx_bytes_nz_group.current()
            mx_vec_nz = mx_vec_nz_group.current()
            result_l1 = result_l1_group.current()
            pl.set_validshape(acc_vec, [valid_rows, nt])
            pl.set_validshape(bf16_vec, [valid_rows, nt])
            pl.set_validshape(mx_vec_nd, [valid_rows, nt])
            pl.set_validshape(mx_vec_nz, [valid_rows, nt])
            pl.set_validshape(result_l1, [mt, nt])
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            # The GM output verifies the ACC -> UB transfer numerically.
            pl.store(out, acc_vec, [row_offset, 0])
            if is_fp4:
                # A5 converts FP32 to packed FP4 through BF16.
                pl.cast(bf16_vec, acc_vec, mode=pl.RoundMode.CAST_ROUND)
                pl.cast(mx_vec_nd, bf16_vec, mode=pl.RoundMode.CAST_ROUND)
                # Follow PTO-ISA's MXFP4 quant path: reinterpret the packed
                # FP4 buffer as uint8 [rows, logical_cols/2] for ND -> NZ,
                # then reinterpret the NZ bytes as logical FP4 for TINSERT.
                pl.set_validshape(mx_bytes_nd, [valid_rows, nt // 2])
                pl.set_validshape(mx_bytes_nz, [valid_rows, nt // 2])
                pl.move(mx_bytes_nz, mx_bytes_nd)
                # mx_bytes_nz and mx_vec_nz alias the same UB address, but are
                # distinct IR values. Make the V -> MTE3 dependency explicit.
                pl.system.sync_src(
                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3
                )
                pl.system.sync_dst(
                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3
                )
            else:
                pl.cast(mx_vec_nd, acc_vec, mode=pl.RoundMode.CAST_ROUND)
                pl.move(mx_vec_nz, mx_vec_nd)
            # TINSERT source and destination are both NZ and have the same dtype.
            pl.insert(result_l1, mx_vec_nz, [row_offset, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    return matmul_mx_insert


@functools.lru_cache(maxsize=None)
def _make_mx_scale_ub_kernel(rows, cols):
    """Build t13: view physical [M, G/2, 2] scale storage as UB [M, G]."""

    @pl.jit(auto_mutex=True, name=f"mx_scale_mg_to_ub_{rows}x{cols}")
    def mx_scale_mg_to_ub(
        scale: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    ):
        rows = scale.shape[0]
        cols = scale.shape[1] * 2
        scale_mg = pl.make_tensor(scale, [rows, cols], [cols, 1])
        out_mg = pl.make_tensor(out, [rows, cols], [cols, 1])
        scale_ub_group = pl.make_tile_group(
            type=pl.TileType(
                # A5 TSTORE requires an ND Vec Tile's static row width to be
                # 32-byte aligned. E8M0 is B8; valid width remains the runtime G.
                shape=[TILE, E8M0_UB_ALIGNED_COLS], dtype=pl.DT_FP8E8M0,
                target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                valid_shape=[-1, -1],
            ),
            addrs=0x0,
            mutex_ids=[0],
        )

        with pl.section_vector():
            scale_ub = scale_ub_group.current()
            for row_offset in pl.range(0, rows, TILE):
                valid_rows = pl.min(TILE, rows - row_offset)
                pl.set_validshape(scale_ub, [valid_rows, cols])
                pl.load(scale_ub, scale_mg, [row_offset, 0])
                pl.store(out_mg, scale_ub, [row_offset, 0])

    return mx_scale_mg_to_ub


def _fp8_tensor_from_codes(codes, element_format, device):
    torch_dtype = torch.float8_e4m3fn if element_format == "e4m3" else torch.float8_e5m2
    return torch.from_numpy(np.ascontiguousarray(codes)).view(torch_dtype).to(device)


_FP4_VALUES = {
    "e2m1": np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    ),
    "e1m2": np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
         -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75],
        dtype=np.float32,
    ),
}


def _pack_fp4(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"FP4 packing requires an even last dimension, got {codes.shape[-1]}")
    return ((codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)).astype(np.uint8)


def _unpack_fp4(packed, logical_shape):
    packed = np.asarray(packed, dtype=np.uint8)
    logical_shape = tuple(logical_shape)
    codes = np.empty(logical_shape, dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = (packed >> 4) & 0x0F
    return codes


def _decode_fp8(codes, element_format):
    codes = np.asarray(codes, dtype=np.uint8)
    if element_format == "e4m3":
        mantissa_bits, exponent_bits, bias = 3, 4, 7
    else:
        mantissa_bits, exponent_bits, bias = 2, 5, 15
    sign = np.where((codes & 0x80) != 0, -1.0, 1.0)
    exponent = (codes >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = codes & ((1 << mantissa_bits) - 1)
    values = np.where(
        exponent == 0,
        (mantissa.astype(np.float32) / (1 << mantissa_bits)) * (2.0 ** (1 - bias)),
        (1.0 + mantissa.astype(np.float32) / (1 << mantissa_bits))
        * np.exp2(exponent.astype(np.int16) - bias),
    )
    special_exp = (1 << exponent_bits) - 1
    if element_format == "e4m3":
        values = np.where((exponent == special_exp) & (mantissa == 7), np.nan, values)
    else:
        values = np.where(
            (exponent == special_exp) & (mantissa == 0),
            np.inf,
            np.where(exponent == special_exp, np.nan, values),
        )
    return (sign * values).astype(np.float32)


def _generate_mantissa_codes(shape, element_format, is_fp4, rng):
    if is_fp4:
        return rng.integers(0, 16, size=shape, dtype=np.uint8)
    all_codes = np.arange(256, dtype=np.uint8)
    all_values = _decode_fp8(all_codes, element_format)
    code_pool = all_codes[np.isfinite(all_values) & (np.abs(all_values) >= 0.25) & (np.abs(all_values) <= 4.0)]
    return rng.choice(code_pool, size=shape).astype(np.uint8)


def _make_quantized_operand(shape, element_format, is_fp4, group_axis, seed):
    """Generate already-quantized MX codes, logical E8M0, and exact dequantized values."""
    rng = np.random.default_rng(seed)
    logical_codes = _generate_mantissa_codes(shape, element_format, is_fp4, rng)
    scale_shape = list(shape)
    scale_shape[group_axis] //= GROUP_SIZE
    e8m0 = rng.integers(126, 130, size=scale_shape, dtype=np.uint8)
    if is_fp4:
        mantissa = _FP4_VALUES[element_format][logical_codes]
        storage_codes = _pack_fp4(logical_codes)
    else:
        mantissa = _decode_fp8(logical_codes, element_format)
        storage_codes = logical_codes
    scale = np.exp2(e8m0.astype(np.int16) - 127).astype(np.float32)
    values = (mantissa * np.repeat(scale, GROUP_SIZE, axis=group_axis)).astype(np.float32)
    return storage_codes, e8m0, values


def _pack_mx_scale_axis(e8m0, group_axis):
    """Return the real physical MX shape [..., G/2, ..., 2]."""
    scale = np.asarray(e8m0, dtype=np.uint8)
    group_axis %= scale.ndim
    groups = scale.shape[group_axis]
    assert groups % 2 == 0
    split_shape = list(scale.shape)
    split_shape[group_axis:group_axis + 1] = [groups // 2, 2]
    split = scale.reshape(split_shape)
    phase_axis = group_axis + 1
    physical_axes = [axis for axis in range(split.ndim) if axis != phase_axis] + [phase_axis]
    packed = np.ascontiguousarray(split.transpose(physical_axes))
    return packed


def _mantissa_storage(codes, logical_shape, is_fp4, transpose):
    if not transpose:
        return codes
    if is_fp4:
        return _pack_fp4(_unpack_fp4(codes, logical_shape).T)
    return codes.T


def _make_mx_operand_pair(a_dtype_label, b_dtype_label, m, k, n, device, *,
                          transpose_a=False, direct_b_kn=False, transpose_scale_a=False):
    """Return device inputs, dequantized CPU operands, and E8M0 ranges."""
    _, a_format, a_is_fp4 = _MX_DTYPES[a_dtype_label]
    _, b_format, b_is_fp4 = _MX_DTYPES[b_dtype_label]
    assert a_is_fp4 == b_is_fp4
    a_codes, a_e8m0, a_values = _make_quantized_operand((m, k), a_format, a_is_fp4, 1, 42)
    # Generate B in the user-facing [N, K] form.  The normal kernel path
    # transposes it with order=[1, 0]; direct_b_kn explicitly materializes the
    # already-transposed [K, N] alternative.
    b_codes, b_e8m0, b_values_nk = _make_quantized_operand((n, k), b_format, b_is_fp4, 1, 43)
    a_storage = _mantissa_storage(a_codes, (m, k), a_is_fp4, transpose_a)
    b_storage = _mantissa_storage(b_codes, (n, k), b_is_fp4, direct_b_kn)

    if a_is_fp4:
        a_dev = torch.from_numpy(np.ascontiguousarray(a_storage)).to(device)
        b_dev = torch.from_numpy(np.ascontiguousarray(b_storage)).to(device)
    else:
        a_dev = _fp8_tensor_from_codes(a_storage, a_format, device)
        b_dev = _fp8_tensor_from_codes(b_storage, b_format, device)

    # MX scale GlobalTensor currently keeps the full 64x2 / 2x64 access window.
    # Pad only scale buffers so the final M/N tail load remains in bounds.
    scale_m = -(-m // TILE) * TILE
    scale_n = -(-n // TILE) * TILE
    scale_a_logical = np.full((scale_m, k // GROUP_SIZE), 127, dtype=np.uint8)
    scale_b_logical = np.full((k // GROUP_SIZE, scale_n), 127, dtype=np.uint8)
    scale_a_logical[:m, :] = a_e8m0
    scale_b_logical[:, :n] = b_e8m0.T
    scale_a_source = scale_a_logical.T if transpose_scale_a else scale_a_logical
    scale_a_group_axis = 0 if transpose_scale_a else 1
    scale_b_source = scale_b_logical if direct_b_kn else scale_b_logical.T
    scale_b_group_axis = 0 if direct_b_kn else 1
    scale_a = torch.from_numpy(_pack_mx_scale_axis(scale_a_source, scale_a_group_axis)).to(device)
    scale_b = torch.from_numpy(_pack_mx_scale_axis(scale_b_source, scale_b_group_axis)).to(device)
    a_golden = torch.from_numpy(np.ascontiguousarray(a_values)).to(torch.float32)
    b_golden = torch.from_numpy(np.ascontiguousarray(b_values_nk.T)).to(torch.float32)
    scale_ranges = (
        int(a_e8m0.min()), int(a_e8m0.max()),
        int(b_e8m0.min()), int(b_e8m0.max()),
    )
    return a_dev, b_dev, scale_a, scale_b, a_golden, b_golden, scale_ranges


def _check_output(got, golden, out_dtype_label, label):
    if out_dtype_label == "int8":
        # QF322B8_PRE/VQF322B8_PRE: scale, round to nearest-even, then
        # saturate to the signed 8-bit range. Small FP32 accumulation
        # differences can cross an x.5 rounding boundary, so allow one LSB
        # while requiring nearly all codes to remain bit-exact.
        expected = torch.clamp(torch.round(golden), -128, 127).to(torch.int8)
        diff = (got.to(torch.int32) - expected.to(torch.int32)).abs()
        exact = float((diff == 0).to(torch.float32).mean())
        logging.info(
            "%s: INT8 exact-match=%g, max|out-golden_int8|=%d",
            label, exact, int(diff.max())
        )
        assert exact >= 0.98
        assert int(diff.max()) <= 1
        return

    if out_dtype_label == "fp16":
        actual_fp16 = got.to(torch.float16)
        expected_fp16 = golden.to(torch.float16)
        actual_bits = actual_fp16.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        expected_bits = expected_fp16.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        actual_codes = torch.where((actual_bits & 0x8000) != 0, 0x8000 - (actual_bits & 0x7FFF),
                                   0x8000 + (actual_bits & 0x7FFF))
        expected_codes = torch.where((expected_bits & 0x8000) != 0, 0x8000 - (expected_bits & 0x7FFF),
                                     0x8000 + (expected_bits & 0x7FFF))
        ulp_diff = (actual_codes - expected_codes).abs()
        logging.info("%s: FP16 exact-match=%g, max-ULP=%d", label,
                     float((ulp_diff == 0).to(torch.float32).mean()), int(ulp_diff.max()))
        assert int(ulp_diff.max()) <= 1
        return

    expected = golden.to(torch.float32)
    actual = got.to(torch.float32)
    diff = (actual - expected).abs()
    rel_diff = diff / expected.abs().clamp_min(1e-6)
    logging.info(
        "%s: max|out-golden|=%g, max-relative-error=%g",
        label, float(diff.max()), float(rel_diff.max())
    )
    # Keep the FP32 tolerance tight: 2% could conceal a layout or scale regression.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def _run_mx(a_dtype_label, b_dtype_label, device, mkn, *, use_phase, addressing="offset",
            transpose_a=False, direct_b_kn=False, transpose_scale_a=False, multicore_atomic=False,
            scalar_scale=False, out_dtype_label="fp32"):
    m, k, n = mkn
    assert k % TILE == 0
    kernel = _make_mx_kernel(
        a_dtype_label, b_dtype_label, m, k, n, use_phase=use_phase, addressing=addressing,
        transpose_a=transpose_a, direct_b_kn=direct_b_kn, transpose_scale_a=transpose_scale_a,
        multicore_atomic=multicore_atomic,
        scalar_scale=scalar_scale, out_dtype_label=out_dtype_label
    )
    (a, b, scale_a, scale_b, a_golden, b_golden, scale_ranges) = _make_mx_operand_pair(
        a_dtype_label, b_dtype_label, m, k, n, device,
        transpose_a=transpose_a, direct_b_kn=direct_b_kn, transpose_scale_a=transpose_scale_a
    )
    _, out_torch_dtype = _OUT_DTYPES[out_dtype_label]
    out = torch.zeros((m, n), dtype=out_torch_dtype, device=device)
    if multicore_atomic:
        kernel[None, k // TILE](a, b, scale_a, scale_b, out)
    else:
        kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)

    golden = torch.matmul(a_golden, b_golden)
    if scalar_scale:
        golden = golden * _SCALAR_SCALE
    got = out.cpu()
    mode = "multicore_atomic" if multicore_atomic else ("phase" if use_phase else "no_phase")
    label = (
        f"A={a_dtype_label},B={b_dtype_label}"
        f"[{m}x{k}x{n},{mode},{addressing},B_{'KN' if direct_b_kn else 'NK'},"
        f"scaleA_{'GM' if transpose_scale_a else 'MG'},{out_dtype_label}]"
    )
    logging.info(
        "%s: E8M0 ranges A=[%d,%d], B=[%d,%d]",
        label, *scale_ranges
    )
    _check_output(got, golden, out_dtype_label, label)


def _make_scale_params(scales, device, out_dtype_label):
    scale_bits = np.asarray(scales, dtype=np.float32).view(np.uint32)
    payload = scale_bits.astype(np.int64)
    if out_dtype_label == "int8":
        payload |= 1 << 46
    params = torch.from_numpy(payload).reshape(1, -1).to(device)

    # A5 FixPipe deqTensor keeps sign + exponent + the top 10 mantissa bits
    # from each FP32 scale parameter (pto-isa masks with 0xFFFFE000).
    effective_scales = (scale_bits & np.uint32(0xFFFFE000)).view(np.float32)
    return params, effective_scales.copy()


def _run_mx_scale_tile(a_dtype_label, b_dtype_label, device, mkn, out_dtype_label):
    m, k, n = mkn
    kernel = _make_mx_scale_tile_kernel(a_dtype_label, b_dtype_label, out_dtype_label, m, k, n)
    raw_kernel = _make_mx_kernel(
        a_dtype_label, b_dtype_label, m, k, n, use_phase=False, out_dtype_label="fp32",
        case_tag=f"mx_scale_tile_raw_{out_dtype_label}"
    )
    (a, b, scale_a, scale_b, a_golden, b_golden, _) = _make_mx_operand_pair(
        a_dtype_label, b_dtype_label, m, k, n, device
    )
    scales = np.linspace(0.5, 1.5, n).astype(np.float32)
    scale_params, effective_scales = _make_scale_params(scales, device, out_dtype_label)
    _, out_torch_dtype = _OUT_DTYPES[out_dtype_label]
    out = torch.zeros((m, n), dtype=out_torch_dtype, device=device)
    raw_out = torch.zeros((m, n), dtype=torch.float32, device=device)
    raw_kernel(a, b, scale_a, scale_b, raw_out)
    kernel(a, b, scale_a, scale_b, scale_params, out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(raw_kernel)
    _assert_single_dynamic_variant(kernel)

    label = f"A={a_dtype_label},B={b_dtype_label}[{m}x{k}x{n},scale_tile,{out_dtype_label}]"
    raw_golden = torch.matmul(a_golden, b_golden)
    raw_cpu = raw_out.cpu()
    _check_output(raw_cpu, raw_golden, "fp32", label + ",raw_acc")

    # Isolate the FixPipe deqTensor path from Cube/Torch accumulation-order
    # differences: use the actual unscaled Cube FP32 result as its input golden.
    scaled_golden = raw_cpu * torch.from_numpy(effective_scales).reshape(1, n)
    _check_output(out.cpu(), scaled_golden, out_dtype_label, label)


def _make_batched_mx_operands(a_dtype_label, b_dtype_label, m, k, n, device):
    _, a_format, a_is_fp4 = _MX_DTYPES[a_dtype_label]
    _, b_format, b_is_fp4 = _MX_DTYPES[b_dtype_label]
    assert a_is_fp4 == b_is_fp4
    a_codes = []
    b_codes = []
    a_values = []
    b_values = []
    scale_m = -(-m // TILE) * TILE
    scale_n = -(-n // TILE) * TILE
    kmx = k // GROUP_SIZE
    scale_a_logical = np.full((BATCH, scale_m, HEADS, kmx), 127, dtype=np.uint8)
    scale_b_logical = np.full((BATCH, scale_n, HEADS, kmx), 127, dtype=np.uint8)

    for bi in range(BATCH):
        a_code_heads = []
        b_code_heads = []
        a_value_heads = []
        b_value_heads = []
        for hi in range(HEADS):
            a_code, a_e8m0, a_value = _make_quantized_operand(
                (m, k), a_format, a_is_fp4, 1, 42 + bi * HEADS + hi
            )
            b_code, b_e8m0, b_value_nk = _make_quantized_operand(
                (n, k), b_format, b_is_fp4, 1, 43 + bi * HEADS + hi
            )
            scale_a_logical[bi, :m, hi, :] = a_e8m0
            scale_b_logical[bi, :n, hi, :] = b_e8m0
            a_code_heads.append(a_code)
            b_code_heads.append(b_code)
            a_value_heads.append(a_value)
            b_value_heads.append(b_value_nk.T)
        a_codes.append(a_code_heads)
        b_codes.append(b_code_heads)
        a_values.append(a_value_heads)
        b_values.append(b_value_heads)

    # GM data uses [B, M, H, K] for A and [B, N, H, K] for B.  The B load
    # maps Tile [K, N] to Tensor axes [3, 1] and therefore transposes in flight.
    a_codes = np.ascontiguousarray(np.asarray(a_codes).transpose(0, 2, 1, 3))
    b_codes = np.ascontiguousarray(np.asarray(b_codes).transpose(0, 2, 1, 3))
    if a_is_fp4:
        a_dev = torch.from_numpy(a_codes).to(device)
        b_dev = torch.from_numpy(b_codes).to(device)
    else:
        a_dev = _fp8_tensor_from_codes(a_codes, a_format, device)
        b_dev = _fp8_tensor_from_codes(b_codes, b_format, device)
    scale_a = torch.from_numpy(_pack_mx_scale_axis(scale_a_logical, 3)).to(device)
    scale_b = torch.from_numpy(_pack_mx_scale_axis(scale_b_logical, 3)).to(device)
    a_golden = torch.from_numpy(np.ascontiguousarray(np.asarray(a_values))).to(torch.float32)
    b_golden = torch.from_numpy(np.ascontiguousarray(np.asarray(b_values))).to(torch.float32)
    return a_dev, b_dev, scale_a, scale_b, a_golden, b_golden


def _run_mx_batched(a_dtype_label, b_dtype_label, device, mkn):
    m, k, n = mkn
    kernel = _make_mx_batched_kernel(a_dtype_label, b_dtype_label, m, k, n)
    a, b, scale_a, scale_b, a_golden, b_golden = _make_batched_mx_operands(
        a_dtype_label, b_dtype_label, m, k, n, device
    )
    out = torch.zeros((BATCH, HEADS, m, n), dtype=torch.float32, device=device)
    kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)
    golden = torch.matmul(a_golden, b_golden)
    label = f"A={a_dtype_label},B={b_dtype_label}[{m}x{k}x{n},batched_tile_order]"
    _check_output(out.cpu(), golden, "fp32", label)


def _run_mx_insert(dtype_label, device, mkn):
    m, k, n = mkn
    _, element_format, is_fp4 = _MX_DTYPES[dtype_label]
    assert k == n
    kernel = _make_mx_insert_kernel(dtype_label, m, k, n)

    # Use an identity RHS and unit E8M0 scales. The first matmul therefore
    # reproduces A exactly, and A already consists of representable MX values.
    # The cast and insert round trip must reproduce the same matrix without a
    # separate software model for FP4/FP8 rounding.
    rng = np.random.default_rng(42)
    a_codes = _generate_mantissa_codes((m, k), element_format, is_fp4, rng)
    b_codes = np.zeros((k, n), dtype=np.uint8)
    one_code = {"e4m3": 0x38, "e5m2": 0x3C, "e2m1": 0x02, "e1m2": 0x04}[element_format]
    np.fill_diagonal(b_codes, one_code)
    if is_fp4:
        a = torch.from_numpy(_pack_fp4(a_codes)).to(device)
        b = torch.from_numpy(_pack_fp4(b_codes)).to(device)
        a_golden = torch.from_numpy(_FP4_VALUES[element_format][a_codes])
    else:
        a = _fp8_tensor_from_codes(a_codes, element_format, device)
        b = _fp8_tensor_from_codes(b_codes, element_format, device)
        a_golden = torch.from_numpy(_decode_fp8(a_codes, element_format))

    scale_m = -(-m // TILE) * TILE
    scale_a_logical = np.full((scale_m, k // GROUP_SIZE), 127, dtype=np.uint8)
    scale_b_logical = np.full((k // GROUP_SIZE, n), 127, dtype=np.uint8)
    scale_a = torch.from_numpy(_pack_mx_scale_axis(scale_a_logical, 1)).to(device)
    scale_b = torch.from_numpy(_pack_mx_scale_axis(scale_b_logical, 0)).to(device)
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    roundtrip_out = torch.zeros((m, n), dtype=torch.float32, device=device)
    kernel(a, b, scale_a, scale_b, out, roundtrip_out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)
    label = f"A={dtype_label},B={dtype_label}[{m}x{k}x{n},insert]"
    _check_output(out.cpu(), a_golden, "fp32", label)
    _check_output(roundtrip_out.cpu(), a_golden, "fp32", label + ",roundtrip")


def _run_e8m0_boundaries(device):
    """Exercise finite E8M0 endpoints without creating extreme FP32 intermediates."""
    m = n = TILE
    k = len(_E8M0_BOUNDARY_PAIRS) * GROUP_SIZE
    kernel = _make_mx_kernel(
        "mxfp8_e4m3", "mxfp8_e5m2", m, k, n, use_phase=True
    )

    # Each K group uses an exactly representable E4M3 weight. Distinct weights
    # ensure that ignoring or reordering scale groups cannot accidentally pass.
    a_group_codes = np.array([0x30, 0x38, 0x40, 0x30, 0x38, 0x40, 0x30, 0x38], dtype=np.uint8)
    a_group_values = _decode_fp8(a_group_codes, "e4m3")
    a_codes = np.repeat(a_group_codes, GROUP_SIZE).reshape(1, k).repeat(m, axis=0)
    b_codes = np.full((n, k), 0x3C, dtype=np.uint8)  # User-facing [N, K], E5M2 1.0

    scale_a_logical = np.empty((m, k // GROUP_SIZE), dtype=np.uint8)
    scale_b_logical = np.empty((n, k // GROUP_SIZE), dtype=np.uint8)
    for group, (a_exp, b_exp) in enumerate(_E8M0_BOUNDARY_PAIRS):
        scale_a_logical[:, group] = a_exp
        scale_b_logical[:, group] = b_exp

    a = _fp8_tensor_from_codes(a_codes, "e4m3", device)
    b = _fp8_tensor_from_codes(b_codes, "e5m2", device)
    scale_a = torch.from_numpy(_pack_mx_scale_axis(scale_a_logical, 1)).to(device)
    scale_b = torch.from_numpy(_pack_mx_scale_axis(scale_b_logical, 1)).to(device)
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)

    # Combine the two group exponents before applying the scale. This avoids
    # materializing 2^-127 or 2^127 operands in FP32.
    golden_value = np.float32(0.0)
    for mantissa, (a_exp, b_exp) in zip(a_group_values, _E8M0_BOUNDARY_PAIRS):
        group_scale = np.exp2(np.int16(a_exp) + np.int16(b_exp) - 254).astype(np.float32)
        golden_value += np.float32(GROUP_SIZE) * mantissa * group_scale
    golden = torch.full((m, n), float(golden_value), dtype=torch.float32)
    label = "A=mxfp8_e4m3,B=mxfp8_e5m2[64x256x64,e8m0_boundaries]"
    logging.info("%s: exponent pairs=%s", label, _E8M0_BOUNDARY_PAIRS)
    _check_output(out.cpu(), golden, "fp32", label)

    # 0xFF is the E8M0 NaN/Inf sentinel rather than a finite power of two.
    # Keep it out of the finite-value comparison and verify propagation separately.
    nan_scale_a = torch.from_numpy(
        _pack_mx_scale_axis(np.full_like(scale_a_logical, 0xFF), 1)
    ).to(device)
    unit_scale_b = torch.from_numpy(
        _pack_mx_scale_axis(np.full_like(scale_b_logical, 127), 1)
    ).to(device)
    nan_out = torch.zeros((m, n), dtype=torch.float32, device=device)
    kernel(a, b, nan_scale_a, unit_scale_b, nan_out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)
    nan_out_cpu = nan_out.cpu()
    logging.info("%s: 0xFF NaN ratio=%g", label, float(torch.isnan(nan_out_cpu).float().mean()))
    assert torch.isnan(nan_out_cpu).all()


def _run_fp8_boundaries(device):
    """Check E4M3/E5M2 zero, subnormal, normal, and maximum finite codes."""
    m = k = n = TILE
    kernel = _make_mx_kernel(
        "mxfp8_e4m3", "mxfp8_e5m2", m, k, n, use_phase=True
    )
    a_codes = np.zeros((m, k), dtype=np.uint8)
    b_codes = np.zeros((n, k), dtype=np.uint8)
    expected = np.zeros((m, n), dtype=np.float32)

    e4m3_codes = _FP8_BOUNDARY_CODES["e4m3"]
    e4m3_values = _decode_fp8(e4m3_codes, "e4m3")
    for index, (code, value) in enumerate(zip(e4m3_codes, e4m3_values)):
        # One nonzero path per output: E4M3 boundary * E5M2 1.0.
        a_codes[index, index] = code
        b_codes[index, index] = 0x3C
        expected[index, index] = value

    e5m2_codes = _FP8_BOUNDARY_CODES["e5m2"]
    e5m2_values = _decode_fp8(e5m2_codes, "e5m2")
    start = 16
    for offset, (code, value) in enumerate(zip(e5m2_codes, e5m2_values)):
        index = start + offset
        # A disjoint path checks E4M3 1.0 * each E5M2 boundary.
        a_codes[index, index] = 0x38
        b_codes[index, index] = code
        expected[index, index] = value

    scale_a_logical = np.full((m, k // GROUP_SIZE), 127, dtype=np.uint8)
    scale_b_logical = np.full((n, k // GROUP_SIZE), 127, dtype=np.uint8)
    a = _fp8_tensor_from_codes(a_codes, "e4m3", device)
    b = _fp8_tensor_from_codes(b_codes, "e5m2", device)
    scale_a = torch.from_numpy(_pack_mx_scale_axis(scale_a_logical, 1)).to(device)
    scale_b = torch.from_numpy(_pack_mx_scale_axis(scale_b_logical, 1)).to(device)
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)

    actual = out.cpu()
    golden = torch.from_numpy(expected)
    label = "A=mxfp8_e4m3,B=mxfp8_e5m2[64x64x64,fp8_boundaries]"
    diff = (actual - golden).abs()
    logging.info("%s: max|out-golden|=%g", label, float(diff.max()))
    torch.testing.assert_close(actual, golden, rtol=0.0, atol=0.0)


def _run_mx_scale_mg_to_ub(device):
    rows, cols = 73, 6
    kernel = _make_mx_scale_ub_kernel(rows, cols)
    physical_codes = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols // 2, 2).astype(np.uint8)
    scale = torch.from_numpy(physical_codes).to(device)
    out = torch.zeros_like(scale)
    kernel(scale, out)
    torch.npu.synchronize()
    _assert_single_dynamic_variant(kernel)
    torch.testing.assert_close(out.cpu(), torch.from_numpy(physical_codes), rtol=0.0, atol=0.0)


@pytest.fixture(scope="module")
def _device():
    _require_a5(ST_DEVICE)
    return ST_DEVICE


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "a_dtype_label,b_dtype_label,shape_id,mkn",
    _T01_CASES,
    ids=[f"{_combo_id((item[0], item[1]))}__{item[2]}" for item in _T01_CASES],
)
def test_t01_mx_mnk_phase(a_dtype_label, b_dtype_label, shape_id, mkn, _device):
    """Three K rounds with Partial/Final phase and M/N tail coverage."""
    _run_mx(a_dtype_label, b_dtype_label, _device, mkn, use_phase=True)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,mkn", _FEATURE_MKN_VARIANTS,
                         ids=[item[0] for item in _FEATURE_MKN_VARIANTS])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t02_mx_mnk_no_phase(a_dtype_label, b_dtype_label, shape_id, mkn, _device):
    """Three K rounds through matmul_mx/matmul_mx_acc without phase annotations."""
    _run_mx(a_dtype_label, b_dtype_label, _device, mkn, use_phase=False)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,mkn", _FEATURE_MKN_VARIANTS,
                         ids=[item[0] for item in _FEATURE_MKN_VARIANTS])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t03_mx_multicore_atomic(a_dtype_label, b_dtype_label, shape_id, mkn, _device):
    """Three AI cores split K and atomically accumulate their partial products in GM."""
    _run_mx(
        a_dtype_label, b_dtype_label, _device, mkn,
        use_phase=False, multicore_atomic=True
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,m,k,n", _TT_SHAPES, ids=[item[0] for item in _TT_SHAPES])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t04_mx_direct_b_kn(a_dtype_label, b_dtype_label, shape_id, m, k, n, _device):
    """B is materialized directly as [K, N] instead of using the default [N, K] transpose load."""
    _run_mx(
        a_dtype_label, b_dtype_label, _device, (m, k, n), use_phase=True,
        direct_b_kn=True
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,mkn", _FEATURE_MKN_VARIANTS,
                         ids=[item[0] for item in _FEATURE_MKN_VARIANTS])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t05_mx_scale_a_gm_order(a_dtype_label, b_dtype_label, shape_id, mkn, _device):
    """A scale uses physical [G/2, M, 2] storage and order=[1, 0] for the ZZ view."""
    _run_mx(
        a_dtype_label, b_dtype_label, _device, mkn, use_phase=True,
        transpose_scale_a=True
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("out_dtype_label", _QUANT_OUTS)
@pytest.mark.parametrize("shape_id,mkn", _MKN_VARIANTS, ids=[item[0] for item in _MKN_VARIANTS])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t06_mx_scale_scalar(a_dtype_label, b_dtype_label, shape_id, mkn, out_dtype_label, _device):
    """Non-unit per-tensor scalar scale store to FP32/FP16/INT8."""
    _run_mx(
        a_dtype_label, b_dtype_label, _device, mkn, use_phase=True,
        scalar_scale=True, out_dtype_label=out_dtype_label
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("out_dtype_label", _QUANT_OUTS)
@pytest.mark.parametrize("shape_id,mkn", _MKN_VARIANTS, ids=[item[0] for item in _MKN_VARIANTS])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t07_mx_scale_tile(a_dtype_label, b_dtype_label, shape_id, mkn, out_dtype_label, _device):
    """Per-column scale Tile store to FP32/FP16/INT8."""
    _run_mx_scale_tile(a_dtype_label, b_dtype_label, _device, mkn, out_dtype_label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,mkn", _FEATURE_MKN_VARIANTS,
                         ids=[item[0] for item in _FEATURE_MKN_VARIANTS])
@pytest.mark.parametrize("case", _MX_ADDRESSING_CASES, ids=[case[0] for case in _MX_ADDRESSING_CASES])
def test_t08_mx_addressing_variants(case, shape_id, mkn, _device):
    """Cover tile, transposing-tile, and FP8 mixed element/tile addressing."""
    _, a_dtype_label, b_dtype_label, addressing, transpose_a = case
    _run_mx(
        a_dtype_label, b_dtype_label, _device, mkn, use_phase=True,
        addressing=addressing, transpose_a=transpose_a
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,mkn", _FEATURE_MKN_VARIANTS,
                         ids=[item[0] for item in _FEATURE_MKN_VARIANTS])
@pytest.mark.parametrize("a_dtype_label,b_dtype_label", _MX_REP_COMBOS,
                         ids=[_combo_id(c) for c in _MX_REP_COMBOS])
def test_t09_mx_batched_tile_order(a_dtype_label, b_dtype_label, shape_id, mkn, _device):
    """4-D A/B select axes [1, 3]/[3, 1]; packed FP4 uses explicit byte offsets."""
    _run_mx_batched(a_dtype_label, b_dtype_label, _device, mkn)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,mkn", _INSERT_MKN_VARIANTS,
                         ids=[item[0] for item in _INSERT_MKN_VARIANTS])
@pytest.mark.parametrize("dtype_label", _MX_INSERT_DTYPES)
def test_t10_mx_insert(dtype_label, shape_id, mkn, _device):
    """ACC-UB-cast-L1 insert round trip with an uneven two-subcore M tail."""
    _run_mx_insert(dtype_label, _device, mkn)


@pytest.mark.soc("950")
def test_t11_mx_e8m0_boundaries(_device):
    """Finite E8M0 scale-encoding boundaries."""
    _run_e8m0_boundaries(_device)


@pytest.mark.soc("950")
def test_t12_mx_fp8_boundaries(_device):
    """E4M3/E5M2 data-encoding boundaries."""
    _run_fp8_boundaries(_device)


@pytest.mark.soc("950")
def test_t13_mx_scale_mg_to_ub(_device):
    """Copy physical [M, G/2, 2] scale storage through its 2-D UB [M, G] view."""
    _run_mx_scale_mg_to_ub(_device)


if __name__ == "__main__":
    _require_a5(ST_DEVICE)
    _cases = []

    # t01 owns the exhaustive dtype-direction matrix.
    for _a_dtype, _b_dtype, _shape_id, _shape in _T01_CASES:
        _cases.append((
            f"t01_phase[A={_a_dtype},B={_b_dtype},{_shape_id}]",
            _run_mx, (_a_dtype, _b_dtype, ST_DEVICE, _shape), dict(use_phase=True)
        ))

    # All other feature paths use one mixed pair per MX bit width.
    for _a_dtype, _b_dtype in _MX_REP_COMBOS:
        for _shape_id, _shape in _FEATURE_MKN_VARIANTS:
            for _tag, _kwargs in (
                ("t02_no_phase", dict(use_phase=False)),
                ("t03_multicore_atomic", dict(use_phase=False, multicore_atomic=True)),
            ):
                _cases.append((
                    f"{_tag}[A={_a_dtype},B={_b_dtype},{_shape_id}]",
                    _run_mx, (_a_dtype, _b_dtype, ST_DEVICE, _shape), _kwargs
                ))
            _cases.append((
                f"t09_batched_tile_order[A={_a_dtype},B={_b_dtype},{_shape_id}]",
                _run_mx_batched, (_a_dtype, _b_dtype, ST_DEVICE, _shape), {}
            ))
        for _shape_id, _shape in _MKN_VARIANTS:
            for _out_dtype in _QUANT_OUTS:
                _cases.append((
                    f"t06_scale_scalar[A={_a_dtype},B={_b_dtype},out={_out_dtype},{_shape_id}]",
                    _run_mx, (_a_dtype, _b_dtype, ST_DEVICE, _shape),
                    dict(use_phase=True, scalar_scale=True, out_dtype_label=_out_dtype)
                ))
                _cases.append((
                    f"t07_scale_tile[A={_a_dtype},B={_b_dtype},out={_out_dtype},{_shape_id}]",
                    _run_mx_scale_tile,
                    (_a_dtype, _b_dtype, ST_DEVICE, _shape, _out_dtype), {}
                ))
        for _shape_id, _m, _k, _n in _TT_SHAPES:
            _cases.append((
                f"t04_direct_b_kn[A={_a_dtype},B={_b_dtype},{_shape_id}]",
                _run_mx, (_a_dtype, _b_dtype, ST_DEVICE, (_m, _k, _n)),
                dict(use_phase=True, direct_b_kn=True)
            ))
        for _shape_id, _shape in _FEATURE_MKN_VARIANTS:
            _cases.append((
                f"t05_scale_a_gm_order[A={_a_dtype},B={_b_dtype},{_shape_id}]",
                _run_mx, (_a_dtype, _b_dtype, ST_DEVICE, _shape),
                dict(use_phase=True, transpose_scale_a=True)
            ))

    for _addressing_id, _a_dtype, _b_dtype, _addressing, _transpose_a in _MX_ADDRESSING_CASES:
        for _shape_id, _shape in _FEATURE_MKN_VARIANTS:
            _cases.append((
                f"t08_addressing_{_addressing_id}[A={_a_dtype},B={_b_dtype},{_shape_id}]",
                _run_mx, (_a_dtype, _b_dtype, ST_DEVICE, _shape),
                dict(use_phase=True, addressing=_addressing, transpose_a=_transpose_a)
            ))

    for _dtype in _MX_INSERT_DTYPES:
        for _shape_id, _shape in _INSERT_MKN_VARIANTS:
            _cases.append((
                f"t10_insert[A={_dtype},B={_dtype},{_shape_id}]",
                _run_mx_insert, (_dtype, ST_DEVICE, _shape), {}
            ))

    _cases.extend((
        ("t11_e8m0_boundaries", _run_e8m0_boundaries, (ST_DEVICE,), {}),
        ("t12_fp8_boundaries", _run_fp8_boundaries, (ST_DEVICE,), {}),
        ("t13_scale_mg_to_ub", _run_mx_scale_mg_to_ub, (ST_DEVICE,), {}),
    ))

    logging.info("MX datatype-generic matmul: %d cases", len(_cases))
    _passed = 0
    _failed = 0
    for _case_name, _runner, _args, _kwargs in _cases:
        try:
            _runner(*_args, **_kwargs)
            _passed += 1
            logging.info("PASS: %s", _case_name)
        except Exception as _exc:  # keep running so one dtype does not hide later coverage
            _failed += 1
            logging.error("FAIL: %s -- %s", _case_name, _exc)
    logging.info("Results: %d passed, %d failed, %d total", _passed, _failed, len(_cases))
