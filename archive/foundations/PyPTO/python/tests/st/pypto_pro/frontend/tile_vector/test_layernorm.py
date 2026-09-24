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

"""Row-wise LayerNorm with **tile ops** + make_tile_group -- FULLY DYNAMIC (rows AND cols), multicore.

A single compiled kernel serves any ``[ROWS, N]`` fp32 matrix: **both** dims are
``pl.DYNAMIC``.

* Rows are tiled by ``TILE_ROWS`` and the row-tiles are spread across all vector
  cores (``pl.get_block_idx()`` strides by ``pl.get_block_num()``), so larger row
  counts scale out over the cores.  The last row-tile of a core may be partial, so
  its valid window is narrowed with ``pl.set_validshape([valid_rows, N])``.

* ``N`` (the reduce axis) is dynamic.  Instead of hand-managing 64-lane VF
  registers and per-register masks, the reduction over ``N`` is expressed with the
  block-level *tile* reduction ``pl.row_sum`` and the row/col broadcast ops
  (``pl.row_expand_sub`` / ``pl.row_expand_mul`` for the per-row mean/std,
  ``pl.col_expand_mul`` / ``pl.col_expand_sub`` for the per-column gamma/beta).
  These operate over the tile's runtime ``valid_shape`` (set via ``set_validshape``),
  so a single kernel handles any ``N`` without register/mask bookkeeping.
  ``gamma``/``beta`` are ``[1, N]`` (per column) and broadcast along the rows.

Numerics (per row):
    mean = sum(x) / N
    xc   = x - mean
    var  = sum(xc^2) / N
    y    = xc / sqrt(var + eps) * gamma + beta
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ================================================================
# Constants
# ================================================================
MAX_N = 512  # max supported columns == compile-time UB tile width
TILE_ROWS = 16  # rows processed per tile-group slot (row count is dynamic)
EPS = 1e-5

SLOT_BYTES = TILE_ROWS * MAX_N * 4  # fp32 [TILE_ROWS, MAX_N]
VEC_BYTES = MAX_N * 4  # fp32 [1, MAX_N]
RED_BYTES = 512  # fp32 [TILE_ROWS, 1] reduction result (padded/aligned)

# UB addresses: double-buffered in/out groups, single-slot workspace/scratch,
# reduction results, and single-slot gamma/beta/(-beta) groups.
VA_IN0 = 0
VA_IN1 = VA_IN0 + SLOT_BYTES
VA_OUT0 = VA_IN1 + SLOT_BYTES
VA_OUT1 = VA_OUT0 + SLOT_BYTES
VA_TMP = VA_OUT1 + SLOT_BYTES  # row-reduce workspace / xc^2 scratch
VA_XC = VA_TMP + SLOT_BYTES  # holds (x - mean), then the normalized result
VA_WS = VA_XC + SLOT_BYTES  # dedicated row-reduce workspace (2nd reduction)
VA_RED0 = VA_WS + SLOT_BYTES  # mean
VA_RED1 = VA_RED0 + RED_BYTES  # var + eps -> 1/std (dual-viewed)
VA_GAMMA = VA_RED1 + RED_BYTES
VA_BETA = VA_GAMMA + VEC_BYTES
VA_NEGBETA = VA_BETA + VEC_BYTES


@pl.jit(auto_mutex=True)
def layernorm_tile_group_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    gamma: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    beta: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    # valid_shape=[-1, -1] makes the per-tile valid window dynamic (set at runtime via
    # set_validshape): the tail row-tile carries fewer rows and N narrows the columns.
    tile_type = pl.TileType(
        shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    vec_type = pl.TileType(shape=[1, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    # Row reductions write a [TILE_ROWS, 1] column vector (layout=pl.DN), which is the
    # layout the row-broadcast ops (row_expand_*) consume.
    red_type = pl.TileType(
        shape=[TILE_ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    # Row-major dual view of the same [TILE_ROWS] scalars: scalar elementwise ops (div/add/
    # rsqrt) require a row-major layout, so we view the reduction memory as [1, TILE_ROWS].
    # (Running those ops on the DN view faults the device at runtime.)
    red_rm_type = pl.TileType(
        shape=[1, TILE_ROWS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )

    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    tmp_group = pl.make_tile_group(type=tile_type, addrs=[VA_TMP], mutex_ids=[4])
    xc_group = pl.make_tile_group(type=tile_type, addrs=[VA_XC], mutex_ids=[5])
    ws_group = pl.make_tile_group(type=tile_type, addrs=[VA_WS], mutex_ids=[6])
    # Each reduction tile is exposed as both a DN view (for row_expand_*) and a row-major
    # view (for the scalar div/add/rsqrt), aliasing the same address.
    red0_group = pl.make_tile_group(type=red_type, addrs=[VA_RED0], mutex_ids=[7])
    red0_rm_group = pl.make_tile_group(type=red_rm_type, addrs=[VA_RED0], mutex_ids=[8])
    red1_group = pl.make_tile_group(type=red_type, addrs=[VA_RED1], mutex_ids=[9])
    red1_rm_group = pl.make_tile_group(type=red_rm_type, addrs=[VA_RED1], mutex_ids=[10])
    gamma_group = pl.make_tile_group(type=vec_type, addrs=[VA_GAMMA], mutex_ids=[11])
    beta_group = pl.make_tile_group(type=vec_type, addrs=[VA_BETA], mutex_ids=[12])
    negbeta_group = pl.make_tile_group(type=vec_type, addrs=[VA_NEGBETA], mutex_ids=[13])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        # gamma/beta are per-column and reused by every row-tile: load once per core.
        # beta is only ever needed with a '+', so precompute -beta and add it via
        # col_expand_sub (there is no col_expand_add).
        gamma_slot = gamma_group.next()
        pl.set_validshape(gamma_slot, [1, cols])
        pl.load(gamma_slot, gamma, [0, 0])
        beta_slot = beta_group.next()
        pl.set_validshape(beta_slot, [1, cols])
        pl.load(beta_slot, beta, [0, 0])
        negbeta_slot = negbeta_group.next()
        pl.set_validshape(negbeta_slot, [1, cols])
        pl.mul(negbeta_slot, beta_slot, -1.0)

        # Each core strides over the row-tile grid; the tail row-tile is partial.
        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])

            out_slot = out_group.next()
            tmp_slot = tmp_group.next()
            xc_slot = xc_group.next()
            ws_slot = ws_group.next()
            red0_slot = red0_group.next()
            red0_rm_slot = red0_rm_group.next()
            red1_slot = red1_group.next()
            red1_rm_slot = red1_rm_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            pl.set_validshape(tmp_slot, [valid_rows, cols])
            pl.set_validshape(xc_slot, [valid_rows, cols])
            pl.set_validshape(ws_slot, [valid_rows, cols])
            pl.set_validshape(red0_slot, [valid_rows, 1])
            pl.set_validshape(red1_slot, [valid_rows, 1])

            pl.row_sum(red0_slot, in_slot, tmp_slot)  # red0 = sum(x); tmp is workspace
            pl.set_validshape(red0_rm_slot, [1, valid_rows])
            pl.div(red0_rm_slot, red0_rm_slot, cols)  # red0 = mean (scalar op on row-major view)

            pl.row_expand_sub(xc_slot, in_slot, red0_slot)  # xc = x - mean (row broadcast, DN view)

            pl.mul(tmp_slot, xc_slot, xc_slot)  # tmp = xc^2
            pl.row_sum(red1_slot, tmp_slot, ws_slot)  # red1 = sum(xc^2); ws is workspace
            pl.set_validshape(red1_rm_slot, [1, valid_rows])
            pl.div(red1_rm_slot, red1_rm_slot, cols)  # red1 = var (row-major view)
            pl.add(red1_rm_slot, red1_rm_slot, EPS)  # red1 = var + eps (row-major view)
            pl.rsqrt(red1_rm_slot, red1_rm_slot)  # 1 / sqrt(var + eps) (row-major view)

            pl.row_expand_mul(xc_slot, xc_slot, red1_slot)  # normalize: xc / std (row broadcast)
            pl.col_expand_mul(xc_slot, xc_slot, gamma_slot)  # * gamma (per-column, col broadcast)
            pl.col_expand_sub(out_slot, xc_slot, negbeta_slot)  # - (-beta) == + beta

            pl.store(y, out_slot, [row_off, 0])

    return


def _run_case(rows, cols):
    assert cols <= MAX_N, f"cols {cols} exceeds MAX_N {MAX_N}"
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    logging.info("\n=== Test LayerNorm dynamic rows=%s cols=%s (make_tile_group + tile ops, multicore) ===", rows, cols)
    x = torch.rand([rows, cols], device=device, dtype=torch.float32) * 8.0 - 4.0
    gamma = torch.rand([1, cols], device=device, dtype=torch.float32) + 0.5
    beta = torch.rand([1, cols], device=device, dtype=torch.float32) - 0.5
    y = torch.empty([rows, cols], device=device, dtype=torch.float32)

    num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, num_tiles)
    layernorm_tile_group_kernel[None, num_cores](x, gamma, beta, y)
    torch.npu.synchronize()

    xc = x.cpu().float()
    mean = xc.mean(dim=-1, keepdim=True)
    var = xc.var(dim=-1, unbiased=False, keepdim=True)
    golden = (xc - mean) / torch.sqrt(var + EPS) * gamma.cpu().float() + beta.cpu().float()
    npu = y.cpu().float()

    diff = (npu - golden).abs().max().item()
    logging.info("  cores=%s  max abs diff: %.3e", num_cores, diff)
    torch.testing.assert_close(npu, golden, rtol=1e-3, atol=1e-3)
    logging.info("  PASS")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_layernorm_tile_group_vf():
    # partial row-tiles, and multicore.
    cases = [
        (2048, 64),  # 128 tiles -> 32 cores
        (4096, 128),
        (1000, 200),  # N unaligned + partial row-tile
        (777, 300),  # N unaligned + partial row-tile
        (100, 512),  # full MAX_N
        (2049, 100),  # odd rows + N unaligned
    ]
    for rows, cols in cases:
        _run_case(rows, cols)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("LayerNorm make_tile_group tile-op Test (dynamic rows + dynamic cols, multicore)")
    logging.info("%s", "=" * 60)
    test_layernorm_tile_group_vf()
    logging.info("\nTest completed!")
