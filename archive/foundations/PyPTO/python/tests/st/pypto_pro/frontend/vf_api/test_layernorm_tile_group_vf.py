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

"""Row-wise LayerNorm with the VF API + make_tile_group -- FULLY DYNAMIC (rows AND cols), multicore.

A single compiled kernel serves any ``[ROWS, N]`` fp32 matrix: **both** dims are
``pl.DYNAMIC``.

* Rows are tiled by ``TILE_ROWS`` and the row-tiles are spread across all vector
  cores (``pl.get_block_idx()`` strides by ``pl.get_block_num()``), so larger row
  counts scale out over the cores.  The last row-tile of a core may be partial, so
  its valid window is narrowed with ``pl.set_validshape([valid_rows, N])``.

* ``N`` (the reduce axis) is dynamic and generally spans **multiple** 64-lane fp32
  VF registers -- this is what makes the VF hard.  Each row occupies
  ``n_regs = ceil(N / 64)`` registers inside a ``[TILE_ROWS, MAX_N]`` UB tile
  (``MAX_N`` a compile-time multiple of 64, so every row is register aligned).  The
  last register of a row is only partially valid: its live lane count is
  ``min(64, N - r*64)`` and ``vf.update_mask`` builds the per-register predicate so
  padding lanes never pollute the reduction or the store.  ``gamma``/``beta`` are
  ``[1, N]`` (per column), so they too span ``n_regs`` registers.

Numerics (per row):
    mean = sum(x) / N
    xc   = x - mean
    var  = sum(xc^2) / N
    y    = xc / sqrt(var + eps) * gamma + beta
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ================================================================
# Constants
# ================================================================
LANES = 64  # fp32 VF register width (lanes)
MAX_N = 512  # max supported columns == compile-time UB tile width (mult. of LANES)
TILE_ROWS = 16  # rows processed per tile-group slot (row count is dynamic)
EPS = 1e-5

SLOT_BYTES = TILE_ROWS * MAX_N * 4  # fp32 [TILE_ROWS, MAX_N]
VEC_BYTES = MAX_N * 4  # fp32 [1, MAX_N]

# UB addresses: double-buffered in/out groups + single-slot gamma/beta groups
VA_IN0 = 0
VA_IN1 = VA_IN0 + SLOT_BYTES
VA_OUT0 = VA_IN1 + SLOT_BYTES
VA_OUT1 = VA_OUT0 + SLOT_BYTES
VA_GAMMA = VA_OUT1 + SLOT_BYTES
VA_BETA = VA_GAMMA + VEC_BYTES


@pl.vector_function
def layernorm_rows_vf(in_tile, out_tile, gamma_tile, beta_tile, n_rows: pl.DT_INT64, n_cols: pl.DT_INT64):
    """LayerNorm over the ``n_cols`` columns of each of ``n_rows`` rows.

    ``n_cols`` may exceed one register, so each row is reduced across
    ``n_regs = ceil(n_cols / 64)`` registers; the tail register is masked to its
    live lane count.  Row ``m`` starts at UB element offset ``m * MAX_N`` (rows are
    register aligned because ``MAX_N`` is a multiple of ``LANES``); gamma/beta
    register ``r`` lives at element offset ``r * LANES``.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES - 1) // LANES
    n_reg_f = vf.full(n_cols, preg, dtype=pl.DT_FP32)  # broadcast (float)N for the /N divides

    for m in pl.range(0, n_rows):
        base = m * MAX_N

        # ---- pass 1: mean = sum(x) / N ----
        row_sum = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)  # live lanes in this register
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            part = vf.reduce_sum(reg, mreg)  # per-reg sum -> lane0
            row_sum = vf.add(row_sum, part, preg)  # combine into lane0 accumulator
        mean_b = vf.full(row_sum, preg)  # broadcast lane0 -> all lanes
        mean_b = vf.div(mean_b, n_reg_f, preg)  # sum / N

        # ---- pass 2: var = sum((x - mean)^2) / N ----
        var_sum = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            xc = vf.sub(reg, mean_b, mreg)  # x - mean (padding lanes masked)
            sq = vf.mul(xc, xc, mreg)
            part = vf.reduce_sum(sq, mreg)  # per-reg sum -> lane0
            var_sum = vf.add(var_sum, part, preg)  # combine into lane0 accumulator
        var_b = vf.full(var_sum, preg)  # broadcast lane0 -> all lanes
        var_b = vf.div(var_b, n_reg_f, preg)  # sum(xc^2) / N
        var_b = vf.adds(var_b, EPS, preg)  # var + eps
        std_b = vf.sqrt(var_b, preg)  # sqrt(var + eps)

        # ---- pass 3: y = (x - mean) / std * gamma + beta ----
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            gamma = vf.load_align(gamma_tile, r * LANES)  # per-column scale for this register
            beta = vf.load_align(beta_tile, r * LANES)  # per-column shift for this register
            xc = vf.sub(reg, mean_b, mreg)  # x - mean
            norm = vf.div(xc, std_b, mreg)  # (x - mean) / sqrt(var + eps)
            out = vf.mul(norm, gamma, mreg)
            out = vf.add(out, beta, mreg)  # * gamma + beta
            vf.store_align(out_tile + (base + r * LANES), out, mreg)


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
    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    gamma_group = pl.make_tile_group(type=vec_type, addrs=[VA_GAMMA], mutex_ids=[4])
    beta_group = pl.make_tile_group(type=vec_type, addrs=[VA_BETA], mutex_ids=[5])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        # gamma/beta are per-column and reused by every row-tile: load once per core.
        gamma_slot = gamma_group.next()
        pl.set_validshape(gamma_slot, [1, cols])
        pl.load(gamma_slot, gamma, [0, 0])
        beta_slot = beta_group.next()
        pl.set_validshape(beta_slot, [1, cols])
        pl.load(beta_slot, beta, [0, 0])

        # Each core strides over the row-tile grid; the tail row-tile is partial.
        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])

            out_slot = out_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            layernorm_rows_vf(in_slot, out_slot, gamma_slot, beta_slot, valid_rows, cols)

            pl.store(y, out_slot, [row_off, 0])

    return


def _run_case(rows, cols):
    assert cols <= MAX_N, f"cols {cols} exceeds MAX_N {MAX_N}"
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(0)

    logging.info(f"\n=== Test LayerNorm dynamic rows={rows} cols={cols} (make_tile_group + VF, multicore) ===")
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
    logging.info(f"  cores={num_cores}  max abs diff: {diff:.3e}")
    torch.testing.assert_close(npu, golden, rtol=1e-3, atol=1e-3)
    logging.info("  PASS")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_layernorm_tile_group_vf():
    # (rows, cols): both dynamic. Cover single-register N, multi-register N,
    # register-unaligned N (partial tail lane), partial row-tiles, and multicore.
    cases = [
        (2048, 64),  # single register per row, 128 tiles -> 32 cores
        (4096, 128),  # 2 registers per row
        (1000, 200),  # N unaligned (3 regs, tail 8 lanes) + partial row-tile
        (777, 300),  # N unaligned (5 regs, tail 44 lanes) + partial row-tile
        (100, 512),  # full MAX_N (8 registers per row)
        (2049, 100),  # odd rows + N unaligned (2 regs, tail 36 lanes)
    ]
    for rows, cols in cases:
        _run_case(rows, cols)


if __name__ == "__main__":
    logging.info("LayerNorm make_tile_group VF Test (dynamic rows + dynamic cols, multicore)")
    logging.info("=" * 60)
    test_layernorm_tile_group_vf()
    logging.info("\nTest completed!")
