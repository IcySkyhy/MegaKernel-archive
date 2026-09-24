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

"""Row-wise softmax with **tile ops** + make_tile_group -- FULLY DYNAMIC (rows AND cols), multicore.

A single compiled kernel serves any ``[ROWS, N]`` fp32 matrix: **both** dims are
``pl.DYNAMIC``.

* Rows are tiled by ``TILE_ROWS`` and the row-tiles are spread across all vector
  cores (``pl.get_block_idx()`` strides by ``pl.get_block_num()``), so larger row
  counts scale out over the cores.  The last row-tile of a core may be partial, so
  its valid window is narrowed with ``pl.set_validshape([valid_rows, N])``.

* ``N`` (the reduce axis) is dynamic.  Instead of hand-managing 64-lane VF
  registers and per-register masks, the reduction over ``N`` is expressed with the
  block-level *tile* reductions ``pl.row_max`` / ``pl.row_sum`` and the row-broadcast
  ops ``pl.row_expand_sub`` / ``pl.row_expand_div``.  These operate over the tile's
  runtime ``valid_shape`` (set via ``set_validshape``), so a single kernel handles
  any ``N`` without register/mask bookkeeping.

Numerics (per row):
    m   = row_max(x)            # max across the N valid columns
    e   = exp(x - m)
    s   = row_sum(e)            # sum across the N valid columns
    y   = e / s
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

SLOT_BYTES = TILE_ROWS * MAX_N * 4  # fp32 [TILE_ROWS, MAX_N]
RED_BYTES = 512  # fp32 [TILE_ROWS, 1] reduction result (padded/aligned)

# UB addresses: double-buffered input / output / workspace groups + reduction result.
VA_IN0 = 0
VA_IN1 = VA_IN0 + SLOT_BYTES
VA_OUT0 = VA_IN1 + SLOT_BYTES
VA_OUT1 = VA_OUT0 + SLOT_BYTES
VA_TMP0 = VA_OUT1 + SLOT_BYTES
VA_TMP1 = VA_TMP0 + SLOT_BYTES
VA_RED0 = VA_TMP1 + SLOT_BYTES
VA_RED1 = VA_RED0 + RED_BYTES


@pl.jit(auto_mutex=True)
def softmax_tile_group_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    # valid_shape=[-1, -1] makes the per-tile valid window dynamic (set at runtime via
    # set_validshape): the tail row-tile carries fewer rows and N narrows the columns.
    tile_type = pl.TileType(
        shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    # Row reductions write a [TILE_ROWS, 1] column vector (layout=pl.DN).
    red_type = pl.TileType(
        shape=[TILE_ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    tmp_group = pl.make_tile_group(type=tile_type, addrs=[VA_TMP0, VA_TMP1], mutex_ids=[4, 5])
    red_group = pl.make_tile_group(type=red_type, addrs=[VA_RED0, VA_RED1], mutex_ids=[6, 7])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        # Each core strides over the row-tile grid; the tail row-tile is partial.
        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])

            out_slot = out_group.next()
            tmp_slot = tmp_group.next()
            red_slot = red_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            pl.set_validshape(tmp_slot, [valid_rows, cols])
            pl.set_validshape(red_slot, [valid_rows, 1])

            # ---- pass 1: row max ----
            pl.row_max(red_slot, in_slot, tmp_slot)  # red = max over N valid cols
            pl.row_expand_sub(out_slot, in_slot, red_slot)  # out = x - max (row broadcast)

            # ---- pass 2: exp then row sum ----
            pl.exp(out_slot, out_slot)  # out = exp(x - max)
            pl.row_sum(red_slot, out_slot, tmp_slot)  # red = sum over N valid cols

            # ---- pass 3: normalize ----
            pl.row_expand_div(out_slot, out_slot, red_slot)  # out = exp(x - max) / sum

            pl.store(y, out_slot, [row_off, 0])

    return


def _run_case(rows, cols):
    assert cols <= MAX_N, f"cols {cols} exceeds MAX_N {MAX_N}"
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    logging.info("\n=== Test Softmax dynamic rows=%s cols=%s (make_tile_group + tile ops, multicore) ===", rows, cols)
    x = torch.rand([rows, cols], device=device, dtype=torch.float32) * 8.0 - 4.0
    y = torch.empty([rows, cols], device=device, dtype=torch.float32)

    num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, num_tiles)
    softmax_tile_group_kernel[None, num_cores](x, y)
    torch.npu.synchronize()

    golden = torch.softmax(x.cpu().float(), dim=-1)
    npu = y.cpu().float()

    diff = (npu - golden).abs().max().item()
    logging.info("  cores=%s  max abs diff: %.3e", num_cores, diff)
    torch.testing.assert_close(npu, golden, rtol=1e-3, atol=1e-3)
    logging.info("  PASS")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_softmax_tile_group_vf():
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
    logging.info("Softmax make_tile_group tile-op Test (dynamic rows + dynamic cols, multicore)")
    logging.info("%s", "=" * 60)
    test_softmax_tile_group_vf()
    logging.info("\nTest completed!")
