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

"""Dynamic-rank element-wise op on Ascend A5 (rank 2..4, all inputs same shape).

Unlike ``test_add.py`` (dynamic shape, *fixed* rank 2), this kernel accepts the
inputs as raw pointers (``pl.Ptr``) and receives the logical shape through a
tiling struct, so a single compiled kernel handles rank-2, rank-3 and rank-4
inputs.

The element-wise operation (add / sub / mul) is also selected at runtime from a
tiling array element ``opkind[4]`` (mirroring ``OpTiling.opkind`` in
``test_tiling_op.py``): 0 -> add, 1 -> sub, otherwise -> mul. The same compiled
kernel therefore serves every (rank, op) combination.

Key idea: element-wise add is *rank-agnostic*. Regardless of the logical rank,
the result only depends on the flat element order, so the kernel collapses the
logical shape to a 2D matrix ``[M, N]`` and tiles it with square
``TILE_M x TILE_N`` (128 x 128) blocks:

  - ``N`` is the innermost (last) dim and ``M`` the product of the rest. The host
    pads the shape to length 4 with *leading* 1s (``AddTiling.shape``), so
    ``N = shape[3]`` and ``M = shape[0]*shape[1]*shape[2]`` for any rank in
    [2, 4] without branching on ``tiling.rank``.
  - ``pl.make_tensor`` re-views each flat buffer as 2D ``[M, N]`` so we can reuse
    the tile-indexed ``load_tile``/``store_tile`` from ``test_add.py``.

Arbitrary ``M``/``N`` (e.g. 513 x 513, neither a multiple of 128) is supported by
rounding the tile counts up with ``ceildiv`` and narrowing each edge tile's valid
window with ``pl.set_validshape``. With a 128 x 128 tile and ``513 = 4*128 + 1``,
the four tail kinds are exactly:

  - interior tiles    -> 128 x 128
  - right-edge tiles  -> 128 x 1   (full rows, partial cols)
  - bottom-edge tiles -> 1   x 128 (partial rows, full cols)
  - the corner tile   -> 1   x 1   (both partial)

Either coding style works: a *nested* ``if`` that calls ``set_validshape``
directly in each of the four branches, or a flat form that computes
``valid_rows``/``valid_cols`` in Python (defaulting to the full ``TILE`` size and
clamping each down to the axis remainder on the last row / column tile) and then
issues a single ``set_validshape``. Both are correct because the backend now
derives the GlobalTensor ``SetShape`` for each load/store from the tile's runtime
valid window (``tile.GetValidRow()``/``GetValidCol()``) rather than from the
emit-order ``set_validshape`` argument expressions -- so the row count the TLOAD
uses always matches the branch the runtime actually took. (The nested form is
used below; the flat-clamp equivalent is kept commented out for reference.)

Uses make_tile_group with auto_mutex for synchronization (no manual sync).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

MAX_RANK = 4
TILE_M = 128
TILE_N = 128


@dataclass
class AddTiling:
    shape: int[4]  # per-dim sizes; unused *leading* dims are 1
    opkind: int[8]  # operation selector array; real value at opkind[4]


@pl.jit(auto_mutex=True)
def add_dynrank_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    # Collapse the (rank 2..4) logical shape to a 2D matrix [M, N]. Leading dims
    # are 1 on the host, so N is the innermost dim and M the product of the rest,
    # for any rank without branching on tiling.rank.
    n = tiling.shape[3]
    m = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]

    tensor_x = pl.make_tensor(x, [m, n])
    tensor_y = pl.make_tensor(y, [m, n])
    tensor_z = pl.make_tensor(z, [m, n])

    # valid_shape=[-1, -1] makes the per-tile valid window dynamic (set at runtime
    # via set_validshape) so edge tiles can process a partial [rows, cols] block.
    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        # ceildiv rounds the per-axis tile counts up so edge tiles are covered.
        m_tiles = (m + TILE_M - 1) // TILE_M
        n_tiles = (n + TILE_N - 1) // TILE_N
        total_tiles = m_tiles * n_tiles

        # Each core strides over the flattened (row-tile, col-tile) grid.
        for idx in pl.range(core_id, total_tiles, num_cores):
            i = idx // n_tiles  # row-tile index
            j = idx % n_tiles  # col-tile index
            tile_a = a_db.next()
            tile_b = b_db.next()
            tile_c = c_db.next()

            # Per-axis live extent of this tile. rem_r / rem_c are < TILE only for
            # the last row / column tile; the nested branch picks one of the four
            # rectangular valid shapes (128x128, 128xrem_c, rem_rx128, rem_rxrem_c).
            rem_r = m - i * TILE_M
            rem_c = n - j * TILE_N
            if rem_r >= TILE_M:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [TILE_M, TILE_N])
                    pl.set_validshape(tile_b, [TILE_M, TILE_N])
                    pl.set_validshape(tile_c, [TILE_M, TILE_N])
                else:
                    pl.set_validshape(tile_a, [TILE_M, rem_c])
                    pl.set_validshape(tile_b, [TILE_M, rem_c])
                    pl.set_validshape(tile_c, [TILE_M, rem_c])
            else:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [rem_r, TILE_N])
                    pl.set_validshape(tile_b, [rem_r, TILE_N])
                    pl.set_validshape(tile_c, [rem_r, TILE_N])
                else:
                    pl.set_validshape(tile_a, [rem_r, rem_c])
                    pl.set_validshape(tile_b, [rem_r, rem_c])
                    pl.set_validshape(tile_c, [rem_r, rem_c])

            pl.load_tile(tile_a, tensor_x, [i, j])
            pl.load_tile(tile_b, tensor_y, [i, j])

            # Runtime op selection driven by the tiling array element (opkind[4]):
            # 0 -> add, 1 -> sub, otherwise -> mul.
            if tiling.opkind[4] == 0:
                pl.add(tile_c, tile_a, tile_b)
            elif tiling.opkind[4] == 1:
                pl.sub(tile_c, tile_a, tile_b)
            else:
                pl.mul(tile_c, tile_a, tile_b)

            pl.store_tile(tensor_z, tile_c, [i, j])


# Operation selector value -> (reference fn, name). opkind[4] holds the selector.
OP_CASES = [
    (0, lambda a, b:a + b, "add"),
    (1, lambda a, b:a - b, "sub"),
    (2, lambda a, b:a * b, "mul"),
]


def ceildiv(a, b):
    """Round-up integer division, a.k.a. ``-(-a // b)`` — mirrors the in-kernel
    ``(a + b - 1) // b`` so host and device agree on the tile count.
    """
    return (a + b - 1) // b


def _run_case(shape, opkind, ref_fn, op_name):
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16

    rank = len(shape)
    assert 2 <= rank <= MAX_RANK, f"rank must be in [2, {MAX_RANK}], got {rank}"

    numel = 1
    for s in shape:
        numel *= s
    # No alignment requirement: arbitrary numel is handled via a partial tail tile.

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    # Pad the shape to length 4 with *leading* 1s so the innermost dim lands at
    # shape[3] (= N) and the rest collapse into M = shape[0]*shape[1]*shape[2].
    dims = [1] * MAX_RANK
    for i in range(rank):
        dims[MAX_RANK - rank + i] = shape[i]
    # Operation selector lives at opkind[4] (mirrors test_tiling_op.py).
    opkind_arr = [0] * 8
    opkind_arr[4] = opkind
    tiling = AddTiling(shape=dims, opkind=opkind_arr)

    # 2D collapse: N = innermost dim, M = product of the rest. Tile counts round up.
    n = shape[-1]
    m = numel // n
    total_tiles = ceildiv(m, TILE_M) * ceildiv(n, TILE_N)
    num_cores = min(32, total_tiles)

    add_dynrank_kernel[None, num_cores](x, y, z, tiling)
    torch.npu.synchronize()

    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("dynamic-rank %s %s (rank=%d, numel=%d) passed!", op_name, list(shape), rank, numel)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add_dynamic_rank():
    shapes = [
        [512, 512],  # rank 2 -> M=512,  N=512   -> 4x4 full tiles (no tail)
        [8, 256, 256],  # rank 3 -> M=2048, N=256   -> exact tiles (no tail)
        [2, 4, 256, 256],  # rank 4 -> M=2048, N=256   -> exact tiles (no tail)
        # --- unaligned shapes: exercise the partial edge / corner tiles ---
        [513, 513],  # rank 2 -> M=513,  N=513   -> 128x1, 1x128, 1x1 tails
        [513, 511],  # rank 2 -> M=513,  N=511   -> 128x127, 1x128, 1x127 tails
        [200, 300],  # rank 2 -> M=200,  N=300   -> mixed partial rows & cols
        [2, 3, 513],  # rank 3 -> M=6,    N=513   -> 6x128 body + 6x1 col tail
        [2, 2, 3, 200],  # rank 4 -> M=12,   N=200   -> 12x128 body + 12x72 col tail
    ]
    for shape in shapes:
        for opkind, ref_fn, op_name in OP_CASES:
            _run_case(shape, opkind, ref_fn, op_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_add_dynamic_rank()
    logging.info("\nAll tests passed!")
