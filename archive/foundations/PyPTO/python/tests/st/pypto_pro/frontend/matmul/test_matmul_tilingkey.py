#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: F821
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 matmul specialization through parser-side tiling-key constant propagation.

``Dtype`` selects the input tile dtype (0: FP16, 1: BF16), while ``TileBlock``
selects the M/N block size (0: 64, 1: 128).  The reduction dimension stays at
128, so both concrete keys execute the same load/move/matmul/store flow.  The
64 key emits four M/N output blocks; the 128 key emits one.
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


M_DIM = 1024
N_DIM = 1024
D_DIM = 128


class MatmulTilingKey:
    """Compile-time choices consumed by ``matmul_tilingkey``."""

    Dtype = TilingKeyField(bits=1, values=[0, 1])
    TileBlock = TilingKeyField(bits=1, values=[0, 1])


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


@pl.jit(auto_mutex=True, tiling_key=MatmulTilingKey, compile_timeout=300)
def matmul_tilingkey(
    a_ptr: pl.Ptr[pl.DT_UINT8],
    b_ptr: pl.Ptr[pl.DT_UINT8],
    out_ptr: pl.Ptr[pl.DT_UINT8],
):
    # Both choices must be parser-side constants before their values are used
    # in TileType.  This is the behavior under test.
    if Dtype == 0:
        io_dtype = pl.DT_FP16
    else:
        io_dtype = pl.DT_BF16

    if TileBlock == 0:
        mn_tile_size = 64
    else:
        mn_tile_size = 128

    a_tensor_shape = [M_DIM, D_DIM]
    b_tensor_shape = [D_DIM, N_DIM]
    out_tensor_shape = [M_DIM, N_DIM]
    a_tile_shape = [mn_tile_size, D_DIM]
    b_tile_shape = [D_DIM, mn_tile_size]
    out_tile_shape = [mn_tile_size, mn_tile_size]

    a = pl.make_tensor(a_ptr, a_tensor_shape, dtype=io_dtype)
    b = pl.make_tensor(b_ptr, b_tensor_shape, dtype=io_dtype)
    out = pl.make_tensor(out_ptr, out_tensor_shape, dtype=pl.DT_FP32)

    with pl.section_cube():
        a_mat_db = pl.make_tile_group(
            type=pl.TileType(shape=a_tile_shape, dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000,
            mutex_ids=[0, 1],
        )
        b_mat_db = pl.make_tile_group(
            type=pl.TileType(shape=b_tile_shape, dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=a_tile_shape, dtype=io_dtype, target_memory=pl.MemorySpace.Left),
            addrs=0x00000,
            mutex_ids=[4, 5],
        )
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=b_tile_shape, dtype=io_dtype, target_memory=pl.MemorySpace.Right),
            addrs=0x00000,
            mutex_ids=[6, 7],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=out_tile_shape, dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0x00000,
            mutex_ids=[8, 9],
        )

        m_tiles = M_DIM // mn_tile_size
        n_tiles = N_DIM // mn_tile_size
        for m_tile_idx in pl.range(0, m_tiles):
            for n_tile_idx in pl.range(0, n_tiles):
                a_mat = a_mat_db.next()
                b_mat = b_mat_db.next()
                a_left = a_left_db.next()
                b_right = b_right_db.next()
                acc = acc_db.next()
                pl.load_tile(a_mat, a, [m_tile_idx, 0])
                pl.load_tile(b_mat, b, [0, n_tile_idx])
                pl.move(a_left, a_mat)
                pl.move(b_right, b_mat)
                pl.matmul(acc, a_left, b_right)
                pl.store_tile(out, acc, [m_tile_idx, n_tile_idx])


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("dtype_key", "tile_key", "torch_dtype", "rtol", "atol"),
    [
        (0, 0, torch.float16, 1e-2, 1e-2),
        (0, 1, torch.float16, 1e-2, 1e-2),
        (1, 0, torch.bfloat16, 5e-2, 5e-2),
        (1, 1, torch.bfloat16, 5e-2, 5e-2),
    ],
)
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_tilingkey(dtype_key, tile_key, torch_dtype, rtol, atol):
    """All dtype/block key combinations use the matching specialized TileTypes."""
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(42)
    a = torch.randn(M_DIM, D_DIM, device=device, dtype=torch_dtype)
    b = torch.randn(D_DIM, N_DIM, device=device, dtype=torch_dtype)
    out = torch.zeros(M_DIM, N_DIM, device=device, dtype=torch.float32)

    matmul_tilingkey[None, 1, {"Dtype": dtype_key, "TileBlock": tile_key}](a, b, out)
    torch.npu.synchronize()

    golden = torch.matmul(a.float(), b.float())
    max_diff = (out - golden).abs().max().item()
    logging.info(
        "Dtype=%d TileBlock=%d: max|out - A@B| = %.6f",
        dtype_key,
        tile_key,
        max_diff,
    )
    torch.testing.assert_close(out, golden, rtol=rtol, atol=atol)


logging.basicConfig(level=logging.INFO, format="%(message)s")
