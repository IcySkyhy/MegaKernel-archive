# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Histogram op for radix sort preprocessing.

Doc: docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/histogram.md

make_tile_group + auto_mutex style. Verifies UINT16 MSB histogram
(bits 15-8) with ND/DN layout constraints.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

ROWS = 32
COLS = 128
IDX_COLS_DN = 1


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def histogram_kernel(
    src: pl.Tensor[[ROWS, COLS], pl.DT_UINT16],
    idx: pl.Tensor[[ROWS, IDX_COLS_DN], pl.DT_UINT8],
    out: pl.Tensor[[ROWS, 256], pl.DT_UINT32],
):
    pl.system.bar_all()
    tt_src = pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    tt_idx = pl.TileType(shape=[ROWS, IDX_COLS_DN], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tt_dst = pl.TileType(shape=[ROWS, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x2000, mutex_ids=[1])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x2020, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, idx, [0, 0])
        pl.histogram(cur_dst, cur_src, cur_idx, is_msb=True)
        pl.store(out, cur_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_histogram():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(42)
    src = torch.randint(0, 65536, (ROWS, COLS), device=device, dtype=torch.int32).to(torch.uint16)
    idx = torch.zeros((ROWS, IDX_COLS_DN), device=device, dtype=torch.uint8)
    out = torch.zeros((ROWS, 256), device=device, dtype=torch.int32)
    histogram_kernel(src, idx, out)
    torch.npu.synchronize()

    expected = torch.zeros((ROWS, 256), device=device, dtype=torch.int32)
    for r in range(ROWS):
        for c in range(COLS):
            msb = (src[r, c].item() >> 8) & 0xFF
            expected[r, msb] += 1
        acc = 0
        for b in range(256):
            acc += expected[r, b].item()
            expected[r, b] = acc

    torch.testing.assert_close(out, expected)
    logging.info("histogram result equal!")


def _doc_histogram(ctx, kernel):
    src = (torch.arange(ROWS * COLS, device=ctx.device, dtype=torch.int32).reshape(ROWS, COLS) * 257).to(torch.uint16)
    idx = torch.zeros((ROWS, IDX_COLS_DN), device=ctx.device, dtype=torch.uint8)
    out = torch.zeros((ROWS, 256), device=ctx.device, dtype=torch.int32)
    kernel(src, idx, out)
    ctx.synchronize()
    return ctx.snippet("histogram", {"src": src, "idx": idx}, {"out": out})


DOC_OUTPUT_CASES = {"histogram": _doc_histogram}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_histogram()
    logging.info("\nAll histogram doc examples passed!")
