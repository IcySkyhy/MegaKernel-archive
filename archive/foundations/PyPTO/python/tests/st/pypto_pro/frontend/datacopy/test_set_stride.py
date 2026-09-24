# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for pl.set_stride(tensor, stride) on A5 CCE.

pl.set_stride overrides the row/col stride of a global (GM) tensor descriptor in
place, so a single pl.load can gather rows that are non-contiguous in GM. The
stride is supplied at runtime (here read from an input tensor via pl.getval).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


LINE = 128  # elements per line (fp16)
N_LINES = 512  # total lines in the source GM tensor
N_PAIRS = 32  # number of pl.load ops, each loading a [2, LINE] tile
OUT_LINES = N_PAIRS * 2  # 64 gathered lines


def _is_a5() -> bool:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        logging.info("NPU unavailable: %s", exc)
        return False
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        logging.info("Current device is %s, not A5 (Ascend950). Skip.", name)
        return False
    return True


# ---------------------------------------------------------------------------
# Test 1: basic two-line gather using a single pl.load + pl.set_stride.
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def set_stride_basic_kernel(
    x: pl.Tensor[[N_LINES, LINE], pl.DT_FP16],
    strides: pl.Tensor[[1, 1], pl.DT_INT32],
    out: pl.Tensor[[2, LINE], pl.DT_FP16],
):
    ub_type = pl.TileType(shape=[2, LINE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub_db = pl.make_tile_group(type=ub_type, addrs=0x0000, mutex_ids=[0, 1])

    with pl.section_vector():
        s = strides[0, 0]  # element (row) stride, read from GM input
        tile = ub_db.next()
        pl.set_stride(x, [s, 1])  # override the GM row stride
        pl.load(tile, x, [0, 0])  # row0 = line 0, row1 = line 0 + s/LINE
        pl.store(out, tile, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_set_stride_basic():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(0)

    x = torch.randn(N_LINES, LINE, device=device, dtype=torch.float16)
    stride_lines = 137
    strides = torch.tensor([[stride_lines * LINE]], device=device, dtype=torch.int32)
    out = torch.zeros(2, LINE, device=device, dtype=torch.float16)

    set_stride_basic_kernel(x, strides, out)
    torch.npu.synchronize()

    expected = torch.stack([x[0], x[stride_lines]], dim=0)

    logging.info("basic set_stride: line strides = %d", stride_lines)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    logging.info("test_set_stride_basic passed.")


# ---------------------------------------------------------------------------
# Test 2: gather 64 separate lines from a [512, 128] GM tensor into UB using
# 32 pl.load operations. Each load fetches a [2, 128] tile whose two lines are
# spaced by a random, positive stride read from an input via pl.getval.
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def gather_stride_kernel(
    x: pl.Tensor[[N_LINES, LINE], pl.DT_FP16],
    strides: pl.Tensor[[1, N_PAIRS], pl.DT_INT32],
    out: pl.Tensor[[OUT_LINES, LINE], pl.DT_FP16],
):
    ub_type = pl.TileType(shape=[2, LINE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub_db = pl.make_tile_group(type=ub_type, addrs=0x0000, mutex_ids=[0, 1])

    with pl.section_vector():
        for i in pl.range(0, N_PAIRS, 1):
            s = strides[0, i]  # per-pair element (row) stride from GM input
            tile = ub_db.next()
            pl.set_stride(x, [s, 1])  # positive stride between the two loaded lines
            pl.load(tile, x, [i, 0])  # row0 = line i, row1 = line i + s/LINE
            pl.store(out, tile, [2 * i, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gather_64_lines_with_set_stride():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(1)

    x = torch.randn(N_LINES, LINE, device=device, dtype=torch.float16)

    # Random positive line strides; keep i + stride_line < N_LINES for every pair.
    stride_lines = torch.randint(1, 400, (N_PAIRS,), dtype=torch.int64)
    for i in range(N_PAIRS):
        if i + int(stride_lines[i]) >= N_LINES:
            stride_lines[i] = N_LINES - 1 - i
    stride_elems = (stride_lines * LINE).to(torch.int32).reshape(1, N_PAIRS).to(device)

    out = torch.zeros(OUT_LINES, LINE, device=device, dtype=torch.float16)

    gather_stride_kernel(x, stride_elems, out)
    torch.npu.synchronize()

    expected = torch.zeros(OUT_LINES, LINE, device=device, dtype=torch.float16)
    for i in range(N_PAIRS):
        k = int(stride_lines[i])
        expected[2 * i] = x[i]
        expected[2 * i + 1] = x[i + k]

    logging.info("gather line strides: %s", stride_lines.tolist())
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    logging.info("test_gather_64_lines_with_set_stride passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_set_stride_basic()
    test_gather_64_lines_with_set_stride()
    logging.info("\nAll set_stride tests passed!")
