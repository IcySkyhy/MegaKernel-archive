# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end test for direct SIMT GM Tensor access."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 256
GM_ROWS = 2
GM_COLS = 64
GM_DTYPE_THREADS = GM_ROWS * GM_COLS


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function(max_threads=THREADS)
def gm_add(
    dst: pl.Tensor[[1, THREADS], pl.DT_FP32],
    src: pl.Tensor[[1, THREADS], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.jit(arch="a5")
def simt_gm_add(
    x: pl.Tensor[[1, THREADS], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    with pl.section_vector():
        pl.simt.launch(gm_add, threads=THREADS, args=(out, x, n, delta))


@pl.simt.function(max_threads=GM_DTYPE_THREADS)
def gm_copy_dtypes(
    src_int8: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT8],
    src_fp16: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_FP16],
    src_int64: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT64],
    dst_int8: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT8],
    dst_fp16: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_FP16],
    dst_int64: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    row = tid // GM_COLS
    col = tid % GM_COLS
    dst_int8[row, col] = src_int8[row, col]
    dst_fp16[row, col] = src_fp16[row, col]
    dst_int64[row, col] = src_int64[row, col]


@pl.jit(arch="a5")
def simt_gm_copy_dtypes(
    src_int8: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT8],
    src_fp16: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_FP16],
    src_int64: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT64],
    dst_int8: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT8],
    dst_fp16: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_FP16],
    dst_int64: pl.Tensor[[GM_ROWS, GM_COLS], pl.DT_INT64],
):
    with pl.section_vector():
        pl.simt.launch(
            gm_copy_dtypes,
            threads=GM_DTYPE_THREADS,
            args=(src_int8, src_fp16, src_int64, dst_int8, dst_fp16, dst_int64),
        )


@pytest.mark.soc("950")
@pytest.mark.parametrize("n", [0, 193, THREADS])
def test_gm_tensor_access(n):
    _require_a5()

    delta = 1.5
    sentinel = -7.0
    x = torch.arange(THREADS, dtype=torch.float32).reshape(1, THREADS).to(ST_DEVICE)
    out = torch.full_like(x, sentinel)

    simt_gm_add(x, out, n, delta)
    torch.npu.synchronize()

    expected = torch.full((1, THREADS), sentinel, dtype=torch.float32)
    expected[:, :n] = x.cpu()[:, :n] + delta
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_gm_two_dimensional_addressing_for_multiple_byte_widths():
    _require_a5()
    base = torch.arange(GM_DTYPE_THREADS, dtype=torch.int64).reshape(GM_ROWS, GM_COLS)
    src_int8 = (base % 127).to(torch.int8)
    src_fp16 = (base.to(torch.float32) / 8.0).to(torch.float16)
    src_int64 = base * (2**32) + 17
    device_inputs = [src_int8.to(ST_DEVICE), src_fp16.to(ST_DEVICE), src_int64.to(ST_DEVICE)]
    device_outputs = [
        torch.empty_like(device_inputs[0]),
        torch.empty_like(device_inputs[1]),
        torch.empty_like(device_inputs[2]),
    ]

    simt_gm_copy_dtypes(*device_inputs, *device_outputs)
    torch.npu.synchronize()

    for actual, expected in zip(device_outputs, (src_int8, src_fp16, src_int64)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    for active_threads in (0, 193, THREADS):
        test_gm_tensor_access(active_threads)
