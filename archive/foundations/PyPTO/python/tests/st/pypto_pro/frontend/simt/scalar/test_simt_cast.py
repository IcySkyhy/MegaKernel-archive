# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT cast interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
ELEMENTS = 256


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function(max_threads=ELEMENTS)
def cast_values(
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_odd: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_rint: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_round: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_floor: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_ceil: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_trunc: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    fp16_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    bf16_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_to_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    value = src_fp32[0, tid]
    out_fp16[0, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    out_odd[0, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_ODD)
    out_bf16[0, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    out_rint[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_round[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_ROUND)
    out_floor[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_FLOOR)
    out_ceil[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_CEIL)
    out_trunc[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_TRUNC)
    fp16_to_fp32[0, tid] = pl.simt.cast(src_fp16[0, tid], pl.DT_FP32)
    bf16_to_fp32[0, tid] = pl.simt.cast(src_bf16[0, tid], pl.DT_FP32)
    int64_to_int32[0, tid] = pl.simt.cast(src_int64[0, tid], pl.DT_INT32)


@pl.jit(arch="a5")
def simt_cast_values(
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_odd: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_rint: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_round: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_floor: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_ceil: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_trunc: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    fp16_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    bf16_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_to_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    with pl.section_vector():
        pl.simt.launch(
            cast_values,
            threads=ELEMENTS,
            args=(
                src_fp32,
                src_fp16,
                src_bf16,
                src_int64,
                out_fp16,
                out_odd,
                out_bf16,
                out_rint,
                out_round,
                out_floor,
                out_ceil,
                out_trunc,
                fp16_to_fp32,
                bf16_to_fp32,
                int64_to_int32,
            ),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_wide_integer_values(
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    src_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    int32_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    uint32_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    uint64_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    value = src_fp32[0, tid]
    out_int32[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_uint32[0, tid] = pl.simt.cast(value, pl.DT_UINT32, mode=pl.RoundMode.CAST_RINT)
    out_int64[0, tid] = pl.simt.cast(value, pl.DT_INT64, mode=pl.RoundMode.CAST_RINT)
    out_uint64[0, tid] = pl.simt.cast(value, pl.DT_UINT64, mode=pl.RoundMode.CAST_RINT)
    int32_to_fp32[0, tid] = pl.simt.cast(src_int32[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    uint32_to_fp32[0, tid] = pl.simt.cast(src_uint32[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    int64_to_fp32[0, tid] = pl.simt.cast(src_int64[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    uint64_to_fp32[0, tid] = pl.simt.cast(src_uint64[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)


@pl.jit(arch="a5")
def simt_cast_wide_integer_values(
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    src_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    int32_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    uint32_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    uint64_to_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(
            cast_wide_integer_values,
            threads=ELEMENTS,
            args=(
                src_fp32,
                src_int32,
                src_uint32,
                src_int64,
                src_uint64,
                out_int32,
                out_uint32,
                out_int64,
                out_uint64,
                int32_to_fp32,
                uint32_to_fp32,
                int64_to_fp32,
                uint64_to_fp32,
            ),
        )


def _repeat(values, dtype):
    values = torch.tensor(values, dtype=dtype)
    repeats = (ELEMENTS + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:ELEMENTS].reshape(1, ELEMENTS)


def _saturating_rint(values, dtype):
    info = torch.iinfo(dtype)
    result = []
    for value in values.reshape(-1).tolist():
        rounded = round(value)
        result.append(min(max(rounded, info.min), info.max))
    return torch.tensor(result, dtype=dtype).reshape_as(values)


@pytest.mark.soc("950")
def test_cast_numeric_conversions():
    _require_a5()

    src_fp32 = ((torch.arange(ELEMENTS, dtype=torch.float32) - 128) / 2).reshape(1, ELEMENTS)
    src_fp32[0, 0] = 1.0001
    src_fp16 = src_fp32.to(torch.float16)
    src_bf16 = src_fp32.to(torch.bfloat16)
    src_int64 = (torch.arange(ELEMENTS, dtype=torch.int64) - 128).reshape(1, ELEMENTS)

    out_fp16 = torch.empty_like(src_fp16, device=ST_DEVICE)
    out_odd = torch.empty_like(src_fp16, device=ST_DEVICE)
    out_bf16 = torch.empty_like(src_bf16, device=ST_DEVICE)
    out_rint = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_round = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_floor = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_ceil = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_trunc = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    fp16_to_fp32 = torch.empty_like(src_fp32, device=ST_DEVICE)
    bf16_to_fp32 = torch.empty_like(src_fp32, device=ST_DEVICE)
    int64_to_int32 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)

    simt_cast_values(
        src_fp32.to(ST_DEVICE),
        src_fp16.to(ST_DEVICE),
        src_bf16.to(ST_DEVICE),
        src_int64.to(ST_DEVICE),
        out_fp16,
        out_odd,
        out_bf16,
        out_rint,
        out_round,
        out_floor,
        out_ceil,
        out_trunc,
        fp16_to_fp32,
        bf16_to_fp32,
        int64_to_int32,
    )
    torch.npu.synchronize()

    round_away = torch.sign(src_fp32) * torch.floor(torch.abs(src_fp32) + 0.5)
    expected_odd = src_fp32.to(torch.float16)
    expected_odd[0, 0] = 1.0009765625
    torch.testing.assert_close(out_fp16.cpu(), src_fp32.to(torch.float16), rtol=0, atol=0)
    torch.testing.assert_close(out_odd.cpu(), expected_odd, rtol=0, atol=0)
    torch.testing.assert_close(out_bf16.cpu(), src_fp32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(out_rint.cpu(), torch.round(src_fp32).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_round.cpu(), round_away.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_floor.cpu(), torch.floor(src_fp32).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_ceil.cpu(), torch.ceil(src_fp32).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_trunc.cpu(), torch.trunc(src_fp32).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(fp16_to_fp32.cpu(), src_fp16.to(torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(bf16_to_fp32.cpu(), src_bf16.to(torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(int64_to_int32.cpu(), src_int64.to(torch.int32), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_cast_wide_integer_intrinsics_and_saturation():
    _require_a5()
    src_fp32 = _repeat([-1.0e30, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 1.0e30], torch.float32)
    src_int32 = _repeat(
        [torch.iinfo(torch.int32).min, -(2**24 + 1), -1, 0, 1, 2**24 + 1, torch.iinfo(torch.int32).max],
        torch.int32,
    )
    src_uint32 = _repeat([0, 1, 2**24 + 1, torch.iinfo(torch.uint32).max], torch.uint32)
    src_int64 = _repeat(
        [torch.iinfo(torch.int64).min, -(2**53 + 1), -1, 0, 1, 2**53 + 1, torch.iinfo(torch.int64).max],
        torch.int64,
    )
    src_uint64 = _repeat([0, 1, 2**53 + 1, torch.iinfo(torch.uint64).max], torch.uint64)

    device_inputs = [
        src_fp32.to(ST_DEVICE),
        src_int32.to(ST_DEVICE),
        src_uint32.to(ST_DEVICE),
        src_int64.to(ST_DEVICE),
        src_uint64.to(ST_DEVICE),
    ]
    integer_outputs = [
        torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE),
        torch.empty((1, ELEMENTS), dtype=torch.int32).to(torch.uint32).to(ST_DEVICE),
        torch.empty((1, ELEMENTS), dtype=torch.int64, device=ST_DEVICE),
        torch.empty((1, ELEMENTS), dtype=torch.int64).to(torch.uint64).to(ST_DEVICE),
    ]
    float_outputs = [torch.empty((1, ELEMENTS), dtype=torch.float32, device=ST_DEVICE) for _ in range(4)]

    simt_cast_wide_integer_values(*device_inputs, *integer_outputs, *float_outputs)
    torch.npu.synchronize()

    integer_expected = [
        _saturating_rint(src_fp32, torch.int32),
        _saturating_rint(src_fp32, torch.uint32),
        _saturating_rint(src_fp32, torch.int64),
        _saturating_rint(src_fp32, torch.uint64),
    ]
    for actual, expected in zip(integer_outputs, integer_expected):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    float_expected = [
        src_int32.to(torch.float32),
        src_uint32.to(torch.float32),
        src_int64.to(torch.float32),
        src_uint64.to(torch.float32),
    ]
    for actual, expected in zip(float_outputs, float_expected):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
