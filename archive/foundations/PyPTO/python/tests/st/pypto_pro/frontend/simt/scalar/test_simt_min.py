# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 generalized system tests for the SIMT min interface."""

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64


def _run_integer_case(kernel, dtypes, lhs_values, rhs_values, a5_device, assert_simt_close):
    lhs_sources = tuple(lhs_values.to(dtype) for dtype in dtypes)
    rhs_sources = tuple(rhs_values.to(dtype) for dtype in dtypes)
    outputs = tuple(torch.empty_like(source, device=a5_device) for source in lhs_sources)
    arguments = [tensor.to(a5_device) for pair in zip(lhs_sources, rhs_sources) for tensor in pair]
    kernel(*arguments, *outputs)
    torch.npu.synchronize()
    expected = torch.minimum(lhs_values, rhs_values)
    for dtype, output in zip(dtypes, outputs):
        assert_simt_close(output, expected.to(dtype))


@pl.simt.function(max_threads=ELEMENTS)
def min_float_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.min(lhs_fp16[0, tid], rhs_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.min(lhs_bf16[0, tid], rhs_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.min(lhs_fp32[0, tid], rhs_fp32[0, tid])


@pl.jit(arch="a5")
def simt_min_float_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(
            min_float_dtypes,
            threads=ELEMENTS,
            args=(
                lhs_fp16,
                rhs_fp16,
                lhs_bf16,
                rhs_bf16,
                lhs_fp32,
                rhs_fp32,
                out_fp16,
                out_bf16,
                out_fp32,
            ),
        )


@pl.simt.function(max_threads=ELEMENTS)
def min_signed_dtypes(
    lhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    rhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    lhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    rhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    lhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    rhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    out_int8[0, tid] = pl.simt.min(lhs_int8[0, tid], rhs_int8[0, tid])
    out_int16[0, tid] = pl.simt.min(lhs_int16[0, tid], rhs_int16[0, tid])
    out_int32[0, tid] = pl.simt.min(lhs_int32[0, tid], rhs_int32[0, tid])
    out_int64[0, tid] = pl.simt.min(lhs_int64[0, tid], rhs_int64[0, tid])


@pl.jit(arch="a5")
def simt_min_signed_dtypes(
    lhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    rhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    lhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    rhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    lhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    rhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    with pl.section_vector():
        pl.simt.launch(
            min_signed_dtypes,
            threads=ELEMENTS,
            args=(
                lhs_int8,
                rhs_int8,
                lhs_int16,
                rhs_int16,
                lhs_int32,
                rhs_int32,
                lhs_int64,
                rhs_int64,
                out_int8,
                out_int16,
                out_int32,
                out_int64,
            ),
        )


@pl.simt.function(max_threads=ELEMENTS)
def min_unsigned_dtypes(
    lhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    rhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    lhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    rhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    lhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    lhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    rhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    tid = pl.simt.linear_thread_idx()
    out_uint8[0, tid] = pl.simt.min(lhs_uint8[0, tid], rhs_uint8[0, tid])
    out_uint16[0, tid] = pl.simt.min(lhs_uint16[0, tid], rhs_uint16[0, tid])
    out_uint32[0, tid] = pl.simt.min(lhs_uint32[0, tid], rhs_uint32[0, tid])
    out_uint64[0, tid] = pl.simt.min(lhs_uint64[0, tid], rhs_uint64[0, tid])


@pl.jit(arch="a5")
def simt_min_unsigned_dtypes(
    lhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    rhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    lhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    rhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    lhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    lhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    rhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    with pl.section_vector():
        pl.simt.launch(
            min_unsigned_dtypes,
            threads=ELEMENTS,
            args=(
                lhs_uint8,
                rhs_uint8,
                lhs_uint16,
                rhs_uint16,
                lhs_uint32,
                rhs_uint32,
                lhs_uint64,
                rhs_uint64,
                out_uint8,
                out_uint16,
                out_uint32,
                out_uint64,
            ),
        )


@pytest.mark.soc("950")
def test_min_all_supported_float_dtypes(run_float_binary):
    lhs = torch.linspace(-4.0, 4.0, ELEMENTS)
    rhs = torch.linspace(2.0, -2.0, ELEMENTS)
    lhs[:7] = torch.tensor([float("nan"), 1.0, float("nan"), -0.0, 0.0, float("inf"), float("-inf")])
    rhs[:7] = torch.tensor([1.0, float("nan"), float("nan"), 0.0, -0.0, 2.0, 2.0])
    run_float_binary(simt_min_float_dtypes, lhs, rhs, torch.fmin, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_min_all_supported_signed_dtypes(a5_device, assert_simt_close):
    lhs = (torch.arange(ELEMENTS, dtype=torch.int64) - 32).reshape(1, ELEMENTS)
    rhs = (12 - lhs).reshape(1, ELEMENTS)
    _run_integer_case(
        simt_min_signed_dtypes,
        (torch.int8, torch.int16, torch.int32, torch.int64),
        lhs,
        rhs,
        a5_device,
        assert_simt_close,
    )


@pytest.mark.soc("950")
def test_min_all_supported_unsigned_dtypes(a5_device, assert_simt_close):
    lhs = torch.arange(ELEMENTS, dtype=torch.int64).reshape(1, ELEMENTS)
    rhs = (ELEMENTS - 1 - lhs).reshape(1, ELEMENTS)
    _run_integer_case(
        simt_min_unsigned_dtypes,
        (torch.uint8, torch.uint16, torch.uint32, torch.uint64),
        lhs,
        rhs,
        a5_device,
        assert_simt_close,
    )
