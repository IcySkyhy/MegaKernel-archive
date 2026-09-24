# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 generalized system test for the SIMT fma interface."""

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64


def _fma_golden(lhs, rhs, addend):
    return (lhs.to(torch.float64) * rhs.to(torch.float64) + addend.to(torch.float64)).to(lhs.dtype)


@pl.simt.function(max_threads=ELEMENTS)
def fma_all_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    addend_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    addend_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    addend_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.fma(lhs_fp16[0, tid], rhs_fp16[0, tid], addend_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.fma(lhs_bf16[0, tid], rhs_bf16[0, tid], addend_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.fma(lhs_fp32[0, tid], rhs_fp32[0, tid], addend_fp32[0, tid])


@pl.jit(arch="a5")
def simt_fma_all_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    addend_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    addend_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    addend_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(
            fma_all_dtypes,
            threads=ELEMENTS,
            args=(
                lhs_fp16,
                rhs_fp16,
                addend_fp16,
                lhs_bf16,
                rhs_bf16,
                addend_bf16,
                lhs_fp32,
                rhs_fp32,
                addend_fp32,
                out_fp16,
                out_bf16,
                out_fp32,
            ),
        )


@pytest.mark.soc("950")
def test_fma_all_supported_dtypes(run_float_ternary):
    lhs = torch.linspace(-2.0, 2.0, ELEMENTS)
    rhs = torch.linspace(0.5, 1.5, ELEMENTS)
    addend = torch.linspace(-0.75, 0.75, ELEMENTS)
    lhs[:4] = torch.tensor([1.0 + 2.0**-23, -(1.0 + 2.0**-23), 4097.0, -4097.0])
    rhs[:4] = torch.tensor([1.0 + 2.0**-23, 1.0 + 2.0**-23, 4097.0, 4097.0])
    addend[:4] = torch.tensor([-(1.0 + 2.0**-22), 1.0 + 2.0**-22, -16785408.0, 16785408.0])
    run_float_ternary(simt_fma_all_dtypes, lhs, rhs, addend, _fma_golden, rtol=0, atol=0)
