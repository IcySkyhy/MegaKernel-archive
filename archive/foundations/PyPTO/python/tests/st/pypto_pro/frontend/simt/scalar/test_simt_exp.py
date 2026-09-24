# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 generalized system test for the SIMT exp interface."""

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64


@pl.simt.function(max_threads=ELEMENTS)
def exp_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.exp(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.exp(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.exp(src_fp32[0, tid])


@pl.jit(arch="a5")
def simt_exp_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(
            exp_all_dtypes,
            threads=ELEMENTS,
            args=(src_fp16, src_bf16, src_fp32, out_fp16, out_bf16, out_fp32),
        )


@pytest.mark.soc("950")
def test_exp_all_supported_dtypes(run_float_unary):
    values = torch.linspace(-3.0, 3.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -104.0, -88.0, -1.0e-7, 0.0, 1.0e-7, 88.0, float("inf")])
    run_float_unary(simt_exp_all_dtypes, values, torch.exp)
