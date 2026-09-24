# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 generalized system test for the SIMT log1p interface."""

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64


@pl.simt.function(max_threads=ELEMENTS)
def log1p_fp32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log1p(source[0, tid])


@pl.jit(arch="a5")
def simt_log1p_fp32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(log1p_fp32, threads=ELEMENTS, args=(source, output))


@pytest.mark.soc("950")
def test_log1p_supported_dtype(run_fp32_unary):
    values = torch.linspace(-0.75, 8.0, ELEMENTS)
    values[:8] = torch.tensor([-1.0, -0.0, 0.0, -1.0e-7, 1.0e-7, -(2.0**-20), 2.0**-20, 1.0])
    run_fp32_unary(simt_log1p_fp32, values, torch.log1p, rtol=2e-5)
