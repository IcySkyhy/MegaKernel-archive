# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 generalized system test for the SIMT abs interface."""

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64


@pl.simt.function(max_threads=ELEMENTS)
def abs_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.abs(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.abs(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.abs(src_fp32[0, tid])
    out_int64[0, tid] = pl.simt.abs(src_int64[0, tid])


@pl.jit(arch="a5")
def simt_abs_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    with pl.section_vector():
        pl.simt.launch(
            abs_all_dtypes,
            threads=ELEMENTS,
            args=(
                src_fp16,
                src_bf16,
                src_fp32,
                src_int64,
                out_fp16,
                out_bf16,
                out_fp32,
                out_int64,
            ),
        )


@pytest.mark.soc("950")
def test_abs_all_supported_dtypes(a5_device, assert_simt_close):
    fp32 = torch.linspace(-8.0, 8.0, ELEMENTS).reshape(1, ELEMENTS)
    fp32[0, :6] = torch.tensor([-0.0, 0.0, float("-inf"), float("inf"), float("nan"), -1.0])
    float_sources = (fp32.to(torch.float16), fp32.to(torch.bfloat16), fp32)
    int64_source = (torch.arange(ELEMENTS, dtype=torch.int64) - 32).reshape(1, ELEMENTS)
    float_outputs = tuple(torch.empty_like(source, device=a5_device) for source in float_sources)
    int64_output = torch.empty_like(int64_source, device=a5_device)

    simt_abs_all_dtypes(
        *(source.to(a5_device) for source in float_sources),
        int64_source.to(a5_device),
        *float_outputs,
        int64_output,
    )
    torch.npu.synchronize()

    for source, output in zip(float_sources, float_outputs):
        expected = torch.abs(source.to(torch.float32)).to(source.dtype)
        assert_simt_close(output, expected)
    assert_simt_close(int64_output, torch.abs(int64_source))
    assert not torch.signbit(float_outputs[2].cpu()[0, 0]).item()
