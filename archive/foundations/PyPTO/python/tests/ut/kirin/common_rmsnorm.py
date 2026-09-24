#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test rmsnorm codegen - common functions for Kirin9030 and KirinX90
"""

import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def _make_rmsnorm_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor([...], dtype),
        out: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.rms_norm(a)

    kernel.__name__ = name
    return kernel


def _make_rmsnorm_gamma_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor([...], dtype),
        out: pypto.Tensor([...], dtype),
        gamma: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.rms_norm(a, gamma)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape: input tensor shape
    # has_gamma: whether to use gamma parameter
    # marks: pytest marks
    pytest.param(
        "rmsnorm_kernel_001",
        torch.float16,
        pypto.DT_FP16,
        (32,),
        (32,),
        False,
        marks=[pytest.mark.skip()],
        id="001"
    ),
    pytest.param(
        "rmsnorm_kernel_002",
        torch.float16,
        pypto.DT_FP16,
        (96,),
        (96,),
        False,
        marks=[pytest.mark.skip()],
        id="002"
    ),
    pytest.param(
        "rmsnorm_kernel_003",
        torch.float16,
        pypto.DT_FP16,
        (2, 32),
        (2, 32),
        False,
        marks=[],
        id="003",
    ),
    pytest.param(
        "rmsnorm_kernel_004",
        torch.float16,
        pypto.DT_FP16,
        (1, 128),
        (4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "rmsnorm_kernel_005",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 32),
        (2, 4, 160),
        False,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "rmsnorm_kernel_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 144),
        (2, 4, 144),
        False,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "rmsnorm_kernel_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 5, 32),
        (2, 5, 160),
        False,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "rmsnorm_kernel_008",
        torch.float16,
        pypto.DT_FP16,
        (1, 3, 176),
        (2, 3, 176),
        False,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "rmsnorm_kernel_009",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 2, 128),
        (5, 2, 4, 176),
        False,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "rmsnorm_kernel_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 128),
        (5, 2, 4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "rmsnorm_kernel_011",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 5, 48),
        (2, 3, 5, 144),
        False,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "rmsnorm_kernel_012",
        torch.float16,
        pypto.DT_FP16,
        (2, 2, 3, 48),
        (4, 2, 6, 144),
        False,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "rmsnorm_kernel_013",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 4, 128),
        (6, 2, 4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "rmsnorm_kernel_014",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 1, 144),
        (3, 2, 3, 144),
        False,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "rmsnorm_kernel_015",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 48),
        (6, 3, 5, 144),
        False,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "rmsnorm_kernel_016", torch.float32, pypto.DT_FP32, (32,), (112,), False, marks=[pytest.mark.skip()], id="016"
    ),
    pytest.param(
        "rmsnorm_kernel_017", torch.float32, pypto.DT_FP32, (96,), (96,), False, marks=[pytest.mark.skip()], id="017"
    ),
    pytest.param(
        "rmsnorm_kernel_018",
        torch.float32,
        pypto.DT_FP32,
        (2, 32),
        (4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "rmsnorm_kernel_019",
        torch.float32,
        pypto.DT_FP32,
        (1, 128),
        (4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "rmsnorm_kernel_020",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32),
        (2, 4, 160),
        False,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "rmsnorm_kernel_021",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 144),
        (2, 4, 144),
        False,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "rmsnorm_kernel_022",
        torch.float32,
        pypto.DT_FP32,
        (1, 5, 32),
        (2, 5, 160),
        False,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "rmsnorm_kernel_023",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 176),
        (2, 3, 176),
        False,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "rmsnorm_kernel_024",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 2, 128),
        (5, 2, 4, 176),
        False,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "rmsnorm_kernel_025",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 128),
        (5, 2, 4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "rmsnorm_kernel_026",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 5, 48),
        (2, 3, 5, 144),
        False,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "rmsnorm_kernel_027",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 3, 48),
        (4, 2, 6, 144),
        False,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "rmsnorm_kernel_028",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 4, 128),
        (6, 2, 4, 128),
        False,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "rmsnorm_kernel_029",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 1, 144),
        (3, 2, 3, 144),
        False,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "rmsnorm_kernel_030",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 48),
        (6, 3, 5, 144),
        False,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "rmsnorm_kernel_031",
        torch.float16,
        pypto.DT_FP16,
        (2, 32),
        (4, 128),
        True,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "rmsnorm_kernel_032",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 32),
        (2, 4, 160),
        True,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "rmsnorm_kernel_033",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 2, 128),
        (5, 2, 4, 176),
        True,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "rmsnorm_kernel_034",
        torch.float32,
        pypto.DT_FP32,
        (2, 32),
        (4, 128),
        True,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "rmsnorm_kernel_035",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32),
        (2, 4, 160),
        True,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "rmsnorm_kernel_036",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 2, 128),
        (5, 2, 4, 176),
        True,
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "rmsnorm_kernel_037",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 64, 128),
        (1, 2, 64, 2048),
        False,
        marks=[pytest.mark.skip()],
        id="037",
    ),
    pytest.param(
        "rmsnorm_kernel_038",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 64, 128),
        (1, 2, 64, 2048),
        False,
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "rmsnorm_kernel_039",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 64, 128),
        (2, 1, 64, 1024),
        False,
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "rmsnorm_kernel_040",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 64, 128),
        (2, 1, 64, 1024),
        False,
        marks=[pytest.mark.skip()],
        id="040",
    ),
]


def _compute_golden_rmsnorm(a, gamma=None, eps=1e-6):
    n = a.shape[-1]
    a_f = a.float()
    rms = torch.sqrt(torch.sum(a_f * a_f * (1.0 / n), -1, keepdim=True) + eps)
    y = a_f / rms
    if gamma is not None:
        y = y * gamma.float()
    return y.to(a.dtype)


def run_rmsnorm_test(kernels, kernel_name, dtype, shape, has_gamma):
    device = "cpu"

    a = torch.rand(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)

    if has_gamma:
        gamma_tensor = torch.rand(shape[-1], dtype=dtype, device=device)
        kernels[kernel_name](a, out, gamma_tensor)
        golden = _compute_golden_rmsnorm(a, gamma_tensor)
    else:
        kernels[kernel_name](a, out)
        golden = _compute_golden_rmsnorm(a)

    check_nan(out, name=kernel_name)
    cos_val = abs(compare_cos(out.numpy(), golden.numpy()))
    if cos_val < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_val {cos_val} < 0.9999")


def create_rmsnorm_kernels(soc_version):
    kernels = {}
    for p in TEST_CASES:
        name = p.values[0]
        pypto_dtype = p.values[2]
        tile_shapes = p.values[3]
        has_gamma = p.values[5]
        if has_gamma:
            kernels[name] = _make_rmsnorm_gamma_kernel(soc_version, name, pypto_dtype, tile_shapes)
        else:
            kernels[name] = _make_rmsnorm_kernel(soc_version, name, pypto_dtype, tile_shapes)
    return kernels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
