#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test comparison ops (eq/ne/lt/le/gt/ge) codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan
import pypto

# op_name -> (pypto_op, torch_op). Pypto and torch expose the same six comparison
# operators with identical semantics, so a single dispatch table drives both the
# kernel body and the golden reference.
CMP_OPS = {
    "eq": (pypto.eq, torch.eq),
    "ne": (pypto.ne, torch.ne),
    "lt": (pypto.lt, torch.lt),
    "le": (pypto.le, torch.le),
    "gt": (pypto.gt, torch.gt),
    "ge": (pypto.ge, torch.ge),
}


def make_cmp_kernel(soc_version, name, op_name, dtype, tile_shapes):
    """Build a jit kernel that computes ``out = pypto_op(a, b)`` for the given op."""
    pypto_op = CMP_OPS[op_name][0]

    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": pypto.RunMode.SIM}
    )
    def kernel(
        a: pypto.Tensor([...], dtype),
        b: pypto.Tensor([...], dtype),
        out: pypto.Tensor([...], pypto.DT_BOOL),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto_op(a, b)
    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # op_name: comparison op ("eq"/"ne"/"lt"/"le"/"gt"/"ge") - selects pypto/torch op via CMP_OPS
    # torch_dtype: torch data type (float16, float32)
    # pypto_dtype: pypto data type (DT_FP16, DT_FP32, etc.)
    # tile_shape: tile shape for pypto kernel
    # shape_a: first input tensor shape
    # shape_b: second input tensor shape (or None for scalar)
    # scalar_val: scalar value (or None for tensor)
    # marks: pytest marks
    pytest.param(
        "eq_kernel_fp16_001",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (50,),
        (112,),
        (112,),
        None,
        marks=[],
        id="001"
    ),
    pytest.param(
        "eq_kernel_fp16_002",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (32,),
        (64,),
        (64,),
        0.5,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "eq_kernel_fp16_003",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        (32,),
        (32,),
        None,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "eq_kernel_fp16_004",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (16, 8),
        (16, 16),
        (16, 16),
        0.5,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "eq_kernel_fp16_005",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (2, 40),
        (4, 80),
        (4, 80),
        None,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "eq_kernel_fp16_006",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (1, 48),
        (2, 96),
        (1, 96),
        None,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "eq_kernel_fp16_007",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (2, 16),
        (4, 1),
        (4, 32),
        None,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "eq_kernel_fp16_008",
        "eq",
        torch.float16,
        pypto.DT_FP16,
        (2, 64),
        (4, 128),
        (4, 128),
        None,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "ne_kernel_fp16_009",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (32, 64),
        (64, 1),
        (64, 64),
        None,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "ne_kernel_fp16_010",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (64, 32),
        (1, 64),
        (64, 64),
        None,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "ne_kernel_fp16_011",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 32),
        (2, 64, 64),
        (2, 64, 64),
        None,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "ne_kernel_fp16_012",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 24),
        (2, 1, 48),
        (2, 3, 48),
        None,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "ne_kernel_fp16_013",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 24),
        (3, 64, 1),
        (3, 64, 48),
        None,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "ne_kernel_fp16_014",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (1, 24, 24),
        (1, 48, 48),
        (1, 1, 48),
        None,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "ne_kernel_fp16_015",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 24),
        (2, 64, 48),
        (2, 64, 48),
        None,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "ne_kernel_fp16_016",
        "ne",
        torch.float16,
        pypto.DT_FP16,
        (3, 32, 32),
        (3, 32, 64),
        (3, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "lt_kernel_fp16_017",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 32),
        (2, 32, 1),
        (2, 1, 64),
        None,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "lt_kernel_fp16_018",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (16, 16, 16),
        (1, 48, 64),
        (48, 48, 64),
        None,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "lt_kernel_fp16_019",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 16, 16),
        (2, 2, 32, 32),
        (2, 2, 32, 32),
        None,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "lt_kernel_fp16_020",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 8, 8),
        (2, 4, 1, 16),
        (2, 1, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "lt_kernel_fp16_021",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 12, 12),
        (2, 1, 24, 24),
        (2, 3, 24, 1),
        None,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "lt_kernel_fp16_022",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 20, 20),
        (2, 2, 40, 1),
        (2, 2, 1, 40),
        None,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "lt_kernel_fp16_023",
        "lt",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 8, 8),
        (2, 3, 16, 16),
        (2, 3, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "lt_kernel_fp32_024",
        "lt",
        torch.float32,
        pypto.DT_FP32,
        (4, 1),
        (8, 4),
        (8, 4),
        0.5,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "le_kernel_fp32_025",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 32, 32),
        (4, 1, 32, 32),
        (4, 1, 32, 32),
        None,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "le_kernel_fp32_026",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (1, 4, 16, 16),
        (1, 8, 16, 16),
        (1, 8, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "le_kernel_fp32_027",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 24, 48),
        (2, 2, 48, 48),
        (2, 2, 48, 48),
        None,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "le_kernel_fp32_028",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (1, 4, 32, 32),
        (1, 4, 32, 64),
        (1, 4, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "le_kernel_fp32_029",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 16, 16),
        (2, 4, 16, 1),
        (2, 1, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "le_kernel_fp32_030",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 16, 32),
        (2, 1, 32, 32),
        (2, 2, 1, 32),
        None,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "le_kernel_fp32_031",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 24, 24),
        (2, 3, 24, 1),
        (1, 3, 24, 48),
        None,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "le_kernel_fp32_032",
        "le",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 8, 16),
        (1, 4, 1, 16),
        (1, 1, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "gt_kernel_fp32_033",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 32, 32),
        (2, 2, 32, 1),
        (2, 2, 1, 64),
        None,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "gt_kernel_fp32_034",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 24, 32),
        (1, 1, 48, 64),
        (1, 1, 48, 64),
        None,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "gt_kernel_fp32_035",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (8, 8, 8, 4),
        (8, 8, 8, 8),
        (8, 8, 8, 8),
        0.5,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "gt_kernel_fp32_036",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (8, 8, 4),
        (8, 8, 8),
        (8, 8, 8),
        0.5,
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "gt_kernel_fp32_037",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (8, 1, 48),
        (16, 1, 48),
        (1, 1, 48),
        None,
        marks=[pytest.mark.skip()],
        id="037",
    ),
    pytest.param(
        "gt_kernel_fp32_038",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (1, 64, 24),
        (2, 64, 48),
        (2, 64, 48),
        None,
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "gt_kernel_fp32_039",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (2, 16, 16),
        (2, 32, 1),
        (2, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "gt_kernel_fp32_040",
        "gt",
        torch.float32,
        pypto.DT_FP32,
        (16, 24, 32),
        (1, 48, 64),
        (48, 48, 64),
        None,
        marks=[pytest.mark.skip()],
        id="040",
    ),
    pytest.param(
        "ge_kernel_fp32_041",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 16, 48),
        (2, 4, 32, 48),
        (2, 4, 32, 48),
        None,
        marks=[pytest.mark.skip()],
        id="041",
    ),
    pytest.param(
        "ge_kernel_fp32_042",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32, 32),
        (2, 4, 32, 64),
        (2, 4, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="042",
    ),
    pytest.param(
        "ge_kernel_fp32_043",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 16, 32),
        (4, 2, 32, 64),
        (4, 2, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="043",
    ),
    pytest.param(
        "ge_kernel_fp32_044",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 16, 32),
        (1, 4, 32, 64),
        (1, 4, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="044",
    ),
    pytest.param(
        "ge_kernel_fp32_045",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (4, 2),
        (1, 4),
        (8, 4),
        None,
        marks=[pytest.mark.skip()],
        id="045",
    ),
    pytest.param(
        "ge_kernel_fp32_046",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (16, 12, 8),
        (1, 24, 16),
        (32, 24, 16),
        None,
        marks=[pytest.mark.skip()],
        id="046",
    ),
    pytest.param(
        "ge_kernel_fp32_047",
        "ge",
        torch.float32,
        pypto.DT_FP32,
        (8, 8, 8, 8),
        (1, 32, 32, 16),
        (16, 32, 32, 16),
        None,
        marks=[pytest.mark.skip()],
        id="047",
    ),
]


def run_cmp_test(kernels, kernel_name, op_name, dtype, shape_a, shape_b, scalar_val):
    """Run a single comparison kernel test; ``op_name`` picks the op from CMP_OPS."""
    torch_op = CMP_OPS[op_name][1]

    device = "cpu"

    a = torch.rand(shape_a, dtype=dtype, device=device)

    if scalar_val is not None:
        b = torch.full_like(a, scalar_val, dtype=dtype, device=device)
    else:
        b = torch.rand(shape_b, dtype=dtype, device=device)

    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    out = torch.zeros(out_shape, dtype=torch.bool, device=device)

    kernels[kernel_name](a, b, out)

    expect = torch_op(a, b)
    out_np = np.array(out.cpu())
    expect_np = np.array(expect.cpu())

    check_nan(out, name=kernel_name)
    np.testing.assert_array_equal(out_np, expect_np)


def create_test_cmp_module(soc_version):
    """Build the comparison-op kernel registry for the given soc_version.

    Returns (kernels_dict, smoke_runner); the per-case runner is invoked directly
    by the per-soc test files, which parametrize over TEST_CASES.
    """
    kernels = {
        p.values[0]: make_cmp_kernel(soc_version, p.values[0], p.values[1], p.values[3], p.values[4])
        for p in TEST_CASES
    }
    return kernels, lambda: run_cmp_test(kernels, None, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
