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
Test full codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def _make_full_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        output: pypto.Tensor([...], dtype),
        shape: tuple,
        input_: dtype,
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output.move(pypto.full(list(shape), input_, dtype))

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32, int8, int16, int32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape: output tensor shape
    # fill_value: scalar value to fill the output tensor
    # marks: pytest marks
    pytest.param("full_kernel_001", torch.float16, pypto.DT_FP16, (120,), (112,), 2.0, marks=[], id="001"),
    pytest.param(
        "full_kernel_002", torch.float32, pypto.DT_FP32, (50,), (100,), 2.0, marks=[pytest.mark.skip()], id="002"
    ),
    pytest.param("full_kernel_003", torch.int8, pypto.DT_INT8, (136,), (137,), 2, marks=[pytest.mark.skip()], id="003"),
    pytest.param(
        "full_kernel_004", torch.int16, pypto.DT_INT16, (8, 256), (4, 128), 2, marks=[pytest.mark.skip()], id="004"
    ),
    pytest.param(
        "full_kernel_005", torch.int32, pypto.DT_INT32, (10, 100), (4, 130), 2, marks=[pytest.mark.skip()], id="005"
    ),
    pytest.param(
        "full_kernel_006", torch.float16, pypto.DT_FP16, (5, 32), (15, 31), 2.0, marks=[pytest.mark.skip()], id="006"
    ),
    pytest.param(
        "full_kernel_007", torch.float32, pypto.DT_FP32, (2, 70), (4, 140), 2.0, marks=[pytest.mark.skip()], id="007"
    ),
    pytest.param(
        "full_kernel_008", torch.int8, pypto.DT_INT8, (5, 5, 32), (10, 5, 12), 2, marks=[pytest.mark.skip()], id="008"
    ),
    pytest.param(
        "full_kernel_009",
        torch.int16,
        pypto.DT_INT16,
        (5, 5, 100),
        (7, 3, 170),
        2,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "full_kernel_010",
        torch.int32,
        pypto.DT_INT32,
        (5, 4, 120),
        (9, 8, 100),
        2,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "full_kernel_011",
        torch.float16,
        pypto.DT_FP16,
        (10, 10, 4),
        (20, 40, 10),
        2.0,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "full_kernel_012",
        torch.float32,
        pypto.DT_FP32,
        (16, 5, 5, 16),
        (32, 3, 5, 14),
        2.0,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "full_kernel_013",
        torch.int8,
        pypto.DT_INT8,
        (2, 10, 9, 8),
        (8, 10, 6, 16),
        2,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "full_kernel_014",
        torch.int16,
        pypto.DT_INT16,
        (3, 40, 4, 40),
        (6, 20, 9, 31),
        2,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "full_kernel_015",
        torch.int32,
        pypto.DT_INT32,
        (3, 3, 30, 20),
        (6, 9, 21, 10),
        2,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "full_kernel_016",
        torch.float16,
        pypto.DT_FP16,
        (5, 10, 5, 5),
        (6, 9, 21, 10),
        2.0,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "full_kernel_017",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 40, 5),
        (6, 9, 21, 10),
        2.0,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "full_kernel_018",
        torch.int8,
        pypto.DT_INT8,
        (5, 5, 12, 20),
        (6, 9, 21, 10),
        2,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "full_kernel_019",
        torch.int16,
        pypto.DT_INT16,
        (5, 8, 12, 5),
        (6, 9, 21, 10),
        2,
        marks=[pytest.mark.skip()],
        id="019",
    ),
]


def run_full_test(kernels, kernel_name, dtype, shape, input_val):
    output = torch.empty(shape, dtype=dtype, device="cpu")
    kernels[kernel_name](output, shape, input_val)
    golden = torch.full(shape, input_val, dtype=dtype, device="cpu")
    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_full_kernels(soc_version):
    return {p.values[0]: _make_full_kernel(soc_version, p.values[0], p.values[2], p.values[3]) for p in TEST_CASES}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
