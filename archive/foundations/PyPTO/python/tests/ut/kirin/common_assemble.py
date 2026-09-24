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
Test assemble codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def pypto_assemble_in_torch(source, target, offsets=None):
    if isinstance(source, list):
        for s_tensor, s_off in source:
            slices = []
            for x, s in enumerate(s_off):
                slices.append(slice(s, s + s_tensor.shape[x]))
            target[tuple(slices)] = s_tensor
    else:
        slices = []
        for x, s in enumerate(offsets):
            slices.append(slice(s, s + source.shape[x]))
        target[tuple(slices)] = source


def make_single_assemble_kernel(soc_version, name, dtype, tile_shapes, offsets):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input_: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        pypto.assemble(input_, offsets, output)

    kernel.__name__ = name
    return kernel


def make_double_assemble_kernel(soc_version, name, dtype, tile_shapes, offsets_list):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input1: pypto.Tensor([...], dtype),
        input2: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        pypto.assemble([(input1, offsets_list[0]), (input2, offsets_list[1])], output, parallel=True)

    kernel.__name__ = name
    return kernel


def _is_double_input_offsets(offsets_or_offsets_list):
    """Check if offsets_or_offsets_list represents a double-input assemble."""
    return (
        isinstance(offsets_or_offsets_list, tuple)
        and len(offsets_or_offsets_list) == 2
        and isinstance(offsets_or_offsets_list[0], list)
    )


def _compute_output_shape_for_two_inputs(input1_shape, input2_shape, offsets1, offsets2):
    """Compute output shape when assembling two tensors at different offsets."""
    result = list(input1_shape)
    for x, _ in enumerate(result):
        result[x] = max(offsets1[x] + input1_shape[x], offsets2[x] + input2_shape[x])
    return tuple(result)


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32, etc.)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # input_shape: input tensor shape
    # output_shape: output tensor shape (assembled result)
    # offsets_or_offsets_list: single offset list for single-input, tuple of two for double-input
    # marks: pytest marks
    pytest.param(
        "assemble_kernel_fp16", torch.float16, pypto.DT_FP16, (1, 16), (2, 2), (4, 4), ([0, 0],), marks=[], id="001"
    ),
    pytest.param(
        "assemble_kernel_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (3, 3),
        (5, 5),
        ([1, 1],),
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "assemble_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 16),
        (2, 2, 2),
        (3, 3, 3),
        ([0, 0, 0],),
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "assemble_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        (4,),
        (6,),
        ([1],),
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "assemble_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (2, 4),
        (3, 6),
        ([0, 2],),
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "assemble_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 16),
        (1, 2, 2),
        (2, 3, 3),
        ([0, 1, 1],),
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "assemble_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 16),
        (2, 2, 2, 2),
        (3, 3, 3, 3),
        ([0, 0, 0, 0],),
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "assemble_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        (3,),
        (5,),
        ([0],),
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "assemble_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (1, 3),
        (2, 5),
        ([0, 0],),
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "assemble_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (2, 2),
        (4, 4),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "assemble_kernel_fp32",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 2),
        (4, 4),
        ([0, 0],),
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "assemble_kernel_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (3, 3),
        (5, 5),
        ([1, 1],),
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "assemble_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        (2, 2, 2),
        (3, 3, 3),
        ([0, 0, 0],),
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "assemble_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        (4,),
        (6,),
        ([1],),
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "assemble_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 4),
        (3, 6),
        ([0, 2],),
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "assemble_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        (1, 2, 2),
        (2, 3, 3),
        ([0, 1, 1],),
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "assemble_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 8),
        (2, 2, 2, 2),
        (3, 3, 3, 3),
        ([0, 0, 0, 0],),
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "assemble_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        (3,),
        (5,),
        ([0],),
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "assemble_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (1, 3),
        (2, 5),
        ([0, 0],),
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "assemble_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 2),
        (4, 4),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "assemble_kernel_int8",
        torch.int8,
        pypto.DT_INT8,
        (1, 32),
        (2, 2),
        (4, 4),
        ([0, 0],),
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "assemble_kernel_int8_002",
        torch.int8,
        pypto.DT_INT8,
        (1, 32),
        (3, 3),
        (5, 5),
        ([1, 1],),
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "assemble_kernel_int8_003",
        torch.int8,
        pypto.DT_INT8,
        (32,),
        (4,),
        (6,),
        ([1],),
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "assemble_kernel_int8_004",
        torch.int8,
        pypto.DT_INT8,
        (1, 32),
        (2, 4),
        (3, 6),
        ([0, 2],),
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "assemble_kernel_int8_005",
        torch.int8,
        pypto.DT_INT8,
        (1, 32),
        (2, 2),
        (4, 4),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "assemble_kernel_int16",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        (2, 2),
        (4, 4),
        ([0, 0],),
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "assemble_kernel_int16_002",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        (3, 3),
        (5, 5),
        ([1, 1],),
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "assemble_kernel_int16_003",
        torch.int16,
        pypto.DT_INT16,
        (16,),
        (4,),
        (6,),
        ([1],),
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "assemble_kernel_int16_004",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        (2, 4),
        (3, 6),
        ([0, 2],),
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "assemble_kernel_int16_005",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        (2, 2),
        (4, 4),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "assemble_kernel_int32",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        (2, 2),
        (4, 4),
        ([0, 0],),
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "assemble_kernel_int32_002",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        (3, 3),
        (5, 5),
        ([1, 1],),
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "assemble_kernel_int32_003",
        torch.int32,
        pypto.DT_INT32,
        (8,),
        (4,),
        (6,),
        ([1],),
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "assemble_kernel_int32_004",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        (2, 4),
        (3, 6),
        ([0, 2],),
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "assemble_kernel_int32_005",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        (2, 2),
        (4, 4),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "assemble_kernel_list_fp32_001",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 2),
        (4, 4),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "assemble_kernel_list_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        (2,),
        (8,),
        ([1], [3]),
        marks=[pytest.mark.skip()],
        id="037",
    ),
    pytest.param(
        "assemble_kernel_list_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        (2, 2, 2),
        (3, 3, 3),
        ([0, 0, 0], [1, 1, 1]),
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "assemble_kernel_list_multi_shape_001",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 2),
        (4, 6),
        ([0, 0], [2, 2]),
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "assemble_kernel_list_multi_shape_002",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (3, 2),
        (5, 4),
        ([0, 0], [2, 0]),
        marks=[pytest.mark.skip()],
        id="040",
    ),
    pytest.param(
        "assemble_kernel_list_multi_shape_003",
        torch.float32,
        pypto.DT_FP32,
        (10, 80),
        (300, 200),
        (500, 400),
        ([0, 0], [2, 0]),
        marks=[pytest.mark.skip()],
        id="041",
    ),
]


def run_assemble_test(kernels, kernel_name, dtype, input_shape, output_shape, offsets_or_offsets_list):
    """Run a single assemble kernel test with given kernels dict."""
    device = "cpu"
    use_int = dtype in (torch.int8, torch.int16, torch.int32)

    if _is_double_input_offsets(offsets_or_offsets_list):
        offsets1, offsets2 = offsets_or_offsets_list
        input1 = (
            torch.randint(0, 100, input_shape, dtype=dtype, device=device)
            if use_int
            else torch.rand(input_shape, dtype=dtype, device=device)
        )

        if "multi_shape" in kernel_name:
            input2_shape = output_shape
            for x in range(len(output_shape)):
                input2_shape = tuple(
                    max(offsets1[x] + input_shape[x], offsets2[x] + input2_shape[x]) for x in range(len(input_shape))
                )
                break
        else:
            input2_shape = input_shape

        input2 = (
            torch.randint(0, 100, input2_shape, dtype=dtype, device=device)
            if use_int
            else torch.rand(input2_shape, dtype=dtype, device=device)
        )

        computed_output_shape = _compute_output_shape_for_two_inputs(input_shape, input2_shape, offsets1, offsets2)
        out = (
            torch.zeros(computed_output_shape, dtype=dtype, device=device)
            if dtype not in (torch.int8, torch.int16, torch.int32)
            else torch.randint(0, 100, computed_output_shape, dtype=dtype, device=device)
        )
        golden_out = out.clone()
        pypto_assemble_in_torch([(input1, offsets1), (input2, offsets2)], golden_out)
        kernels[kernel_name](input1, input2, out)
        check_nan(out, name=kernel_name)
        cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden_out.cpu())))
    else:
        offsets = offsets_or_offsets_list if isinstance(offsets_or_offsets_list, list) else offsets_or_offsets_list[0]
        input_ = (
            torch.randint(0, 100, input_shape, dtype=dtype, device=device)
            if use_int
            else torch.rand(input_shape, dtype=dtype, device=device)
        )
        out = (
            torch.zeros(output_shape, dtype=dtype, device=device)
            if dtype not in (torch.int8, torch.int16, torch.int32)
            else torch.randint(0, 100, output_shape, dtype=dtype, device=device)
        )
        golden_out = out.clone()
        pypto_assemble_in_torch(input_, golden_out, offsets)
        kernels[kernel_name](input_, out)
        check_nan(out, name=kernel_name)
        cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden_out.cpu())))

    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_assemble_module(soc_version):
    """Create a test module for assemble with specified soc_version."""

    def _make_kernel(p):
        kernel_name = p.values[0]
        dtype = p.values[2]
        tile_shapes = p.values[3]
        offsets_or_offsets_list = p.values[6]
        if _is_double_input_offsets(offsets_or_offsets_list):
            return make_double_assemble_kernel(soc_version, kernel_name, dtype, tile_shapes, offsets_or_offsets_list)
        else:
            offsets = (
                offsets_or_offsets_list[0] if isinstance(offsets_or_offsets_list, tuple) else offsets_or_offsets_list
            )
            return make_single_assemble_kernel(soc_version, kernel_name, dtype, tile_shapes, offsets)

    kernels = {p.values[0]: _make_kernel(p) for p in TEST_CASES}
    return kernels, lambda: run_assemble_test(kernels, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
