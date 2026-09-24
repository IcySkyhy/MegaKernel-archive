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
Test index_put_ codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def create_index_put_kernels(soc_version):
    """Factory function to create index_put kernels with specified soc_version."""

    def make_index_put_kernel_1(x_dtype, indices_dtype, vec_tile_shape, accumulate):
        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def kernel(
            x: pypto.Tensor([...], x_dtype),
            indices: pypto.Tensor([...], indices_dtype),
            values: pypto.Tensor([...], x_dtype),
        ):
            pypto.set_vec_tile_shapes(vec_tile_shape)
            pypto.index_put_(x, (indices,), values, accumulate)

        return kernel

    def make_index_put_kernel_2(x_dtype, indices_dtype, vec_tile_shape, accumulate):
        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def kernel(
            x: pypto.Tensor([...], x_dtype),
            indices0: pypto.Tensor([...], indices_dtype),
            indices1: pypto.Tensor([...], indices_dtype),
            values: pypto.Tensor([...], x_dtype),
        ):
            pypto.set_vec_tile_shapes(vec_tile_shape)
            pypto.index_put_(x, (indices0, indices1), values, accumulate)

        return kernel

    def make_index_put_kernel_3(x_dtype, indices_dtype, vec_tile_shape, accumulate):
        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def kernel(
            x: pypto.Tensor([...], x_dtype),
            indices0: pypto.Tensor([...], indices_dtype),
            indices1: pypto.Tensor([...], indices_dtype),
            indices2: pypto.Tensor([...], indices_dtype),
            values: pypto.Tensor([...], x_dtype),
        ):
            pypto.set_vec_tile_shapes(vec_tile_shape)
            pypto.index_put_(x, (indices0, indices1, indices2), values, accumulate)

        return kernel

    def make_index_put_kernel_4(x_dtype, indices_dtype, vec_tile_shape, accumulate):
        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def kernel(
            x: pypto.Tensor([...], x_dtype),
            indices0: pypto.Tensor([...], indices_dtype),
            indices1: pypto.Tensor([...], indices_dtype),
            indices2: pypto.Tensor([...], indices_dtype),
            indices3: pypto.Tensor([...], indices_dtype),
            values: pypto.Tensor([...], x_dtype),
        ):
            pypto.set_vec_tile_shapes(vec_tile_shape)
            pypto.index_put_(x, (indices0, indices1, indices2, indices3), values, accumulate)

        return kernel

    return {
        "one_indice": make_index_put_kernel_1,
        "two_indices": make_index_put_kernel_2,
        "three_indices": make_index_put_kernel_3,
        "four_indices": make_index_put_kernel_4,
    }


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type
    # pypto_dtype: pypto data type
    # index_dtype: torch index data type
    # pypto_index_dtype: pypto index data type
    #       shape, indices_shape, values_shape, index_count, accumulate, marks
    # - kernel_name: name of the kernel in create_index_put_kernels dict
    # - torch_dtype: torch data type for input tensor
    # - pypto_dtype: pypto data type for input tensor
    # - index_dtype: torch data type for indices
    # - pypto_index_dtype: pypto data type for indices
    # - shape: input tensor shape
    # - indices_shape: indices tensor shape
    # - values_shape: values tensor shape
    # - index_count: number of indices
    # - accumulate: whether to accumulate values
    pytest.param(
        "index_put_kernel_001",
        torch.float16,
        pypto.DT_FP16,
        torch.int32,
        pypto.DT_INT32,
        (60,),
        (4,),
        (4,),
        2,
        False,
        marks=[],
        id="001",
    ),
    pytest.param(
        "index_put_kernel_002",
        torch.float32,
        pypto.DT_FP32,
        torch.int64,
        pypto.DT_INT64,
        (3, 3),
        (2,),
        (2, 3),
        3,
        False,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "index_put_kernel_003",
        torch.int8,
        pypto.DT_INT8,
        torch.int8,
        pypto.DT_INT8,
        (3, 3),
        (2, 2),
        (2,),
        3,
        False,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "index_put_kernel_004",
        torch.int16,
        pypto.DT_INT16,
        torch.uint8,
        pypto.DT_UINT8,
        (64, 128),
        (32,),
        (32, 128),
        64,
        False,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "index_put_kernel_005",
        torch.int8,
        pypto.DT_INT8,
        torch.int16,
        pypto.DT_INT16,
        (64, 140),
        (20,),
        (20, 140),
        80,
        False,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "index_put_kernel_006",
        torch.int32,
        pypto.DT_INT32,
        torch.uint16,
        pypto.DT_UINT16,
        (16, 32, 120),
        (8,),
        (8, 32, 120),
        100,
        False,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "index_put_kernel_007",
        torch.int16,
        pypto.DT_INT16,
        torch.uint32,
        pypto.DT_UINT32,
        (16, 32, 120),
        (8, 8),
        (8, 120),
        64,
        False,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "index_put_kernel_008",
        torch.int8,
        pypto.DT_INT8,
        torch.uint32,
        pypto.DT_UINT32,
        (16, 32, 120),
        (10, 10, 10),
        (10,),
        10,
        False,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "index_put_kernel_009",
        torch.float16,
        pypto.DT_FP16,
        torch.int32,
        pypto.DT_INT32,
        (10, 20, 16, 112),
        (2,),
        (2, 20, 16, 112),
        1,
        False,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "index_put_kernel_010",
        torch.float32,
        pypto.DT_FP32,
        torch.int32,
        pypto.DT_INT32,
        (10, 20, 16, 112),
        (5, 5),
        (5, 16, 112),
        70,
        False,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "index_put_kernel_011",
        torch.float16,
        pypto.DT_FP16,
        torch.uint32,
        pypto.DT_UINT32,
        (10, 20, 16, 112),
        (5, 5, 5),
        (5, 112),
        80,
        False,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "index_put_kernel_012",
        torch.float16,
        pypto.DT_FP16,
        torch.int32,
        pypto.DT_INT32,
        (10, 20, 16, 112),
        (5, 5, 5, 5),
        (5,),
        32,
        False,
        marks=[pytest.mark.skip()],
        id="012",
    ),
]


def run_index_put_test(
    kernels,
    kernel_name,
    torch_x_dtype,
    pypto_x_dtype,
    torch_indices_dtype,
    pypto_indices_dtype,
    shape_x,
    shape_indices,
    shape_values,
    vec_tile_shape,
    accumulate,
):
    """Run a single index_put kernel test with given kernels dict."""
    device = "cpu"

    if torch_x_dtype in [torch.float16, torch.float32]:
        x = torch.rand(shape_x, dtype=torch_x_dtype, device=device)
        values = torch.rand(shape_values, dtype=torch_x_dtype, device=device)
    elif torch_x_dtype == torch.int8:
        x = torch.randint(-128, 127, shape_x, dtype=torch_x_dtype, device=device)
        values = torch.randint(-128, 127, shape_values, dtype=torch_x_dtype, device=device)
    elif torch_x_dtype == torch.int16:
        x = torch.randint(-32768, 32767, shape_x, dtype=torch_x_dtype, device=device)
        values = torch.randint(-32768, 32767, shape_values, dtype=torch_x_dtype, device=device)
    elif torch_x_dtype == torch.int32:
        x = torch.randint(-2147483648, 2147483647, shape_x, dtype=torch_x_dtype, device=device)
        values = torch.randint(-2147483648, 2147483647, shape_values, dtype=torch_x_dtype, device=device)

    indices_list = []
    torch_int32_indices_list = []
    for shape_indice, max_indice in zip(shape_indices, shape_x):
        rand = torch.randperm(max_indice, device=device, dtype=torch.int32)[:shape_indice].to(torch_indices_dtype)
        torch_int32_indices_list.append(rand.to(torch.int32))
        indices_list.append(rand)
    indices = tuple(indices_list)
    torch_int32_indices_tuple = tuple(torch_int32_indices_list)

    golden = x.clone()
    golden.index_put_(torch_int32_indices_tuple, values, accumulate=accumulate)

    num_indices = len(shape_indices)
    if num_indices == 1:
        kernels["one_indice"](pypto_x_dtype, pypto_indices_dtype, vec_tile_shape, accumulate)(x, indices[0], values)
    elif num_indices == 2:
        kernels["two_indices"](pypto_x_dtype, pypto_indices_dtype, vec_tile_shape, accumulate)(
            x, indices[0], indices[1], values
        )
    elif num_indices == 3:
        kernels["three_indices"](pypto_x_dtype, pypto_indices_dtype, vec_tile_shape, accumulate)(
            x, indices[0], indices[1], indices[2], values
        )
    elif num_indices == 4:
        kernels["four_indices"](pypto_x_dtype, pypto_indices_dtype, vec_tile_shape, accumulate)(
            x, indices[0], indices[1], indices[2], indices[3], values
        )

    check_nan(x, name=kernel_name)
    cos_value = abs(compare_cos(np.array(x.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_index_put_module(soc_version):
    """Create a test module for index_put with specified soc_version."""
    kernels = create_index_put_kernels(soc_version)
    return kernels, lambda: run_index_put_test(kernels, None, None, None, None, None, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
