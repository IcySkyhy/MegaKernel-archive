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
Test scatternd codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def get_index_put_kernel(x_dtype, indices_dtype, vec_tile_shape, accumulate, soc_version):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def index_put_kernel(
        x: pypto.Tensor([...], x_dtype),
        indices: pypto.Tensor([...], indices_dtype),
        values: pypto.Tensor([...], x_dtype),
    ):
        pypto.set_vec_tile_shapes(vec_tile_shape)
        pypto.index_put_(x, (indices,), values, accumulate)

    return index_put_kernel


def create_scatter_kernels(soc_version):
    """Factory function to create scatter kernels with specified soc_version."""

    def index_put_kernel(x_dtype, indices_dtype, vec_tile_shape, accumulate):
        return get_index_put_kernel(x_dtype, indices_dtype, vec_tile_shape, accumulate, soc_version)

    return index_put_kernel


TEST_CASES = [
    # data_shape: data tensor shape
    # indices_shape: indices tensor shape
    # update_shape: update values tensor shape
    # vec_tile_shape: vector tile shape for the kernel
    # torch_data_dtype: torch data type for data tensor
    # torch_indices_dtype: torch data type for indices tensor
    # pypto_data_dtype: pypto data type for data tensor
    # pypto_indices_dtype: pypto data type for indices tensor
    # accumulate: whether to accumulate values
    # marks: pytest marks
    # - data_shape: data tensor shape
    # - indices_shape: indices tensor shape
    # - update_shape: update values tensor shape
    # - vec_tile_shape: vector tile shape for the kernel
    # - torch_data_dtype: torch data type for data tensor
    # - torch_indices_dtype: torch data type for indices tensor
    # - pypto_data_dtype: pypto data type for data tensor
    # - pypto_indices_dtype: pypto data type for indices tensor
    # - accumulate: whether to accumulate values
    pytest.param(
        (2048, 8, 1, 128),
        (64, 1),
        (64, 8, 1, 128),
        16,
        torch.float16,
        torch.int32,
        pypto.DT_FP16,
        pypto.DT_INT32,
        False,
        marks=[],
        id="001",
    ),
]


def get_golden(data, indices, update, accumulate):
    indices = indices.squeeze(1)
    golden = data.clone().index_put_((indices,), update, accumulate=accumulate)
    return golden


def run_scatter_test(
    kernels,
    data_shape,
    indices_shape,
    update_shape,
    vec_tile_shape,
    torch_data_dtype,
    torch_indices_dtype,
    pypto_data_dtype,
    pypto_indices_dtype,
    accumulate,
):
    data = torch.rand(data_shape, dtype=torch_data_dtype, device="cpu")
    indices = torch.randperm(data_shape[0], device="cpu", dtype=torch_indices_dtype)[:indices_shape[0]].unsqueeze(1)
    update = torch.rand(update_shape, dtype=torch_data_dtype, device="cpu")

    golden = get_golden(data.clone(), indices, update, accumulate)

    kernels["index_put_kernel"](pypto_data_dtype, pypto_indices_dtype, vec_tile_shape, accumulate)(
        data, indices.squeeze(1), update
    )

    check_nan(data)
    cos_value = abs(compare_cos(np.array(data.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"cos_value {cos_value} < 0.9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
