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
Test index_put_ codegen for Kirin9030
"""

from kirin.common_index_put_ import TEST_CASES, create_index_put_kernels, run_index_put_test
import pytest

KERNELS = create_index_put_kernels("Kirin9030")


@pytest.mark.parametrize(
    "kernel_name,torch_x_dtype,pypto_x_dtype,torch_indices_dtype,"
    "pypto_indices_dtype,shape_x,shape_indices,shape_values,"
    "vec_tile_shape,accumulate",
    TEST_CASES,
)
def test_index_put_(
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
    run_index_put_test(
        KERNELS,
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
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
