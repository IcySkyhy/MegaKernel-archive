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
Test matmul codegen for KirinX90
"""

from kirin.common_matmul import TEST_CASES, create_test_matmul_module, run_matmul_test
import pytest

KERNELS, _ = create_test_matmul_module("KirinX90")


@pytest.mark.parametrize(
    "kernel_name,torch_in_dtype,torch_out_dtype,pypto_in_dtype,pypto_out_dtype,"
    "shape_a,shape_b,shape_c,cube_tile_shape,has_bias,bias_shape,a_trans,b_trans",
    TEST_CASES,
)
def test_matmul(
    kernel_name,
    torch_in_dtype,
    torch_out_dtype,
    pypto_in_dtype,
    pypto_out_dtype,
    shape_a,
    shape_b,
    shape_c,
    cube_tile_shape,
    has_bias,
    bias_shape,
    a_trans,
    b_trans,
):
    run_matmul_test(
        KERNELS,
        kernel_name,
        torch_in_dtype,
        torch_out_dtype,
        shape_a,
        shape_b,
        shape_c,
        cube_tile_shape,
        has_bias,
        bias_shape,
        a_trans,
        b_trans,
        "KirinX90",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
