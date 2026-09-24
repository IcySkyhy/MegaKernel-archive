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
Test amin codegen for KirinX90
"""

from kirin.common_amin import TEST_CASES, create_test_amin_module, run_amin_test
import pytest

KERNELS, _ = create_test_amin_module("KirinX90")


@pytest.mark.parametrize(
    "kernel_name,torch_dtype,pypto_dtype,tile_shapes,input_shape,output_shape,dim",
    TEST_CASES,
)
def test_amin(kernel_name, torch_dtype, pypto_dtype, tile_shapes, input_shape, output_shape, dim):
    run_amin_test(KERNELS, kernel_name, torch_dtype, input_shape, output_shape, dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
