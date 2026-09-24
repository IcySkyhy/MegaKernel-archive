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
Test abs codegen for Kirin9030
"""

from kirin.common_abs import TEST_CASES, create_test_abs_module
import pytest

KERNELS, _ = create_test_abs_module("Kirin9030")


@pytest.mark.parametrize(
    "kernel_name,torch_dtype,pypto_dtype,tile_shapes,shape",
    TEST_CASES,
)
def test_abs(kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape):
    from kirin.common_abs import run_abs_test

    run_abs_test(KERNELS, kernel_name, torch_dtype, shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
