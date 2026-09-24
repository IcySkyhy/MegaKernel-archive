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
Test silu_mul codegen for KirinX90
"""

from kirin.common_silu_mul import (
    TEST_CASES,
    create_silu_mul_kernels,
    run_silu_mul_test,
)
import pytest


def create_test_module(soc_version):
    kernels = create_silu_mul_kernels(soc_version)

    @pytest.mark.parametrize("shape,dtype", TEST_CASES)
    def _test_silu_mul(shape, dtype):
        run_silu_mul_test(kernels, shape, dtype)

    return _test_silu_mul


test_silu_mul = create_test_module("KirinX90")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
