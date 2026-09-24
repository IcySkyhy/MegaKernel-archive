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
Test qkv_rmsnorm_rope_scatternd codegen for KirinX90
"""

from kirin.common_qkv_rmsnorm_rope_scatternd import (
    TEST_CASES,
    create_fused_qkv_kernels,
    run_qkv_fused_test,
)
import pytest


def create_test_module(soc_version):
    kernels = {"fused_qkv_kernel_fp16": create_fused_qkv_kernels(soc_version)}

    @pytest.mark.parametrize(
        "dtype,q_input_shape,k_input_shape,v_input_shape,cos_shape,"
        "sin_shape,past_key_shape,past_value_shape,indices_shape",
        TEST_CASES,
    )
    def _test_qkv_fused(
        dtype,
        q_input_shape,
        k_input_shape,
        v_input_shape,
        cos_shape,
        sin_shape,
        past_key_shape,
        past_value_shape,
        indices_shape,
    ):
        run_qkv_fused_test(
            kernels,
            dtype,
            q_input_shape,
            k_input_shape,
            v_input_shape,
            cos_shape,
            sin_shape,
            past_key_shape,
            past_value_shape,
            indices_shape,
        )

    return _test_qkv_fused


test_qkv_fused = create_test_module("KirinX90")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
