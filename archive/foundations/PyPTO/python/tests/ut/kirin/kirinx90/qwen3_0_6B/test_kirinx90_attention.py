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
Test attention codegen for KirinX90
"""

from kirin.common_attention import (
    TEST_CASES_4INPUT,
    TEST_CASES_ATTENTION,
    create_attention_kernels,
    run_4input_test,
    run_attention_test,
)
import pytest

KERNELS = create_attention_kernels("KirinX90")


@pytest.mark.parametrize("op_type,a_shape,b_shape,c_shape,out_shape,vec_tile,cube_tile", TEST_CASES_4INPUT)
def test_attention_4input(op_type, a_shape, b_shape, c_shape, out_shape, vec_tile, cube_tile):
    shapes = (a_shape, b_shape, c_shape, out_shape)
    run_4input_test(KERNELS, op_type, shapes)


@pytest.mark.parametrize(
    "op_type,q_shape,k_shape,v_shape,attn_mask_shape,output_shape,vec_tile,cube_tile", TEST_CASES_ATTENTION
)
def test_attention_5input(op_type, q_shape, k_shape, v_shape, attn_mask_shape, output_shape, vec_tile, cube_tile):
    shapes = (q_shape, k_shape, v_shape, attn_mask_shape)
    run_attention_test(KERNELS, op_type, shapes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
