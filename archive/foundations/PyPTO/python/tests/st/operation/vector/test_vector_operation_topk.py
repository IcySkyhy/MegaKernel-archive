#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

from test_case_class_vector_operations import TopKTestCase

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_topk():
    original_shape = (64, 64)
    k = 10
    axis = 1
    output_shape = tuple([k if index == axis else value for index, value in enumerate(original_shape)])
    input_tensors = [
        {
            "name": "A",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-100, 100],
        }
    ]
    output_tensors = [
        {
            "name": "Value",
            "shape": output_shape,
            "dtype": "fp32",
        },
        {
            "name": "Index",
            "shape": output_shape,
            "dtype": "int32",
        },
    ]
    view_shape = (32, 64)
    tile_shape = (32, 64)
    test_case = TopKTestCase(
        0,
        "TopK_test_0",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"count": k, "dims": "[-1]", "islargest": "[1]"},
    )
    test_case.exec(True)
