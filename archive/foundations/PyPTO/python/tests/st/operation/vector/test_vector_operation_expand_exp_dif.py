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

from test_case_class_vector_operations import ExpandExpDifTestCase

import pypto


def exec_test_tensor_expand_exp_dif(input_shapes, view_shape, tile_shape):
    input_tensors = [
        {
            "name": "A",
            "shape": input_shapes[0],
            "dtype": "fp32",
            "data_range": [-10, 10],
        },
        {
            "name": "B",
            "shape": input_shapes[1],
            "dtype": "fp32",
            "data_range": [-10, 10],
        },
    ]
    output_tensors = [
        {
            "name": "C",
            "shape": input_shapes[0],
            "dtype": "fp32",
        }
    ]
    test_case = ExpandExpDifTestCase(
        0,
        "Expand_exp_dif_test_0",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {},
    )
    test_case.exec(False)


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_expand_exp_dif_0():
    input_shapes = [(16, 128), (16, 1)]
    view_shape = (16, 128)
    tile_shape = (16, 32)
    exec_test_tensor_expand_exp_dif(input_shapes, view_shape, tile_shape)


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_expand_exp_dif_0():  # noqa: F811
    input_shapes = [(16, 128), (1, 128)]
    view_shape = (16, 128)
    tile_shape = (16, 32)
    exec_test_tensor_expand_exp_dif(input_shapes, view_shape, tile_shape)
