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

import numpy as np

import pypto
from st.pypto_test import TestBuilder


def op_record_if_branch(params, a, b, c):
    tile_shape = params[1]
    pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
    for k in pypto.loop(2, name="LOOP", idx_name="k"):
        if pypto.cond(k < 10):
            c.move(pypto.add(a, b))


def golden_if_branch(params, a, b, c):
    return a + b


class AddIfTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def get_input_from_param(self):
        shape = self.params[0]
        n, m = shape
        a_tensor = np.random.uniform(0, 100, [n, m]).astype(np.float32)
        b_tensor = np.random.uniform(0, 100, [n, m]).astype(np.float32)
        self.setup_inputs(a_tensor, b_tensor)
        return (a_tensor, b_tensor)


def test():
    st = AddIfTest(((32, 32), (8, 8)), op_record_if_branch, golden_if_branch, tiling=32)
    st(False)


if __name__ == "__main__":
    test()
