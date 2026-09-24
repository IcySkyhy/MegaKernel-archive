#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import numpy as np
from st.pypto_test import TestBuilder
import torch

import pypto


# pypto op define, need args: params, tensors
def op_sign(params, a, b):
    n, m = a.shape
    view_shape, tile_shape = params
    for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SIGN_L0", idx_name="b_idx"):
        for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SIGN_L1", idx_name="s_idx"):
            tile_a = pypto.view(
                a,
                view_shape,
                [b_idx * view_shape[0], s_idx * view_shape[1]],
                valid_shape=[
                    pypto.min(pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(n)),
                    pypto.min(pypto.symbolic_scalar(m) - b_idx * view_shape[1], pypto.symbolic_scalar(m)),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tile_a.move(pypto.sign(tile_a))
            pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)


def op_sign_golden(param, a, b):
    return torch.sign(a)


class SignTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def get_input_from_param(self):
        n, m = self.tiling * 1, self.tiling * 1
        a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
        self.setup_inputs(a_tensor)
        self.set_tol(rtol=3e-3, atol=3e-3)
        return (a_tensor,)


def test():
    st = SignTest(((16, 16), (8, 8)), op_sign, op_sign_golden, tiling=32)
    st()
