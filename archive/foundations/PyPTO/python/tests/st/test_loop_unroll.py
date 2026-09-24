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
Add Constant Operator

This file implements and tests the add_constant operator which computes y = x + constant
with dynamic shape and nested loop_unroll.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import torch

import pypto


@pypto.frontend.jit()
def add_constant_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """
    Add constant operator: y = x + constant

    Args:
        x: Input tensor with dynamic shape [M, N]
        y: Output tensor with dynamic shape [M, N]
        constant: Constant value to add
    """
    m_dyn = x.shape[0]
    n_dyn = x.shape[1]
    constant = 0.5

    tile_m = 16
    tile_n = 16

    m_loop = (m_dyn + tile_m - 1) // tile_m

    for m_idx, m_unroll_factor in pypto.loop_unroll(m_loop, unroll_list=[4, 2, 1]):
        m_offset = m_idx * tile_m
        m_offset_end = min(m_offset + tile_m * m_unroll_factor, m_dyn)
        valid_m = m_offset_end - m_offset

        n_loop = (n_dyn + tile_n - 1) // tile_n

        for n_idx, n_unroll_factor in pypto.loop_unroll(n_loop, unroll_list=[4, 2, 1]):
            n_offset = n_idx * tile_n
            n_offset_end = min(n_offset + tile_n * n_unroll_factor, n_dyn)
            valid_n = n_offset_end - n_offset

            actual_tile_m = tile_m * m_unroll_factor
            actual_tile_n = tile_n * n_unroll_factor

            x_view = pypto.view(x, [actual_tile_m, actual_tile_n], [m_offset, n_offset], valid_shape=[valid_m, valid_n])

            pypto.set_vec_tile_shapes(actual_tile_m, actual_tile_n)
            result = pypto.add(x_view, constant)

            pypto.assemble(result, [m_offset, n_offset], y)


if __name__ == "__main__":
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)

    m, n = 128, 128
    constant = 0.5

    x = torch.ones(m, n, dtype=torch.float32)
    y = torch.zeros(m, n, dtype=torch.float32)

    x_npu = x.npu()
    y_npu = y.npu()

    add_constant_kernel(x_npu, y_npu)
    torch.npu.synchronize()

    y = y_npu.cpu()

    golden = x.cpu() + constant

    assert_allclose(np.array(y), np.array(golden.cpu()), rtol=1e-3, atol=1e-3)
