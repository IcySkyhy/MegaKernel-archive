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
""" """

import os

import numpy as np
from numpy.testing import assert_allclose
import torch
import torch_npu

import pypto


@pypto.frontend.jit()
def reshape_kernel(
    in_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    _b = 3
    n1 = 64
    d = 64

    pypto.set_vec_tile_shapes(64, 64)

    tile_b = 1
    real_b = out_tensor.shape[0]
    loop_b_times = (real_b + tile_b - 1) // tile_b

    tmp = pypto.reshape(out_tensor, [real_b * n1, d], inplace=True)
    for b_idx in pypto.loop(loop_b_times, name="b_loop", idx_name="b_idx"):
        a0 = pypto.view(in_tensor, [tile_b * n1, d], [b_idx * n1, 0])
        a1 = a0 + 1.0
        pypto.assemble(a1, [b_idx * n1, 0], tmp)


@pypto.options(pass_options={"enable_slice": True})
def test_reshape():
    b = 3
    n1 = 64
    d = 64
    _shape = (b, n1, d)
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    # prepare data
    input_cpu = torch.rand((b * n1, d), dtype=torch.float32)
    input_npu = input_cpu.to(device=f'npu:{device_id}')
    output_npu = torch.zeros((b, n1, d), dtype=torch.float32, device=f'npu:{device_id}')

    reshape_kernel(input_npu, output_npu)
    torch_npu.npu.synchronize()

    output_cpu = output_npu.cpu()

    ## golden
    output_golde = (input_cpu + 1).reshape(b, n1, d)

    assert_allclose(np.array(output_cpu), np.array(output_golde), rtol=1e-3, atol=1e-3)


@pypto.frontend.jit()
def reshape_infer_shape_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out_tensor: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(4, 8)
    y = pypto.reshape(x, [-1])
    out_tensor.move(y)


@pypto.options(pass_options={"enable_slice": True})
def test_reshape_infer_shape():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x = torch.rand((4, 8), dtype=torch.float32)
    x_tensor = x.npu()
    y_tensor = torch.zeros((32), dtype=torch.float32, device=f'npu:{device_id}')
    reshape_infer_shape_kernel(x_tensor, y_tensor)
    torch_npu.npu.synchronize()
    y = y_tensor.cpu()
    assert_allclose(np.array(y.flatten()), np.array(x.flatten()), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_reshape_infer_shape()
    test_reshape()
