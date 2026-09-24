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

import os

from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_softmax_shape_dim():
    """Test whether the output shape is correct"""

    x_shape = [4, 4]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)
    dim = -1

    with pypto.function("SOFTMAX_SHAPE", x):
        pypto.set_vec_tile_shapes(32, 32)
        res = pypto.softmax(x, dim)
        torch_case_tensor = torch.randn((4, 4), dtype=torch.float32)
        torch_case_res = torch.softmax(torch_case_tensor, dim)
        assert res.shape == list(torch_case_res.shape)


@pypto.options(pass_options={"enable_slice": True})
def test_softmax_fp32():
    """Test whether the output of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    res = pypto.tensor(x_shape, dtype)
    dim = -1

    with pypto.function("SOFTMAX_CONTENT_FP32", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(32, 32)
            res.move(pypto.softmax(x, dim))

    x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_x_tensor = pypto.from_torch(x_tensor, "x_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_x_tensor, pto_res_tensor)
    expected = torch.softmax(x_tensor, dim)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_softmax_fp32():
    """Test whether the output of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    res = pypto.tensor(x_shape, dtype)
    dim = -1

    with pypto.function("TENSOR_SOFTMAX_CONTENT_FP32", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(32, 32)
            res.move(x.softmax(dim))

    x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_x_tensor = pypto.from_torch(x_tensor, "x_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_x_tensor, pto_res_tensor)
    expected = torch.softmax(x_tensor, dim)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pypto.runtime._device_fini()
