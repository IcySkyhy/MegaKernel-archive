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

import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_view_basic_shape():
    """Test whether the output shape is correct"""

    x_shape = [32, 64]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)
    view_shape = [32, 32]
    offset = [0, 32]
    with pypto.function("VIEW_SHAPE", x):
        pypto.set_vec_tile_shapes(32, 32)
        res = pypto.view(x, view_shape, offset)

    assert res.shape == view_shape


@pypto.options(pass_options={"enable_slice": True})
def test_view_content_equal():
    """Test whether the output content has changed"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 8]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [0, 4]
    res = pypto.tensor(view_shape, dtype)

    with pypto.function("VIEW_CONTENT", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(pypto.view(x, view_shape, offset))

    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_input_tensor = pypto.from_torch(torch_tensor, "pto_input_tensor")
    pto_output_tensor = pypto.from_torch(res_tensor, "pto_output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)

    expected = torch_tensor[0:4, 4:8]
    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_view_content_equal_validshape():
    """Test whether the output content has changed with validshape"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [2, 0]
    validshape = [2, 4]
    res = pypto.tensor(view_shape, dtype)

    with pypto.function("VIEW_CONTENT_VALIDSHAPE", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(pypto.view(x, view_shape, offset, valid_shape=validshape))

    torch_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_input_tensor = pypto.from_torch(torch_tensor, "pto_input_tensor")
    pto_output_tensor = pypto.from_torch(res_tensor, "pto_output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)

    expected = torch_tensor[2:4, 0:4]
    assert torch.equal(res_tensor.flatten()[:2 * 4], expected.flatten())

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_view_content_equal():
    """Test whether the output content has changed"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 8]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [0, 4]
    res = pypto.tensor(view_shape, dtype)

    with pypto.function("Tensor_VIEW_CONTENT", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(x.view(view_shape, offset))

    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_input_tensor = pypto.from_torch(torch_tensor, "pto_input_tensor")
    pto_output_tensor = pypto.from_torch(res_tensor, "pto_output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)

    expected = torch_tensor[0:4, 4:8]
    assert torch.equal(res_tensor.flatten(), expected.flatten())

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_view_content_validshape_equal():
    """Test whether the output content has changed"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [2, 0]
    validshape = [2, 4]
    res = pypto.tensor(view_shape, dtype)

    with pypto.function("Tensor_VIEW_CONTENT_VALIDSHAPE", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(x.view(view_shape, offset, valid_shape=validshape))

    torch_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_input_tensor = pypto.from_torch(torch_tensor, "pto_input_tensor")
    pto_output_tensor = pypto.from_torch(res_tensor, "pto_output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)

    expected = torch_tensor[2:4, 0:4]
    assert torch.equal(res_tensor.flatten()[:2 * 4], expected.flatten())

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_syntactic_sugar_view_content_equal():
    """Test whether the output content has changed"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 8]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [0, 4]
    res = pypto.tensor(view_shape, dtype)

    with pypto.function("SURGER_VIEW_CONTENT", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(x[:offset[0] + view_shape[0], offset[1]:offset[1] + view_shape[1]])

    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_input_tensor = pypto.from_torch(torch_tensor, "pto_input_tensor")
    pto_output_tensor = pypto.from_torch(res_tensor, "pto_output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)

    expected = torch_tensor[:, 4:8]
    assert torch.equal(res_tensor.flatten(), expected.flatten())

    pypto.runtime._device_fini()
