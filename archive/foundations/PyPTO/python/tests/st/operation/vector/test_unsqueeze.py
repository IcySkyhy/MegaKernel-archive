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
def test_unsqueeze_shape_dim():
    """Test whether the output shape is correct"""
    shape = [8, 16, 16]
    dtype = pypto.DT_FP32
    # Test each valid dim:[-4, -3, -2, -1, 0, 1, 2, 3]
    for dim in range(-4, 4):
        x = pypto.tensor(shape, dtype)
        pypto.runtime._device_init()
        with pypto.function(f"UNSQUEEZE_SHAPE_DIM_{dim}", x):
            pypto.set_vec_tile_shapes(8, 8, 8, 8)
            res = pypto.unsqueeze(x, dim)
            x_torch = torch.randn((8, 16, 16), dtype=torch.float32)
            res_torch = torch.unsqueeze(x_torch, dim)
            assert res.shape == list(res_torch.shape)


@pypto.options(pass_options={"enable_slice": True})
def test_unsqueeze_content_equal():
    """Test whether the output content has changed"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = [2, 2]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(shape, dtype)
    res = pypto.tensor([1, 2, 2], dtype)
    dim = 0
    with pypto.function("UNSQUEEZE_CONTENT", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(2, 2, 8)
            res.move(pypto.unsqueeze(x, dim))

    torch_case_tensor = torch.rand(2, 2, dtype=torch.float32)
    res_tensor = torch.zeros((1,) + torch_case_tensor.shape, dtype=torch.float32)

    pto_case_tensor = pypto.from_torch(torch_case_tensor, "torch_case_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_case_tensor, pto_res_tensor)

    torch_case_res = torch.unsqueeze(torch_case_tensor, dim)

    assert torch.equal(res_tensor.flatten(), torch_case_res.flatten())
    pypto.runtime._device_fini()
