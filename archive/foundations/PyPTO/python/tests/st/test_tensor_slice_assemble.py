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

import math
import os

import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_slice_neg_index():
    """Test negative index"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [2, 1]
    res_shape = [4, 8]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    res = pypto.tensor(res_shape, dtype)

    with pypto.function("SLICE_NEG_INDEX", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res[-3:-1, -2:-1] = x  # equivalent to a[1:3, 6:7]

    torch_tensor = torch.rand(x_shape, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(res_shape, dtype=torch.float32)
    expected = res_tensor.clone()
    expected[-3:-1, -2:-1] = torch_tensor

    pto_tensor = pypto.from_torch(torch_tensor, "torch_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_tensor, pto_res_tensor)

    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_1d_assemble_to_2d():
    """1d tensor assemble to 2d tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4]
    res_shape = [4, 8]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    res = pypto.tensor(res_shape, dtype)

    with pypto.function("SLICE_NEG_INDEX", x, res):
        for a_idx in pypto.loop(res_shape[1], name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res[0:, a_idx] = x

    torch_tensor = torch.rand(x_shape, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(res_shape, dtype=torch.float32)
    expected = res_tensor.clone()
    for k in range(res_shape[1]):
        expected[0:, k] = torch_tensor

    pto_tensor = pypto.from_torch(torch_tensor, "torch_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_tensor, pto_res_tensor)

    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_2d_assemble_to_3d():
    """2d tensor assemble to 3d tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    res_shape = [8, 8, 4]
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    res = pypto.tensor(res_shape, dtype)

    with pypto.function("SLICE_NEG_INDEX", x, res):
        for b_idx in pypto.loop(res.shape[0], name="LOOP_L0", idx_name="a_idx"):
            s_loop = math.ceil(res.shape[1] / x.shape[0])
            for s_idx in pypto.loop(s_loop, name="LOOP_L1", idx_name="s_idx"):
                pypto.set_vec_tile_shapes(4, 4, 8)
                offset = s_idx * x.shape[1]
                res[b_idx, offset:, :] = x

    torch_tensor = torch.rand(x_shape, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(res_shape, dtype=torch.float32)
    expected = res_tensor.clone()
    for k in range(res_shape[0]):
        for m in range(2):
            expected[k, m * 4:m * 4 + 4, 0:4] = torch_tensor

    pto_tensor = pypto.from_torch(torch_tensor, "torch_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_tensor, pto_res_tensor)

    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_slice_int_index():
    """Test mix use of slice and int"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [3, 8, 3]
    res_shape = [4, 8, 8, 8, 8]

    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x = pypto.tensor(x_shape, dtype)
    res = pypto.tensor(res_shape, dtype)

    with pypto.function("SLICE_INT_INDEX", x, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 4, 4, 4, 8)
            res[-2, -3:8, :, 1:4, 2] = x  # reshape x to (1, 3, 8, 3, 1), res[2:, 5:8, 0:8, 1:4, 2:3] = x

    torch_tensor = torch.rand(x_shape, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(res_shape, dtype=torch.float32)
    expected = res_tensor.clone()
    expected[-2, -3:8, :, 1:4, 2] = torch_tensor
    pto_tensor = pypto.from_torch(torch_tensor, "torch_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_tensor, pto_res_tensor)

    assert torch.equal(res_tensor, expected)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_slice_ellipsis_index():
    """Test mix use of ellipsis, slice and int"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    x_shape = [[4, 8, 8], [1, 8, 8, 2], [8, 8], [4, 8, 8, 8]]
    res_shape = [4, 8, 8, 8]

    x0 = pypto.tensor(x_shape[0], dtype)
    x1 = pypto.tensor(x_shape[1], dtype)
    x2 = pypto.tensor(x_shape[2], dtype)
    x3 = pypto.tensor(x_shape[3], dtype)

    res0 = pypto.tensor(res_shape, dtype)
    res1 = pypto.tensor(res_shape, dtype)
    res2 = pypto.tensor(res_shape, dtype)
    res3 = pypto.tensor(res_shape, dtype)

    with pypto.function("SLICE_INT_ELLIPSIS_INDEX", x0, x1, x2, x3, res0, res1, res2, res3):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 4, 4, 8)
            res0[..., 2] = x0
            res1[1:2, :, ..., 3:5] = x1
            res2[2, 3, ...] = x2
            res3[...] = x3 + 0.0

    x_tensor = [torch.rand(shape, dtype=torch.float32) * 200 - 100 for shape in x_shape]
    res_tensor = [torch.zeros(res_shape, dtype=torch.float32) for _ in range(4)]

    res0_copy = res_tensor[0].clone()
    res1_copy = res_tensor[1].clone()
    res2_copy = res_tensor[2].clone()
    res3_copy = res_tensor[3].clone()

    res0_copy[..., 2] = x_tensor[0]
    res1_copy[1:2, :, ..., 3:5] = x_tensor[1]
    res2_copy[2, 3, ...] = x_tensor[2]
    res3_copy[...] = x_tensor[3]

    pto_x_tensor = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(x_tensor)]
    pto_res_tensor = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(res_tensor)]
    pypto.runtime._device_run_once_data_from_host(
        pto_x_tensor[0],
        pto_x_tensor[1],
        pto_x_tensor[2],
        pto_x_tensor[3],
        pto_res_tensor[0],
        pto_res_tensor[1],
        pto_res_tensor[2],
        pto_res_tensor[3],
    )

    assert torch.equal(res_tensor[0].flatten(), res0_copy.flatten())
    assert torch.equal(res_tensor[1].flatten(), res1_copy.flatten())
    assert torch.equal(res_tensor[2].flatten(), res2_copy.flatten())
    assert torch.equal(res_tensor[3].flatten(), res3_copy.flatten())
    pypto.runtime._device_fini()
