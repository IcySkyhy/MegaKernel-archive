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

from numpy.testing import assert_allclose
import pytest
import torch

import pypto


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tensor_deinterleave():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    shape = (8, 64)
    view_shape = (4, 64)
    tile_shape = (2, 64)
    pypto.runtime._device_init()

    input0 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_input0")
    input1 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_input1")
    output0 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_output0")
    output1 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_output1")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    with pypto.function("MAIN", input0, input1, output0, output1):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                offset = [b_idx * view_shape[0], s_idx * view_shape[1]]
                valid_shape = [
                    pypto.min(
                        pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                        pypto.symbolic_scalar(view_shape[0]),
                    ),
                    pypto.min(
                        pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                        pypto.symbolic_scalar(view_shape[1]),
                    ),
                ]
                view_tensor0 = pypto.view(input0, view_shape, offset, valid_shape=valid_shape)
                view_tensor1 = pypto.view(input1, view_shape, offset, valid_shape=valid_shape)
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                result0, result1 = pypto.deinterleave(view_tensor0, view_tensor1)
                pypto.assemble(result0, offset, output0)
                pypto.assemble(result1, offset, output1)

    assert isinstance(output0, pypto.tensor)
    assert isinstance(output1, pypto.tensor)

    input0_tensor = torch.rand(shape[0], shape[1], dtype=torch.float32) * 20 - 10
    input1_tensor = torch.rand(shape[0], shape[1], dtype=torch.float32) * 20 - 10
    output0_tensor = torch.zeros(shape[0], shape[1], dtype=torch.float32)
    output1_tensor = torch.zeros(shape[0], shape[1], dtype=torch.float32)
    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_input1_tensor = pypto.from_torch(input1_tensor, "input1_tensor")
    pto_output0_tensor = pypto.from_torch(output0_tensor, "output0_tensor")
    pto_output1_tensor = pypto.from_torch(output1_tensor, "output1_tensor")
    pypto.runtime._device_run_once_data_from_host(
        pto_input0_tensor, pto_input1_tensor, pto_output0_tensor, pto_output1_tensor
    )

    interleaved = torch.cat((input0_tensor, input1_tensor), dim=-1)
    golden0 = interleaved[..., 0::2]
    golden1 = interleaved[..., 1::2]
    assert_allclose(output0_tensor.flatten(), golden0.flatten(), rtol=3e-3, atol=3e-3)
    assert_allclose(output1_tensor.flatten(), golden1.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tensor_deinterleave_single_input():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    input_shape = (8, 64)
    output_shape = (8, 32)
    view_shape = (4, 64)
    tile_shape = (2, 32)
    pypto.runtime._device_init()

    input0 = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_input0")
    output0 = pypto.tensor(output_shape, pypto.DT_FP32, "PTO_TENSOR_output0")
    output1 = pypto.tensor(output_shape, pypto.DT_FP32, "PTO_TENSOR_output1")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])
    s_loop_num = math.ceil(input_shape[1] / view_shape[1])
    with pypto.function("MAIN", input0, output0, output1):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                input_offset = [b_idx * view_shape[0], s_idx * view_shape[1]]
                output_offset = [b_idx * view_shape[0], s_idx * view_shape[1] // 2]
                valid_shape = [
                    pypto.min(
                        pypto.symbolic_scalar(input_shape[0]) - b_idx * view_shape[0],
                        pypto.symbolic_scalar(view_shape[0]),
                    ),
                    pypto.min(
                        pypto.symbolic_scalar(input_shape[1]) - s_idx * view_shape[1],
                        pypto.symbolic_scalar(view_shape[1]),
                    ),
                ]
                view_tensor0 = pypto.view(input0, view_shape, input_offset, valid_shape=valid_shape)
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                result0, result1 = pypto.deinterleave(view_tensor0)
                pypto.assemble(result0, output_offset, output0)
                pypto.assemble(result1, output_offset, output1)

    assert isinstance(output0, pypto.tensor)
    assert isinstance(output1, pypto.tensor)

    input0_tensor = torch.rand(input_shape[0], input_shape[1], dtype=torch.float32) * 20 - 10
    output0_tensor = torch.zeros(output_shape[0], output_shape[1], dtype=torch.float32)
    output1_tensor = torch.zeros(output_shape[0], output_shape[1], dtype=torch.float32)
    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_output0_tensor = pypto.from_torch(output0_tensor, "output0_tensor")
    pto_output1_tensor = pypto.from_torch(output1_tensor, "output1_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_output0_tensor, pto_output1_tensor)

    golden0 = input0_tensor[..., 0::2]
    golden1 = input0_tensor[..., 1::2]
    assert_allclose(output0_tensor.flatten(), golden0.flatten(), rtol=3e-3, atol=3e-3)
    assert_allclose(output1_tensor.flatten(), golden1.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()
