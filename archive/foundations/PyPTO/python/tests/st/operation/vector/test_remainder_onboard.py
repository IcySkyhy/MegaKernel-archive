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

import math
import os

import torch

import pypto


def test_remainder_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (72, 71)
    view_shape = (32, 32)
    tile_shape = (32, 32)
    pypto.runtime._device_init()

    input1 = pypto.tensor(shape, pypto.DT_INT16, "PTO_TENSOR_input1")
    input2 = pypto.tensor(shape, pypto.DT_INT16, "PTO_TENSOR_input2")
    output = pypto.tensor(shape, pypto.DT_INT16, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    with pypto.function("MAIN", input1, input2, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                valid_shape = [
                    pypto.min(
                        pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                    ),
                    pypto.min(
                        pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                    ),
                ]
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]
                view_tensor_a = pypto.view(input1, view_shape, offsets, valid_shape=valid_shape)
                view_tensor_b = pypto.view(input2, view_shape, offsets, valid_shape=valid_shape)
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_tensor_c = pypto.remainder(view_tensor_a, view_tensor_b)
                pypto.assemble(view_tensor_c, offsets, output)
                del view_tensor_a, view_tensor_b, view_tensor_c

    a_tensor = torch.randint(low=1, high=100, size=[shape[0], shape[1]], dtype=torch.int16)
    b_tensor = torch.randint(low=-100, high=-1, size=[shape[0], shape[1]], dtype=torch.int16)
    out_tensor = torch.zeros(shape[0], shape[1], dtype=torch.int16)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_out_tensor = pypto.from_torch(out_tensor, "out_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_out_tensor)

    golden = torch.remainder(a_tensor, b_tensor)
    assert torch.allclose(out_tensor.flatten(), golden.flatten(), rtol=1e-4, atol=1e-5)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": False})
def test_remainders_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (72, 71)
    scalar = 3
    view_shape = (32, 32)
    tile_shape = (16, 16)
    pypto.runtime._device_init()

    input1 = pypto.tensor(shape, pypto.DT_INT16, "PTO_TENSOR_input1")
    output = pypto.tensor(shape, pypto.DT_INT16, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    with pypto.function("MAIN", input1, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                valid_shape = [
                    pypto.min(
                        pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                    ),
                    pypto.min(
                        pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                    ),
                ]
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]
                view_tensor_a = pypto.view(input1, view_shape, offsets, valid_shape=valid_shape)
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_tensor_b = pypto.remainder(view_tensor_a, scalar)
                pypto.assemble(view_tensor_b, offsets, output)
                del view_tensor_a, view_tensor_b

    a_tensor = torch.randint(low=1, high=100, size=[shape[0], shape[1]], dtype=torch.int16)
    out_tensor = torch.zeros(shape[0], shape[1], dtype=torch.int16)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_out_tensor = pypto.from_torch(out_tensor, "out_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_out_tensor)

    golden = torch.remainder(a_tensor, scalar)
    assert torch.allclose(out_tensor.flatten(), golden.flatten(), rtol=1e-4, atol=1e-5)
    pypto.runtime._device_fini()


def test_remainderrs_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (72, 71)
    scalar = 3.2
    view_shape = (32, 32)
    tile_shape = (16, 16)
    pypto.runtime._device_init()

    input1 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_input1")
    output = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    with pypto.function("MAIN", input1, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                valid_shape = [
                    pypto.min(
                        pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                    ),
                    pypto.min(
                        pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                    ),
                ]
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]
                view_tensor_a = pypto.view(input1, view_shape, offsets, valid_shape=valid_shape)
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_tensor_b = pypto.remainder(scalar, view_tensor_a)
                pypto.assemble(view_tensor_b, offsets, output)
                del view_tensor_a, view_tensor_b

    a_tensor = torch.randn(shape, dtype=torch.float32)
    out_tensor = torch.zeros(shape[0], shape[1], dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_out_tensor = pypto.from_torch(out_tensor, "out_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_out_tensor)

    golden = torch.remainder(scalar, a_tensor)
    assert torch.allclose(out_tensor.flatten(), golden.flatten(), rtol=1e-4, atol=1e-5)
    pypto.runtime._device_fini()
