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
import torch._prims as prims

import pypto


class VarParam:
    def __init__(self, _dim, _correction, _keepdim, _is_tensor_func):
        self.dim = _dim
        self.correction = _correction
        self.keepdim = _keepdim
        self.is_tensor_func = _is_tensor_func


def var_2dim_tensor_proc(input_shape, dst_shape, param):
    pypto.runtime._device_init()

    b, s = input_shape
    view_shape = (b, s)
    tile_shape = (b, s)

    input_tensor = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_SELF")
    dst_tensor = pypto.tensor(dst_shape, pypto.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])
    s_loop_num = math.ceil(input_shape[1] / view_shape[1])
    with pypto.function("MAIN", input_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                view_tensor_input = pypto.view(
                    input_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(input_shape[0] - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])),
                        pypto.min(input_shape[1] - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                if param.is_tensor_func:
                    dst_tensor.move(
                        view_tensor_input.var(param.dim, correction=param.correction, keepdim=param.keepdim)
                    )
                else:
                    dst_tensor.move(
                        pypto.var(view_tensor_input, param.dim, correction=param.correction, keepdim=param.keepdim)
                    )

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*input_shape, dtype=torch.float32)
    c_tensor = torch.zeros(*dst_shape)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_c_tensor)

    result = torch.var(input0_tensor, param.dim, correction=param.correction, keepdim=param.keepdim)

    assert torch.allclose(c_tensor, result)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_var0_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [1], VarParam(None, 1, False, False))


@pypto.options(pass_options={"enable_slice": True})
def test_var1_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [b], VarParam(1, 1, False, False))


@pypto.options(pass_options={"enable_slice": True})
def test_var2_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [1], VarParam((), 1, False, False))


@pypto.options(pass_options={"enable_slice": True})
def test_var3_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [1], VarParam([], 1, False, False))


@pypto.options(pass_options={"enable_slice": True})
def test_var4_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [1], VarParam([0, 1], 1, False, False))


@pypto.options(pass_options={"enable_slice": True})
def test_var5_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [8], VarParam([-2], 1, False, False))


@pypto.options(pass_options={"enable_slice": True})
def test_var6_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [8], VarParam([-2], 1, False, True))


@pypto.options(pass_options={"enable_slice": True})
def test_var7_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    var_2dim_tensor_proc([b, s], [8], VarParam((-2,), 1, True, False))


def prims_var_2dim_tensor_proc(input_shape, dst_shape, dim, correction):
    pypto.runtime._device_init()

    b, s = input_shape
    view_shape = (b, s)
    tile_shape = (b, s)

    input_tensor = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_SELF")
    dst_tensor = pypto.tensor(dst_shape, pypto.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])
    s_loop_num = math.ceil(input_shape[1] / view_shape[1])
    with pypto.function("MAIN", input_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                view_tensor_input = pypto.view(
                    input_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(input_shape[0] - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])),
                        pypto.min(input_shape[1] - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                dst_tensor.move(pypto.var(view_tensor_input, dim, correction))

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*input_shape, dtype=torch.float32)
    c_tensor = torch.zeros(*dst_shape)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_c_tensor)

    result = prims.var(input0_tensor, dim, correction)

    assert torch.allclose(c_tensor, result)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_var8_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    dim = [0]
    correction = 1
    prims_var_2dim_tensor_proc([b, s], [8], dim, correction)
