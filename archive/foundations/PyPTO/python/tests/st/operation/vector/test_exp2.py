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

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_exp2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXP2_TENSOR_a")
    b = pypto.tensor(shape, dtype, "EXP2_TENSOR_b")

    with pypto.function("EXP2", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXP2_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXP2_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.exp2(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 4 - 2
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.exp2(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_exp2_int8():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_INT8
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXP2_INT8_TENSOR_a")
    b = pypto.tensor(shape, pypto.DT_FP32, "EXP2_INT8_TENSOR_b")

    with pypto.function("EXP2_INT8", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXP2_INT8_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXP2_INT8_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.exp2(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.randint(-5, 6, (n, m), dtype=torch.int8)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.exp2(a_tensor.to(torch.float32))
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_exp2_uint8():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_UINT8
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXP2_UINT8_TENSOR_a")
    b = pypto.tensor(shape, pypto.DT_FP32, "EXP2_UINT8_TENSOR_b")

    with pypto.function("EXP2_UINT8", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXP2_UINT8_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXP2_UINT8_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.exp2(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.randint(0, 11, (n, m), dtype=torch.uint8)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.exp2(a_tensor.to(torch.float32))
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()
