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

import numpy as np
from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_greater():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "Greater_TENSOR_a")
    b = pypto.tensor(shape, dtype, "Greater_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "Greater_TENSOR_c")
    with pypto.function("Greater", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_GREATER_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_GREATER_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(m) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                tile_b = pypto.view(
                    b,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.greater(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)

    a_tensor = torch.from_numpy(np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    b_tensor = torch.from_numpy(np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    c_tensor = torch.zeros(n, m, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)
    expected = torch.greater(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_greater_scalar():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 64
    n, m = tiling * 2, tiling * 2
    shape = (n, m)
    view_shape = (32, 32)
    tile_shape = (16, 16)
    scalar_value = 50.0
    pypto.runtime._device_init()
    input_tensor = pypto.tensor(shape, dtype, "GREATER_SCALAR_INPUT")
    output_tensor = pypto.tensor(shape, pypto.DT_BOOL, "GREATER_SCALAR_OUTPUT")
    with pypto.function("GreaterScalar", input_tensor, output_tensor):
        b_loop_num = int(np.ceil(n / view_shape[0]))
        s_loop_num = int(np.ceil(m / view_shape[1]))
        for b_idx in pypto.loop(b_loop_num, name="BLOCK_LOOP", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="SLICE_LOOP", idx_name="s_idx"):
                view_block = pypto.view(
                    input_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(m) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_block.move(pypto.greater(view_block, scalar_value))
                pypto.assemble(view_block, [b_idx * view_shape[0], s_idx * view_shape[1]], output_tensor)

    input_data = torch.randn(n, m, dtype=torch.float32) * 100
    output_data = torch.zeros(n, m, dtype=torch.bool)

    pto_input_tensor = pypto.from_torch(input_data, "input_data")
    pto_output_tensor = pypto.from_torch(output_data, "output_data")

    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)
    expected_result = torch.greater(input_data, scalar_value)
    assert_allclose(output_data.flatten(), expected_result.flatten(), rtol=1e-6, atol=1e-6)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_equal():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "Equal_TENSOR_a")
    b = pypto.tensor(shape, dtype, "Equal_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "Equal_TENSOR_c")
    with pypto.function("Equal", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EQUAL_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EQUAL_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(m) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                tile_b = pypto.view(
                    b,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.eq(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
    a_tensor = torch.from_numpy(np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    b_tensor = torch.from_numpy(np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    c_tensor = torch.zeros(n, m, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)
    expected = torch.eq(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_equal_scalar():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 64
    n, m = tiling * 2, tiling * 2
    shape = (n, m)
    view_shape = (32, 32)
    tile_shape = (16, 16)
    scalar_value = 50.0
    pypto.runtime._device_init()
    input_tensor = pypto.tensor(shape, dtype, "EQUAL_SCALAR_INPUT")
    output_tensor = pypto.tensor(shape, pypto.DT_BOOL, "EQUAL_SCALAR_OUTPUT")
    with pypto.function("EqualScalar", input_tensor, output_tensor):
        b_loop_num = int(np.ceil(n / view_shape[0]))
        s_loop_num = int(np.ceil(m / view_shape[1]))
        for b_idx in pypto.loop(b_loop_num, name="BLOCK_LOOP", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="SLICE_LOOP", idx_name="s_idx"):
                view_block = pypto.view(
                    input_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(m) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_block.move(pypto.eq(view_block, scalar_value))
                pypto.assemble(view_block, [b_idx * view_shape[0], s_idx * view_shape[1]], output_tensor)

    input_data = torch.randn(n, m, dtype=torch.float32) * 100
    output_data = torch.zeros(n, m, dtype=torch.bool)

    pto_input_tensor = pypto.from_torch(input_data, "input_data")
    pto_output_tensor = pypto.from_torch(output_data, "output_data")

    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)
    expected_result = torch.eq(input_data, scalar_value)
    assert_allclose(output_data.flatten(), expected_result.flatten(), rtol=1e-6, atol=1e-6)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_less():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "Less_TENSOR_a")
    b = pypto.tensor(shape, dtype, "Less_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "Less_TENSOR_c")
    with pypto.function("Less", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LESS_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_LESS_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(m) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                tile_b = pypto.view(
                    b,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.lt(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
    a_tensor = torch.from_numpy(np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    b_tensor = torch.from_numpy(np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    c_tensor = torch.zeros(n, m, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)
    expected = torch.lt(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_less_scalar():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 64
    n, m = tiling * 2, tiling * 2
    shape = (n, m)
    view_shape = (32, 32)
    tile_shape = (16, 16)
    scalar_value = 50.0
    pypto.runtime._device_init()
    input_tensor = pypto.tensor(shape, dtype, "LESS_SCALAR_INPUT")
    output_tensor = pypto.tensor(shape, pypto.DT_BOOL, "LESS_SCALAR_OUTPUT")
    with pypto.function("LessScalar", input_tensor, output_tensor):
        b_loop_num = int(np.ceil(n / view_shape[0]))
        s_loop_num = int(np.ceil(m / view_shape[1]))
        for b_idx in pypto.loop(b_loop_num, name="BLOCK_LOOP", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="SLICE_LOOP", idx_name="s_idx"):
                view_block = pypto.view(
                    input_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(m) - s_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_block.move(pypto.lt(view_block, scalar_value))
                pypto.assemble(view_block, [b_idx * view_shape[0], s_idx * view_shape[1]], output_tensor)

    input_data = torch.randn(n, m, dtype=torch.float32) * 100
    output_data = torch.zeros(n, m, dtype=torch.bool)
    pto_input_tensor = pypto.from_torch(input_data, "input_data")
    pto_output_tensor = pypto.from_torch(output_data, "output_data")

    pypto.runtime._device_run_once_data_from_host(pto_input_tensor, pto_output_tensor)
    expected_result = torch.lt(input_data, scalar_value)
    assert_allclose(output_data.flatten(), expected_result.flatten(), rtol=1e-6, atol=1e-6)

    pypto.runtime._device_fini()
