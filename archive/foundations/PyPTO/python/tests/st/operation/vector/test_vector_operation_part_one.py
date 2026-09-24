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
def test_vector_operation_add():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ADD_TENSOR_a")
    b = pypto.tensor(shape, dtype, "ADD_TENSOR_b")
    c = pypto.tensor(shape, dtype, "ADD_TENSOR_c")

    with pypto.function("ADD", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ADD_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ADD_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
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
                tile_a.move(pypto.add(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = a_tensor + b_tensor
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_div():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "DIV_TENSOR_a")
    b = pypto.tensor(shape, dtype, "DIV_TENSOR_b")
    c = pypto.tensor(shape, dtype, "DIV_TENSOR_c")

    with pypto.function("DIV", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_DIV_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_DIV_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
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
                tile_a.move(pypto.div(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.rand(n, m, dtype=torch.float32) * 99 + 1
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.div(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_mul():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "MUL_TENSOR_a")
    b = pypto.tensor(shape, dtype, "MUL_TENSOR_b")
    c = pypto.tensor(shape, dtype, "MUL_TENSOR_c")

    with pypto.function("MUL", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MUL_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_MUL_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
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
                tile_a.move(pypto.mul(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.mul(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_sub():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SUB_TENSOR_a")
    b = pypto.tensor(shape, dtype, "SUB_TENSOR_b")
    c = pypto.tensor(shape, dtype, "SUB_TENSOR_c")

    with pypto.function("SUB", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SUB_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SUB_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
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
                tile_a.move(pypto.sub(tile_a, tile_b))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = a_tensor - b_tensor
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_abs():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ABS_TENSOR_a")
    b = pypto.tensor(shape, dtype, "ABS_TENSOR_b")

    with pypto.function("ABS", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ABS_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ABS_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.abs(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.abs(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_sqrt():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pypto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pypto.function("SQRT", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SQRT_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SQRT_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.sqrt(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.sqrt(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_ceil():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "CEIL_TENSOR_a")
    b = pypto.tensor(shape, dtype, "CEIL_TENSOR_b")

    with pypto.function("CEIL", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CEIL_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_CEIL_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.ceil(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) * 200) - 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.ceil(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_floor():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "FLOOR_TENSOR_a")
    b = pypto.tensor(shape, dtype, "FLOOR_TENSOR_b")

    with pypto.function("FLOOR", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_FLOOR_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_FLOOR_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.floor(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) * 200) - 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.floor(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_trunc():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "TRUNC_TENSOR_a")
    b = pypto.tensor(shape, dtype, "TRUNC_TENSOR_b")

    with pypto.function("TRUNC", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_TRUNC_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_TRUNC_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.trunc(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) * 200) - 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.trunc(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_exp():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pypto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pypto.function("EXP", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SQRT_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SQRT_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.exp(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.exp(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_neg():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor((n, m), dtype, "NEG_TENSOR_a")
    b = pypto.tensor((n, m), dtype, "NEG_TENSOR_b")

    with pypto.function("NEG", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_NEG_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_NEG_L1", idx_name="s_idx"):
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
                tile_a.move(pypto.neg(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = -a_tensor
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_full():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor((n, m), dtype, "VEC_DUP_TENSOR_a")
    b = 2.0

    with pypto.function("VEC_DUP", a):
        pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_VEC_DUP_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_VEC_DUP_L1", idx_name="s_idx"):
                tile_a = pypto.tensor()
                tile_a.move(
                    pypto.full(
                        view_shape,
                        b,
                        dtype,
                        valid_shape=[
                            (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                pypto.symbolic_scalar(view_shape[0])
                            ),
                            (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                pypto.symbolic_scalar(view_shape[1])
                            ),
                        ],
                    )
                )
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], a)

    a_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor)

    expected = torch.full((n, m), 2, dtype=torch.float32)
    assert_allclose(a_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_logical_not():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_FP32, "LOGICALNOT_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_BOOL, "LOGICALNOT_TENSOR_b")

    with pypto.function("LOGICALNOT", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOGICALNOT_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_LOGICALNOT_L1", idx_name="s_idx"):
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
                tmp_a = pypto.tensor(view_shape, pypto.DT_BOOL)
                tmp_a.move(pypto.logical_not(tile_a))
                pypto.assemble(tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 6 - 1.5  # 生成 [-3, 3] 范围
    b_tensor = torch.ones(n, m, dtype=torch.bool)  # 使用 torch.bool 类型

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.logical_not(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_expand():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor((n, 1), dtype, "EXPAND_TENSOR_a")
    b = pypto.tensor((n, m), dtype, "EXPAND_TENSOR_b")

    with pypto.function("EXPAND", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPAND_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXPAND_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    [16, 1],
                    [b_idx * view_shape[0], 0],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        1,
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pypto.tensor()
                tmp_a.move(
                    pypto.expand_clone(
                        tile_a,
                        view_shape,
                        valid_shape=[
                            (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                pypto.symbolic_scalar(view_shape[0])
                            ),
                            (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                pypto.symbolic_scalar(view_shape[1])
                            ),
                        ],
                    )
                )
                pypto.assemble(tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.full((n, 1), -16, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.full((n, m), -16, dtype=torch.float32)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_concat():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 32)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "CONCAT_TENSOR_a")
    b = pypto.tensor(shape, dtype, "CONCAT_TENSOR_b")
    c = pypto.tensor([n, m * 2], dtype, "CONCAT_TENSOR_c")

    with pypto.function("CONCAT", a, b, c):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CONCAT_L0", idx_name="b_idx"):
            tile_a = pypto.view(
                a,
                view_shape,
                [b_idx * view_shape[0], 0],
                valid_shape=[
                    (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                    n,
                ],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [b_idx * view_shape[0], 0],
                valid_shape=[
                    (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                    n,
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_c = pypto.tensor([16, 64], dtype)
            tmp_c.move(pypto.concat([tile_a, tile_b], -1))
            pypto.assemble(tmp_c, [b_idx * view_shape[0], 0], c)

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, 2 * m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.cat([a_tensor, b_tensor], dim=-1)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_rowmaxsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ROWMAXSINGLE_TENSOR_a")
    b = pypto.tensor(output_shape, dtype, "ROWMAXSINGLE_TENSOR_b")
    dim = 0

    with pypto.function("ROWMAXSINGLE", a, b):
        for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWMAXSINGLE_L1", idx_name="s_idx"):
            tile_a = pypto.view(
                a,
                [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(n),
                    (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pypto.tensor([1, view_shape[1]], dtype)
            tmp_a.move(pypto.amax(tile_a, dim, True))
            pypto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)

    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = a_tensor.max(dim=dim, keepdim=True)[0].reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_rowsumsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ROWSUMSINGLE_TENSOR_a")
    b = pypto.tensor(output_shape, dtype, "ROWSUMSINGLE_TENSOR_b")
    dim = 0

    with pypto.function("ROWSUMSINGLE", a, b):
        for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWSUMSINGLE_L1", idx_name="s_idx"):
            tile_a = pypto.view(
                a,
                [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(n),
                    (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pypto.tensor([1, view_shape[1]], dtype)
            tmp_a.move(pypto.sum(tile_a, dim, True))
            pypto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)

    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = a_tensor.sum(dim=dim, keepdim=True).reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_rowminsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ROWMINSINGLE_TENSOR_a")
    b = pypto.tensor(output_shape, dtype, "ROWMINSINGLE_TENSOR_b")
    dim = 0

    with pypto.function("ROWMINSINGLE", a, b):
        for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWMINSINGLE_L1", idx_name="s_idx"):
            tile_a = pypto.view(
                a,
                [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(n),
                    (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pypto.tensor([1, view_shape[1]], dtype)
            tmp_a.move(pypto.amin(tile_a, dim, True))
            pypto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)

    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = a_tensor.min(dim=dim, keepdim=True)[0].reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_rowargmaxsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (n, 16)
    tile_shape = (n, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ROWARGMAXSINGLE_TENSOR_a")
    b = pypto.tensor(output_shape, pypto.DT_INT32, "ROWARGMAXSINGLE_TENSOR_b")
    dim = 0

    with pypto.function("ROWARGMAXSINGLE", a, b):
        for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWARGMAXSINGLE_L1", idx_name="s_idx"):
            tile_a = pypto.view(
                a,
                [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(n),
                    (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pypto.tensor([1, view_shape[1]], pypto.DT_INT32)
            tmp_a.move(pypto.argmax(tile_a, dim, True))
            pypto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)

    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = a_tensor.argmax(dim=dim, keepdim=True).reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_rowargminsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (n, 16)
    tile_shape = (n, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ROWARGMINSINGLE_TENSOR_a")
    b = pypto.tensor(output_shape, pypto.DT_INT32, "ROWARGMINSINGLE_TENSOR_b")
    dim = 0

    with pypto.function("ROWARGMINSINGLE", a, b):
        for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWARGMINSINGLE_L1", idx_name="s_idx"):
            tile_a = pypto.view(
                a,
                [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(n),
                    (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pypto.tensor([1, view_shape[1]], pypto.DT_INT32)
            tmp_a.move(pypto.argmin(tile_a, dim, True))
            pypto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)

    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = a_tensor.argmin(dim=dim, keepdim=True).reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_tensor_operation_expand():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor((n, 1), dtype, "EXPAND_TENSOR_a")
    b = pypto.tensor((n, m), dtype, "EXPAND_TENSOR_b")

    with pypto.function("EXPAND", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPAND_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXPAND_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    [16, 1],
                    [b_idx * view_shape[0], 0],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        1,
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pypto.tensor()
                tmp_a.move(
                    tile_a.expand_clone(
                        view_shape,
                        valid_shape=[
                            (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                pypto.symbolic_scalar(view_shape[0])
                            ),
                            (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                pypto.symbolic_scalar(view_shape[1])
                            ),
                        ],
                    )
                )
                pypto.assemble(tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.full((n, 1), -16, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.full((n, m), -16, dtype=torch.float32)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()
