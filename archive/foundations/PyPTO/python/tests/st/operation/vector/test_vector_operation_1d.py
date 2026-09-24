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

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

import pypto


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_add():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ADD_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "ADD_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "ADD_1D_TENSOR_c")

    with pypto.function("ADD_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ADD_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.add(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = a_tensor + b_tensor
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_sub():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SUB_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "SUB_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "SUB_1D_TENSOR_c")

    with pypto.function("SUB_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SUB_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.sub(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = a_tensor - b_tensor
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_mul():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "MUL_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "MUL_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "MUL_1D_TENSOR_c")

    with pypto.function("MUL_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MUL_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.mul(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.mul(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_div():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "DIV_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "DIV_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "DIV_1D_TENSOR_c")

    with pypto.function("DIV_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_DIV_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.div(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.rand(n, dtype=torch.float32) * 99 + 1
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.div(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_abs():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ABS_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "ABS_1D_TENSOR_b")

    with pypto.function("ABS_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ABS_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.abs(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.abs(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_sqrt():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SQRT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "SQRT_1D_TENSOR_b")

    with pypto.function("SQRT_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SQRT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.sqrt(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.sqrt(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_ceil():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "CEIL_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "CEIL_1D_TENSOR_b")

    with pypto.function("CEIL_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CEIL_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.ceil(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) * 200) - 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.ceil(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_floor():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "FLOOR_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "FLOOR_1D_TENSOR_b")

    with pypto.function("FLOOR_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_FLOOR_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.floor(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) * 200) - 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.floor(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_trunc():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, dtype, "TRUNC_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "TRUNC_1D_TENSOR_b")

    with pypto.function("TRUNC_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_TRUNC_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.trunc(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) * 200) - 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.trunc(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_exp():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXP_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "EXP_1D_TENSOR_b")

    with pypto.function("EXP_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXP_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.exp(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.exp(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_neg():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()

    a = pypto.tensor((n,), dtype, "NEG_1D_TENSOR_a")
    b = pypto.tensor((n,), dtype, "NEG_1D_TENSOR_b")

    with pypto.function("NEG_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_NEG_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.neg(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = -a_tensor
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_reciprocal():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "RECIPROCAL_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "RECIPROCAL_1D_TENSOR_b")

    with pypto.function("RECIPROCAL_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_RECIPROCAL_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.reciprocal(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = torch.rand(n, dtype=torch.float32) * 99 + 1
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.reciprocal(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_rsqrt():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "RSQRT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "RSQRT_1D_TENSOR_b")

    with pypto.function("RSQRT_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_RSQRT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.rsqrt(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.rsqrt(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_relu():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "RELU_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "RELU_1D_TENSOR_b")

    with pypto.function("RELU_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_RELU_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.relu(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.relu(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_lrelu():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "LRELU_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "LRELU_1D_TENSOR_b")

    with pypto.function("LRELU_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LRELU_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.lrelu(tile_a, 0.01))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.nn.functional.leaky_relu(a_tensor, 0.01)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_bitwise_not():
    dtype = pypto.DT_INT16
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "BITWISE_NOT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "BITWISE_NOT_1D_TENSOR_b")

    with pypto.function("BITWISE_NOT_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_BITWISE_NOT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.bitwise_not(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    b_tensor = torch.zeros(n, dtype=torch.int16)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.bitwise_not(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_bitwise_and():
    dtype = pypto.DT_INT16
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "BITWISE_AND_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "BITWISE_AND_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "BITWISE_AND_1D_TENSOR_c")

    with pypto.function("BITWISE_AND_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_BITWISE_AND_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.bitwise_and(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    b_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    c_tensor = torch.zeros(n, dtype=torch.int16)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.bitwise_and(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_bitwise_or():
    dtype = pypto.DT_INT16
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "BITWISE_OR_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "BITWISE_OR_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "BITWISE_OR_1D_TENSOR_c")

    with pypto.function("BITWISE_OR_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_BITWISE_OR_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.bitwise_or(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    b_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    c_tensor = torch.zeros(n, dtype=torch.int16)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.bitwise_or(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_bitwise_xor():
    dtype = pypto.DT_INT16
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "BITWISE_XOR_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "BITWISE_XOR_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "BITWISE_XOR_1D_TENSOR_c")

    with pypto.function("BITWISE_XOR_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_BITWISE_XOR_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.bitwise_xor(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    b_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    c_tensor = torch.zeros(n, dtype=torch.int16)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.bitwise_xor(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_bitwise_left_shift():
    dtype = pypto.DT_INT16
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "BITWISE_LEFT_SHIFT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "BITWISE_LEFT_SHIFT_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "BITWISE_LEFT_SHIFT_1D_TENSOR_c")

    with pypto.function("BITWISE_LEFT_SHIFT_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_BITWISE_LEFT_SHIFT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.bitwise_left_shift(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    b_tensor = torch.randint(0, 5, (n,), dtype=torch.int16)  # 避免位移过大
    c_tensor = torch.zeros(n, dtype=torch.int16)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.bitwise_left_shift(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_bitwise_right_shift():
    dtype = pypto.DT_INT16
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "BITWISE_RIGHT_SHIFT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "BITWISE_RIGHT_SHIFT_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "BITWISE_RIGHT_SHIFT_1D_TENSOR_c")

    with pypto.function("BITWISE_RIGHT_SHIFT_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_BITWISE_RIGHT_SHIFT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.bitwise_right_shift(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int16)
    b_tensor = torch.randint(0, 5, (n,), dtype=torch.int16)  # 避免位移过大
    c_tensor = torch.zeros(n, dtype=torch.int16)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.bitwise_right_shift(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_eq():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EQ_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "EQ_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "EQ_1D_TENSOR_c")

    with pypto.function("EQ_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EQ_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.eq(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.eq(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_ne():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "NE_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "NE_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "NE_1D_TENSOR_c")

    with pypto.function("NE_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_NE_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.ne(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.ne(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_ge():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "GE_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "GE_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "GE_1D_TENSOR_c")

    with pypto.function("GE_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_GE_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.ge(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.ge(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_gt():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "GT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "GT_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "GT_1D_TENSOR_c")

    with pypto.function("GT_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_GT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.gt(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.gt(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_le():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "LE_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "LE_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "LE_1D_TENSOR_c")

    with pypto.function("LE_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LE_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.le(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.le(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_lt():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "LT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "LT_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "LT_1D_TENSOR_c")

    with pypto.function("LT_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.lt(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.lt(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_logical_and():
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()

    a = pypto.tensor(shape, pypto.DT_BOOL, "LOGICAL_AND_1D_TENSOR_a")
    b = pypto.tensor(shape, pypto.DT_BOOL, "LOGICAL_AND_1D_TENSOR_b")
    c = pypto.tensor(shape, pypto.DT_BOOL, "LOGICAL_AND_1D_TENSOR_c")

    with pypto.function("LOGICAL_AND_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOGICAL_AND_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.logical_and(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 2, (n,), dtype=torch.bool)
    b_tensor = torch.randint(0, 2, (n,), dtype=torch.bool)
    c_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.logical_and(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_maximum():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "MAXIMUM_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "MAXIMUM_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "MAXIMUM_1D_TENSOR_c")

    with pypto.function("MAXIMUM_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MAXIMUM_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.maximum(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.maximum(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_minimum():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "MINIMUM_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "MINIMUM_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "MINIMUM_1D_TENSOR_c")

    with pypto.function("MINIMUM_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MINIMUM_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.minimum(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.minimum(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_cbrt():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "CBRT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "CBRT_1D_TENSOR_b")

    with pypto.function("CBRT_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CBRT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.cbrt(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch._prims.cbrt(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_clip():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "CLIP_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "CLIP_1D_TENSOR_b")

    with pypto.function("CLIP_1D", a, b):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CLIP_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.clip(tile_a, 0.0, 100.0))
            pypto.assemble(tile_a, [i * view_shape[0]], b)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.clip(a_tensor, 0.0, 100.0)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_copysign():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "COPYSIGN_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "COPYSIGN_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "COPYSIGN_1D_TENSOR_c")

    with pypto.function("COPYSIGN_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_COPYSIGN_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.copysign(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.copysign(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_fmod():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "FMOD_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "FMOD_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "FMOD_1D_TENSOR_c")

    with pypto.function("FMOD_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_FMOD_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.fmod(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 99 + 1
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.fmod(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_gcd():
    dtype = pypto.DT_INT32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "GCD_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "GCD_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "GCD_1D_TENSOR_c")

    with pypto.function("GCD_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_GCD_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.gcd(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(1, 100, (n,), dtype=torch.int32)
    b_tensor = torch.randint(1, 100, (n,), dtype=torch.int32)
    c_tensor = torch.zeros(n, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.gcd(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_hypot():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "HYPOT_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "HYPOT_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "HYPOT_1D_TENSOR_c")

    with pypto.function("HYPOT_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_HYPOT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.hypot(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.hypot(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_floor_div():
    dtype = pypto.DT_INT32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "FLOOR_DIV_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "FLOOR_DIV_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "FLOOR_DIV_1D_TENSOR_c")

    with pypto.function("FLOOR_DIV_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_FLOOR_DIV_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.floor_div(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(-100, 100, (n,), dtype=torch.int32)
    b_tensor = torch.randint(1, 100, (n,), dtype=torch.int32)
    c_tensor = torch.zeros(n, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.floor_divide(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_expand_exp_dif():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    b_shape = (1,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXPAND_EXP_DIF_1D_TENSOR_a")
    b = pypto.tensor(b_shape, dtype, "EXPAND_EXP_DIF_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "EXPAND_EXP_DIF_1D_TENSOR_c")

    with pypto.function("EXPAND_EXP_DIF_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPAND_EXP_DIF_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.expand_exp_dif(tile_a, b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.rand(n, dtype=torch.float32) * 10
    b_tensor = torch.rand(1, dtype=torch.float32) * 10
    c_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.exp(a_tensor - b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_ceil_div():
    dtype = pypto.DT_INT32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "CEIL_DIV_1D_TENSOR_a")
    b = pypto.tensor(shape, dtype, "CEIL_DIV_1D_TENSOR_b")
    c = pypto.tensor(shape, dtype, "CEIL_DIV_1D_TENSOR_c")

    with pypto.function("CEIL_DIV_1D", a, b, c):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CEIL_DIV_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_b = pypto.view(
                b,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.ceil_div(tile_a, tile_b))
            pypto.assemble(tile_a, [i * view_shape[0]], c)

    a_tensor = torch.randint(0, 100, (n,), dtype=torch.int32)
    b_tensor = torch.randint(1, 100, (n,), dtype=torch.int32)
    c_tensor = torch.zeros(n, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    expected = torch.ceil(torch.div(a_tensor, b_tensor)).to(torch.int32)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_argsort():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ARGSORT_1D_TENSOR_a")
    indices = pypto.tensor(shape, pypto.DT_INT32, "ARGSORT_1D_TENSOR_indices")

    b_loop_num = int(np.ceil(n / view_shape[0]))
    with pypto.function("ARGSORT_1D", a, indices):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_ARGSORT_1D_L0", idx_name="b_idx"):
            view_tensor_a = pypto.view(
                a,
                view_shape,
                [b_idx * view_shape[0]],
                valid_shape=[
                    pypto.min(pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0]))
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            view_tensor_a.move(pypto.argsort(view_tensor_a, 0, True))
            pypto.assemble(view_tensor_a, [b_idx * view_shape[0]], indices)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    indices_tensor = torch.zeros(n, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_indices_tensor = pypto.from_torch(indices_tensor, "indices_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_indices_tensor)

    expected_indices = torch.argsort(a_tensor, dim=0, descending=True)
    assert_allclose(indices_tensor.flatten(), expected_indices.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_index_put():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    values_size = n // 2
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "INDEX_PUT_1D_TENSOR_a")
    indices = pypto.tensor((values_size,), pypto.DT_INT32, "INDEX_PUT_1D_TENSOR_indices")
    values = pypto.tensor((values_size,), dtype, "INDEX_PUT_1D_TENSOR_values")

    b_loop_num = int(np.ceil(values_size / view_shape[0]))
    with pypto.function("INDEX_PUT_1D", a, indices, values):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_INDEX_PUT_1D_L0", idx_name="b_idx"):
            pypto.set_vec_tile_shapes(tile_shape[0])
            view_values = pypto.view(
                values,
                [view_shape[0]],
                [b_idx * view_shape[0]],
                valid_shape=[
                    pypto.min(
                        pypto.symbolic_scalar(values_size) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                    )
                ],
            )
            view_indices = pypto.view(
                indices,
                [view_shape[0]],
                [b_idx * view_shape[0]],
                valid_shape=[
                    pypto.min(
                        pypto.symbolic_scalar(values_size) - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])
                    )
                ],
            )
            pypto.index_put_(a, (view_indices,), view_values)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    indices_tensor = torch.from_numpy(np.random.choice(range(0, n), (values_size,), False).astype(np.int32))
    values_tensor = torch.rand(values_size, dtype=torch.float32) * 100

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_indices_tensor = pypto.from_torch(indices_tensor, "indices_tensor")
    pto_values_tensor = pypto.from_torch(values_tensor, "values_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_indices_tensor, pto_values_tensor)

    expected = a_tensor.clone()
    expected[indices_tensor] = values_tensor
    assert_allclose(a_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_where():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    condition = pypto.tensor(shape, pypto.DT_BOOL, "WHERE_1D_TENSOR_condition")
    x = pypto.tensor(shape, dtype, "WHERE_1D_TENSOR_x")
    y = pypto.tensor(shape, dtype, "WHERE_1D_TENSOR_y")
    output = pypto.tensor(shape, dtype, "WHERE_1D_TENSOR_output")

    with pypto.function("WHERE_1D", condition, x, y, output):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_WHERE_1D_L0", idx_name="i"):
            tile_condition = pypto.view(
                condition,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_x = pypto.view(
                x,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            tile_y = pypto.view(
                y,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_x.move(pypto.where(tile_condition, tile_x, tile_y))
            pypto.assemble(tile_x, [i * view_shape[0]], output)

    condition_tensor = torch.randint(0, 2, (n,)).bool()
    x_tensor = torch.rand(n, dtype=torch.float32) * 100
    y_tensor = torch.rand(n, dtype=torch.float32) * 100
    output_tensor = torch.zeros(n, dtype=torch.float32)

    pto_condition_tensor = pypto.from_torch(condition_tensor, "condition_tensor")
    pto_x_tensor = pypto.from_torch(x_tensor, "x_tensor")
    pto_y_tensor = pypto.from_torch(y_tensor, "y_tensor")
    pto_output_tensor = pypto.from_torch(output_tensor, "output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_condition_tensor, pto_x_tensor, pto_y_tensor, pto_output_tensor)

    expected = torch.where(condition_tensor, x_tensor, y_tensor)
    assert_allclose(output_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_topk():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    k = 5
    shape = (n,)
    view_shape = (n,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "TOPK_1D_TENSOR_a")
    values = pypto.tensor((k,), dtype, "TOPK_1D_TENSOR_values")
    indices = pypto.tensor((k,), pypto.DT_INT32, "TOPK_1D_TENSOR_indices")

    with pypto.function("TOPK_1D", a, values, indices):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_TOPK_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            result = pypto.topk(tile_a, k)
            pypto.assemble(result[0], [i * view_shape[0]], values)
            pypto.assemble(result[1], [i * view_shape[0]], indices)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    values_tensor = torch.zeros(k, dtype=torch.float32)
    indices_tensor = torch.zeros(k, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_values_tensor = pypto.from_torch(values_tensor, "values_tensor")
    pto_indices_tensor = pypto.from_torch(indices_tensor, "indices_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_values_tensor, pto_indices_tensor)

    expected_values, expected_indices = torch.topk(a_tensor, k)
    assert_allclose(values_tensor.flatten(), expected_values.flatten(), rtol=1e-3, atol=1e-3)
    assert_allclose(indices_tensor.flatten(), expected_indices.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_expand():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    input_shape = (1,)
    expand_shape = (n,)
    view_shape = (16,)
    tile_shape = (8,)
    pypto.runtime._device_init()
    a = pypto.tensor(input_shape, dtype, "EXPAND_1D_TENSOR_a")
    output = pypto.tensor(expand_shape, dtype, "EXPAND_1D_TENSOR_output")

    with pypto.function("EXPAND_1D", a, output):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPAND_1D_L0", idx_name="i"):
            pypto.set_vec_tile_shapes(tile_shape[0])
            result = pypto.expand_clone(a, view_shape)
            pypto.assemble(result, [i * view_shape[0]], output)

    a_tensor = torch.rand(1, dtype=torch.float32) * 100
    output_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_output_tensor = pypto.from_torch(output_tensor, "output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_output_tensor)

    expected = a_tensor.expand(n)
    assert_allclose(output_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_sum():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SUM_1D_TENSOR_a")
    result = pypto.tensor((1,), dtype, "SUM_1D_TENSOR_result")

    with pypto.function("SUM_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SUM_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.sum(tile_a, 0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.sum(a_tensor, dim=0, keepdim=True)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_max():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "MAX_1D_TENSOR_a")
    result = pypto.tensor((1,), dtype, "MAX_1D_TENSOR_result")

    with pypto.function("MAX_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MAX_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.amax(tile_a, 0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.amax(a_tensor, dim=0, keepdim=True)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_min():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "MIN_1D_TENSOR_a")
    result = pypto.tensor((1,), dtype, "MIN_1D_TENSOR_result")

    with pypto.function("MIN_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MIN_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.amin(tile_a, 0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.amin(a_tensor, dim=0, keepdim=True)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_prod():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "PROD_1D_TENSOR_a")
    result = pypto.tensor((1,), dtype, "PROD_1D_TENSOR_result")

    with pypto.function("PROD_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_PROD_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.prod(tile_a, 0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.prod(a_tensor, dim=0, keepdim=True)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_argmax():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (n,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ARGMAX_1D_TENSOR_a")
    result = pypto.tensor((1,), pypto.DT_INT32, "ARGMAX_1D_TENSOR_result")

    with pypto.function("ARGMAX_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ARGMAX_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.argmax(tile_a, 0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.argmax(a_tensor, dim=0, keepdim=True)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_argmin():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (n,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ARGMIN_1D_TENSOR_a")
    result = pypto.tensor((1,), pypto.DT_INT32, "ARGMIN_1D_TENSOR_result")

    with pypto.function("ARGMIN_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ARGMIN_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.argmin(tile_a, 0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.argmin(a_tensor, dim=0, keepdim=True)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_logicnot():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "LOGICNOT_1D_TENSOR_a")
    result = pypto.tensor(shape, pypto.DT_BOOL, "LOGICNOT_1D_TENSOR_result")

    with pypto.function("LOGICNOT_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOGICNOT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.logical_not(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.logical_not(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_sign():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SIGN_1D_TENSOR_a")
    result = pypto.tensor(shape, dtype, "SIGN_1D_TENSOR_result")

    with pypto.function("SIGN_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SIGN_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.sign(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.randn(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.sign(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_signbit():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SIGNBIT_1D_TENSOR_a")
    result = pypto.tensor(shape, pypto.DT_BOOL, "SIGNBIT_1D_TENSOR_result")

    with pypto.function("SIGNBIT_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SIGNBIT_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.signbit(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.randn(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(n, dtype=torch.bool)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.signbit(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_log1p():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "LOG1P_1D_TENSOR_a")
    result = pypto.tensor(shape, dtype, "LOG1P_1D_TENSOR_result")

    with pypto.function("LOG1P_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOG1P_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.log1p(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.log1p(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_var():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (n,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "VAR_1D_TENSOR_a")
    result = pypto.tensor((1,), dtype, "VAR_1D_TENSOR_result")

    with pypto.function("VAR_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_VAR_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_result = pypto.var(tile_a, 0, 1.0, True)
            pypto.assemble(tile_result, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(1, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.var(a_tensor, dim=0, keepdim=True, correction=1)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_exp2():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXP2_1D_TENSOR_a")
    result = pypto.tensor(shape, dtype, "EXP2_1D_TENSOR_result")

    with pypto.function("EXP2_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXP2_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.exp2(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.rand(n, dtype=torch.float32) * 10
    result_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.exp2(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_round():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "ROUND_1D_TENSOR_a")
    result = pypto.tensor(shape, dtype, "ROUND_1D_TENSOR_result")

    with pypto.function("ROUND_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ROUND_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.round(tile_a, 0))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.randn(n, dtype=torch.float32) * 100
    result_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.round(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="only local test")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_1d_expm1():
    dtype = pypto.DT_FP32
    tiling = 32
    n = tiling * 4
    shape = (n,)
    view_shape = (tiling,)
    tile_shape = (32,)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "EXPM1_1D_TENSOR_a")
    result = pypto.tensor(shape, dtype, "EXPM1_1D_TENSOR_result")

    with pypto.function("EXPM1_1D", a, result):
        for i in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPM1_1D_L0", idx_name="i"):
            tile_a = pypto.view(
                a,
                view_shape,
                [i * view_shape[0]],
                valid_shape=[(pypto.symbolic_scalar(n) - i * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            tile_a.move(pypto.expm1(tile_a))
            pypto.assemble(tile_a, [i * view_shape[0]], result)

    a_tensor = torch.randn(n, dtype=torch.float32) * 5
    result_tensor = torch.zeros(n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_result_tensor)

    expected = torch.expm1(a_tensor)
    assert_allclose(result_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


if __name__ == "__main__":
    test_vector_operation_1d_add()
    test_vector_operation_1d_sub()
    test_vector_operation_1d_mul()
    test_vector_operation_1d_div()
    test_vector_operation_1d_abs()
    test_vector_operation_1d_sqrt()
    test_vector_operation_1d_ceil()
    test_vector_operation_1d_floor()
    test_vector_operation_1d_trunc()
    test_vector_operation_1d_exp()
    test_vector_operation_1d_neg()
    test_vector_operation_1d_reciprocal()
    test_vector_operation_1d_rsqrt()
    test_vector_operation_1d_relu()
    test_vector_operation_1d_lrelu()
    test_vector_operation_1d_bitwise_not()
    test_vector_operation_1d_bitwise_and()
    test_vector_operation_1d_bitwise_or()
    test_vector_operation_1d_bitwise_xor()
    test_vector_operation_1d_bitwise_left_shift()
    test_vector_operation_1d_bitwise_right_shift()
    test_vector_operation_1d_eq()
    test_vector_operation_1d_ne()
    test_vector_operation_1d_ge()
    test_vector_operation_1d_gt()
    test_vector_operation_1d_le()
    test_vector_operation_1d_lt()
    test_vector_operation_1d_logical_and()
    test_vector_operation_1d_maximum()
    test_vector_operation_1d_minimum()
    test_vector_operation_1d_cbrt()
    test_vector_operation_1d_clip()
    test_vector_operation_1d_copysign()
    test_vector_operation_1d_fmod()
    test_vector_operation_1d_gcd()
    test_vector_operation_1d_hypot()
    test_vector_operation_1d_floor_div()
    test_vector_operation_1d_expand_exp_dif()
    test_vector_operation_1d_ceil_div()
    test_vector_operation_1d_argsort()
    test_vector_operation_1d_index_put()
    test_vector_operation_1d_where()
    test_vector_operation_1d_topk()
    test_vector_operation_1d_expand()
    test_vector_operation_1d_sum()
    test_vector_operation_1d_max()
    test_vector_operation_1d_min()
    test_vector_operation_1d_prod()
    test_vector_operation_1d_argmax()
    test_vector_operation_1d_argmin()
    test_vector_operation_1d_logicnot()
    test_vector_operation_1d_sign()
    test_vector_operation_1d_signbit()
    test_vector_operation_1d_log1p()
    test_vector_operation_1d_var()
    test_vector_operation_1d_exp2()
    test_vector_operation_1d_round()
    test_vector_operation_1d_expm1()
