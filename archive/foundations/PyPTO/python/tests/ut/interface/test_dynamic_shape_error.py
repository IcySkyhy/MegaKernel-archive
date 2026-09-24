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
"""Test pypto.frontend.jit kernel dynamic shape error behavior."""

import logging

import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="", force=True)
SIM_RUNTIME_OPTIONS = {"run_mode": pypto.RunMode.SIM}
DYNAMIC_SHAPE_ERROR = "Dynamic shape tensors are not allowed as operation operands"
ASSEMBLE_DYNAMIC_SHAPE_ERROR = "Assemble: shape of src tensor requires integer"


@pypto.frontend.jit(runtime_options=SIM_RUNTIME_OPTIONS)
def kernel_with_dynamic(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    out[:] = pypto.exp(a)


def test_kernel_dynamic_shape_error():
    """Test that dynamic shape (-1) in exp kernel operand causes CheckTensorDynamicShape error."""
    a = torch.ones(1, 8, dtype=torch.float32)
    out = torch.zeros(1, 8, dtype=torch.float32)

    with pytest.raises(Exception, match=DYNAMIC_SHAPE_ERROR):
        kernel_with_dynamic(a, out)


@pypto.frontend.jit(runtime_options=SIM_RUNTIME_OPTIONS)
def matmul_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    out[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP16)


def test_matmul_dynamic_shape_error():
    """Test that dynamic shape (-1) in matmul kernel operand causes CheckTensorDynamicShape error."""
    m, k, n = 128, 128, 128
    a = torch.rand([m, k], dtype=torch.float16)
    b = torch.rand([k, n], dtype=torch.float16)
    out = torch.zeros([m, n], dtype=torch.float16)

    with pytest.raises(Exception, match=DYNAMIC_SHAPE_ERROR):
        matmul_kernel(a, b, out)


@pypto.frontend.jit(runtime_options=SIM_RUNTIME_OPTIONS)
def one_hot_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(4, 5, 32)
    out[:] = pypto.one_hot(a, 5)


def test_one_hot_dynamic_shape_error():
    """Test that dynamic shape (-1) in one_hot kernel operand causes CheckTensorDynamicShape error."""
    x = torch.tensor([0, 2, 4], dtype=torch.int32)
    out = torch.zeros([3, 5], dtype=torch.int32)

    with pytest.raises(Exception, match=DYNAMIC_SHAPE_ERROR):
        one_hot_kernel(x, out)


@pypto.frontend.jit(runtime_options=SIM_RUNTIME_OPTIONS)
def assemble_kernel_with_dynamic_input(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.assemble(a, [0, 0], out)


def test_assemble_dynamic_shape_error():
    """Test that Assemble rejects a dynamic-shape source tensor at its interface."""
    a = torch.ones(1, 8, dtype=torch.float32)
    out = torch.zeros(1, 8, dtype=torch.float32)

    with pytest.raises(Exception, match=ASSEMBLE_DYNAMIC_SHAPE_ERROR):
        assemble_kernel_with_dynamic_input(a, out)


@pypto.frontend.jit(runtime_options=SIM_RUNTIME_OPTIONS)
def kernel_dynamic_reshape(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    sq = 128
    d = 64
    b = q.shape[0]
    q_reshaped = pypto.reshape(q, [b * sq, d])  # inplace=False, dynamic shape will raise an error.

    pypto.set_vec_tile_shapes(64, 64)
    for idx in pypto.loop(b):
        temp = pypto.view(q_reshaped, [sq, d], [idx * sq, 0])
        temp1 = pypto.exp(temp)
        pypto.assemble(temp1, [idx * sq, 0], out)


def test_dynamic_reshape_error():
    """Test that dynamic reshape without inplace=True causes an error."""
    q = torch.ones(1, 128, 64, dtype=torch.float32)
    out = torch.zeros(128, 64, dtype=torch.float32)

    with pytest.raises(Exception, match="reshape\\(\\) requires integer shape when using non-inplace reshape"):
        kernel_dynamic_reshape(q, out)


@pypto.frontend.jit(runtime_options=SIM_RUNTIME_OPTIONS)
def kernel_view_valid_shape_reshape_inplace_error(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    sq = 128
    d = 64
    b = q.shape[0]

    pypto.set_vec_tile_shapes(64, 64)
    for idx in pypto.loop(b):
        q_view = pypto.view(q, [1, sq, d], [idx, 0, 0], valid_shape=[1, sq, d])
        q_reshaped = pypto.reshape(q_view, [sq, d], valid_shape=[sq, d], inplace=True)
        pypto.assemble(q_reshaped, [idx * sq, 0], out)


def test_view_valid_shape_reshape_inplace_error():
    """Test that reshape with valid_shape and inplace=True after dynamic view causes an error."""
    q = torch.ones(1, 128, 64, dtype=torch.float32)
    out = torch.zeros(128, 64, dtype=torch.float32)

    with pytest.raises(
        Exception,
        match="Reshape\\(Inplace=true\\) is not supported for tensors derived from dynamic view",
    ):
        kernel_view_valid_shape_reshape_inplace_error(q, out)


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    test_kernel_dynamic_shape_error()
    test_matmul_dynamic_shape_error()
    test_one_hot_dynamic_shape_error()
    test_assemble_dynamic_shape_error()
    test_dynamic_reshape_error()
    test_view_valid_shape_reshape_inplace_error()

    logging.info("All dynamic shape error tests passed.")
