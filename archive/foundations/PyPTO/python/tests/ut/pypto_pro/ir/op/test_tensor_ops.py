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
"""Block-facing smoke tests for pypto_pro.ir.op.tensor helpers."""

from pypto_pro import DataType, ir
import pytest


def _tensor_var(name: str, shape=None, dtype=DataType.FP32):
    span = ir.Span.unknown()
    dims = [ir.ConstInt(dim, DataType.INDEX, span) for dim in (shape or [64, 128])]
    return ir.Var(name, ir.TensorType(dims, dtype), span)


@pytest.mark.parametrize(
    "factory, expected_name",
    [
        (lambda:ir.op.tensor.create([4, 8], DataType.FP32), "tensor.create"),
        (lambda:ir.op.tensor.view(_tensor_var("t"), [8, 16], [0, 0]), "tensor.view"),
        (lambda:ir.op.tensor.assemble(_tensor_var("dst"), _tensor_var("src"), [0, 0]), "tensor.assemble"),
    ],
)
def test_tensor_memory_helpers_create_calls(factory, expected_name):
    call = factory()

    assert isinstance(call, ir.Call)
    assert call.name == expected_name
    assert isinstance(call.type, ir.TensorType)


@pytest.mark.parametrize(
    "factory, expected_name",
    [
        (lambda x, y:ir.op.tensor.add(x, y), "tensor.add"),
        (lambda x, y:ir.op.tensor.sub(x, y), "tensor.sub"),
        (lambda x, y:ir.op.tensor.mul(x, y), "tensor.mul"),
        (lambda x, y:ir.op.tensor.div(x, y), "tensor.div"),
        (lambda x, y:ir.op.tensor.maximum(x, y), "tensor.maximum"),
    ],
)
def test_tensor_binary_helpers_create_calls(factory, expected_name):
    x = _tensor_var("x")
    y = _tensor_var("y")
    call = factory(x, y)

    assert isinstance(call, ir.Call)
    assert call.name == expected_name
    assert isinstance(call.type, ir.TensorType)


@pytest.mark.parametrize(
    "factory, expected_name",
    [
        (lambda x:ir.op.tensor.exp(x), "tensor.exp"),
        (lambda x:ir.op.tensor.cast(x, DataType.FP32), "tensor.cast"),
        (lambda x:ir.op.tensor.row_max(x), "tensor.row_max"),
        (lambda x:ir.op.tensor.row_sum(x), "tensor.row_sum"),
    ],
)
def test_tensor_unary_and_reduction_helpers_create_calls(factory, expected_name):
    x = _tensor_var("x", dtype=DataType.FP16)
    call = factory(x)

    assert isinstance(call, ir.Call)
    assert call.name == expected_name
    assert isinstance(call.type, ir.TensorType)


@pytest.mark.parametrize(
    "factory, expected_name",
    [
        (lambda x:ir.op.tensor.reshape(x, [32]), "tensor.reshape"),
        (
            lambda x:ir.op.tensor.reshape(
                x, [ir.Var('n', ir.ScalarType(DataType.INT64), ir.Span.unknown())]
            ),
            "tensor.reshape",
        ),
        (lambda x:ir.op.tensor.transpose(x, 0, 1), "tensor.transpose"),
        (lambda x:ir.op.tensor.transpose(x, -2, -1), "tensor.transpose"),
    ],
)
def test_tensor_shape_transform_helpers_create_calls(factory, expected_name):
    x = _tensor_var("x", shape=[4, 8])
    call = factory(x)

    assert isinstance(call, ir.Call)
    assert call.name == expected_name
    assert isinstance(call.type, ir.TensorType)


def test_tensor_matmul_helper_accepts_kwargs():
    lhs = _tensor_var("lhs", [4, 8], DataType.FP16)
    rhs = _tensor_var("rhs", [8, 16], DataType.FP16)

    call = ir.op.tensor.matmul(lhs, rhs, out_dtype=DataType.FP32, a_trans=False, b_trans=False)

    assert isinstance(call, ir.Call)
    assert call.name == "tensor.matmul"
    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == DataType.FP32


def test_const_float_is_available_for_tensor_helpers():
    value = ir.ConstFloat(3.14, DataType.FP32, ir.Span.unknown())

    assert isinstance(value, ir.ConstFloat)
    assert value.value == 3.14
    assert value.dtype == DataType.FP32


def test_tensor_ops_are_registered_without_removed_dim_helper():
    expected = {
        "tensor.create",
        "tensor.view",
        "tensor.matmul",
        "tensor.row_max",
        "tensor.row_sum",
        "tensor.exp",
        "tensor.cast",
        "tensor.assemble",
        "tensor.maximum",
        "tensor.reshape",
        "tensor.transpose",
    }

    assert all(ir.is_op_registered(name) for name in expected)
    assert not hasattr(ir.op.tensor, "dim")
    assert not ir.is_op_registered("tensor.dim")
