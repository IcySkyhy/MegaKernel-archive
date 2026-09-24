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
"""Block-facing smoke tests for pypto_pro.ir op registration."""

from pypto_pro import DataType, ir
import pytest


def _tensor_var(name: str, dtype=DataType.FP16):
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([ir.ConstInt(64, DataType.INDEX, span)], dtype)
    return ir.Var(name, tensor_type, span)


@pytest.mark.parametrize(
    "op_name",
    ["tensor.add", "tensor.matmul", "tensor.cast", "tensor.row_max", "block.load", "block.store"],
)
def test_block_visible_ops_are_registered(op_name):
    assert ir.is_op_registered(op_name)
    assert ir.get_op(op_name).name == op_name


@pytest.mark.parametrize(
    "op_name, attrs",
    [
        ("tensor.matmul", {"out_dtype", "a_trans", "b_trans", "c_matrix_nz"}),
        ("tensor.cast", {"target_type", "mode"}),
        ("tensor.row_max", {"axis", "keep_dim"}),
        ("tensor.row_sum", {"axis", "keep_dim"}),
    ],
)
def test_block_op_kwarg_schema_is_exposed(op_name, attrs):
    op = ir.get_op(op_name)

    assert attrs.issubset(set(op.get_attr_keys()))
    assert all(op.has_attr(attr) for attr in attrs)


def test_create_op_call_accepts_registered_kwargs():
    lhs = _tensor_var("lhs")
    rhs = _tensor_var("rhs")
    call = ir.create_op_call(
        "tensor.matmul",
        [lhs, rhs],
        {"out_dtype": DataType.FP32, "a_trans": False, "b_trans": False},
        ir.Span.unknown(),
    )

    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == DataType.FP32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unknown_param": 123},
        {"a_trans": "true"},
    ],
)
def test_create_op_call_rejects_invalid_kwargs(kwargs):
    lhs = _tensor_var("lhs")
    rhs = _tensor_var("rhs")

    with pytest.raises(Exception):
        ir.create_op_call("tensor.matmul", [lhs, rhs], kwargs, ir.Span.unknown())
