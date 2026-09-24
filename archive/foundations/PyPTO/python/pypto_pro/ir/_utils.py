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

"""Utility functions for IR construction."""

from __future__ import annotations

__all__ = ["_get_span_or_capture", "_is_int", "_normalize_expr", "_normalize_shape", "_to_make_tuple"]


from collections.abc import Sequence
import inspect

from pypto.pypto_impl import ir as _ir
from pypto.pypto_impl.ir import DataType


def _get_span_or_capture(span: _ir.Span | None = None, frame_offset: int = 1) -> _ir.Span:
    """Get explicit span or capture from caller.

    Args:
        span: Explicit span if provided
        frame_offset: Additional frames to skip beyond immediate caller

    Returns:
        Provided span or captured span from call site
    """
    if span is not None:
        return span

    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back

    for _ in range(frame_offset):
        if frame is None:
            break
        frame = frame.f_back

    if frame is not None:
        info = inspect.getframeinfo(frame)
        return _ir.Span(info.filename, info.lineno, -1)

    return _ir.Span.unknown()


def _is_int(value: object) -> bool:
    """Whether *value* is a plain integer (bool is not, despite subclassing int)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_expr(
    value: int | float | _ir.Expr,
    span: _ir.Span | None = None,
    int_dtype: DataType = DataType.INDEX,
    float_dtype: DataType = DataType.FP32,
) -> _ir.Expr:
    """Convert Python values to IR expressions.

    Args:
        value: Python int/float or existing Expr
        span: Optional span for created constants
        int_dtype: Data type to use for integer constants (default: INDEX)
        float_dtype: Data type to use for float constants (default: FP32)

    Returns:
        IR expression node

    Raises:
        TypeError: If value is not int, float, or ir.Expr
    """
    if isinstance(value, _ir.Expr):
        return value

    actual_span = span if span is not None else _ir.Span.unknown()

    if isinstance(value, int):
        return _ir.ConstInt(value, int_dtype, actual_span)
    elif isinstance(value, float):
        return _ir.ConstFloat(value, float_dtype, actual_span)
    else:
        raise TypeError(f"Cannot convert {type(value)} to IR expression")


def _normalize_shape(
    shape: Sequence[int | _ir.Expr],
    span: _ir.Span | None = None,
) -> list[_ir.Expr]:
    """Convert shape dimensions to IR expressions.

    Args:
        shape: Sequence of integers or Expr nodes representing shape dimensions
        span: Optional span for created constants

    Returns:
        List of IR expression nodes

    Raises:
        TypeError: If shape contains non-int, non-Expr values
    """
    return [_normalize_expr(dim, span, int_dtype=DataType.INDEX) for dim in shape]


def _to_make_tuple(
    value: _ir.MakeTuple | Sequence[int | float | _ir.Expr],
    span: _ir.Span | None = None,
) -> _ir.MakeTuple:
    """Normalize a sequence or MakeTuple into a MakeTuple IR node.

    Args:
        value: Either an existing MakeTuple (returned as-is) or a sequence
            of ints/floats/Exprs to wrap
        span: Optional span for created constants

    Returns:
        MakeTuple IR expression
    """
    if isinstance(value, _ir.MakeTuple):
        return value
    actual_span = span if span is not None else _ir.Span.unknown()
    # A tuple-typed Expr (e.g. a helper-function parameter bound to a ``[i, j]`` offset
    # list at the call site) is not a Python sequence and cannot be iterated directly.
    # Expand it into per-element ``GetItemExpr`` accesses so offsets can be passed through
    # helper functions.
    if isinstance(value, _ir.Expr):
        value_type = getattr(value, "type", None)
        if isinstance(value_type, _ir.TupleType):
            elements = [
                _ir.GetItemExpr(value, _ir.ConstInt(i, DataType.INDEX, actual_span), actual_span)
                for i in range(len(value_type.types))
            ]
            return _ir.MakeTuple(elements, actual_span)
        raise TypeError(
            f"Cannot convert Expr of type {value_type} to an offset tuple; expected a MakeTuple or a tuple-typed value"
        )
    elements = [_normalize_expr(v, actual_span) for v in value]
    return _ir.MakeTuple(elements, actual_span)
