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

"""Operator overloading with automatic span capture and expression normalization."""

from __future__ import annotations

__all__ = ["make_binary", "truediv"]


import inspect

from pypto.pypto_impl import ir as _ir

from ._utils import _normalize_expr


def make_binary(op_name: str, left: _ir.Expr, right: _ir.Expr, span: _ir.Span | None = None) -> _ir.Expr:
    """Build a binary scalar expression through the C++ promotion rules."""
    actual_span = span if span is not None else _ir.Span.unknown()
    return getattr(_ir, op_name)(left, right, actual_span)


def truediv(left: _ir.Expr, right: _ir.Expr, span: _ir.Span | None = None) -> _ir.Expr:
    """True division with Python-consistent promotion (kept for backward compat)."""
    return make_binary("truediv", left, right, span)


def capture_call_span() -> _ir.Span:
    """Capture span from the caller's location.

    Goes back through the call stack to find the actual user code that invoked the operator.

    Returns:
        Span: Source location of the caller
    """
    # Go back through frames to find user code:
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        frame = frame.f_back.f_back

    if frame is not None:
        info = inspect.getframeinfo(frame)
        return _ir.Span(info.filename, info.lineno, -1)

    return _ir.Span.unknown()


def _make_binary_op(op_name: str):
    """Create a binary operator wrapper with span capture and normalization.

    Args:
        op_name: Name of the operator (e.g., "add")

    Returns:
        Wrapper function
    """

    def wrapper(self: _ir.Expr, other: int | float | _ir.Expr) -> _ir.Expr:
        span = capture_call_span()
        other_expr = _normalize_expr(other, span)
        return make_binary(op_name, self, other_expr, span)

    return wrapper


def _make_reverse_binary_op(op_name: str):
    """Create a binary operator wrapper with span capture and normalization.

    Args:
        op_name: Name of the operator (e.g., "add")

    Returns:
        Wrapper function
    """

    def wrapper(self: _ir.Expr, other: int | float | _ir.Expr) -> _ir.Expr:
        span = capture_call_span()
        other_expr = _normalize_expr(other, span)
        return make_binary(op_name, other_expr, self, span)

    return wrapper


def _make_unary_op(op_name: str):
    """Create a unary operator wrapper with span capture.

    Args:
        op_name: Name of the operator (e.g., "neg")

    Returns:
        Wrapper function
    """

    def wrapper(self) -> _ir.Expr:
        span = capture_call_span()
        return getattr(_ir, op_name)(self, span)

    return wrapper


def _patch_operators():
    """Patch Expr and Var classes with operator overloads."""

    # Binary operators for Expr
    _ir.Expr.__add__ = _make_binary_op("add")
    _ir.Expr.__sub__ = _make_binary_op("sub")
    _ir.Expr.__mul__ = _make_binary_op("mul")
    _ir.Expr.__truediv__ = _make_binary_op("truediv")
    _ir.Expr.__floordiv__ = _make_binary_op("floordiv")
    _ir.Expr.__mod__ = _make_binary_op("mod")
    _ir.Expr.__pow__ = _make_binary_op("pow")

    # Comparison operators for Expr
    _ir.Expr.__eq__ = _make_binary_op("eq")
    _ir.Expr.__ne__ = _make_binary_op("ne")
    _ir.Expr.__lt__ = _make_binary_op("lt")
    _ir.Expr.__le__ = _make_binary_op("le")
    _ir.Expr.__gt__ = _make_binary_op("gt")
    _ir.Expr.__ge__ = _make_binary_op("ge")

    # Bitwise operators for Expr
    _ir.Expr.__and__ = _make_binary_op("bit_and")
    _ir.Expr.__or__ = _make_binary_op("bit_or")
    _ir.Expr.__xor__ = _make_binary_op("bit_xor")
    _ir.Expr.__lshift__ = _make_binary_op("bit_shift_left")
    _ir.Expr.__rshift__ = _make_binary_op("bit_shift_right")

    # Unary operators for Expr
    _ir.Expr.__neg__ = _make_unary_op("neg")
    _ir.Expr.__invert__ = _make_unary_op("bit_not")

    # Reverse operators for Expr (when Expr is on right side)
    _ir.Expr.__radd__ = _make_reverse_binary_op("add")
    _ir.Expr.__rmul__ = _make_reverse_binary_op("mul")
    _ir.Expr.__rsub__ = _make_reverse_binary_op("sub")
    _ir.Expr.__rtruediv__ = _make_reverse_binary_op("truediv")
    _ir.Expr.__rfloordiv__ = _make_reverse_binary_op("floordiv")
    _ir.Expr.__rmod__ = _make_reverse_binary_op("mod")
    _ir.Expr.__rpow__ = _make_reverse_binary_op("pow")
    _ir.Expr.__rand__ = _make_reverse_binary_op("bit_and")
    _ir.Expr.__ror__ = _make_reverse_binary_op("bit_or")
    _ir.Expr.__rxor__ = _make_reverse_binary_op("bit_xor")
    _ir.Expr.__rlshift__ = _make_reverse_binary_op("bit_shift_left")
    _ir.Expr.__rrshift__ = _make_reverse_binary_op("bit_shift_right")


# Automatically patch operators when this module is imported
_patch_operators()
