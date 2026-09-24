# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar op parse handlers (min, max, const) and unified dispatchers."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Expr, Span

from ._op_registry import OpSpec, op_impl, register_table


@op_impl("min")
def _parse_min(self, call: ast.Call):
    from pypto_pro.language.parser.diagnostics import InvalidOperationError

    call_span = self.span_tracker.get_span(call)
    if call.keywords:
        raise InvalidOperationError("Scalar operation 'min' does not accept keyword arguments", span=call_span)
    if len(call.args) != 2:
        raise InvalidOperationError(
            f"Scalar binary operation 'min' requires exactly 2 arguments, got {len(call.args)}",
            span=call_span,
        )
    lhs = self.parse_expression(call.args[0])
    rhs = self.parse_expression(call.args[1])
    operation = _ir_core.min_(lhs, rhs, call_span)
    folded = self._fold_const_binop("min", operation, call_span)
    return folded if folded is not None else operation


@op_impl("max")
def _parse_max(self, call: ast.Call):
    from pypto_pro.language.parser.diagnostics import InvalidOperationError

    call_span = self.span_tracker.get_span(call)
    if call.keywords:
        raise InvalidOperationError("Scalar operation 'max' does not accept keyword arguments", span=call_span)
    if len(call.args) != 2:
        raise InvalidOperationError(
            f"Scalar binary operation 'max' requires exactly 2 arguments, got {len(call.args)}",
            span=call_span,
        )
    lhs = self.parse_expression(call.args[0])
    rhs = self.parse_expression(call.args[1])
    operation = _ir_core.max_(lhs, rhs, call_span)
    folded = self._fold_const_binop("max", operation, call_span)
    return folded if folded is not None else operation


# ---------------------------------------------------------------------------
# Builders for binary ops (tile-tile / tile-scalar dispatch)
#   rhs is Tile   -> block.add / block.sub / ... / block.and
#   rhs is Scalar -> block.adds / block.subs / ... / block.ands
# ---------------------------------------------------------------------------
def _ir_add(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import _BINARY_DTYPES, _check_dtype, _check_dtype_match, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    _check_dtype("add", dt, _BINARY_DTYPES)
    _check_dtype_match("add", dt, getattr(lhs.type, "dtype", None))
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="add", scalar_op="adds", span=span, **kwargs)


def _ir_sub(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import _BINARY_DTYPES, _check_dtype, _check_dtype_match, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    _check_dtype("sub", dt, _BINARY_DTYPES)
    _check_dtype_match("sub", dt, getattr(lhs.type, "dtype", None))
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="sub", scalar_op="subs", span=span, **kwargs)


def _ir_mul(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import _MUL_DTYPES, _check_dtype, _check_dtype_match, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    _check_dtype("mul", dt, _MUL_DTYPES)
    _check_dtype_match("mul", dt, getattr(lhs.type, "dtype", None))
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="mul", scalar_op="muls", span=span, **kwargs)


def _ir_div(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import _DIV_DTYPES, _check_dtype, _check_dtype_match, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    _check_dtype("div", dt, _DIV_DTYPES)
    _check_dtype_match("div", dt, getattr(lhs.type, "dtype", None))
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="div", scalar_op="divs", span=span, **kwargs)


def _ir_and(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import _BITWISE_DTYPES, _check_dtype, _check_dtype_match, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    _check_dtype("and", dt, _BITWISE_DTYPES)
    _check_dtype_match("and", dt, getattr(lhs.type, "dtype", None))
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="and", scalar_op="ands", span=span, **kwargs)


# ---------------------------------------------------------------------------
# Builders for min/max (element-wise + reduce overload via dim kwarg)
# ---------------------------------------------------------------------------
def _ir_minimum(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, dim: int | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import (
        _BINARY_DTYPES,
        _check_dtype,
        _check_dtype_match,
        _create_dim_op,
        _create_tile_scalar_op,
    )

    dt = getattr(out.type, "dtype", None)
    _check_dtype("minimum", dt, _BINARY_DTYPES)
    _check_dtype_match("minimum", dt, getattr(lhs.type, "dtype", None))
    if dim is not None:
        return _create_dim_op([out, lhs, rhs], row_op="row_min", col_op="col_min", dim=dim, span=span)
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="minimum", scalar_op="mins", span=span, **kwargs)


def _ir_maximum(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, dim: int | None = None, **kwargs) -> Expr:
    from pypto_pro.ir.op.block_ops import (
        _BINARY_DTYPES,
        _check_dtype,
        _check_dtype_match,
        _create_dim_op,
        _create_tile_scalar_op,
    )

    dt = getattr(out.type, "dtype", None)
    _check_dtype("maximum", dt, _BINARY_DTYPES)
    _check_dtype_match("maximum", dt, getattr(lhs.type, "dtype", None))
    if dim is not None:
        return _create_dim_op([out, lhs, rhs], row_op="row_max", col_op="col_max", dim=dim, span=span)
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="maximum", scalar_op="maxs", span=span, **kwargs)


register_table(
    {
        "add": OpSpec(builder=_ir_add),
        "sub": OpSpec(builder=_ir_sub),
        "mul": OpSpec(builder=_ir_mul),
        "div": OpSpec(builder=_ir_div),
        "and_": OpSpec(builder=_ir_and),
        "minimum": OpSpec(builder=_ir_minimum),
        "maximum": OpSpec(builder=_ir_maximum),
    }
)


@op_impl("const")
def _parse_typed_constant(self, call: ast.Call):
    from pypto_pro.language.parser.diagnostics import ParserSyntaxError

    span = self.span_tracker.get_span(call)

    if len(call.args) != 2:
        raise ParserSyntaxError(
            "pl.const() requires exactly 2 arguments: value and dtype",
            span=span,
            hint="Use pl.const(42, pl.DT_INT32) or pl.const(1.0, pl.DT_FP16)",
        )

    value_node = call.args[0]
    negate = False
    if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
        negate = True
        value_node = value_node.operand

    if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
        raise ParserSyntaxError(
            "pl.const() first argument must be a numeric literal",
            span=span,
            hint="Use an int or float literal: pl.const(42, pl.DT_INT32)",
        )

    value = value_node.value
    if negate:
        value = -value

    dtype = self.parse_expression(call.args[1])
    if not isinstance(dtype, _ir_core.DataType):
        raise ParserSyntaxError(
            f"pl.const() second argument must be a dtype (pl.DT_*), "
            f"got {type(dtype).__name__}",
            span=span,
            hint="Use a dtype: pl.DT_FP32, pl.DT_INT32, etc.",
        )

    if dtype == _ir_core.DataType.BOOL:
        return _ir_core.ConstBool(bool(value), span)
    if dtype.is_float():
        return _ir_core.ConstFloat(float(value), dtype, span)
    return _ir_core.ConstInt(int(value), dtype, span)
