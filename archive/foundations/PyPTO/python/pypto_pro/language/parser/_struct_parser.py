# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Struct parsing helpers for ASTParser (pl.struct / pl.struct_array / pl.make_tuple)."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir
from pypto_pro.ir.op._op_registry import op_impl

from .diagnostics import ParserSyntaxError


class StructParserMixin:
    """Mixin containing ``pl.struct`` / ``pl.struct_array`` / ``pl.make_tuple`` helpers."""

    @op_impl("make_tuple")
    def _parse_pl_make_tuple_expr(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if call.args:
            raise ParserSyntaxError(
                "pl.make_tuple() does not accept positional arguments; use keyword args",
                span=span,
            )
        if not call.keywords:
            raise ParserSyntaxError(
                "pl.make_tuple() requires at least one keyword argument (field=value)",
                span=span,
            )
        field_names: list[str] = []
        elements: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.make_tuple() does not support **kwargs", span=span)
            field_names.append(kw.arg)
            elements.append(self.parse_expression(kw.value, nested=True))
        return self.make_named_tuple(elements, field_names, span)

    @op_impl("struct")
    def _parse_pl_struct_expr(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        struct_name = call.args[0].value
        if not call.keywords:
            raise ParserSyntaxError(
                'pl.struct("Name", ...) requires at least one keyword field',
                span=span,
            )
        field_names: list[str] = []
        elements: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.struct() does not support **kwargs", span=span)
            field_names.append(kw.arg)
            elements.append(self.parse_expression(kw.value, nested=True))
        call = ir.create_op_call(
            "struct.create",
            elements,
            {"name": struct_name, "fields": field_names},
            span,
        )
        self.register_tuple_name(call, struct_name)
        return self.register_tuple_fields(call, field_names)

    @op_impl("struct_array")
    def _parse_struct_array_expr(self, call: ast.Call) -> ir.Expr:
        """Handle ``var = pl.struct_array(N, "StructName", field1=val1, ...)``.

        Lowers to N ``struct.create`` slots wrapped in a MakeTuple.
        The caller (_parse_name_assignment) emits the final let-binding.
        """
        span = self.span_tracker.get_span(call)
        if not call.args or not isinstance(call.args[0], ast.Constant):
            raise ParserSyntaxError(
                "pl.struct_array() requires an integer size as first argument",
                span=span,
                hint='Use pl.struct_array(N, "Name", field1=0, field2=0, ...)',
            )
        arr_size = call.args[0].value
        if not isinstance(arr_size, int) or arr_size < 1:
            raise ParserSyntaxError(
                f"pl.struct_array() size must be a positive integer, got {arr_size}",
                span=span,
            )
        if not (len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
            raise ParserSyntaxError(
                'pl.struct_array(N, "Name", ...) requires a string struct name as second arg',
                span=span,
            )
        struct_name = call.args[1].value
        if not call.keywords:
            raise ParserSyntaxError(
                'pl.struct_array(N, "Name", ...) requires at least one keyword field',
                span=span,
            )
        var_name = self.current_target_name
        field_names: list[str] = []
        field_inits: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.struct_array() does not support **kwargs", span=span)
            field_names.append(kw.arg)
            field_inits.append(self.parse_expression(kw.value, nested=True))
        slot_vars: list[ir.Expr] = []
        for i in range(arr_size):
            slot_call = ir.create_op_call(
                "struct.create",
                field_inits,
                {"name": struct_name, "fields": field_names},
                span,
            )
            self.register_tuple_name(slot_call, struct_name)
            self.register_tuple_fields(slot_call, field_names)
            slot_var = self.builder.let(f"{var_name}_{i}", slot_call, span=span)
            slot_vars.append(slot_var)
        result = ir.MakeTuple(slot_vars, span)
        self._struct_array_tuples.add(result)
        return result
