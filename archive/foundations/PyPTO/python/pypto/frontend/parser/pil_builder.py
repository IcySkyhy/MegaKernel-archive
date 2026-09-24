#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Factory layer building PIL data-model nodes (pil.py) from simplified pieces.

Mirrors the ``create_pil_XXX`` surface of ``PILBuilder`` in pil_parser.py,
but returns PilCode subclasses instead of ast nodes.
"""

import ast
from typing import Optional

from pypto.error import FeError

from .pil import (
    PIL_BOP_TO_CALLEE,
    PIL_CALLEE_DICT_OP,
    PIL_CALLEE_JOINED_STR,
    PIL_CALLEE_LIST,
    PIL_CALLEE_SET_OP,
    PIL_CALLEE_TUPLE,
    PIL_CALLEE_YIELD,
    PIL_CALLEE_YIELD_FROM,
    PIL_COP_TO_CALLEE,
    PIL_UOP_TO_CALLEE,
    PilAssert,
    PilAssignAttribute,
    PilAssignSubscript,
    PilAssignSymbolList,
    PilAttribute,
    PilBreak,
    PilCall,
    PilClassDef,
    PilConstantExpr,
    PilContinue,
    PilDeleteAttribute,
    PilDeleteIdentifier,
    PilDeleteSubscript,
    PilExpr,
    PilExprAssignSymbol,
    PilExprIndex,
    PilExprListItem,
    PilFor,
    PilFunctionDef,
    PilFunctionParameterDef,
    PilGlobal,
    PilIdentifierExpr,
    PilIf,
    PilImport,
    PilImportFrom,
    PilKeyword,
    PilNonlocal,
    PilParameter,
    PilPass,
    PilRaise,
    PilReturn,
    PilSlice,
    PilStarred,
    PilStmtList,
    PilSubscript,
    PilTry,
    PilWhile,
    PilWith,
)


def _checked_expr(value: PilExpr) -> PilExpr:
    """Validate a PilExpr; call sites must pass PilIdentifierExpr for identifiers."""
    if isinstance(value, PilExpr):
        return value
    raise FeError(TypeError(f"Expected PilExpr, but got {type(value).__name__}"))


def _checked_expr_or_none(value: Optional[PilExpr]) -> Optional[PilExpr]:
    return None if value is None else _checked_expr(value)


def _checked_index(value: PilExprIndex) -> PilExprIndex:
    """Validate an index slot (expr or slice)."""
    if isinstance(value, PilExprIndex):
        return value
    raise FeError(TypeError(f"Expected PilExprIndex, but got {type(value).__name__}"))


def _checked_assign_symbol(value: PilExprAssignSymbol) -> PilExprAssignSymbol:
    """Validate an assign-symbol slot (expr, attribute, subscript or call)."""
    if isinstance(value, PilExprAssignSymbol):
        return value
    raise FeError(TypeError(f"Expected PilExprAssignSymbol, but got {type(value).__name__}"))


def _checked_list_item(value: PilExprListItem) -> PilExprListItem:
    """Validate a list-item slot (expr or starred)."""
    if isinstance(value, PilExprListItem):
        return value
    raise FeError(TypeError(f"Expected PilExprListItem, but got {type(value).__name__}"))


class PilBuilder:
    """Builds PIL data-model nodes (PilCode subclasses) from simplified pieces."""

    # ---- Expression factories ----

    def create_constant(self, value: object, kind: Optional[str] = None, **_kwargs) -> PilConstantExpr:
        """Constant(constant value, string? kind)"""
        return PilConstantExpr(value, kind)

    def create_name(self, identifier: str, **_kwargs) -> PilIdentifierExpr:
        """Name(identifier id)"""
        return PilIdentifierExpr(identifier)

    def create_starred(self, value: str, **_kwargs) -> PilStarred:
        """Starred(identifier value)"""
        return PilStarred(value)

    def create_slice(self, lower: Optional[PilExpr], upper: Optional[PilExpr],
                     step: Optional[PilExpr], **_kwargs) -> PilSlice:
        """Slice(identifier? lower, identifier? upper, identifier? step)"""
        return PilSlice(lower, upper, step)

    def create_attribute(self, value_expr: PilExpr, attr_name: str, **_kwargs) -> PilAttribute:
        """Attribute(expr target, identifier attr)"""
        return PilAttribute(_checked_expr(value_expr), attr_name)

    def create_subscript(self, value_expr: PilExpr, indices: list[PilExprIndex],
                         **_kwargs) -> PilSubscript:
        """Subscript(expr target, PilExprIndex* index)"""
        return PilSubscript(_checked_expr(value_expr),
                            [_checked_index(i) for i in indices])

    def create_bin_op(self, left_expr: PilExpr, op: ast.operator, right_expr: PilExpr, **_kwargs) -> PilCall:
        """BinOp(identifier left, operator op, identifier right) -> ``!bop.<op>`` call"""
        return PilCall(PIL_BOP_TO_CALLEE[type(op)], [_checked_expr(left_expr), _checked_expr(right_expr)])

    def create_unary_op(self, op: ast.unaryop, operand_expr: PilExpr, **_kwargs) -> PilCall:
        """UnaryOp(unaryop op, identifier operand) -> ``!uop.<op>`` call"""
        return PilCall(PIL_UOP_TO_CALLEE[type(op)], [_checked_expr(operand_expr)])

    def create_compare(self, left_expr: PilExpr, op: ast.cmpop, comparator_expr: PilExpr, **_kwargs) -> PilCall:
        """Compare(identifier left, cmpop op, identifier comparator) -> ``!cop.<op>`` call"""
        return PilCall(PIL_COP_TO_CALLEE[type(op)], [_checked_expr(left_expr), _checked_expr(comparator_expr)])

    def create_dict(self, keys: list[Optional[PilExpr]], values: list[PilExpr],
                    **_kwargs) -> PilCall:
        """Dict(identifier?* keys, identifier* values) -> ``!dict`` call

        ``!dict`` args alternate ``key0?, value0, key1?, value1, ...``;
        a ``None`` key slot means ``**value`` unpacking.
        """
        if len(keys) != len(values):
            raise FeError(ValueError(f"dict keys/values length mismatch: {len(keys)} vs {len(values)}"))
        interleaved = []
        for key, value in zip(keys, values):
            interleaved.append(_checked_expr_or_none(key))
            interleaved.append(_checked_expr(value))
        return PilCall(PIL_CALLEE_DICT_OP, interleaved)

    def create_set(self, elts: list[PilExprListItem], **_kwargs) -> PilCall:
        """Set(expr* elts) -> ``!set`` call"""
        return PilCall(PIL_CALLEE_SET_OP, [_checked_list_item(e) for e in elts])

    def create_list(self, elts: list[PilExprListItem], **_kwargs) -> PilCall:
        """List(expr* elts) -> ``!list`` call"""
        return PilCall(PIL_CALLEE_LIST, [_checked_list_item(e) for e in elts])

    def create_tuple(self, elts: list[PilExprListItem], **_kwargs) -> PilCall:
        """Tuple(expr* elts) -> ``!tuple`` call"""
        return PilCall(PIL_CALLEE_TUPLE, [_checked_list_item(e) for e in elts])

    def create_call(self, func_expr: PilIdentifierExpr, args: list[PilExprListItem],
                    keywords: list[PilKeyword], **_kwargs) -> PilCall:
        """Call(identifier func, identifier* args, keyword* keywords)"""
        return PilCall(func_expr.id, [_checked_list_item(a) for a in args], list(keywords))

    def create_yield(self, expr: Optional[PilExpr], **_kwargs) -> PilCall:
        """Yield(identifier? value) -> ``!yield`` call"""
        return PilCall(PIL_CALLEE_YIELD, [_checked_expr(expr)] if expr is not None else [])

    def create_yield_from(self, expr: PilExpr, **_kwargs) -> PilCall:
        """YieldFrom(identifier value) -> ``!yield_from`` call"""
        return PilCall(PIL_CALLEE_YIELD_FROM, [_checked_expr(expr)])

    def create_joined_str(self, values: list[PilExpr], **_kwargs) -> PilCall:
        """JoinedStr(expr* values) -> ``!joined_str`` call"""
        return PilCall(PIL_CALLEE_JOINED_STR, [_checked_expr(v) for v in values])

    # ---- Statement factories ----

    def create_assign_identifier(self, target_name: str, value: PilExprAssignSymbol,
                                 type_comment: Optional[str] = None, **_kwargs) -> PilAssignSymbolList:
        """Assign(identifier target, expr value, string? type_comment) — single-name form"""
        return PilAssignSymbolList([PilIdentifierExpr(target_name)], _checked_assign_symbol(value), type_comment)

    def create_assign_symbol_list(self, targets: list[PilExprListItem], value: PilExprAssignSymbol,
                                  type_comment: Optional[str] = None, **_kwargs) -> PilAssignSymbolList:
        """Assign(expr target*, expr value, string? type_comment) for multiple names"""
        return PilAssignSymbolList(
            [_checked_list_item(t) for t in targets], _checked_assign_symbol(value), type_comment)

    def create_assign_attribute(self, target_expr: PilExpr, attr_name: str, value: PilExpr,
                                type_comment: Optional[str] = None, **_kwargs) -> PilAssignAttribute:
        """Assign(target.attr = value, string? type_comment)"""
        return PilAssignAttribute(_checked_expr(target_expr), attr_name, _checked_expr(value), type_comment)

    def create_assign_subscript(self, target_expr: PilExpr, indices: list[PilExprIndex],
                                value: PilExpr, type_comment: Optional[str] = None,
                                **_kwargs) -> PilAssignSubscript:
        """Assign(target[index] = value, string? type_comment)"""
        return PilAssignSubscript(_checked_expr(target_expr),
                                  [_checked_index(i) for i in indices],
                                  _checked_expr(value), type_comment)

    def create_function_def(self, name: str, args, body: PilStmtList,
                            decorator_list: list[PilExpr] = (),
                            return_expr: Optional[PilExpr] = None,
                            type_comment: Optional[str] = None,
                            **_kwargs) -> PilFunctionDef:
        """FunctionDef(identifier name, PilFunctionParameterDef args, stmt* body,
                       expr* decorator_list, expr? returns, string? type_comment)"""
        return PilFunctionDef(name, args, body, [_checked_expr(d) for d in decorator_list], return_expr, type_comment)

    def create_class_def(self, name: str, bases: list[PilExprListItem],
                         keywords: list, body: PilStmtList,
                         decorator_list: list[PilExpr] = (), **_kwargs) -> PilClassDef:
        """ClassDef(identifier name, expr* bases, keyword* keywords, stmt* body, expr* decorator_list)"""
        keywords = [kw if isinstance(kw, PilKeyword) else PilKeyword(kw[0], _checked_expr(kw[1])) for kw in keywords]
        return PilClassDef(name, [_checked_list_item(b) for b in bases], keywords, body,
                           [_checked_expr(d) for d in decorator_list])

    def create_return(self, expr: Optional[PilExpr], **_kwargs) -> PilReturn:
        """Return(identifier? value)"""
        return PilReturn(_checked_expr_or_none(expr))

    def create_delete_identifier(self, target: PilIdentifierExpr, **_kwargs) -> PilDeleteIdentifier:
        """Delete(expr target) — identifier form"""
        return PilDeleteIdentifier(target)

    def create_delete_attribute(self, target: PilIdentifierExpr, attr: str, **_kwargs) -> PilDeleteAttribute:
        """Delete(expr target, identifier attr)"""
        return PilDeleteAttribute(target, attr)

    def create_delete_subscript(self, target: PilIdentifierExpr, indices: list[PilExprIndex],
                                **_kwargs) -> PilDeleteSubscript:
        """Delete(expr target, PilExprIndex* index)"""
        return PilDeleteSubscript(target, [_checked_index(i) for i in indices])

    def create_for(self, target: PilExpr, iter_expr: PilExpr, body: PilStmtList,
                   orelse: PilStmtList, type_comment: Optional[str] = None, **_kwargs) -> PilFor:
        """For(identifier target, expr iter, stmt* body, stmt* orelse, string? type_comment)

        A starred tuple target is decomposed into plain symbol + assign symbol list
        statements inside the loop body, so target here is always a plain PilExpr.
        """
        return PilFor(target, _checked_expr(iter_expr), body, orelse, type_comment)

    def create_while(self, test_expr: PilExpr, body: PilStmtList, orelse: PilStmtList, **_kwargs) -> PilWhile:
        """While(identifier test, stmt* body, stmt* orelse)"""
        return PilWhile(_checked_expr(test_expr), body, orelse)

    def create_if(self, test_expr: PilExpr, body: PilStmtList, orelse: PilStmtList, **_kwargs) -> PilIf:
        """If(identifier test, stmt* body, stmt* orelse)"""
        return PilIf(_checked_expr(test_expr), body, orelse)

    def create_with(self, context_expr: PilExpr, body: PilStmtList,
                    optional_vars: Optional[str] = None,
                    type_comment: Optional[str] = None, **_kwargs) -> PilWith:
        """With(expr context_expr, stmt* body, identifier? optional_vars, string? type_comment)

        One clause per statement; multiple context managers nest via body.
        """
        return PilWith(_checked_expr(context_expr), body, optional_vars, type_comment)

    def create_raise(self, exc: Optional[PilExpr], cause: Optional[PilExpr], **_kwargs) -> PilRaise:
        """Raise(identifier? exc, identifier? cause)"""
        return PilRaise(_checked_expr_or_none(exc), _checked_expr_or_none(cause))

    def create_try(self, body: PilStmtList, handler: tuple[Optional[str], PilStmtList],
                   orelse: PilStmtList, finalbody: PilStmtList, **_kwargs) -> PilTry:
        """Try(stmt* body, (identifier? exc_var, stmt* handler_body), stmt* orelse, stmt* finalbody)"""
        return PilTry(body, handler, orelse, finalbody)

    def create_assert(self, test_expr: PilExpr, msg: Optional[PilExpr], **_kwargs) -> PilAssert:
        """Assert(identifier test, identifier? msg)"""
        return PilAssert(_checked_expr(test_expr), _checked_expr_or_none(msg))

    def create_import(self, name: str, asname: Optional[str] = None, **_kwargs) -> PilImport:
        """Import(identifier name, identifier? asname)"""
        return PilImport(name, asname)

    def create_import_from(self, module: Optional[str], name: str, asname: Optional[str] = None,
                           level: Optional[int] = None, **_kwargs) -> PilImportFrom:
        """ImportFrom(identifier? module, identifier name, identifier? asname, int? level)"""
        return PilImportFrom(module, name, asname, level)

    def create_global(self, name: str, **_kwargs) -> PilGlobal:
        """Global(identifier name); ``global a, b`` is two statements."""
        return PilGlobal(name)

    def create_nonlocal(self, name: str, **_kwargs) -> PilNonlocal:
        """Nonlocal(identifier name); ``nonlocal a, b`` is two statements."""
        return PilNonlocal(name)

    def create_pass(self) -> PilPass:
        """Pass"""
        return PilPass()

    def create_break(self) -> PilBreak:
        """Break"""
        return PilBreak()

    def create_continue(self) -> PilContinue:
        """Continue"""
        return PilContinue()

    # ---- Parameter factories ----

    def create_keyword(self, arg: Optional[str], value: PilExpr, **_kwargs) -> PilKeyword:
        """keyword(arg?, value)"""
        return PilKeyword(arg, value)

    def create_parameter(self, name: str, default: Optional[PilExpr] = None,
                         annotation: Optional[PilExpr] = None,
                         type_comment: Optional[str] = None, **_kwargs) -> PilParameter:
        """arg(identifier arg, expr? annotation, expr? default, string? type_comment)"""
        return PilParameter(name, default, annotation, type_comment)

    def create_function_parameter_def(self, posonlyargs: list[PilParameter] = (),
                                      args: list[PilParameter] = (),
                                      vararg: Optional[PilParameter] = None,
                                      kwonlyargs: list[PilParameter] = (),
                                      kwarg: Optional[PilParameter] = None) -> PilFunctionParameterDef:
        """arguments(arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs, arg? kwarg)"""
        return PilFunctionParameterDef(posonlyargs, args, vararg, kwonlyargs, kwarg)
