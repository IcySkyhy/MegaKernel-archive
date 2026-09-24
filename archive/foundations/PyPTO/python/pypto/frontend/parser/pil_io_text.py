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

"""Lower PilCode nodes back to Python ast nodes and unparse them to legal Python code.

Every PilCode class carries a ``kind`` tag (set by ``pil_code_dict`` in pil.py);
conversion is dispatched through the ``ast_builder_dict`` registry keyed by
that tag.

Registered builders use uniform signatures:
    expr builders:     (node) -> ast.expr   (always Load ctx)
    stmt builders:     (node) -> ast.stmt
    parameter builder: (node) -> ast.arguments

Statement builders build Store/Del contexts themselves for their target
slots; nested expression visiting never changes context.
"""

import ast
from typing import Union

from pypto.error import FeError
from pypto.frontend.kind_dict import KindDict

from .pil import (
    PIL_CALLEE_BOP_ADD,
    PIL_CALLEE_BOP_BITAND,
    PIL_CALLEE_BOP_BITOR,
    PIL_CALLEE_BOP_BITXOR,
    PIL_CALLEE_BOP_DIV,
    PIL_CALLEE_BOP_FLOORDIV,
    PIL_CALLEE_BOP_LSHIFT,
    PIL_CALLEE_BOP_MATMULT,
    PIL_CALLEE_BOP_MOD,
    PIL_CALLEE_BOP_MULT,
    PIL_CALLEE_BOP_POW,
    PIL_CALLEE_BOP_RSHIFT,
    PIL_CALLEE_BOP_SUB,
    PIL_CALLEE_COP_EQ,
    PIL_CALLEE_COP_GT,
    PIL_CALLEE_COP_GTE,
    PIL_CALLEE_COP_IN,
    PIL_CALLEE_COP_IS,
    PIL_CALLEE_COP_ISNOT,
    PIL_CALLEE_COP_LT,
    PIL_CALLEE_COP_LTE,
    PIL_CALLEE_COP_NOTEQ,
    PIL_CALLEE_COP_NOTIN,
    PIL_CALLEE_DICT_OP,
    PIL_CALLEE_JOINED_STR,
    PIL_CALLEE_LIST,
    PIL_CALLEE_SET_OP,
    PIL_CALLEE_TUPLE,
    PIL_CALLEE_UOP_INVERT,
    PIL_CALLEE_UOP_NOT,
    PIL_CALLEE_UOP_UADD,
    PIL_CALLEE_UOP_USUB,
    PIL_CALLEE_YIELD,
    PIL_CALLEE_YIELD_FROM,
    PilAssert,
    PilAssignAttribute,
    PilAssignSubscript,
    PilAssignSymbolList,
    PilAttribute,
    PilBreak,
    PilCall,
    PilClassDef,
    PilCode,
    PilConstantExpr,
    PilContinue,
    PilDeleteAttribute,
    PilDeleteIdentifier,
    PilDeleteSubscript,
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
    PilStmt,
    PilStmtList,
    PilSubscript,
    PilTry,
    PilWhile,
    PilWith,
)

_CALLEE_TO_BOP = {
    PIL_CALLEE_BOP_ADD: ast.Add(),
    PIL_CALLEE_BOP_SUB: ast.Sub(),
    PIL_CALLEE_BOP_MULT: ast.Mult(),
    PIL_CALLEE_BOP_MATMULT: ast.MatMult(),
    PIL_CALLEE_BOP_DIV: ast.Div(),
    PIL_CALLEE_BOP_MOD: ast.Mod(),
    PIL_CALLEE_BOP_POW: ast.Pow(),
    PIL_CALLEE_BOP_LSHIFT: ast.LShift(),
    PIL_CALLEE_BOP_RSHIFT: ast.RShift(),
    PIL_CALLEE_BOP_BITOR: ast.BitOr(),
    PIL_CALLEE_BOP_BITXOR: ast.BitXor(),
    PIL_CALLEE_BOP_BITAND: ast.BitAnd(),
    PIL_CALLEE_BOP_FLOORDIV: ast.FloorDiv(),
}

_CALLEE_TO_UOP = {
    PIL_CALLEE_UOP_INVERT: ast.Invert(),
    PIL_CALLEE_UOP_NOT: ast.Not(),
    PIL_CALLEE_UOP_UADD: ast.UAdd(),
    PIL_CALLEE_UOP_USUB: ast.USub(),
}

_CALLEE_TO_COP = {
    PIL_CALLEE_COP_EQ: ast.Eq(),
    PIL_CALLEE_COP_NOTEQ: ast.NotEq(),
    PIL_CALLEE_COP_LT: ast.Lt(),
    PIL_CALLEE_COP_LTE: ast.LtE(),
    PIL_CALLEE_COP_GT: ast.Gt(),
    PIL_CALLEE_COP_GTE: ast.GtE(),
    PIL_CALLEE_COP_IS: ast.Is(),
    PIL_CALLEE_COP_ISNOT: ast.IsNot(),
    PIL_CALLEE_COP_IN: ast.In(),
    PIL_CALLEE_COP_NOTIN: ast.NotIn(),
}

# kind -> builder function.
# Expr builders take (node); stmt/parameter builders take (node).
ast_builder_dict = KindDict()

# callee -> ast.expr builder for PilCall nodes, signature (node) -> ast.expr.
# BOP/UOP/COP are handled by operator maps before consulting this registry.
call_builder_dict = KindDict()


def _store() -> ast.expr_context:
    return ast.Store()


# ---- Generic building helpers ----

def _build_code(node):
    """Build the ast counterpart of any PilCode node via kind dispatch.

    Expr, index, assign-symbol and stmt builders all live in the same
    ``ast_builder_dict`` registry, so one dispatcher serves every slot type.
    """
    return ast_builder_dict[node.kind](node)


def visit_keyword(node: PilKeyword) -> ast.keyword:
    return ast.keyword(arg=node.arg, value=_build_code(node.value))


def visit_stmt_list(node: PilStmtList) -> list:
    return [_build_code(s) for s in node.stmts]


def _body(node) -> list:
    stmts = visit_stmt_list(node)
    return stmts if stmts else [ast.Pass()]


def _clause(node) -> list:
    """Like _body but keeps empty clauses empty so unparse omits them."""
    return visit_stmt_list(node)


def _indices_to_slice(indices) -> ast.expr:
    """Convert an index list to an ast slice expression.

    One index stays bare; multiple indices form a tuple index.
    """
    elts = [_build_code(i) for i in indices]
    if len(elts) == 1:
        return elts[0]
    return ast.Tuple(elts=elts, ctx=ast.Load())


# ---- Expression builders ----

@ast_builder_dict(PilConstantExpr.kind)
def _build_constant(node):
    return ast.Constant(value=node.value, kind=node.constant_kind)


@ast_builder_dict(PilIdentifierExpr.kind)
def _build_identifier(node):
    return ast.Name(id=node.id, ctx=ast.Load())


@ast_builder_dict(PilStarred.kind)
def _build_starred(node):
    return ast.Starred(value=ast.Name(id=node.value, ctx=ast.Load()), ctx=ast.Load())


@ast_builder_dict(PilSlice.kind)
def _build_slice(node):
    return ast.Slice(
        lower=_build_code(node.lower) if node.lower is not None else None,
        upper=_build_code(node.upper) if node.upper is not None else None,
        step=_build_code(node.step) if node.step is not None else None,
    )


@ast_builder_dict(PilAttribute.kind)
def _build_attribute(node):
    return ast.Attribute(value=_build_code(node.target), attr=node.attr, ctx=ast.Load())


@ast_builder_dict(PilSubscript.kind)
def _build_subscript(node):
    return ast.Subscript(
        value=_build_code(node.target),
        slice=_indices_to_slice(node.index),
        ctx=ast.Load(),
    )


@ast_builder_dict(PilCall.kind)
def _build_call(node):
    callee = node.callee
    if callee in _CALLEE_TO_BOP:
        return ast.BinOp(
            left=_build_code(node.args[0]),
            op=_CALLEE_TO_BOP[callee],
            right=_build_code(node.args[1]),
        )
    if callee in _CALLEE_TO_UOP:
        return ast.UnaryOp(op=_CALLEE_TO_UOP[callee], operand=_build_code(node.args[0]))
    if callee in _CALLEE_TO_COP:
        return ast.Compare(
            left=_build_code(node.args[0]),
            ops=[_CALLEE_TO_COP[callee]],
            comparators=[_build_code(node.args[1])],
        )
    builder = call_builder_dict.data.get(callee)
    if builder is not None:
        return builder(node)
    # Plain function call
    return ast.Call(
        func=ast.Name(id=callee, ctx=ast.Load()),
        args=[_build_code(a) for a in node.args],
        keywords=[visit_keyword(kw) for kw in node.kwargs],
    )


# ---- Call-callee builders ----

@call_builder_dict(PIL_CALLEE_DICT_OP)
def _build_dict_call(node):
    # args alternate key0?, value0, key1?, value1, ...; None key slot = **unpack
    if len(node.args) % 2 != 0:
        raise FeError(ValueError(f"dict args must be key/value pairs, got {len(node.args)}"))
    keys = []
    values = []
    for i in range(0, len(node.args), 2):
        key = node.args[i]
        keys.append(None if key is None else _build_code(key))
        values.append(_build_code(node.args[i + 1]))
    return ast.Dict(keys=keys, values=values)


@call_builder_dict(PIL_CALLEE_SET_OP)
def _build_set_call(node):
    return ast.Set(elts=[_build_code(e) for e in node.args])


@call_builder_dict(PIL_CALLEE_LIST)
def _build_list_call(node):
    return ast.List(elts=[_build_code(e) for e in node.args], ctx=ast.Load())


@call_builder_dict(PIL_CALLEE_TUPLE)
def _build_tuple_call(node):
    return ast.Tuple(elts=[_build_code(e) for e in node.args], ctx=ast.Load())


@call_builder_dict(PIL_CALLEE_YIELD)
def _build_yield_call(node):
    return ast.Yield(value=_build_code(node.args[0]) if node.args else None)


@call_builder_dict(PIL_CALLEE_YIELD_FROM)
def _build_yield_from_call(node):
    return ast.YieldFrom(value=_build_code(node.args[0]))


@call_builder_dict(PIL_CALLEE_JOINED_STR)
def _build_joined_str_call(node):
    return ast.JoinedStr(values=[_build_code(a) for a in node.args])


# ---- Parameter builders ----

@ast_builder_dict(PilParameter.kind)
def _build_parameter(node):
    return ast.arg(
        arg=node.name,
        annotation=_build_code(node.annotation) if node.annotation is not None else None,
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilFunctionParameterDef.kind)
def _build_parameter_def(node):
    pos_defaults = [_build_code(p.default) for p in node.args if p.default is not None]
    kw_defaults = [_build_code(p.default) if p.default is not None else None
                   for p in node.kwonlyargs]
    return ast.arguments(
        posonlyargs=[_build_code(p) for p in node.posonlyargs],
        args=[_build_code(p) for p in node.args],
        vararg=_build_code(node.vararg) if node.vararg is not None else None,
        kwonlyargs=[_build_code(p) for p in node.kwonlyargs],
        kw_defaults=kw_defaults,
        kwarg=_build_code(node.kwarg) if node.kwarg is not None else None,
        defaults=pos_defaults,
    )


def _build_code_or_passthrough(node):
    """Like _build_code but tolerates raw ``ast.arguments`` passed through
    from the parser (function signatures not yet PIL-ified)."""
    if not hasattr(node, 'kind'):
        return node
    return _build_code(node)


# ---- Statement builders ----

@ast_builder_dict(PilFunctionDef.kind)
def _build_function_def(node):
    return ast.FunctionDef(
        name=node.name,
        args=_build_code_or_passthrough(node.args),
        body=_body(node.body),
        decorator_list=[_build_code(d) for d in node.decorator_list],
        returns=_build_code(node.return_expr) if node.return_expr is not None else None,
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilClassDef.kind)
def _build_class_def(node):
    return ast.ClassDef(
        name=node.name,
        bases=[_build_code(b) for b in node.bases],
        keywords=[visit_keyword(kw) for kw in node.keywords],
        body=_body(node.body),
        decorator_list=[_build_code(d) for d in node.decorator_list],
    )


def _store_target(node: PilExprListItem) -> ast.expr:
    """Build a Store-context target from an identifier or starred item."""
    if isinstance(node, PilStarred):
        return ast.Starred(value=ast.Name(id=node.value, ctx=ast.Load()), ctx=_store())
    return ast.Name(id=node.id, ctx=_store())


@ast_builder_dict(PilAssignSymbolList.kind)
def _build_assign_symbol_list(node):
    if len(node.target_list) == 1:
        target = _store_target(node.target_list[0])
    else:
        target = ast.Tuple(
            elts=[_store_target(t) for t in node.target_list],
            ctx=_store(),
        )
    return ast.Assign(
        targets=[target],
        value=_build_code(node.value),
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilAssignAttribute.kind)
def _build_assign_attribute(node):
    return ast.Assign(
        targets=[ast.Attribute(
            value=_build_code(node.target), attr=node.attr, ctx=ast.Store())],
        value=_build_code(node.value),
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilAssignSubscript.kind)
def _build_assign_subscript(node):
    return ast.Assign(
        targets=[ast.Subscript(
            value=_build_code(node.target),
            slice=_indices_to_slice(node.index),
            ctx=ast.Store())],
        value=_build_code(node.value),
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilDeleteIdentifier.kind)
def _build_delete_identifier(node):
    return ast.Delete(targets=[ast.Name(id=node.target.id, ctx=ast.Del())])


@ast_builder_dict(PilDeleteAttribute.kind)
def _build_delete_attribute(node):
    return ast.Delete(targets=[ast.Attribute(
        value=ast.Name(id=node.target.id, ctx=ast.Load()), attr=node.attr, ctx=ast.Del())])


@ast_builder_dict(PilDeleteSubscript.kind)
def _build_delete_subscript(node):
    return ast.Delete(targets=[ast.Subscript(
        value=ast.Name(id=node.target.id, ctx=ast.Load()),
        slice=_indices_to_slice(node.index),
        ctx=ast.Del())])


@ast_builder_dict(PilReturn.kind)
def _build_return(node):
    return ast.Return(value=_build_code(node.value) if node.value is not None else None)


@ast_builder_dict(PilFor.kind)
def _build_for(node):
    return ast.For(
        target=_store_target(node.target),
        iter=_build_code(node.iter),
        body=_body(node.body),
        orelse=_clause(node.orelse),
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilWhile.kind)
def _build_while(node):
    return ast.While(
        test=_build_code(node.test),
        body=_body(node.body),
        orelse=_clause(node.orelse),
    )


@ast_builder_dict(PilIf.kind)
def _build_if(node):
    return ast.If(
        test=_build_code(node.test),
        body=_body(node.body),
        orelse=_clause(node.orelse),
    )


@ast_builder_dict(PilWith.kind)
def _build_with(node):
    optional_vars = ast.Name(id=node.optional_vars, ctx=ast.Store()) \
        if node.optional_vars is not None else None
    item = ast.withitem(context_expr=_build_code(node.context_expr), optional_vars=optional_vars)
    return ast.With(
        items=[item],
        body=_body(node.body),
        type_comment=node.type_comment,
    )


@ast_builder_dict(PilRaise.kind)
def _build_raise(node):
    return ast.Raise(
        exc=_build_code(node.exc) if node.exc is not None else None,
        cause=_build_code(node.cause) if node.cause is not None else None,
    )


@ast_builder_dict(PilTry.kind)
def _build_try(node):
    exc_var, handler_body = node.handler
    handler = ast.ExceptHandler(
        type=ast.Name(id='Exception', ctx=ast.Load()),
        name=exc_var,
        body=_body(handler_body),
    )
    return ast.Try(
        body=_body(node.body),
        handlers=[handler],
        orelse=_clause(node.orelse),
        finalbody=_clause(node.finalbody),
    )


@ast_builder_dict(PilAssert.kind)
def _build_assert(node):
    return ast.Assert(
        test=_build_code(node.test),
        msg=_build_code(node.msg) if node.msg is not None else None,
    )


@ast_builder_dict(PilImport.kind)
def _build_import(node):
    return ast.Import(names=[ast.alias(name=node.name, asname=node.asname)])


@ast_builder_dict(PilImportFrom.kind)
def _build_import_from(node):
    return ast.ImportFrom(
        module=node.module,
        names=[ast.alias(name=node.name, asname=node.asname)],
        level=node.level if node.level is not None else 0,
    )


@ast_builder_dict(PilGlobal.kind)
def _build_global(node):
    return ast.Global(names=[node.name])


@ast_builder_dict(PilNonlocal.kind)
def _build_nonlocal(node):
    return ast.Nonlocal(names=[node.name])


@ast_builder_dict(PilPass.kind)
def _build_pass(node):
    return ast.Pass()


@ast_builder_dict(PilBreak.kind)
def _build_break(node):
    return ast.Break()


@ast_builder_dict(PilContinue.kind)
def _build_continue(node):
    return ast.Continue()


def pil_to_ast(node: PilCode) -> Union[ast.expr, ast.stmt, list]:
    """Convert a single PilCode node (expr or stmt) to its ast counterpart."""
    if isinstance(node, PilStmtList):
        module = ast.Module(body=visit_stmt_list(node), type_ignores=[])
        ast.fix_missing_locations(module)
        return module.body
    if isinstance(node, PilStmt):
        module = ast.Module(body=[_build_code(node)], type_ignores=[])
        ast.fix_missing_locations(module)
        return module.body[0]
    result = _build_code(node)
    ast.fix_missing_locations(result)
    return result


def pil_to_python(node: PilCode) -> str:
    """Convert a PilCode node to legal Python source via ast.unparse."""
    result = pil_to_ast(node)
    if isinstance(result, list):
        return "\n".join(ast.unparse(s) for s in result)
    return ast.unparse(result)
