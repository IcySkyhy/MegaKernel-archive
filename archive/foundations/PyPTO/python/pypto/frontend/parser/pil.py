#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PIL (Python Intermediate Language) data model for the pypto frontend.

Compared with the full python ast, PIL is a simplified version of python ast,
which only contains the necessary information for code generation. The main
purpose of PIL is to simplify the code generation process and improve the
performance of code generation.

Simplify rule:
1.  All expr should be replaced by identifier as much as possible
2.  When assigned, only multiple names with starred, single attribute, single subscript
    are allowed in the assignment's lhs and for's target
3.  Lambda is converted to a named FunctionDef
4.  Comprehensions (ListComp, SetComp, GeneratorExp, DictComp) are converted to generator FunctionDef + call
5.  BoolOp, IfExp, Compare chains are converted to if/assign sequences
6.  AugAssign is converted to explicit load + BinOp + store
7.  AnnAssign is converted to annotation expr eval + optional assignment
8.  Assert is converted to if + Assert (preserving __debug__ guard)
9.  With items' optional_vars assignment is placed inside the body
10. Try handlers are unified into a single except Exception with isinstance dispatch
11. Import / ImportFrom / Global / Nonlocal carry exactly one symbol per statement
12. With carries exactly one context manager per statement; multiple managers nest through the body

stmt = FunctionDef(identifier name, PilFunctionParameterDef args, stmt* body,
                    expr* decorator_list, expr? returns, string? type_comment)
        | ClassDef(identifier name, expr* bases, keyword* keywords,
                    stmt* body, expr* decorator_list)
        | AssignSymbolList(PilExprListItem* target_list, PilExprAssignSymbol value, string? type_comment)
          # target only allow for identifier and starred
        | AssignAttribute(expr target, identifier attr, expr value, string? type_comment)
        | AssignSubscript(expr target, PilExprIndex* index, expr value, string? type_comment)
          # one index stays bare, multiple indices form a tuple index
        | DeleteIdentifier(PilIdentifierExpr target)
        | DeleteAttribute(PilIdentifierExpr target, identifier attr)
        | DeleteSubscript(PilIdentifierExpr target, PilExprIndex* index)
        | Return(expr? value)

        | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        | While(expr test, stmt* body, stmt* orelse)
        | If(expr test, stmt* body, stmt* orelse)

        | With(expr context_expr, stmt* body, identifier? optional_vars, string? type_comment)
        | Raise(expr? exc, expr? cause)
        | Try(stmt* body, (identifier? exc_var, stmt* handler_body), stmt* orelse, stmt* finalbody)

        | Assert(expr test, expr? msg)

        | Import(identifier name, identifier? asname)
        | ImportFrom(identifier? module, identifier name, identifier? asname, int? level)

        | Global(identifier name)
        | Nonlocal(identifier name)
        | Pass
        | Break
        | Continue

        -- col_offset is the byte offset in the utf8 string the parser uses
        attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

expr =  Call(callee, PilExprListItem* args, keyword* kwargs)
        # callee is a plain function identifier or one of the !-sentinels below;
        # BinOp / UnaryOp / Compare / Dict / Set / List / Tuple / Yield /
        # YieldFrom / JoinedStr are all carried by Call
        | Constant(constant value, string? constant_kind)
        | Name(identifier id)
        | Attribute(expr target, identifier attr)
        | Subscript(expr target, PilExprIndex* index)
        | Starred(identifier value)
        | Slice(expr? lower, expr? upper, expr? step)

callee-sentinel = !bop.<op> | !uop.<op> | !cop.<op>
                | !dict | !set | !list | !tuple
                | !yield | !yield_from | !joined_str

operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                | RShift | BitOr | BitXor | BitAnd | FloorDiv

unaryop = Invert | Not | UAdd | USub

cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
"""

import ast
from collections.abc import Mapping
import enum
from typing import Optional

from pypto.frontend.kind_dict import KindDict


class PilCodeKind(enum.Enum):
    PIL_CONSTANT_EXPR = 'pil_constant_expr'
    PIL_IDENTIFIER_EXPR = 'pil_identifier_expr'
    PIL_STARRED = 'pil_starred'
    PIL_SLICE = 'pil_slice'
    PIL_ATTRIBUTE = 'pil_attribute'
    PIL_SUBSCRIPT = 'pil_subscript'
    PIL_KEYWORD = 'pil_keyword'
    PIL_CALL = 'pil_call'
    PIL_STMT_LIST = 'pil_stmt_list'
    PIL_PARAMETER = 'pil_parameter'
    PIL_FUNCTION_PARAMETER_DEF = 'pil_function_parameter_def'
    PIL_FUNCTION_DEF = 'pil_function_def'
    PIL_CLASS_DEF = 'pil_class_def'
    PIL_ASSIGN_SYMBOL_LIST = 'pil_assign_symbol_list'
    PIL_ASSIGN_ATTRIBUTE = 'pil_assign_attribute'
    PIL_ASSIGN_SUBSCRIPT = 'pil_assign_subscript'
    PIL_DELETE_IDENTIFIER = 'pil_delete_identifier'
    PIL_DELETE_ATTRIBUTE = 'pil_delete_attribute'
    PIL_DELETE_SUBSCRIPT = 'pil_delete_subscript'
    PIL_RETURN = 'pil_return'
    PIL_FOR = 'pil_for'
    PIL_WHILE = 'pil_while'
    PIL_IF = 'pil_if'
    PIL_WITH = 'pil_with'
    PIL_RAISE = 'pil_raise'
    PIL_TRY = 'pil_try'
    PIL_ASSERT = 'pil_assert'
    PIL_IMPORT = 'pil_import'
    PIL_IMPORT_FROM = 'pil_import_from'
    PIL_GLOBAL = 'pil_global'
    PIL_NONLOCAL = 'pil_nonlocal'
    PIL_PASS = 'pil_pass'
    PIL_BREAK = 'pil_break'
    PIL_CONTINUE = 'pil_continue'

pil_code_dict = KindDict()

class PilCode:
    pass

class PilAttr(Mapping):
    ATTR_LIST = [
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset',
    ]

    def __init__(self, node):
        self._data = {attr: 0 if node is None else getattr(node, attr, 0) for attr in self.ATTR_LIST}

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

class PilExprListItem(PilCode):
    """list item that might be expr, starred"""
    pass

class PilExprIndex(PilCode):
    """index that might be expr, slice"""
    pass

class PilExprAssignSymbol(PilCode):
    """assign symbol that might be expr, call, attr, subscript"""
    pass

class PilExpr(PilExprListItem, PilExprIndex, PilExprAssignSymbol):
    """individual expresion"""
    pass

@pil_code_dict(PilCodeKind.PIL_CONSTANT_EXPR)
class PilConstantExpr(PilExpr):
    """Constant(constant value, string? constant_kind)

    ``constant_kind`` is the legacy u-string marker; renamed so the
    ``kind`` class attribute set by KindDict stays intact.

    Emit code:
        value
    """

    def __init__(self, value, constant_kind: Optional[str] = None):
        self._value = value
        self._constant_kind = constant_kind

    @property
    def value(self):
        return self._value

    @property
    def constant_kind(self) -> Optional[str]:
        return self._constant_kind

@pil_code_dict(PilCodeKind.PIL_IDENTIFIER_EXPR)
class PilIdentifierExpr(PilExpr):
    """Name(identifier id)

    Emit code:
        id
    """

    def __init__(self, id: str):
        self._id = id

    @property
    def id(self) -> str:
        return self._id

@pil_code_dict(PilCodeKind.PIL_STARRED)
class PilStarred(PilExprListItem):
    """Starred(identifier value)

    Emit code:
        *value
    """

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value

@pil_code_dict(PilCodeKind.PIL_SLICE)
class PilSlice(PilExprIndex):
    """Slice(expr? lower, expr? upper, expr? step)

    Emit code:
        lower:upper:step
    """

    def __init__(self, lower: Optional[PilExpr], upper: Optional[PilExpr], step: Optional[PilExpr]):
        self._lower = lower
        self._upper = upper
        self._step = step

    @property
    def lower(self) -> Optional[PilExpr]:
        return self._lower

    @property
    def upper(self) -> Optional[PilExpr]:
        return self._upper

    @property
    def step(self) -> Optional[PilExpr]:
        return self._step


@pil_code_dict(PilCodeKind.PIL_ATTRIBUTE)
class PilAttribute(PilExprAssignSymbol):
    """Attribute(expr target, identifier attr)

    Emit code:
        target.attr
    """

    def __init__(self, target: PilExpr, attr: str):
        self._target = target
        self._attr = attr

    @property
    def target(self) -> PilExpr:
        return self._target

    @property
    def attr(self) -> str:
        return self._attr

@pil_code_dict(PilCodeKind.PIL_SUBSCRIPT)
class PilSubscript(PilExprAssignSymbol):
    """Subscript(expr target, PilExprIndex* index)

    One index stays bare; multiple indices form a tuple index.

    Emit code:
        target[index[0]]
        target[index[0], index[1]]
    """

    def __init__(self, target: PilExpr, index: list[PilExprIndex]):
        self._target = target
        self._index = index

    @property
    def target(self) -> PilExpr:
        return self._target

    @property
    def index(self) -> list[PilExprIndex]:
        return self._index

PIL_CALLEE_DICT_OP = '!dict'
PIL_CALLEE_SET_OP = '!set'
PIL_CALLEE_YIELD = '!yield'
PIL_CALLEE_YIELD_FROM = '!yield_from'
PIL_CALLEE_JOINED_STR = '!joined_str'
PIL_CALLEE_LIST = '!list'
PIL_CALLEE_TUPLE = '!tuple'

PIL_CALLEE_BOP_ADD = '!bop.add'
PIL_CALLEE_BOP_SUB = '!bop.sub'
PIL_CALLEE_BOP_MULT = '!bop.mult'
PIL_CALLEE_BOP_MATMULT = '!bop.matmult'
PIL_CALLEE_BOP_DIV = '!bop.div'
PIL_CALLEE_BOP_MOD = '!bop.mod'
PIL_CALLEE_BOP_POW = '!bop.pow'
PIL_CALLEE_BOP_LSHIFT = '!bop.lshift'
PIL_CALLEE_BOP_RSHIFT = '!bop.rshift'
PIL_CALLEE_BOP_BITOR = '!bop.bitor'
PIL_CALLEE_BOP_BITXOR = '!bop.bitxor'
PIL_CALLEE_BOP_BITAND = '!bop.bitand'
PIL_CALLEE_BOP_FLOORDIV = '!bop.floordiv'

PIL_CALLEE_UOP_INVERT = '!uop.invert'
PIL_CALLEE_UOP_NOT = '!uop.not'
PIL_CALLEE_UOP_UADD = '!uop.uadd'
PIL_CALLEE_UOP_USUB = '!uop.usub'

PIL_CALLEE_COP_EQ = '!cop.eq'
PIL_CALLEE_COP_NOTEQ = '!cop.noteq'
PIL_CALLEE_COP_LT = '!cop.lt'
PIL_CALLEE_COP_LTE = '!cop.lte'
PIL_CALLEE_COP_GT = '!cop.gt'
PIL_CALLEE_COP_GTE = '!cop.gte'
PIL_CALLEE_COP_IS = '!cop.is'
PIL_CALLEE_COP_ISNOT = '!cop.isnot'
PIL_CALLEE_COP_IN = '!cop.in'
PIL_CALLEE_COP_NOTIN = '!cop.notin'

# Map ast.operator / ast.unaryop / ast.cmpop instances to callee sentinels.
PIL_BOP_TO_CALLEE = {
    ast.Add: PIL_CALLEE_BOP_ADD,
    ast.Sub: PIL_CALLEE_BOP_SUB,
    ast.Mult: PIL_CALLEE_BOP_MULT,
    ast.MatMult: PIL_CALLEE_BOP_MATMULT,
    ast.Div: PIL_CALLEE_BOP_DIV,
    ast.Mod: PIL_CALLEE_BOP_MOD,
    ast.Pow: PIL_CALLEE_BOP_POW,
    ast.LShift: PIL_CALLEE_BOP_LSHIFT,
    ast.RShift: PIL_CALLEE_BOP_RSHIFT,
    ast.BitOr: PIL_CALLEE_BOP_BITOR,
    ast.BitXor: PIL_CALLEE_BOP_BITXOR,
    ast.BitAnd: PIL_CALLEE_BOP_BITAND,
    ast.FloorDiv: PIL_CALLEE_BOP_FLOORDIV,
}

PIL_UOP_TO_CALLEE = {
    ast.Invert: PIL_CALLEE_UOP_INVERT,
    ast.Not: PIL_CALLEE_UOP_NOT,
    ast.UAdd: PIL_CALLEE_UOP_UADD,
    ast.USub: PIL_CALLEE_UOP_USUB,
}

PIL_COP_TO_CALLEE = {
    ast.Eq: PIL_CALLEE_COP_EQ,
    ast.NotEq: PIL_CALLEE_COP_NOTEQ,
    ast.Lt: PIL_CALLEE_COP_LT,
    ast.LtE: PIL_CALLEE_COP_LTE,
    ast.Gt: PIL_CALLEE_COP_GT,
    ast.GtE: PIL_CALLEE_COP_GTE,
    ast.Is: PIL_CALLEE_COP_IS,
    ast.IsNot: PIL_CALLEE_COP_ISNOT,
    ast.In: PIL_CALLEE_COP_IN,
    ast.NotIn: PIL_CALLEE_COP_NOTIN,
}

@pil_code_dict(PilCodeKind.PIL_KEYWORD)
class PilKeyword(PilCode):
    """keyword(arg?, value) — ``None`` arg means ``**dict`` unpacking.

    Emit code:
        arg=value    # arg is not None
        **value      # arg is None
    """

    def __init__(self, arg: Optional[str], value: PilExpr):
        self._arg = arg
        self._value = value

    @property
    def arg(self) -> Optional[str]:
        return self._arg

    @property
    def value(self) -> PilExpr:
        return self._value

@pil_code_dict(PilCodeKind.PIL_CALL)
class PilCall(PilExprAssignSymbol):
    """Generic carrier for every other expr kind in the grammar.

    The kind is encoded in ``callee``: either a plain function identifier
    (``Call``) or one of the ``PIL_CALLEE_*`` sentinels (BinOp, UnaryOp, Dict,
    Set, Compare, Yield, YieldFrom, JoinedStr, List, Tuple). Operands go in
    ``args``; named parts go in ``kwargs``.

    Emit code:
        callee(args[0], args[1], kwargs[0].arg=kwargs[0].value)   # plain call
        args[0] <op> args[1]                                      # !bop.<op>
        <op> args[0]                                               # !uop.<op>
        args[0] <op> args[1]                                       # !cop.<op>
        {kwargs}                                                   # !dict
        {args[0], *args[1]}                                        # !set
        [args[0], *args[1]]                                        # !list
        (args[0], *args[1])                                        # !tuple
        yield args[0]                                              # !yield
        yield from args[0]                                         # !yield_from
        f"literal {args[0]}"                                       # !joined_str
    """

    def __init__(self, callee: str, args: list[PilExprListItem], kwargs: list[PilKeyword] = None):
        self._callee = callee
        self._args = list(args)
        self._kwargs = list(kwargs) if kwargs else []

    @property
    def callee(self) -> str:
        return self._callee

    @property
    def args(self) -> list[PilExprListItem]:
        return self._args

    @property
    def kwargs(self) -> list[PilKeyword]:
        return self._kwargs

class PilStmt(PilCode):
    pass

@pil_code_dict(PilCodeKind.PIL_STMT_LIST)
class PilStmtList(PilStmt):
    """Carrier for every ``stmt*`` list field in the grammar.

    Emit code:
        stmts[0]
        stmts[1]
    """

    def __init__(self, stmts: list[PilStmt]):
        self._stmts = list(stmts)

    @property
    def stmts(self) -> list[PilStmt]:
        return self._stmts

@pil_code_dict(PilCodeKind.PIL_PARAMETER)
class PilParameter(PilCode):
    """arg(identifier arg, expr? annotation, expr? default, string? type_comment)

    Emit code:
        arg: annotation = default
    """

    def __init__(self, name: str, default: Optional[PilExpr] = None,
                 annotation: Optional[PilExpr] = None, type_comment: Optional[str] = None):
        self._name = name
        self._default = default
        self._annotation = annotation
        self._type_comment = type_comment

    @property
    def name(self) -> str:
        return self._name

    @property
    def default(self) -> Optional[PilExpr]:
        return self._default

    @property
    def annotation(self) -> Optional[PilExpr]:
        return self._annotation

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_FUNCTION_PARAMETER_DEF)
class PilFunctionParameterDef(PilCode):
    """arguments(arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs, arg? kwarg)

    Emit code:
        def f(posonlyargs, /, args, *vararg, kwonlyargs, **kwarg)
    """

    def __init__(self,
                 posonlyargs: list[PilParameter] = (),
                 args: list[PilParameter] = (),
                 vararg: Optional[PilParameter] = None,
                 kwonlyargs: list[PilParameter] = (),
                 kwarg: Optional[PilParameter] = None):
        self._posonlyargs = list(posonlyargs)
        self._args = list(args)
        self._vararg = vararg
        self._kwonlyargs = list(kwonlyargs)
        self._kwarg = kwarg

    @property
    def posonlyargs(self) -> list[PilParameter]:
        return self._posonlyargs

    @property
    def args(self) -> list[PilParameter]:
        return self._args

    @property
    def vararg(self) -> Optional[PilParameter]:
        return self._vararg

    @property
    def kwonlyargs(self) -> list[PilParameter]:
        return self._kwonlyargs

    @property
    def kwarg(self) -> Optional[PilParameter]:
        return self._kwarg

@pil_code_dict(PilCodeKind.PIL_FUNCTION_DEF)
class PilFunctionDef(PilStmt):
    """FunctionDef(identifier name, PilFunctionParameterDef args, stmt* body, expr* decorator_list,
                   expr? returns, string? type_comment)

    Emit code:
        @decorator_list[0]
        def name(args) -> returns:
            body
    """

    def __init__(self, name: str, args: PilFunctionParameterDef, body: PilStmtList,
                 decorator_list: list[PilExpr] = (),
                 return_expr: Optional[PilExpr] = None, type_comment: Optional[str] = None):
        self._name = name
        self._args = args
        self._body = body
        self._decorator_list = list(decorator_list)
        self._return_expr = return_expr
        self._type_comment = type_comment

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> PilFunctionParameterDef:
        return self._args

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def decorator_list(self) -> list[PilExpr]:
        return self._decorator_list

    @property
    def return_expr(self) -> Optional[PilExpr]:
        return self._return_expr

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_CLASS_DEF)
class PilClassDef(PilStmt):
    """ClassDef(identifier name, expr* bases, keyword* keywords, stmt* body, expr* decorator_list)

    Emit code:
        @decorator_list[0]
        class name(bases, **keywords):
            body
    """

    def __init__(self, name: str, bases: list[PilExprListItem], keywords: list[PilKeyword],
                 body: PilStmtList, decorator_list: list[PilExpr] = ()):
        self._name = name
        self._bases = list(bases)
        self._keywords = list(keywords)
        self._body = body
        self._decorator_list = list(decorator_list)

    @property
    def name(self) -> str:
        return self._name

    @property
    def bases(self) -> list[PilExprListItem]:
        return self._bases

    @property
    def keywords(self) -> list[PilKeyword]:
        return self._keywords

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def decorator_list(self) -> list[PilExpr]:
        return self._decorator_list

@pil_code_dict(PilCodeKind.PIL_ASSIGN_SYMBOL_LIST)
class PilAssignSymbolList(PilStmt):
    """Assign(expr target*, expr value, string? type_comment)

    Emit code:
        target_list[0], *target_list[1], target_list[2] = value
    """

    def __init__(self, target_list: list[PilExprListItem], value: PilExprAssignSymbol,
                 type_comment: Optional[str] = None):
        self._target_list = target_list
        self._value = value
        self._type_comment = type_comment

    @property
    def target_list(self) -> list[PilExprListItem]:
        return self._target_list

    @property
    def value(self) -> PilExpr:
        return self._value

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_ASSIGN_ATTRIBUTE)
class PilAssignAttribute(PilStmt):
    """Assign(expr target, identifier attr, expr value, string? type_comment)

    Emit code:
        target.attr = value
    """

    def __init__(self, target: PilExpr, attr: str, value: PilExpr, type_comment: Optional[str] = None):
        self._target = target
        self._attr = attr
        self._value = value
        self._type_comment = type_comment

    @property
    def target(self) -> PilExpr:
        return self._target

    @property
    def attr(self) -> str:
        return self._attr

    @property
    def value(self) -> PilExpr:
        return self._value

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_ASSIGN_SUBSCRIPT)
class PilAssignSubscript(PilStmt):
    """Assign(expr target, PilExprIndex* index, expr value, string? type_comment)

    One index stays bare; multiple indices form a tuple index.

    Emit code:
        target[index[0]] = value
        target[index[0], index[1]] = value
    """

    def __init__(self, target: PilExpr, index: list[PilExprIndex], value: PilExpr, type_comment: Optional[str] = None):
        self._target = target
        self._index = index
        self._value = value
        self._type_comment = type_comment

    @property
    def target(self) -> PilExpr:
        return self._target

    @property
    def index(self) -> list[PilExprIndex]:
        return self._index

    @property
    def value(self) -> PilExpr:
        return self._value

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_DELETE_IDENTIFIER)
class PilDeleteIdentifier(PilStmt):
    """Delete(expr target)

    target only allows identifier, attribute and subscript.

    Emit code:
        del target
    """

    def __init__(self, target: PilIdentifierExpr):
        self._target = target

    @property
    def target(self) -> PilIdentifierExpr:
        return self._target

@pil_code_dict(PilCodeKind.PIL_DELETE_ATTRIBUTE)
class PilDeleteAttribute(PilStmt):
    """Delete(expr target, attr)

    Emit code:
        del target.attr
    """

    def __init__(self, target: PilIdentifierExpr, attr: str):
        self._target = target
        self._attr = attr

    @property
    def target(self) -> PilIdentifierExpr:
        return self._target

    @property
    def attr(self) -> str:
        return self._attr

@pil_code_dict(PilCodeKind.PIL_DELETE_SUBSCRIPT)
class PilDeleteSubscript(PilStmt):
    """Delete(expr target, PilExprIndex* index)

    One index stays bare; multiple indices form a tuple index.

    Emit code:
        del target[index[0]]
        del target[index[0], index[1]]
    """

    def __init__(self, target: PilIdentifierExpr, index: list[PilExprIndex]):
        self._target = target
        self._index = index

    @property
    def target(self) -> PilIdentifierExpr:
        return self._target

    @property
    def index(self) -> list[PilExprIndex]:
        return self._index

@pil_code_dict(PilCodeKind.PIL_RETURN)
class PilReturn(PilStmt):
    """Return(identifier? value)

    Emit code:
        return value    # value is not None
        return          # value is None
    """

    def __init__(self, value: Optional[PilExpr] = None):
        self._value = value

    @property
    def value(self) -> Optional[PilExpr]:
        return self._value

@pil_code_dict(PilCodeKind.PIL_FOR)
class PilFor(PilStmt):
    """For(identifier target, identifier iter, stmt* body, stmt* orelse, string? type_comment)

    Emit code:
        for target in iter:
            body
        else:
            orelse
    """

    def __init__(self, target: PilExpr, iter: PilExpr, body: PilStmtList,
                 orelse: PilStmtList, type_comment: Optional[str] = None):
        self._target = target
        self._iter = iter
        self._body = body
        self._orelse = orelse
        self._type_comment = type_comment

    @property
    def target(self) -> PilExpr:
        return self._target

    @property
    def iter(self) -> PilExpr:
        return self._iter

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def orelse(self) -> PilStmtList:
        return self._orelse

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_WHILE)
class PilWhile(PilStmt):
    """While(identifier test, stmt* body, stmt* orelse)

    Emit code:
        while test:
            body
        else:
            orelse
    """

    def __init__(self, test: PilExpr, body: PilStmtList, orelse: PilStmtList):
        self._test = test
        self._body = body
        self._orelse = orelse

    @property
    def test(self) -> PilExpr:
        return self._test

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def orelse(self) -> PilStmtList:
        return self._orelse

@pil_code_dict(PilCodeKind.PIL_IF)
class PilIf(PilStmt):
    """If(identifier test, stmt* body, stmt* orelse)

    Emit code:
        if test:
            body
        else:
            orelse
    """

    def __init__(self, test: PilExpr, body: PilStmtList, orelse: PilStmtList):
        self._test = test
        self._body = body
        self._orelse = orelse

    @property
    def test(self) -> PilExpr:
        return self._test

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def orelse(self) -> PilStmtList:
        return self._orelse

@pil_code_dict(PilCodeKind.PIL_WITH)
class PilWith(PilStmt):
    """With(expr context_expr, identifier? optional_vars, stmt* body, string? type_comment)

    One statement holds exactly one ``with xxx [as yyy]`` clause; multiple
    context managers nest through the body.

    Emit code:
        with context_expr as optional_vars:
            body
    """

    def __init__(self, context_expr: PilExpr, body: PilStmtList,
                 optional_vars: Optional[str] = None, type_comment: Optional[str] = None):
        self._context_expr = context_expr
        self._optional_vars = optional_vars
        self._body = body
        self._type_comment = type_comment

    @property
    def context_expr(self) -> PilExpr:
        return self._context_expr

    @property
    def optional_vars(self) -> Optional[str]:
        return self._optional_vars

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def type_comment(self) -> Optional[str]:
        return self._type_comment

@pil_code_dict(PilCodeKind.PIL_RAISE)
class PilRaise(PilStmt):
    """Raise(identifier? exc, identifier? cause)

    Emit code:
        raise exc from cause    # exc and cause are not None
        raise exc               # cause is None
        raise                   # exc is None
    """

    def __init__(self, exc: Optional[PilExpr] = None, cause: Optional[PilExpr] = None):
        self._exc = exc
        self._cause = cause

    @property
    def exc(self) -> Optional[PilExpr]:
        return self._exc

    @property
    def cause(self) -> Optional[PilExpr]:
        return self._cause

@pil_code_dict(PilCodeKind.PIL_TRY)
class PilTry(PilStmt):
    """Try(stmt* body, (identifier exc_var, stmt* handler_body), stmt* orelse, stmt* finalbody)

    Emit code:
        try:
            body
        except Exception as exc_var:
            handler_body
        else:
            orelse
        finally:
            finalbody
    """

    def __init__(self, body: PilStmtList, handler: tuple[Optional[str], PilStmtList],
                 orelse: PilStmtList, finalbody: PilStmtList):
        self._body = body
        self._handler = handler
        self._orelse = orelse
        self._finalbody = finalbody

    @property
    def body(self) -> PilStmtList:
        return self._body

    @property
    def handler(self) -> tuple[Optional[str], PilStmtList]:
        return self._handler

    @property
    def orelse(self) -> PilStmtList:
        return self._orelse

    @property
    def finalbody(self) -> PilStmtList:
        return self._finalbody

@pil_code_dict(PilCodeKind.PIL_ASSERT)
class PilAssert(PilStmt):
    """Assert(identifier test, identifier? msg)

    Emit code:
        assert test, msg    # msg is not None
        assert test         # msg is None
    """

    def __init__(self, test: PilExpr, msg: Optional[PilExpr] = None):
        self._test = test
        self._msg = msg

    @property
    def test(self) -> PilExpr:
        return self._test

    @property
    def msg(self) -> Optional[PilExpr]:
        return self._msg

@pil_code_dict(PilCodeKind.PIL_IMPORT)
class PilImport(PilStmt):
    """Import(identifier name, identifier asname?)

    One statement imports exactly one symbol; ``import a, b`` is lowered to
    two statements.

    Emit code:
        import name as asname    # asname is not None
        import name              # asname is None
    """

    def __init__(self, name: str, asname: Optional[str] = None):
        self._name = name
        self._asname = asname

    @property
    def name(self):
        return self._name

    @property
    def asname(self):
        return self._asname

@pil_code_dict(PilCodeKind.PIL_IMPORT_FROM)
class PilImportFrom(PilStmt):
    """ImportFrom(identifier? module, identifier name, identifier? asname, int? level)

    One statement imports exactly one symbol; ``import a, b`` is lowered to
    two statements.

    Emit code:
        from module import name as asname    # asname is not None
        from module import name              # asname is None
    """

    def __init__(self, module: Optional[str], name: str, asname: Optional[str] = None,
                 level: Optional[int] = None):
        self._module = module
        self._name = name
        self._asname = asname
        self._level = level

    @property
    def module(self) -> Optional[str]:
        return self._module

    @property
    def name(self) -> str:
        return self._name

    @property
    def asname(self) -> Optional[str]:
        return self._asname

    @property
    def level(self) -> Optional[int]:
        return self._level

@pil_code_dict(PilCodeKind.PIL_GLOBAL)
class PilGlobal(PilStmt):
    """Global(identifier name)

    One statement declares exactly one variable; ``global a, b`` is lowered
    to two statements.

    Emit code:
        global name
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

@pil_code_dict(PilCodeKind.PIL_NONLOCAL)
class PilNonlocal(PilStmt):
    """Nonlocal(identifier name)

    One statement declares exactly one variable; ``nonlocal a, b`` is lowered
    to two statements.

    Emit code:
        nonlocal name
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

@pil_code_dict(PilCodeKind.PIL_PASS)
class PilPass(PilStmt):
    """Pass

    Emit code:
        pass
    """

@pil_code_dict(PilCodeKind.PIL_BREAK)
class PilBreak(PilStmt):
    """Break

    Emit code:
        break
    """

@pil_code_dict(PilCodeKind.PIL_CONTINUE)
class PilContinue(PilStmt):
    """Continue

    Emit code:
        continue
    """
