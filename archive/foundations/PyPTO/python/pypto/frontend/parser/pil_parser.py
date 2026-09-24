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


import ast
import re
from typing import Callable, Optional, Union

from pypto.error import FeError

from .pil import (
    PilAssignSymbolList,
    PilAttr,
    PilConstantExpr,
    PilExpr,
    PilExprIndex,
    PilFunctionDef,
    PilIdentifierExpr,
    PilSlice,
    PilStarred,
    PilStmtList,
)
from .pil_builder import PilBuilder
from .pil_io_text import pil_to_ast

PIL_DEFAULT_PREFIX = '_pil_'


class PILContext:
    def __init__(self, prefix=PIL_DEFAULT_PREFIX):
        self._continue_stack = []
        self._temp_count = 0
        self._prefix = prefix
        self._ann_assign_in_function = False

    @property
    def continue_stack(self) -> list[Optional[tuple[ast.expr, str]]]:
        return self._continue_stack

    def create_temp_identifier(self, *_args, **_kwargs) -> str:
        name = f"{self._prefix}{self._temp_count}"
        self._temp_count += 1
        return name

    def ann_assign_in_function_switch(self, switch: bool) -> bool:
        curr = self._ann_assign_in_function
        self._ann_assign_in_function = switch
        return curr

    @property
    def ann_assign_in_function(self):
        return self._ann_assign_in_function


NOATTR = PilAttr(None)


class PythonParser(PilBuilder, ast.NodeVisitor):
    def __init__(self, ctx: PILContext):
        PilBuilder.__init__(self)
        self._ctx = ctx

    @property
    def continue_stack(self) -> list[Optional[tuple[ast.expr, str]]]:
        return self._ctx.continue_stack

    def create_temp_identifier(self) -> str:
        return self._ctx.create_temp_identifier()

    @staticmethod
    def _node_name_to_visitor_suffix(name: str) -> str:
        return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


    @staticmethod
    def create_expr(value):
        """Coerce legacy forms (str | ast.Constant | None) into a PilExpr node."""
        if value is None or isinstance(value, PilExpr):
            return value
        if isinstance(value, str):
            return PilIdentifierExpr(value)
        if isinstance(value, ast.Constant):
            return PilConstantExpr(value.value, value.kind)
        return value



    def create_stmt_list(self, *parts) -> PilStmtList:
        """Flatten parts (PilStmtList / list / single stmt) into one PilStmtList."""
        out = []
        for p in parts:
            if p is None:
                continue
            if isinstance(p, PilStmtList):
                out.extend(p.stmts)
            elif isinstance(p, list):
                out.extend(p)
            else:
                out.append(p)
        return PilStmtList(out)

    def create_assign_symbol(self, name: str, value: PilExpr, **_kwargs) -> PilAssignSymbolList:
        """Assign a single identifier (str name) from a PilExpr value."""
        return self.create_assign_symbol_list([self.create_name(name)], value, **_kwargs)

    def visit_slice_values(
        self,
        slice_expr: Union[ast.Slice, tuple[ast.Slice]],
    ) -> tuple[PilStmtList, list[PilExprIndex]]:
        slice_list = slice_expr.elts if isinstance(slice_expr, ast.Tuple) else [slice_expr]

        slice_stmts: list = []
        pil_slice_list: list[PilExprIndex] = []
        for s in slice_list:
            if isinstance(s, ast.Slice):
                lower_expr = upper_expr = step_expr = None
                if s.lower is not None:
                    stmts, lower_expr = self.visit(s.lower)
                    slice_stmts.extend(stmts.stmts)
                if s.upper is not None:
                    stmts, upper_expr = self.visit(s.upper)
                    slice_stmts.extend(stmts.stmts)
                if s.step is not None:
                    stmts, step_expr = self.visit(s.step)
                    slice_stmts.extend(stmts.stmts)
                pil_slice_list.append(PilSlice(
                    self.create_expr(lower_expr), self.create_expr(upper_expr), self.create_expr(step_expr)))
            else:
                subscript_stmts, subscript_expr = self.visit(s)
                slice_stmts.extend(subscript_stmts.stmts)
                pil_slice_list.append(subscript_expr)
        return PilStmtList(slice_stmts), pil_slice_list

    def visit_lhs(self, target: ast.expr, source_expr: Optional[PilExpr]) -> PilStmtList:
        if isinstance(target, ast.Name):
            if source_expr is None:
                return self.create_stmt_list()
            return self.create_stmt_list(self.create_assign_identifier(target.id, source_expr))

        elif isinstance(target, ast.Attribute):
            obj_stmts, obj_expr = self.visit(target.value)
            if source_expr is None:
                return self.create_stmt_list(obj_stmts)
            return self.create_stmt_list(obj_stmts, self.create_assign_attribute(obj_expr, target.attr, source_expr))

        elif isinstance(target, ast.Subscript):
            obj_stmts, obj_expr = self.visit(target.value)
            slice_stmts, pil_slice_list = self.visit_slice_values(target.slice)
            if source_expr is None:
                return self.create_stmt_list(obj_stmts, slice_stmts)
            index = pil_slice_list
            return self.create_stmt_list(obj_stmts, slice_stmts,
                               self.create_assign_subscript(obj_expr, index, source_expr))

        elif isinstance(target, (ast.Tuple, ast.List)):
            # Step 1: allocate one temp per element, preserving starred-ness
            elt_temps = [
                PilStarred(self.create_temp_identifier())
                if isinstance(elt, ast.Starred)
                else self.create_name(self.create_temp_identifier())
                for elt in target.elts
            ]

            # Step 2: emit the first unpack into per-element temporaries.
            if source_expr is None:
                return self.create_stmt_list()
            unpack_stmts = [self.create_assign_symbol_list(elt_temps, source_expr)]

            # Step 3: recursively handle each element with its temp
            result_stmts = list(unpack_stmts)
            for elt_temp, elt in zip(elt_temps, target.elts):
                if isinstance(elt_temp, PilStarred):
                    result_stmts.extend(self.visit_lhs(elt.value, self.create_name(elt_temp.value)).stmts)
                else:
                    result_stmts.extend(self.visit_lhs(elt, elt_temp).stmts)
            return PilStmtList(result_stmts)

        raise FeError(NotImplementedError(f"LHS target type {type(target).__name__} is not supported"))

    def visit_function_def(
        self,
        name: str,
        args: ast.arguments,
        body: list[ast.stmt],
        decorator_list: list[ast.expr],
        returns: Optional[ast.expr],
        type_comment: Optional[str],
        node_attr: PilAttr = NOATTR,
        **kwargs,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (no decorators):
            Python:
                def name(args):
                    body
            PIL:
                def name(args):
                    body
        Case 2 (with decorators):
            Python:
                @dec_expr
                def name(args):
                    body
            PIL:
                _tmp_0 = dec_expr

                @_tmp_0
                def name(args):
                    body
        """
        decorator_stmts = []
        decorator_names = []
        for dec in decorator_list:
            dec_stmts, dec_name = self.visit(dec)
            decorator_stmts.extend(dec_stmts.stmts)
            decorator_names.append(dec_name)

        curr = self._ctx.ann_assign_in_function_switch(True)
        body_stmts, _ = self.visit_stmts(body)
        self._ctx.ann_assign_in_function_switch(curr)

        if returns is not None and not isinstance(returns, ast.Name):
            raise FeError(NotImplementedError(
                f"Return annotation {type(returns).__name__} is not supported"))
        return_expr = self.create_name(returns.id) if returns is not None else None
        func_def = self.create_function_def(
            name, args, body_stmts, decorator_names, return_expr, type_comment, node_attr=node_attr
        )
        return self.create_stmt_list(decorator_stmts, func_def), None

    def visit_async_function_def(
        self,
        name: str,
        args: ast.arguments,
        body: list[ast.stmt],
        decorator_list: list[ast.expr],
        returns: Optional[ast.expr],
        type_comment: Optional[str],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("AsyncFunctionDef is not supported"))

    def visit_class_def(
        self,
        name: str,
        bases: list[ast.expr],
        keywords: list[ast.keyword],
        body: list[ast.stmt],
        decorator_list: list[ast.expr],
        type_params=None,
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (no decorators):
            Python:
                class name(args):
                    body
            PIL:
                class name(args):
                    body
        Case 2 (with decorators):
            Python:
                @dec_expr
                class name(args):
                    body
            PIL:
                _tmp_0 = dec_expr

                @_tmp_0
                class name(args):
                    body
        """
        result_decorator_stmts = []
        result_decorator_names = []
        for dec in decorator_list:
            dec_stmts, dec_name = self.visit(dec)
            result_decorator_stmts.extend(dec_stmts.stmts)
            result_decorator_names.append(dec_name)

        result_body, _ = self.visit_stmts(body)
        if len(result_body.stmts) == 0:
            result_body = self.create_stmt_list(self.create_pass())

        result_base_stmt_list = []
        result_base_names = []
        for base in bases:
            if isinstance(base, ast.Starred):
                base_stmts, base_name = self.visit(base.value)
                result_base_names.append(PilStarred(base_name.id))
            else:
                base_stmts, base_name = self.visit(base)
                result_base_names.append(base_name)
            result_base_stmt_list.extend(base_stmts.stmts)

        result_keyword_stmt_list = []
        result_keywords = []
        for kw in keywords:
            kw_stmts, kw_name = self.visit(kw.value)
            result_keyword_stmt_list.extend(kw_stmts.stmts)
            result_keywords.append((kw.arg, kw_name))

        class_def = self.create_class_def(
            name, result_base_names, result_keywords, result_body, result_decorator_names, node_attr=node_attr
        )
        return self.create_stmt_list(
            result_decorator_stmts, result_base_stmt_list, result_keyword_stmt_list, class_def), None

    def visit_return(
        self, value: Optional[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1:
            Python:
                return
            PIL:
                return
        Case 2:
            Pythoin:
                return id0
            PIL:
                return id0
        Case 3:
            Python:
                return expr
            PIL:
                _tmp_0 = expr
                return _tmp_0
        """
        if value is not None:
            value_stmt_list, value_name = self.visit(value)
            return self.create_stmt_list(value_stmt_list, self.create_return(value_name, node_attr=node_attr)), None
        return self.create_stmt_list(self.create_return(None, node_attr=node_attr)), None

    def visit_delete(
        self, targets: list[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (delete name):
            Python:
                del name
            PIL:
                del name
        Case 2 (delete attribute):
            Python:
                del obj.attr
            PIL:
                _tmp_0 = obj
                del _tmp_0.attr
        Case 3 (delete subscript):
            Python:
                del obj[key]
            PIL:
                _tmp_0 = obj
                _tmp_1 = key
                del _tmp_0[_tmp_1]
        Case 4 (delete tuple/list - recursive):
            Python:
                del (a, obj.attr, obj[key])
            PIL:
                del a
                _tmp_0 = obj
                del _tmp_0.attr
                _tmp_1 = obj
                _tmp_2 = key
                del _tmp_1[_tmp_2]
        Case 5 (multiple targets):
            Python:
                del a, obj.attr, obj[key]
            PIL:
                del a
                _tmp_0 = obj
                del _tmp_0.attr
                _tmp_1 = obj
                _tmp_2 = key
                del _tmp_1[_tmp_2]
        """
        result_stmts = []
        for target in targets:
            if isinstance(target, ast.Name):
                result_stmts.append(self.create_delete_identifier(self.create_name(target.id), node_attr=node_attr))
            elif isinstance(target, ast.Attribute):
                obj_stmts, obj_expr = self.visit(target.value)
                result_stmts.extend(obj_stmts.stmts)
                result_stmts.append(self.create_delete_attribute(obj_expr, target.attr, node_attr=node_attr))
            elif isinstance(target, ast.Subscript):
                obj_stmts, obj_expr = self.visit(target.value)
                result_stmts.extend(obj_stmts.stmts)
                slice_stmts, slice_exprs = self.visit_slice_values(target.slice)
                result_stmts.extend(slice_stmts.stmts)
                result_stmts.append(self.create_delete_subscript(
                    obj_expr, slice_exprs, node_attr=node_attr))
            elif isinstance(target, (ast.Tuple, ast.List)):
                nested_stmts, _ = self.visit_delete(target.elts, node_attr=node_attr)
                result_stmts.extend(nested_stmts.stmts)
            else:
                raise FeError(NotImplementedError(f"Delete target type {type(target).__name__} is not supported"))
        return PilStmtList(result_stmts), None

    def visit_assign(
        self, targets: list[ast.expr], value: ast.expr, type_comment: Optional[str], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1:
            Python:
                a = b = id0
            PIL:
                a = id0
                b = id0
        Case 2:
            Python:
                a = b = expr
            PIL:
                _tmp_0 = expr
                a = _tmp_0
                b = _tmp_0
        """
        value_stmts, value_name = self.visit(value)
        result_stmts = list(value_stmts.stmts)
        for target in targets:
            result_stmts.extend(self.visit_lhs(target, value_name).stmts)
        return PilStmtList(result_stmts), None

    def visit_aug_assign(
        self, target: ast.expr, op: ast.operator, value: ast.expr, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (name target):
            Python:
                name op= expr
            PIL:
                _tmp_0 = expr
                _tmp_1 = name op _tmp_0
                name = _tmp_1
        Case 2 (attribute target):
            Python:
                obj.attr op= expr
            PIL:
                _tmp_0 = obj_expr       # evaluate obj
                _tmp_1 = _tmp_0.attr    # load
                _tmp_2 = expr           # evaluate rhs
                _tmp_3 = _tmp_1 op _tmp_2
                _tmp_0.attr = _tmp_3    # store
        Case 3 (subscript target):
            Python:
                obj[slice] op= expr
            PIL:
                _tmp_0 = obj_expr       # evaluate obj
                _tmp_1 = slice_expr     # evaluate slice
                _tmp_2 = _tmp_0[_tmp_1] # load
                _tmp_3 = expr           # evaluate rhs
                _tmp_4 = _tmp_2 op _tmp_3
                _tmp_0[_tmp_1] = _tmp_4 # store
        """
        if isinstance(target, ast.Name):
            # target is a bare name - no side effects, visit value first is fine
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_assign_symbol(
                temp_name, self.create_bin_op(self.create_name(target.id), op, value_name))
            store_stmt = self.create_assign_identifier(target.id, self.create_name(temp_name))
            return self.create_stmt_list(value_stmt_list, [binop_stmt, store_stmt]), None

        elif isinstance(target, ast.Attribute):
            # Python evaluates: obj first, then load, then rhs value, then store
            target_stmt_list, target_expr = self.visit(target.value)
            load_temp = self.create_temp_identifier()
            load_stmt = self.create_assign_symbol(load_temp, self.create_attribute(target_expr, target.attr))
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_assign_symbol(
                temp_name, self.create_bin_op(self.create_name(load_temp), op, value_name))
            store_stmt = self.create_assign_attribute(target_expr, target.attr, self.create_name(temp_name))
            return self.create_stmt_list(target_stmt_list, load_stmt, value_stmt_list, [binop_stmt, store_stmt]), None

        elif isinstance(target, ast.Subscript):
            # Python evaluates: obj first, then slice, then load, then rhs value, then store
            target_stmt_list, target_expr = self.visit(target.value)
            # normalize slice into list of (lower, upper, step) tuples
            slice_stmt_list, pil_slice_list = self.visit_slice_values(target.slice)
            load_temp = self.create_temp_identifier()
            load_stmt = self.create_assign_symbol(load_temp, self.create_subscript(target_expr, pil_slice_list))
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_assign_symbol(
                temp_name, self.create_bin_op(self.create_name(load_temp), op, value_name))
            store_stmt = self.create_assign_subscript(target_expr, pil_slice_list, self.create_name(temp_name))
            return self.create_stmt_list(
                target_stmt_list, slice_stmt_list, load_stmt, value_stmt_list, [binop_stmt, store_stmt]), None

        raise FeError(NotImplementedError(f"AugAssign target type {type(target).__name__} is not supported"))

    def visit_ann_assign(
        self,
        target: ast.expr,
        annotation: ast.expr,
        value: Optional[ast.expr],
        simple: int,
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (annotation only):
            Python:
                target: annotation
            PIL:
                _tmp_0 = annotation     # evaluate annotation for side effects
        Case 2 (annotation with value):
            Python:
                target: annotation = expr
            PIL:
                _tmp_0 = annotation     # evaluate annotation for side effects
                _tmp_1 = expr
                target = _tmp_1
        """
        if value is not None:
            value_stmts, value_expr = self.visit(value)
        else:
            value_stmts, value_expr = self.create_stmt_list(), None
        if self._ctx.ann_assign_in_function:
            # In function, annotation is not computed. However, local class level annotations
            # are still computed for side effects. So we only compute annotation if we are not in a function.
            ann_stmts = self.create_stmt_list()
        else:
            ann_stmts, _ = self.visit(annotation)
        return self.create_stmt_list(value_stmts, self.visit_lhs(target, value_expr), ann_stmts), None

    def visit_for(
        self,
        target: ast.expr,
        iter_expr: ast.expr,
        body: list[ast.stmt],
        orelse: list[ast.stmt],
        type_comment: Optional[str],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            for target in iter_expr:
                body
            else:
                orelse
        PIL:
            _tmp_0 = iter_expr
            for _tmp_1 in _tmp_0:
                target = _tmp_1         # visit_lhs unpacking
                body
            else:
                orelse
        """
        self.continue_stack.append(None)

        iter_stmts, iter_name = self.visit(iter_expr)
        target_name = self.create_temp_identifier()
        target_stmts = self.visit_lhs(target, self.create_name(target_name))

        body_stmts, _ = self.visit_stmts(body)
        result_body = self.create_stmt_list(target_stmts, body_stmts)
        orelse_stmts, _ = self.visit_stmts(orelse)
        self.continue_stack.pop()
        return self.create_stmt_list(iter_stmts, self.create_for(
            self.create_name(target_name), iter_name, result_body, orelse_stmts, type_comment)), None

    def visit_async_for(
        self,
        target: ast.expr,
        iter_expr: ast.expr,
        body: list[ast.stmt],
        orelse: list[ast.stmt],
        type_comment: Optional[str],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("AsyncFor is not supported"))

    def visit_while(
        self, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            while test_expr:
                body
            else:
                orelse
        PIL:
            _tmp_0 = test_expr
            while _tmp_0:
                body
                _tmp_1 = test_expr      # re-evaluate for next iteration
                _tmp_0 = _tmp_1
            else:
                orelse
        """
        test_stmts, test_expr = self.visit(test)
        self.continue_stack.append((test, test_expr))
        body_stmts, _ = self.visit_stmts(body)
        orelse_stmts, _ = self.visit_stmts(orelse)

        if isinstance(test_expr, PilIdentifierExpr):
            # re-evaluate test at the end of each iteration to update test_name for the next check
            reeval_stmts, reeval_expr = self.visit(test)
            result_body = self.create_stmt_list(body_stmts, reeval_stmts,
                                      self.create_assign_identifier(test_expr.id, reeval_expr))
        else:
            result_body = body_stmts
        self.continue_stack.pop()
        return self.create_stmt_list(test_stmts, self.create_while(test_expr, result_body, orelse_stmts)), None

    def visit_if(
        self, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            if test_expr:
                body
            else:
                orelse
        PIL:
            _tmp_0 = test_expr
            if _tmp_0:
                body
            else:
                orelse
        """
        test_stmts, test_name = self.visit(test)
        body_stmts, _ = self.visit_stmts(body)
        orelse_stmts, _ = self.visit_stmts(orelse)
        return self.create_stmt_list(test_stmts, self.create_if(test_name, body_stmts, orelse_stmts)), None

    def visit_with(
        self, items: list[ast.withitem], body: list[ast.stmt], type_comment: Optional[str], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            with ctx_expr0 as var0, ctx_expr1 as var1:
                body
        PIL:
            _tmp_0 = ctx_expr0
            with _tmp_0 as _tmp_1
                var0 = _tmp_1

                _tmp_2 = ctx_expr1
                with _tmp_2 as _tmp_3:
                    var1 = _tmp_3
                    body
        """
        body_stmts, _ = self.visit_stmts(body)
        for item in reversed(items):
            ctx_stmts, ctx_name = self.visit(item.context_expr)
            if item.optional_vars is not None:
                target_name = self.create_temp_identifier()
                item_body = self.create_stmt_list(
                    self.visit_lhs(item.optional_vars, self.create_name(target_name)), body_stmts)
            else:
                target_name = None
                item_body = body_stmts
            body_with = self.create_with(ctx_name, item_body, target_name, type_comment,
                                          node_attr=node_attr)
            body_stmts = self.create_stmt_list(ctx_stmts, body_with)
        return body_stmts, None

    def visit_async_with(
        self, items: list[ast.withitem], body: list[ast.stmt], type_comment: Optional[str], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("AsyncWith is not supported"))

    def visit_match(
        self, subject: ast.expr, cases: list, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("Match is not supported"))

    def visit_raise(
        self, exc: Optional[ast.expr], cause: Optional[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (bare re-raise):
            Python:
                raise
            PIL:
                raise
        Case 2 (raise exception):
            Python:
                raise exc_expr
            PIL:
                _tmp_0 = exc_expr
                raise _tmp_0
        Case 3 (raise with cause):
            Python:
                raise exc_expr from cause_expr
            PIL:
                _tmp_0 = exc_expr
                _tmp_1 = cause_expr
                raise _tmp_0 from _tmp_1
        """
        result_stmts = []
        exc_name = None
        if exc is not None:
            exc_stmts, exc_name = self.visit(exc)
            result_stmts.extend(exc_stmts.stmts)
        cause_name = None
        if cause is not None:
            cause_stmts, cause_name = self.visit(cause)
            result_stmts.extend(cause_stmts.stmts)
        result_stmts.append(self.create_raise(exc_name, cause_name, node_attr=node_attr))
        return PilStmtList(result_stmts), None

    def visit_try(
        self,
        body: list[ast.stmt],
        handlers: list[ast.excepthandler],
        orelse: list[ast.stmt],
        finalbody: list[ast.stmt],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            try:
                body
            except TypeA as e:
                handler_a_body
            except TypeB:
                handler_b_body
            except:
                bare_body
            else:
                orelse
            finally:
                finalbody
        PIL:
            try:
                body
            except Exception as _tmp_0:
                _tmp_1 = TypeA
                _tmp_2 = isinstance(_tmp_0, _tmp_1)
                if _tmp_2:
                    e = _tmp_0
                    handler_a_body
                    del e
                else:
                    _tmp_3 = TypeB
                    _tmp_4 = isinstance(_tmp_0, _tmp_3)
                    if _tmp_4:
                        handler_b_body
                    else:
                        bare_body
            else:
                orelse
            finally:
                finalbody
        """
        body_stmts, _ = self.visit_stmts(body)
        exc_var = self.create_temp_identifier()
        # Build if-elif chain from innermost outward; default else: bare raise
        dispatch_body: PilStmtList = self.create_stmt_list(self.create_raise(None, None))

        for handler in reversed(handlers):
            handler_body_stmts, _ = self.visit_stmts(handler.body)

            if handler.type is not None:
                handler_type_stmts, handler_type_name = self.visit(handler.type)
                isinstance_temp = self.create_temp_identifier()
                isinstance_stmts = [
                    self.create_assign_symbol(isinstance_temp, self.create_call(
                        self.create_name('isinstance'), [self.create_name(exc_var), handler_type_name], []))
                ]

                if handler.name is not None:
                    pre_handler_set = [self.create_assign_identifier(handler.name, self.create_name(exc_var))]
                    post_handler_del = [self.create_delete_identifier(self.create_name(handler.name))]
                    result_then_stmts = self.create_stmt_list(pre_handler_set, handler_body_stmts, post_handler_del)
                else:
                    result_then_stmts = handler_body_stmts
            else:
                handler_type_stmts = self.create_stmt_list()
                isinstance_stmts = self.create_stmt_list()
                isinstance_temp = self.create_constant(True, None)
                result_then_stmts = handler_body_stmts

            dispatch_body = self.create_stmt_list(handler_type_stmts, isinstance_stmts,
                                        self.create_if(
                                            self.create_name(isinstance_temp),
                                            result_then_stmts, dispatch_body))
        orelse_stmts, _ = self.visit_stmts(orelse)
        finalbody_stmts, _ = self.visit_stmts(finalbody)
        return self.create_stmt_list([
            self.create_try(
                body_stmts, (exc_var, dispatch_body), orelse_stmts, finalbody_stmts, node_attr=node_attr
            )
        ]), None

    def visit_try_star(
        self,
        body: list[ast.stmt],
        handlers: list[ast.excepthandler],
        orelse: list[ast.stmt],
        finalbody: list[ast.stmt],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("TryStar is not supported"))

    def visit_assert(
        self, test: ast.expr, msg: Optional[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            assert test_expr, msg_expr
        PIL:
            if __debug__:
                _tmp_0 = test_expr
                _tmp_1 = not _tmp_0
                if _tmp_1:
                    _tmp_2 = msg_expr
                    assert _tmp_0, _tmp_2
        """
        test_stmts, test_name = self.visit(test)
        # use prefix "not_test" to keep "not test_name"
        not_test_name = self.create_temp_identifier()
        not_test_stmt = self.create_assign_symbol(not_test_name, self.create_unary_op(ast.Not(), test_name))
        # msg is only evaluated when the assertion fails
        if msg is not None:
            msg_stmts, msg_name = self.visit(msg)
        else:
            msg_stmts, msg_name = self.create_stmt_list(), None
        fail_body = self.create_stmt_list(msg_stmts, self.create_assert(test_name, msg_name))
        debug_body = self.create_stmt_list(test_stmts, not_test_stmt,
                                 self.create_if(self.create_name(not_test_name), fail_body, self.create_stmt_list()))
        return self.create_stmt_list(
            self.create_if(self.create_name('__debug__'), debug_body, self.create_stmt_list())), None

    def visit_import(
        self, names: list[ast.alias], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list([self.create_import(alias.name, alias.asname) for alias in names]), None

    def visit_import_from(
        self, module: Optional[str], names: list[ast.alias], level: Optional[int], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list(
            [self.create_import_from(module, alias.name, alias.asname, level) for alias in names]), None

    def visit_global(self, names: list[str], node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list([self.create_global(name) for name in names]), None

    def visit_nonlocal(self, names: list[str], node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list([self.create_nonlocal(name) for name in names]), None

    def visit_expr(self, value: ast.expr, node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        value_stmt_list, value_name = self.visit(value)
        return value_stmt_list, None

    def visit_pass(self, node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list([self.create_pass()]), None

    def visit_break(self, node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list([self.create_break()]), None

    def visit_continue(self, node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        if self.continue_stack[-1] is None:
            # continue in for-loop
            return self.create_stmt_list(self.create_continue()), None
        # continue in while-loop
        test, test_expr = self.continue_stack[-1]
        if isinstance(test_expr, PilIdentifierExpr):
            reeval_stmts, reeval_expr = self.visit(test)
            return self.create_stmt_list(reeval_stmts,
                               self.create_assign_identifier(test_expr.id, reeval_expr),
                               self.create_continue()), None
        return self.create_stmt_list(self.create_continue()), None

    # expr nodes

    def visit_bool_op(
        self, op: ast.boolop, values: list[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (and):
            Python:
                a and b and c
            PIL:
                _tmp_0 = a
                if _tmp_0:
                    _tmp_1 = b and c    # recursive
                    _tmp_0 = _tmp_1
                else:
                    _tmp_0 = a
        Case 2 (or):
            Python:
                a or b or c
            PIL:
                _tmp_0 = a
                if _tmp_0:
                    _tmp_0 = a
                else:
                    _tmp_1 = b or c    # recursive
                    _tmp_0 = _tmp_1
        """
        # Base case: single value, just visit it directly
        if len(values) == 1:
            return self.visit(values[0])

        temp_name = self.create_temp_identifier()
        first_stmts, first_name = self.visit(values[0])
        rest_stmts, rest_name = self.visit_bool_op(op=op, values=values[1:])

        if isinstance(op, ast.And):
            # if temp is truthy, evaluate the rest and update temp
            rest_stmt = self.create_if(
                first_name,
                self.create_stmt_list(rest_stmts, self.create_assign_identifier(temp_name, rest_name)),
                self.create_stmt_list(self.create_assign_identifier(temp_name, first_name)),
            )
            return self.create_stmt_list(first_stmts, rest_stmt), self.create_name(temp_name)

        if isinstance(op, ast.Or):
            # if temp is falsy, evaluate the rest and update temp
            rest_stmt = self.create_if(
                first_name,
                self.create_stmt_list(self.create_assign_identifier(temp_name, first_name)),
                self.create_stmt_list(rest_stmts, self.create_assign_identifier(temp_name, rest_name)),
            )
            return self.create_stmt_list(first_stmts, rest_stmt), self.create_name(temp_name)

        raise FeError(NotImplementedError(f"BoolOp {type(op).__name__} is not supported"))

    def visit_named_expr(
        self, target: ast.expr, value: ast.expr, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            target := expr
        PIL:
            _tmp_0 = expr
            target = _tmp_0
        """
        if not isinstance(target, ast.Name):
            raise FeError(
                TypeError('Python native ast parser should guarantee that the target of NamedExpr is always ast.Name')
            )
        value_stmts, value_name = self.visit(value)
        assign_stmt = self.create_assign_identifier(target.id, value_name)
        return self.create_stmt_list(value_stmts, assign_stmt), self.create_name(target.id)

    def visit_bin_op(
        self, left: ast.expr, op: ast.operator, right: ast.expr, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            left_expr op right_expr
        PIL:
            _tmp_0 = left_expr
            _tmp_1 = right_expr
            _tmp_2 = _tmp_0 op _tmp_1
        """
        left_stmts, left_name = self.visit(left)
        right_stmts, right_name = self.visit(right)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_bin_op(left_name, op, right_name)
        binop_stmt = self.create_assign_symbol(temp_name, result_expr)
        return self.create_stmt_list(left_stmts, right_stmts, binop_stmt), self.create_name(temp_name)

    def visit_unary_op(
        self, op: ast.unaryop, operand: ast.expr, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            op operand_expr
        PIL:
            _tmp_0 = operand_expr
            _tmp_1 = op _tmp_0
        """
        operand_stmts, operand_name = self.visit(operand)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_unary_op(op, operand_name)
        unaryop_stmt = self.create_assign_symbol(temp_name, result_expr)
        return self.create_stmt_list(operand_stmts, unaryop_stmt), self.create_name(temp_name)

    def visit_lambda(
        self, args: ast.arguments, body: ast.expr, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            lambda args: body_expr
        PIL:

            def _tmp_0(args):
                _tmp_1 = body_expr
                return _tmp_1
        """
        body_stmts, body_name = self.visit(body)
        return_stmt = self.create_return(body_name)
        func_name = self.create_temp_identifier()
        func_def = self.create_function_def(
            name=func_name,
            args=args,
            body=self.create_stmt_list(body_stmts, return_stmt),
            decorator_list=[],
            return_expr=None,
            type_comment=None,
            node_attr=node_attr,
        )
        return self.create_stmt_list(func_def), self.create_name(func_name)

    def visit_if_exp(
        self, test: ast.expr, body: ast.expr, orelse: ast.expr, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            body_expr if test_expr else orelse_expr
        PIL:
            _tmp_0 = test_expr
            if _tmp_0:
                _tmp_1 = body_expr
                _tmp_2 = _tmp_1
            else:
                _tmp_3 = orelse_expr
                _tmp_2 = _tmp_3
        """
        test_stmts, test_name = self.visit(test)
        body_stmts, body_name = self.visit(body)
        orelse_stmts, orelse_name = self.visit(orelse)

        temp_name = self.create_temp_identifier()
        result_body = self.create_stmt_list(body_stmts, self.create_assign_identifier(temp_name, body_name))
        result_orelse = self.create_stmt_list(orelse_stmts, self.create_assign_identifier(temp_name, orelse_name))
        result_if_stmt = self.create_if(test_name, result_body, result_orelse)
        return self.create_stmt_list(test_stmts, result_if_stmt), self.create_name(temp_name)

    def visit_dict(
        self, keys: list[Optional[ast.expr]], values: list[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            {key0: val0, **val1, key2: val2}
        PIL:
            _tmp_0 = key0
            _tmp_1 = val0
            _tmp_2 = val1
            _tmp_3 = key2
            _tmp_4 = val2
            _tmp_5 = {_tmp_0: _tmp_1, **_tmp_2, _tmp_3: _tmp_4}
        """
        result_stmts: list = []
        key_name_list = []
        value_name_list = []
        for key, value in zip(keys, values):
            if key is not None:
                key_stmts, key_name = self.visit(key)
            else:
                key_stmts, key_name = self.create_stmt_list(), None
            value_stmts, value_name = self.visit(value)

            result_stmts.extend(key_stmts.stmts)
            result_stmts.extend(value_stmts.stmts)
            key_name_list.append(key_name)
            value_name_list.append(value_name)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_dict(key_name_list, value_name_list)
        result_stmts.append(self.create_assign_symbol(temp_name, result_expr))
        return PilStmtList(result_stmts), self.create_name(temp_name)

    def visit_set(self, elts: list[ast.expr], node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            {elt0, *elt1, elt2}
        PIL:
            _tmp_0 = elt0
            _tmp_1 = elt1
            _tmp_2 = elt2
            _tmp_3 = {_tmp_0, *_tmp_1, _tmp_2}
        """
        temp_name = self.create_temp_identifier()
        result_stmts, elt_list = self._collect_sequence_elements(elts)
        result_expr = self.create_set(elt_list)
        return self.create_stmt_list(
            result_stmts, self.create_assign_symbol(temp_name, result_expr)), self.create_name(temp_name)

    def visit_comp_generator(self, elt: ast.expr, generators: list[ast.comprehension]) -> PilStmtList:
        gen = generators[0]
        iter_stmts, iter_name = self.visit(gen.iter)

        target_var = self.create_temp_identifier()
        target_stmts = self.visit_lhs(gen.target, self.create_name(target_var))

        if len(generators) == 1:
            elt_stmts, elt_name = self.visit(elt)
            yield_temp = self.create_temp_identifier()
            yield_stmt = self.create_assign_symbol(yield_temp, self.create_yield(elt_name))
            inner_body = self.create_stmt_list(elt_stmts, yield_stmt)
        else:
            inner_body = self.visit_comp_generator(elt, generators[1:])

        # Apply if guards: wrap inner_body from innermost outward
        for cond in reversed(gen.ifs):
            cond_stmts, cond_name = self.visit(cond)
            inner_body = self.create_stmt_list(
                cond_stmts, self.create_if(cond_name, inner_body, self.create_stmt_list()))

        for_stmt = self.create_for(self.create_name(target_var), iter_name,
                                   self.create_stmt_list(target_stmts, inner_body), self.create_stmt_list(), None)
        return self.create_stmt_list(iter_stmts, for_stmt)

    def visit_comp(self, elt: ast.expr, generators: list[ast.comprehension]) -> tuple[PilFunctionDef, str]:
        comp_body = self.visit_comp_generator(elt, generators)
        func_name = self.create_temp_identifier()
        func_args = ast.arguments(
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
        )
        return self.create_function_def(func_name, func_args, comp_body, [], None, None), func_name

    def visit_list_comp(
        self, elt: ast.expr, generators: list[ast.comprehension], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            [elt for x in iter if cond]
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = elt
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
            _tmp_6 = [*_tmp_5]
        """
        func_def, func_name = self.visit_comp(elt, generators)
        gen_name = self.create_temp_identifier()
        gen_stmt = self.create_assign_symbol(gen_name, self.create_call(self.create_name(func_name), [], []))
        result_name = self.create_temp_identifier()
        result_stmt = self.create_assign_symbol(result_name, self.create_list([PilStarred(gen_name)]))
        return self.create_stmt_list([func_def, gen_stmt, result_stmt]), self.create_name(result_name)

    def visit_set_comp(
        self, elt: ast.expr, generators: list[ast.comprehension], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            {elt for x in iter if cond}
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = elt
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
            _tmp_6 = {*_tmp_5}
        """
        func_def, func_name = self.visit_comp(elt, generators)
        gen_name = self.create_temp_identifier()
        gen_stmt = self.create_assign_symbol(gen_name, self.create_call(self.create_name(func_name), [], []))
        result_name = self.create_temp_identifier()
        result_stmt = self.create_assign_symbol(result_name, self.create_set([PilStarred(gen_name)]))
        return self.create_stmt_list([func_def, gen_stmt, result_stmt]), self.create_name(result_name)

    def visit_dict_comp(
        self, key: ast.expr, value: ast.expr, generators: list[ast.comprehension], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            {key: val for x in iter if cond}
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = (key, val)
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
            _tmp_6 = dict(_tmp_5)
        """
        kv_tuple = ast.Tuple(elts=[key, value], ctx=ast.Load())
        func_def, func_name = self.visit_comp(kv_tuple, generators)
        gen_name = self.create_temp_identifier()
        gen_stmt = self.create_assign_symbol(gen_name, self.create_call(self.create_name(func_name), [], []))
        result_name = self.create_temp_identifier()
        result_stmt = self.create_assign_symbol(
            result_name, self.create_call(self.create_name('dict'), [self.create_name(gen_name)], []))
        return self.create_stmt_list([func_def, gen_stmt, result_stmt]), self.create_name(result_name)

    def visit_generator_exp(
        self, elt: ast.expr, generators: list[ast.comprehension], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            (elt for x in iter if cond)
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = elt
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
        """
        func_def, func_name = self.visit_comp(elt, generators)
        result_name = self.create_temp_identifier()
        result_stmt = self.create_assign_symbol(result_name, self.create_call(self.create_name(func_name), [], []))
        return self.create_stmt_list([func_def, result_stmt]), self.create_name(result_name)

    def visit_await(
        self,
        value: ast.expr,
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("Await is not supported"))

    def visit_yield(
        self,
        value: Optional[ast.expr],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1:
            Python:
                yield expr
            PIL:
                _tmp_0 = expr
                _tmp_1 = yield _tmp_0
        Case 2:
            Python:
                yield
            PIL:
                _tmp_0 = yield
        """
        temp_name = self.create_temp_identifier()
        if value is not None:
            value_stmts, value_expr = self.visit(value)
            yield_stmt = self.create_assign_symbol(temp_name, self.create_yield(value_expr))
            return self.create_stmt_list(value_stmts, yield_stmt), self.create_name(temp_name)
        yield_stmt = self.create_assign_symbol(temp_name, self.create_yield(None))
        return self.create_stmt_list(yield_stmt), self.create_name(temp_name)

    def visit_yield_from(self, value: ast.expr, node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            yield from expr
        PIL:
            _tmp_0 = expr
            _tmp_1 = yield from _tmp_0
        """
        value_stmts, value_name = self.visit(value)
        temp_name = self.create_temp_identifier()
        yield_from_stmt = self.create_assign_symbol(temp_name, self.create_yield_from(value_name))
        return self.create_stmt_list(value_stmts, yield_from_stmt), self.create_name(temp_name)

    def visit_compare(
        self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (single comparison):
            Python:
                a < b
            PIL:
                _tmp_0 = a
                _tmp_1 = b
                _tmp_2 = _tmp_0 < _tmp_1
        Case 2 (chained comparison):
            Python:
                a < b < c
            PIL:
                _tmp_0 = a
                _tmp_1 = b
                _tmp_2 = _tmp_0 < _tmp_1
                if _tmp_2:
                    _tmp_3 = c
                    _tmp_4 = _tmp_1 < _tmp_3    # b evaluated only once
                    _tmp_2 = _tmp_4
        """
        left_stmts, left_name = self.visit(left)
        return self._compare_rest(left_stmts, left_name, ops, comparators)

    def _compare_rest(self, left_stmts: PilStmtList, left_name: PilExpr,
                      ops: list[ast.cmpop], comparators: list[ast.expr],
                      node_attr: PilAttr = NOATTR) -> tuple[PilStmtList, Optional[PilExpr]]:
        comp_stmts, comp_name = self.visit(comparators[0])
        temp_name = self.create_temp_identifier()
        first_stmt = self.create_assign_symbol(temp_name, self.create_compare(left_name, ops[0], comp_name))

        if len(ops) == 1:
            return self.create_stmt_list(left_stmts, comp_stmts, first_stmt), self.create_name(temp_name)

        # Recursive case: a op0 b op1 c ... => (a op0 b) and (b op1 c ...)
        # comp_name is reused as left of the next comparison (evaluated only once)
        rest_stmt_list, rest_name = self._compare_rest(
            self.create_stmt_list(), comp_name, ops[1:], comparators[1:], node_attr=node_attr)
        rest_stmt = self.create_if(
            self.create_name(temp_name),
            self.create_stmt_list(rest_stmt_list, self.create_assign_identifier(temp_name, rest_name)),
            self.create_stmt_list(),
            node_attr=node_attr,
        )
        return self.create_stmt_list(left_stmts, comp_stmts, [first_stmt, rest_stmt]), self.create_name(temp_name)

    def visit_call(
        self, func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            func_expr(arg0, arg1, key=kw_expr)
        PIL:
            _tmp_0 = func_expr
            _tmp_1 = arg0
            _tmp_2 = arg1
            _tmp_3 = kw_expr
            _tmp_4 = _tmp_0(_tmp_1, _tmp_2, key=_tmp_3)
        """
        result_stmts: list = []

        # func must be an identifier per PIL spec
        func_stmts, func_name = self.visit(func)
        result_stmts.extend(func_stmts.stmts)

        # visit positional args
        arg_names = []
        for arg in args:
            if isinstance(arg, ast.Starred):
                arg_stmts, arg_name = self.visit(arg.value)
                arg_names.append(PilStarred(arg_name.id))
            else:
                arg_stmts, arg_name = self.visit(arg)
                arg_names.append(arg_name)
            result_stmts.extend(arg_stmts.stmts)

        # visit keyword values, rewrite keyword nodes with resolved names
        pil_keywords = []
        for kw in keywords:
            kw_stmts, kw_name = self.visit(kw.value)
            result_stmts.extend(kw_stmts.stmts)
            pil_keywords.append(self.create_keyword(kw.arg, kw_name))

        temp_name = self.create_temp_identifier()
        result_expr = self.create_call(func_name, arg_names, pil_keywords)
        result_stmts.append(self.create_assign_symbol(temp_name, result_expr))
        return PilStmtList(result_stmts), self.create_name(temp_name)

    def visit_formatted_value(
        self, value: ast.expr, conversion: int, format_spec: Optional[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (no conversion, no format_spec):
            Python:
                {expr}
            PIL:
                _tmp_0 = expr
        Case 2 (with conversion):
            Python:
                {expr!r}
            PIL:
                _tmp_0 = expr
                _tmp_1 = repr(_tmp_0)
        Case 3 (with format_spec):
            Python:
                {expr:fmt}
            PIL:
                _tmp_0 = expr
                _tmp_1 = fmt_expr
                _tmp_2 = format(_tmp_0, _tmp_1)
        Case 4 (with conversion and format_spec):
            Python:
                {expr!r:fmt}
            PIL:
                _tmp_0 = expr
                _tmp_1 = repr(_tmp_0)
                _tmp_2 = fmt_expr
                _tmp_3 = format(_tmp_1, _tmp_2)
        """
        visited_stmts, value_name = self.visit(value)
        result_stmts: list = list(visited_stmts.stmts)

        # Apply the formatted-value conversion code when one is present.
        if conversion == ord('s'):
            conv_name = self.create_temp_identifier()
            result_stmts += [self.create_assign_symbol(
                conv_name, self.create_call(self.create_name('str'), [value_name], []))]
            value_name = self.create_name(conv_name)
        elif conversion == ord('r'):
            conv_name = self.create_temp_identifier()
            result_stmts += [self.create_assign_symbol(
                conv_name, self.create_call(self.create_name('repr'), [value_name], []))]
            value_name = self.create_name(conv_name)
        elif conversion == ord('a'):
            conv_name = self.create_temp_identifier()
            result_stmts += [self.create_assign_symbol(
                conv_name, self.create_call(self.create_name('ascii'), [value_name], []))]
            value_name = self.create_name(conv_name)

        if format_spec is not None:
            spec_stmts, spec_name = self.visit(format_spec)
            result_stmts += spec_stmts.stmts
            temp_name = self.create_temp_identifier()
            result_stmts += [
                self.create_assign_symbol(
                    temp_name, self.create_call(self.create_name('format'), [value_name, spec_name], []))
            ]
            return PilStmtList(result_stmts), self.create_name(temp_name)

        return PilStmtList(result_stmts), value_name

    def visit_interpolation(
        self,
        value: ast.expr,
        literal_text: str,
        conversion: int,
        format_spec: Optional[ast.expr],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("Interpolation is not supported"))

    def visit_joined_str(
        self, values: list[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Case 1 (single part):
            Python:
                f"{expr}"
            PIL:
                _tmp_0 = expr           # via visit_FormattedValue
        Case 2 (multiple parts):
            Python:
                f"prefix{expr}suffix"
            PIL:
                _tmp_0 = expr           # via visit_FormattedValue
                _tmp_1 = ['prefix', _tmp_0, 'suffix']
                _tmp_2 = ''.join
                _tmp_3 = _tmp_2(_tmp_1)
        """
        result_stmts: list = []
        part_names = []
        for part in values:
            if isinstance(part, ast.Constant):
                part_names.append(self.create_constant(str(part.value)))
            else:
                part_stmts, part_name = self.visit(part)
                result_stmts.extend(part_stmts.stmts)
                part_names.append(part_name)
        if len(part_names) == 1:
            return PilStmtList(result_stmts), part_names[0]
        parts_list_name = self.create_temp_identifier()
        result_stmts.append(self.create_assign_symbol(parts_list_name, self.create_list(list(part_names))))
        join_func = self.create_temp_identifier()
        result_stmts.append(
            self.create_assign_symbol(join_func, self.create_attribute(self.create_constant(''), 'join'))
        )
        temp_name = self.create_temp_identifier()
        result_stmts.append(
            self.create_assign_symbol(
                temp_name, self.create_call(self.create_name(join_func), [self.create_name(parts_list_name)], []))
        )
        return PilStmtList(result_stmts), self.create_name(temp_name)

    def visit_template_str(
        self, values: list[ast.expr], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("TemplateStr is not supported"))

    def visit_constant(
        self, value: object, kind: Optional[str], node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        return self.create_stmt_list(), self.create_constant(value, kind)

    def visit_attribute(
        self, value: ast.expr, attr: str, ctx: ast.expr_context, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            value_expr.attr
        PIL:
            _tmp_0 = value_expr
            _tmp_1 = _tmp_0.attr
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))
        value_stmts, value_name = self.visit(value)
        temp_name = self.create_temp_identifier()
        result_expr = self.create_attribute(value_name, attr)
        return self.create_stmt_list(
            value_stmts, self.create_assign_symbol(temp_name, result_expr)), self.create_name(temp_name)

    def visit_subscript(
        self, value: ast.expr, slice_expr: ast.expr, ctx: ast.expr_context, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            value_expr[slice]
        PIL:
            _tmp_0 = value_expr
            _tmp_1 = slice_expr         # lower/upper/step each resolved
            _tmp_2 = _tmp_0[_tmp_1]
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))
        value_stmts, value_name = self.visit(value)

        slice_stmts, pil_slice_list = self.visit_slice_values(slice_expr)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_subscript(value_name, pil_slice_list)
        return self.create_stmt_list(
            value_stmts, slice_stmts, self.create_assign_symbol(temp_name, result_expr)), self.create_name(temp_name)

    def visit_starred(
        self, value: ast.expr, ctx: ast.expr_context, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(RuntimeError("Starred should not be directly accessed"))

    def visit_name(
        self, identifier: str, ctx: ast.expr_context, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))
        return self.create_stmt_list(), self.create_name(identifier)

    def visit_list(
        self, elts: list[ast.expr], ctx: ast.expr_context, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            [elt0, *elt1, elt2]
        PIL:
            _tmp_0 = elt0
            _tmp_1 = elt1
            _tmp_2 = elt2
            _tmp_3 = [_tmp_0, *_tmp_1, _tmp_2]
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))

        return self._visit_sequence_literal(elts, self.create_list)

    def visit_tuple(
        self, elts: list[ast.expr], ctx: ast.expr_context, node_attr: PilAttr = NOATTR
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        """
        Python:
            (elt0, *elt1, elt2)
        PIL:
            _tmp_0 = elt0
            _tmp_1 = elt1
            _tmp_2 = elt2
            _tmp_3 = (_tmp_0, *_tmp_1, _tmp_2)
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))

        return self._visit_sequence_literal(elts, self.create_tuple)

    def visit_slice(
        self,
        lower: Optional[ast.expr],
        upper: Optional[ast.expr],
        step: Optional[ast.expr],
        node_attr: PilAttr = NOATTR,
    ) -> tuple[PilStmtList, Optional[PilExpr]]:
        raise FeError(NotImplementedError("Slice is not supported"))

    def visit_stmts(self, stmts: list[ast.stmt]) -> tuple[PilStmtList, Optional[PilExpr]]:
        stmt_list = []
        for stmt in stmts:
            result_stmt_list, _ = self.visit(stmt)
            stmt_list.extend(result_stmt_list.stmts)
        return PilStmtList(stmt_list), None

    def visit(self, node):
        method = 'visit_' + self._node_name_to_visitor_suffix(node.__class__.__name__)
        visitor = getattr(self, method)
        field_dict = {key: value for key, value in ast.iter_fields(node)}
        # Rename fields to match PilAttr
        field_aliases = {
            'id': 'identifier',
            'iter': 'iter_expr',
            'slice': 'slice_expr',
            'str': 'literal_text',
        }
        field_dict = {field_aliases.get(key, key): value for key, value in field_dict.items()}
        node_attr = PilAttr(node)
        return visitor(**field_dict, node_attr=node_attr)

    def parse_func(self, func: ast.FunctionDef) -> ast.FunctionDef:
        body_ast_list = pil_to_ast(PilStmtList(self.visit_stmts(func.body)[0]))
        result_func = ast.FunctionDef(
            func.name,
            func.args,
            body_ast_list,
            func.decorator_list,
            func.returns,
            func.type_comment,
            **self.create_attribute(func),
        )
        return result_func

    def parse_pil(self, stmts: list[ast.stmt]) -> PilStmtList:
        """Parse statements into a PIL data-model statement list."""
        return self.visit_stmts(stmts)[0]

    def parse_stmts(self, stmts: list[ast.stmt]) -> PilStmtList:
        """Parse statements into PIL, then lower back to ast nodes for execution."""
        return pil_to_ast(self.parse_pil(stmts))

    def _collect_sequence_elements(self, elts: list[ast.expr]) -> tuple[PilStmtList, list]:
        result_stmts: list = []
        elt_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                elt_stmts, elt_name = self.visit(elt.value)
                elt_list.append(PilStarred(elt_name.id))
            else:
                elt_stmts, elt_name = self.visit(elt)
                elt_list.append(elt_name)
            result_stmts.extend(elt_stmts.stmts)
        return PilStmtList(result_stmts), elt_list

    def _visit_sequence_literal(
        self, elts: list[ast.expr], create_expr: Callable[[list], PilExpr]
    ) -> tuple[PilStmtList, PilExpr]:
        temp_name = self.create_temp_identifier()
        result_stmts, elt_list = self._collect_sequence_elements(elts)
        result_expr = create_expr(elt_list)
        return self.create_stmt_list(
            result_stmts, self.create_assign_symbol(temp_name, result_expr)), self.create_name(temp_name)


def parse_func(func: ast.FunctionDef, prefix=PIL_DEFAULT_PREFIX) -> ast.FunctionDef:
    ctx = PILContext(prefix=prefix)
    parser = PythonParser(ctx)
    return parser.parse_func(func)


def parse_stmts(stmts: list[ast.stmt], prefix=PIL_DEFAULT_PREFIX) -> PilStmtList:
    ctx = PILContext(prefix=prefix)
    parser = PythonParser(ctx)
    return parser.parse_stmts(stmts)
