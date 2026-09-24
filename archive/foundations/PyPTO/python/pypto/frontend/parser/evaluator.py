#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO Script Expression Evaluator.

This module provides the expression evaluator for the PTO frontend parser.
The evaluator is responsible for evaluating Python expressions during the
parsing phase, converting them to concrete values or PTO IR constructs.

Key Features:
    - Expression Evaluation: Evaluates Python expressions in the context of
      available variables and the PTO module namespace
    - Symbolic Support: Handles SymbolicScalar objects and converts them to
      concrete values when possible
    - Error Handling: Converts Python evaluation errors to ParserError with
      proper source location information

The evaluator uses Python's built-in compile() and eval() functions to
execute expressions, providing flexibility while maintaining safety through
controlled variable scoping.

Example:
    # During parsing, evaluate an expression like: pypto.Tensor([16], "float32")
    result = ExprEvaluator.eval(expr_node, var_table, diagnostics)
"""

import ast
import builtins
from typing import Any

import pypto
from pypto.error import FeError, ParserError

from .diagnostics import Diagnostics


def _min(*args: Any, **kwargs: Any) -> Any:
    if len(args) == 2 and not kwargs and any(isinstance(arg, pypto.SymbolicScalar) for arg in args):
        return pypto.min(args[0], args[1])
    return builtins.min(*args, **kwargs)


def _max(*args: Any, **kwargs: Any) -> Any:
    if len(args) == 2 and not kwargs and any(isinstance(arg, pypto.SymbolicScalar) for arg in args):
        return pypto.max(args[0], args[1])
    return builtins.max(*args, **kwargs)


_EVAL_BUILTIN_OVERRIDES = {"min": _min, "max": _max}


class ExprEvaluator:
    """Expression evaluator for PTO frontend."""

    var_table: dict[str, Any]
    diag: Diagnostics

    def __init__(self, var_table: dict[str, Any], diag: Diagnostics) -> None:
        self.var_table = var_table
        self.diag = diag

    @staticmethod
    def eval(node: ast.expr, var_table: dict[str, Any], diag: Diagnostics) -> Any:
        """Evaluate an expression node using the provided variable table.

        This is the main entry point for expression evaluation during parsing.
        It creates an ExprEvaluator instance and evaluates the expression using
        Python's compile() and eval() functions.

        Parameters
        ----------
        node : ast.expr
            The expression AST node to evaluate.
        var_table : dict[str, Any]
            Variable table containing available names and their values.
        diag : Diagnostics
            Diagnostics instance for error reporting.

        Returns
        -------
        Any
            The evaluated result of the expression.

        Raises
        ------
        ParserError
            If evaluation fails due to undefined names, type errors, etc.
        """
        self = ExprEvaluator(var_table, diag)
        result = self.visit(node)
        return result

    @staticmethod
    def _is_safe_expression(node: ast.expr) -> bool:
        """Verify that the expression does not contain dangerous operations"""
        dangerous_attrs = {
            '__class__',
            '__bases__',
            '__subclasses__',
            '__dict__',
            '__globals__',
            '__code__',
            '__builtins__',
            '__import__',
            '__loader__',
        }
        dangerous_funcs = {
            'eval',
            'exec',
            'compile',
            '__import__',
            'open',
            'input',
            'breakpoint',
        }

        for node_item in ast.walk(node):
            if isinstance(node_item, (ast.Import, ast.ImportFrom)):
                return False

            if isinstance(node_item, ast.Attribute) and node_item.attr in dangerous_attrs:
                return False

            if isinstance(node_item, ast.Call) and isinstance(node_item.func, ast.Name):
                if node_item.func.id in dangerous_funcs:
                    return False

        return True

    def visit(self, node: ast.expr) -> Any:
        """Visit and evaluate an expression node.

        Parameters
        ----------
        node : ast.expr
            The expression node to visit.

        Returns
        -------
        Any
            The evaluated result.
        """
        return self._eval_by_python(node, self.var_table)

    def _eval_by_python(self, node: ast.expr, var_table: dict[str, Any]) -> Any:
        if not self._is_safe_expression(node):
            raise ParserError(node, "Expression contains unsafe operations")
        if isinstance(node, ast.expr):
            # Case 1: a simple expression
            mod = ast.fix_missing_locations(ast.Expression(body=node))
            exe = compile(mod, filename=self.diag.source.source_name, mode="eval")
            dict_locals = var_table.copy()
            # Replace SymbolicScalars with concrete values if available
            for key, value in dict_locals.items():
                if isinstance(value, pypto.SymbolicScalar) and value.is_concrete():
                    dict_locals[key] = value.concrete()
            dict_locals.update(_EVAL_BUILTIN_OVERRIDES)
            try:
                return eval(exe, {}, dict_locals)  # pylint: disable=eval-used
            except Exception as e:
                raise ParserError(node, e) from e
        elif isinstance(node, ast.Expr):
            # Case 2: a expression in a statement
            mod = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exe = compile(mod, filename=self.diag.source.source_name, mode="exec")
            dict_locals = var_table.copy()
            # Replace SymbolicScalars with concrete values if available
            for key, value in dict_locals.items():
                if isinstance(value, pypto.SymbolicScalar) and value.is_concrete():
                    dict_locals[key] = value.concrete()
            dict_locals.update(_EVAL_BUILTIN_OVERRIDES)
            try:
                return exec(exe, {}, dict_locals)  # pylint: disable=exec-used
            except Exception as e:
                raise ParserError(node, e) from e
        else:
            # Other unsupported expression types, raise python native error,
            # which will be caught by the parser and reported as a bug.
            raise FeError(NotImplementedError("Unsupported expression type."))
