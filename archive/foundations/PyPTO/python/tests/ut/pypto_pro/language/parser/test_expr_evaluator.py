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
"""Unit tests for ExprEvaluator."""

import ast

from pypto_pro import DataType
from pypto_pro.language.parser._expr_evaluator import ExprEvaluator
from pypto_pro.language.parser.diagnostics import ParserTypeError
import pytest

from pypto.enum import DYNAMIC


def _parse_expr(code: str) -> ast.expr:
    """Parse a Python expression string to an AST node."""
    return ast.parse(code, mode="eval").body


@pytest.mark.parametrize(
    ("closure_vars", "expr", "expected"),
    [
        ({"x": 42}, "x", 42),
        ({"shape": [128, 64]}, "shape", [128, 64]),
        ({"dims": (32, 64)}, "dims", (32, 64)),
        ({"dtype": DataType.FP32}, "dtype", DataType.FP32),
        ({"flag": True}, "flag", True),
        ({"val": None}, "val", None),
        ({"base": 64}, "base * 2", 128),
        ({"a": 10, "b": 20}, "a + b", 30),
        ({"total": 256}, "total - 128", 128),
        ({"n": 256}, "n // 4", 64),
        ({"x": 5}, "-x", -5),
        ({"shape": [128, 64, 32]}, "len(shape)", 3),
        ({"a": 10, "b": 20}, "max(a, b)", 20),
        ({"a": 10, "b": 20}, "min(a, b)", 10),
        ({"x": -5}, "abs(x)", 5),
        ({"x": 3.7}, "int(x)", 3),
        ({"t": (1, 2, 3)}, "list(t)", [1, 2, 3]),
        ({"dims": [128, 64, 32]}, "dims[1]", 64),
        ({"dims": [128, 64, 32, 16]}, "dims[0:2]", [128, 64]),
        ({"a": 128, "b": 64}, "[a, b]", [128, 64]),
    ],
)
def test_eval_expr_supported_cases(closure_vars, expr, expected):
    ev = ExprEvaluator(closure_vars=closure_vars)
    result = ev.eval_expr(_parse_expr(expr))
    if expected is DYNAMIC:
        assert result is DYNAMIC
    else:
        assert result == expected


def test_eval_expr_resolves_dynamic_policy_by_identity():
    ev = ExprEvaluator(closure_vars={"policy": DYNAMIC})
    assert ev.eval_expr(_parse_expr("policy")) is DYNAMIC


@pytest.mark.parametrize(
    ("closure_vars", "expr", "match"),
    [
        ({}, "undefined_var", "Cannot resolve expression"),
        ({"x": "hello"}, "x + 1", "Failed to evaluate expression"),
        ({}, "open('file.txt')", "Cannot resolve expression"),
        ({}, "__import__('os')", None),
    ],
)
def test_eval_expr_rejects_unsupported_cases(closure_vars, expr, match):
    ev = ExprEvaluator(closure_vars=closure_vars)
    if match is None:
        with pytest.raises(ParserTypeError):
            ev.eval_expr(_parse_expr(expr))
    else:
        with pytest.raises(ParserTypeError, match=match):
            ev.eval_expr(_parse_expr(expr))


@pytest.mark.parametrize(
    ("closure_vars", "expr", "success", "expected"),
    [
        ({"x": 42}, "x", True, 42),
        ({}, "undefined", False, None),
        ({"base": 64}, "base * 2", True, 128),
    ],
)
def test_try_eval_expr(closure_vars, expr, success, expected):
    ev = ExprEvaluator(closure_vars=closure_vars)
    actual_success, value = ev.try_eval_expr(_parse_expr(expr))
    assert actual_success is success
    assert value == expected
