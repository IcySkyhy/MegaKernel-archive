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
"""Tests for text-entry parsing helpers such as ``pl.parse`` and ``pl.loads``."""

from __future__ import annotations

import os
import tempfile
import textwrap

from pypto_pro import ir
import pypto_pro.language as pl
import pytest


def _code(source: str) -> str:
    """Normalize embedded DSL source snippets."""
    return textwrap.dedent(source).strip() + "\n"


def test_parse_function_with_import():
    """Parse a text function that explicitly imports the DSL namespace."""
    func = pl.parse(
        _code(
            """
            import pypto_pro.language as pl

            @pl.function
            def add_one(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                return result
            """
        )
    )

    assert isinstance(func, ir.Function)
    assert func.name == "add_one"
    assert len(func.params) == 1
    assert len(func.return_types) == 1


def test_parse_function_without_import():
    """``pl.parse`` injects the DSL namespace for simple standalone snippets."""
    func = pl.parse(
        _code(
            """
            @pl.function
            def multiply(x: pl.Tensor[[32], pl.DT_FP32]) -> pl.Tensor[[32], pl.DT_FP32]:
                result: pl.Tensor[[32], pl.DT_FP32] = pl.tensor.mul(x, 2.0)
                return result
            """
        )
    )

    assert isinstance(func, ir.Function)
    assert func.name == "multiply"


def test_parse_program_with_import():
    """Parse a text program containing one method."""
    program = pl.parse_program(
        _code(
            """
            import pypto_pro.language as pl

            @pl.program
            class SimpleProgram:
                @pl.function
                def add_one(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                    result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                    return result
            """
        )
    )

    assert isinstance(program, ir.Program)
    assert program.name == "SimpleProgram"
    assert len(program.functions) == 1


def test_load_function_from_file():
    """``pl.loads`` reads and parses a function from disk."""
    code = _code(
        """
        @pl.function
        def identity(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            return x
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name

    try:
        func = pl.loads(temp_path)
        assert isinstance(func, ir.Function)
        assert func.name == "identity"
    finally:
        os.unlink(temp_path)


def test_parse_reports_syntax_error():
    """Syntax errors from text snippets are reported by ``pl.parse``."""
    code = _code(
        """
        @pl.function
        def bad_syntax(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            return x +
        """
    )

    with pytest.raises(SyntaxError, match="Failed to compile code"):
        pl.parse(code)


def test_parse_rejects_missing_function_or_program():
    """Text entry must contain exactly one supported decorated object."""
    with pytest.raises(ValueError, match="No @pl.function or @pl.program found"):
        pl.parse("x = 1\n")


def test_load_preserves_filename_in_errors():
    """Errors from ``pl.loads`` retain the file-backed parse path."""
    code = _code(
        """
        @pl.function
        def bad_func(x):
            return x
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name

    try:
        with pytest.raises(pl.parser.ParserError):
            pl.loads(temp_path)
    finally:
        os.unlink(temp_path)
