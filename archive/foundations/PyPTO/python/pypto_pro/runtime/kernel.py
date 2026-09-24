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

"""
Kernel decorator for PyPTO Runtime.

The @kernel decorator provides a simplified API that wraps a single function
into a KernelDef, deferring AST parsing to compile time.

Usage:
    import pypto_pro.language as pl

    @pl.jit()
    def vector_add(
        x: pl.Tensor[[1024], pl.DT_FP32],
        y: pl.Tensor[[1024], pl.DT_FP32],
    ) -> pl.Tensor[[1024], pl.DT_FP32]:
        tile_x = pl.load(x, [0], [1024])
        tile_y = pl.load(y, [0], [1024])
        result = pl.add(tile_x, tile_y)
        out = pl.create([1024], dtype=pl.DT_FP32)
        return pl.store(result, [0], [1024], out)

    # Direct call (default stream, block_dim=1):
    vector_add(x, y)

    # Bracket-launch syntax:
    vector_add[stream, block_dim](x, y)
"""

from __future__ import annotations

__all__ = ["kernel"]


import ast
import inspect
import sys
import textwrap
from typing import Any, Callable, Optional

from pypto.pypto_impl import ir
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.language.parser.decorator import (
    _attach_source_lines_to_error,
    _calculate_col_offset,
    _find_ast_node,
    _parse_ast_tree,
)
from pypto_pro.language.parser.diagnostics import ParserError, ParserSyntaxError


def extract_func_source_info(f: Callable):
    """Extract source file, lines, offsets, and parsed func_def from a function.

    Returns:
        tuple of (source_file, source_lines, source_lines_raw, line_offset,
                  col_offset, func_def)
    """
    source_file = inspect.getfile(f)
    source_lines_raw, starting_line = inspect.getsourcelines(f)
    source_code = "".join(source_lines_raw)
    col_offset = _calculate_col_offset(source_lines_raw)
    source_code = textwrap.dedent(source_code)
    source_lines = source_code.split("\n")
    line_offset = starting_line - 1

    try:
        tree = _parse_ast_tree(source_code, "function")
        func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")
    except ParserError as e:
        _attach_source_lines_to_error(e, source_file, source_lines_raw)
        raise

    return (source_file, source_lines, source_lines_raw, line_offset, col_offset, func_def)


class KernelDef:
    """Lazy kernel definition — captures source/AST/closure at decoration time,
    defers AST parsing to compile time.

    The JIT codegen path parses the kernel once for each required target.

    Args:
        func: Original Python function.
        source_file: Path to the source file.
        source_lines: Dedented source lines for the parser.
        source_lines_raw: Raw (non-dedented) source lines for error reporting.
        line_offset: Line number offset in the original file.
        col_offset: Column indentation offset.
        func_def: AST FunctionDef node.
        closure_vars: Captured caller scope for name resolution.
        name: Optional program name.
        func_type: IR function type (Opaque, InCore, Helper).
        strict_ssa: Whether to enforce SSA.
        meta_data: Optional metadata.
        auto_mutex: Whether to enable automatic mutex lock/unlock insertion.
    """

    def __init__(
        self,
        func: Callable,
        source_file: str,
        source_lines: list[str],
        source_lines_raw: list[str],
        line_offset: int,
        col_offset: int,
        func_def: ast.FunctionDef,
        closure_vars: dict[str, Any],
        name: str | None,
        func_type: ir.FunctionType,
        strict_ssa: bool,
        meta_data: Any,
        auto_mutex: bool = True,
        pipeline=None,
        tilingkey_consts: dict[str, int] | None = None,
        datatype_consts: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self._source_file = source_file
        self._source_lines = source_lines
        self._source_lines_raw = source_lines_raw
        self._line_offset = line_offset
        self._col_offset = col_offset
        self._func_def = func_def
        self._closure_vars = closure_vars
        self._name = name
        self._func_type = func_type
        self._strict_ssa = strict_ssa
        self._auto_mutex = auto_mutex
        self._pipeline = pipeline
        self._meta_data = meta_data
        self._tilingkey_consts = tilingkey_consts
        self._datatype_consts = datatype_consts

    @property
    def func_def(self) -> ast.FunctionDef:
        return self._func_def

    @property
    def closure_vars(self) -> dict[str, Any]:
        return self._closure_vars

    @property
    def func_name(self) -> str:
        return self._func.__name__

    def parse_target_program(
        self,
        target: ir.SectionKind,
        bound_signature=None,
    ) -> tuple[ir.Program, bool]:
        """Parse a fresh target Program and report whether its target section was matched."""
        program_name = self._name if self._name is not None else self._func.__name__

        try:
            # The Program owns one IRDebugInfo; create it here and share it with the parser
            # so field names land in the table the Program carries.
            debug_info = ir.IRDebugInfo()
            parser = ASTParser(
                self._source_file,
                self._source_lines,
                target,
                self._line_offset,
                self._col_offset,
                strict_ssa=self._strict_ssa,
                closure_vars=self._closure_vars,
                auto_mutex=self._auto_mutex,
                debug_info=debug_info,
                tilingkey_consts=self._tilingkey_consts,
                datatype_consts=self._datatype_consts,
                bound_signature=bound_signature,
                # Kernels use a void ABI: they may early-return, but cannot return values.
                void_return_only=True,
                void_return_context="@pl.jit/@pl.kernel",
                allow_early_return=True,
            )

            try:
                ir_func = parser.parse_function(self._func_def, func_type=self._func_type)
            except ParserError:
                raise
            except Exception as e:
                span = None
                node = getattr(parser, '_current_node', None)
                if node is not None:
                    span = parser.span_tracker.get_span(node)
                if isinstance(e, (AttributeError, TypeError)):
                    hint = (
                        "an internal type check failed while parsing; an argument may "
                        "have an unsupported type — check that kernel arguments match "
                        "the expected Tile/Tensor/scalar types"
                    )
                else:
                    hint = "Check your function definition for errors"
                raise ParserSyntaxError(
                    f"Failed to parse kernel function '{self._func.__name__}': {type(e).__name__}: {e}",
                    span=span,
                    hint=hint,
                ) from e

            external_funcs = list(parser.external_funcs.values())
            starting_line = self._line_offset + 1
            program_span = ir.Span(self._source_file, starting_line, self._col_offset)
            program = ir.Program(
                external_funcs + [ir_func], program_name, program_span, parser.debug_info
            )
            return program, parser.matched_target

        except ParserError as e:
            _attach_source_lines_to_error(e, self._source_file, self._source_lines_raw)
            raise


def _call_meta_and_capture_env(meta_fn):
    """Run meta_fn() and capture its local namespace (for types etc.). Returns (return_value, env dict)."""
    env = {}

    if meta_fn is None:
        return None, env

    def trace(frame, event, arg):
        if event == "return":
            env.clear()
            env.update(frame.f_locals)
        return trace

    old_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        result = meta_fn()
    finally:
        sys.settrace(old_trace)
    return result, env


def kernel(
    func: Optional[Callable] = None,
    meta_data=None,
    *,
    name: Optional[str] = None,
    func_type: ir.FunctionType = ir.FunctionType.Opaque,
    strict_ssa: bool = False,
    auto_mutex: bool = True,
    pipeline=None,
) -> "KernelDef":
    """Decorator that captures a DSL function for deferred compilation.

    The decorated function becomes a :class:`KernelDef` — a lazy definition
    that records the source code, AST, and closure at decoration time but
    defers parsing until JIT codegen runs.  This allows the target
    architecture to be specified once at compile time.

    Args:
        func: Python function to capture.
        name: Optional program name (defaults to function name).
        func_type: Function type (Opaque, InCore, or Helper).
        strict_ssa: If True, enforce SSA (single assignment per variable).

    Returns:
        KernelDef consumed by the JIT codegen path.

    Example:
        >>> @pl.kernel
        ... def my_kernel(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        ...     tile = pl.load(x, [0], [64])
        ...     result = pl.add(tile, tile)
        ...     return pl.store(result, [0], [64], x)
        >>> isinstance(my_kernel, KernelDef)
        True
    """
    # Capture caller's scope so the parser can resolve names like `pl`, etc.
    # Must be captured here (not inside _decorator) to get the correct call-site frame.
    caller_frame = inspect.currentframe().f_back
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(f: Callable) -> KernelDef:
        (source_file, source_lines, source_lines_raw, line_offset, col_offset, func_def) = extract_func_source_info(f)

        return KernelDef(
            func=f,
            source_file=source_file,
            source_lines=source_lines,
            source_lines_raw=source_lines_raw,
            line_offset=line_offset,
            col_offset=col_offset,
            func_def=func_def,
            closure_vars=closure_vars,
            name=name,
            func_type=func_type,
            strict_ssa=strict_ssa,
            meta_data=meta_data,
            auto_mutex=auto_mutex,
            pipeline=pipeline,
        )

    if func is None:
        return _decorator
    else:
        return _decorator(func)
