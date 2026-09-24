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
IR Builder for incremental IR construction with context management.

Provides a Pythonic API for building IR using context managers with
automatic span tracking via the inspect module.
"""

from __future__ import annotations

__all__ = [
    "IRBuilder",
    "FunctionBuilder",
    "ForLoopBuilder",
    "WhileLoopBuilder",
    "IfStmtBuilder",
    "ProgramBuilder",
]


import builtins
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import inspect

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto.pypto_impl.ir import IRBuilder as CppIRBuilder

from ._utils import _normalize_expr


class IRBuilder:
    """IR Builder with context management and automatic span tracking.

    The IRBuilder provides a convenient API for building IR incrementally
    using context managers. Spans are automatically captured from the call
    site using Python's inspect module, or can be explicitly provided.

    Example:
        >>> ib = IRBuilder()
        >>> with ib.function("my_func") as f:
        ...     x = f.param("x", ir.ScalarType(ir.DataType.INT64))
        ...     y = f.param("y", ir.ScalarType(ir.DataType.INT64))
        ...     f.return_type(ir.ScalarType(ir.DataType.INT64))
        ...     result = ib.var("result", ir.ScalarType(ir.DataType.INT64))
        ...     ib.assign(result, ir.Add(x, y, ir.DataType.INT64, ir.Span.unknown()))
        >>> func = f.get_result()
    """

    def __init__(self) -> None:
        """Initialize the IR builder."""
        # Import here to avoid circular dependency

        self.builder = CppIRBuilder()
        self._begin_spans: dict[int, ir.Span] = {}  # Track begin spans for multi-line contexts
        self._ctx_counter = 0  # Counter for unique context IDs

    @staticmethod
    def capture_call_span() -> ir.Span:
        """Capture span from immediate caller using inspect.

        Returns:
            Span: Source location of the caller
        """
        # Go back 2 frames:
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            frame = frame.f_back.f_back
        if frame is not None:
            info = inspect.getframeinfo(frame)
            return ir.Span(info.filename, info.lineno, -1)
        return ir.Span.unknown()

    @classmethod
    def _combine_spans(cls, begin: ir.Span, end: ir.Span) -> ir.Span:
        """Combine begin and end spans into a multi-line span.

        Args:
            begin: Begin span (from context enter)
            end: End span (from context exit)

        Returns:
            Span: Combined span covering the range
        """
        return ir.Span(
            begin.filename,
            begin.begin_line,
            begin.begin_column,
            end.begin_line,
            end.begin_column,
        )

    @contextmanager
    def function(
        self,
        name: str,
        span: ir.Span | None = None,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
    ) -> Iterator["FunctionBuilder"]:
        """Context manager for building functions.

        Args:
            name: Function name
            span: Optional explicit span. If None, automatically captured from call site.
            func_type: Function type (default: Opaque)

        Yields:
            FunctionBuilder: Helper object for building the function

        Example:
            >>> with ib.function("add") as f:
            ...     x = f.param("x", ir.ScalarType(ir.DataType.INT64))
            ...     f.return_type(ir.ScalarType(ir.DataType.INT64))
        """
        begin_span = span if span is not None else self.capture_call_span()
        ctx_id = self._ctx_counter
        self._ctx_counter += 1
        self._begin_spans[ctx_id] = begin_span

        self.builder.begin_function(name, begin_span, func_type)
        builder_obj = FunctionBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self.capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self.builder.end_function(combined_span)
            builder_obj.result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def for_loop(
        self,
        loop_var: ir.Var,
        start: int | ir.Expr,
        stop: int | ir.Expr,
        step: int | ir.Expr,
        span: ir.Span | None = None,
    ) -> Iterator["ForLoopBuilder"]:
        """Context manager for building for loops.

        Args:
            loop_var: Loop variable
            start: Start value (int or Expr)
            stop: Stop value (int or Expr)
            step: Step value (int or Expr)
            span: Optional explicit span. If None, automatically captured.

        Yields:
            ForLoopBuilder: Helper object for building the loop

        Example:
            >>> i = ib.var("i", ir.ScalarType(ir.DataType.INT64))
            >>> with ib.for_loop(i, 0, 10, 1) as loop:
            ...     sum_iter = loop.iter_arg("sum", init_val)
        """
        begin_span = span if span is not None else self.capture_call_span()
        ctx_id = self._ctx_counter
        self._ctx_counter += 1
        self._begin_spans[ctx_id] = begin_span

        # Normalize all expression parameters
        start_expr = _normalize_expr(start, begin_span)
        stop_expr = _normalize_expr(stop, begin_span)
        step_expr = _normalize_expr(step, begin_span)

        self.builder.begin_for_loop(
            loop_var,
            start_expr,
            stop_expr,
            step_expr,
            begin_span,
        )
        builder_obj = ForLoopBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self.capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self.builder.end_for_loop(combined_span)
            builder_obj.result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def while_loop(self, condition: int | ir.Expr, span: ir.Span | None = None) -> Iterator["WhileLoopBuilder"]:
        """Context manager for building while loops.

        Args:
            condition: Condition expression (int or Expr)
            span: Optional explicit span. If None, automatically captured.

        Yields:
            WhileLoopBuilder: Helper object for building the loop

        Example:
            >>> x = ib.var("x", ir.ScalarType(ir.DataType.INT64))
            >>> with ib.while_loop(ir.Lt(x, ir.ConstInt(10, ir.DataType.INT64, span), span)) as loop:
            ...     x_iter = loop.iter_arg("x_iter", x)
            ...     # ... loop body ...
        """
        begin_span = span if span is not None else self.capture_call_span()
        ctx_id = self._ctx_counter
        self._ctx_counter += 1
        self._begin_spans[ctx_id] = begin_span

        condition_expr = _normalize_expr(condition, begin_span)
        self.builder.begin_while_loop(condition_expr, begin_span)
        builder_obj = WhileLoopBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self.capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self.builder.end_while_loop(combined_span)
            builder_obj.result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def if_stmt(self, condition: int | ir.Expr, span: ir.Span | None = None) -> Iterator["IfStmtBuilder"]:
        """Context manager for building if statements.

        Args:
            condition: Condition expression (int or Expr)
            span: Optional explicit span. If None, automatically captured.

        Yields:
            IfStmtBuilder: Helper object for building the if statement

        Example:
            >>> with ib.if_stmt(condition) as if_builder:
            ...     # then branch
            ...     ib.assign(x, value)
            ...     if_builder.else_()
            ...     # else branch
            ...     ib.assign(x, other_value)
        """
        begin_span = span if span is not None else self.capture_call_span()
        ctx_id = self._ctx_counter
        self._ctx_counter += 1
        self._begin_spans[ctx_id] = begin_span

        condition_expr = _normalize_expr(condition, begin_span)
        self.builder.begin_if(condition_expr, begin_span)
        builder_obj = IfStmtBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self.capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self.builder.end_if(combined_span)
            builder_obj.result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def section(self, section_kind: ir.SectionKind, span: ir.Span | None = None) -> Iterator["SectionBuilder"]:
        """Context manager for building section statements.

        Args:
            section_kind: The kind of section (Vector or Cube)
            span: Optional explicit span. If None, automatically captured.

        Yields:
            SectionBuilder: Helper object for building the section statement

        Example:
            >>> with ib.section(ir.SectionKind.Vector) as section_builder:
            ...     # Vector section body
            ...     ib.assign(y, add_expr)
        """
        begin_span = span if span is not None else self.capture_call_span()
        ctx_id = self._ctx_counter
        self._ctx_counter += 1
        self._begin_spans[ctx_id] = begin_span

        self.builder.begin_section(section_kind, begin_span)
        builder_obj = SectionBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self.capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self.builder.end_section(combined_span)
            builder_obj.result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def program(self, name: str, span: ir.Span | None = None) -> Iterator["ProgramBuilder"]:
        """Context manager for building programs.

        Args:
            name: Program name
            span: Optional explicit span. If None, automatically captured.

        Yields:
            ProgramBuilder: Helper object for building the program

        Example:
            >>> with ib.program("my_program") as p:
            ...     # Build function1
            ...     with ib.function("func1") as f:
            ...         # function body
            ...     func1 = f.get_result()
            ...     p.add_function(func1)
            ...
            ...     # Build function2
            ...     with ib.function("func2") as f:
            ...         # function body
            ...     func2 = f.get_result()
            ...     p.add_function(func2)
            >>> program = p.get_result()
        """
        begin_span = span if span is not None else self.capture_call_span()
        ctx_id = self._ctx_counter
        self._ctx_counter += 1
        self._begin_spans[ctx_id] = begin_span

        self.builder.begin_program(name, begin_span)
        builder_obj = ProgramBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self.capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self.builder.end_program(combined_span)
            builder_obj.result = result
            del self._begin_spans[ctx_id]

    def var(self, name: str, var_type: ir.Type, span: ir.Span | None = None) -> ir.Var:
        """Create a variable with span from call site or explicit span.

        Args:
            name: Variable name
            var_type: Variable type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The created variable
        """
        actual_span = span if span is not None else self.capture_call_span()
        return self.builder.var(name, var_type, actual_span)

    def assign(
        self,
        var: ir.Var,
        value: int | float | ir.Expr,
        span: ir.Span | None = None,
    ) -> ir.AssignStmt:
        """Create assignment statement and emit it.

        Args:
            var: Variable to assign to (must be an ir.Var)
            value: Expression value (int, float, or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            AssignStmt: The created assignment statement
        """
        actual_span = span if span is not None else self.capture_call_span()
        value_expr = _normalize_expr(value, actual_span)
        return self.builder.assign(var, value_expr, actual_span)

    def let(
        self,
        name: str,
        value: int | float | ir.Expr,
        var_type: ir.Type | None = None,
        span: ir.Span | None = None,
    ) -> ir.Var:
        """Create a variable and assign a value to it in one statement.

        This is a convenience method that combines var() and assign() for the
        common pattern of creating a variable and immediately assigning to it.

        The type is automatically inferred from the value expression. If an explicit
        type is provided, it overrides the inferred type -> but must be the same type
        kind (e.g., TensorType can only override TensorType). This supports
        annotation-driven metadata like memref on Tile/Tensor types.

        For Call expressions with cross-function ops, the return type is automatically
        inferred from the function signature if available.

        Args:
            name: Variable name
            value: Expression value (int, float, or Expr)
            var_type: Optional type override. Must be same type kind as inferred type.
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The created variable

        Raises:
            TypeError: If override type is incompatible with inferred type

        Example:
            >>> # Type is inferred from the expression:
            >>> x = ib.let("x", 42)
            >>> # Override with compatible type (same kind, adds memref):
            >>> x = ib.let("x", expr, var_type=ir.TensorType([64], dt, memref))
            >>> # For function calls, type is auto-inferred from function signature:
            >>> result = ib.let("result", ir.Call("helper", [x], span))
        """
        actual_span = span if span is not None else self.capture_call_span()
        value_expr = _normalize_expr(value, actual_span)

        # Auto-infer return type for Call expressions with cross-function calls
        if isinstance(value_expr, ir.Call):
            # Check if the Call has UnknownType and we're inside a program
            if isinstance(value_expr.type, ir.UnknownType) and self.builder.in_program():
                # Try to get return types from the function signature
                return_types = self.builder.get_function_return_types(value_expr.name)
                if len(return_types) == 1:
                    # Recreate the Call with the correct return type
                    value_expr = ir.Call(value_expr.name, value_expr.args, return_types[0], actual_span)
                elif len(return_types) > 1:
                    raise ValueError(
                        f"Function '{value_expr.name}' returns {len(return_types)} values, "
                        f"but let() can only assign single return values. "
                        f"Use explicit tuple unpacking or multiple let() statements."
                    )

        # Infer type from the value expression
        inferred_type = value_expr.type

        # If explicit type is provided, validate compatibility then use as override.
        # Overrides are only allowed when:
        # - inferred type is unknown (no inference available), OR
        # - override type is the same kind as inferred type (e.g., TensorType-> TensorType)
        # This prevents creating vars with incompatible declared types.
        if var_type is not None:
            if not isinstance(inferred_type, ir.UnknownType) and builtins.type(var_type) is not builtins.type(
                inferred_type
            ):
                raise TypeError(
                    f"Type override for '{name}' is incompatible: "
                    f"inferred {builtins.type(inferred_type).__name__} "
                    f"but override is {builtins.type(var_type).__name__}"
                )
            final_type = var_type
        else:
            final_type = inferred_type

        var = self.builder.var(name, final_type, actual_span)
        self.builder.assign(var, value_expr, actual_span)
        return var

    def make_tuple(
        self,
        elements: Sequence[ir.Expr | ir.Var],
        span: ir.Span | None = None,
    ) -> ir.MakeTuple:
        """Create a tuple construction expression.

        Args:
            elements: Expressions to be tuple elements
            span: Optional explicit span. If None, captured from call site.

        Returns:
            MakeTuple: The created tuple expression

        Example:
            >>> with builder.function("my_func") as func:
            ...     x = builder.func_arg("x", ir.ScalarType(DataType.INT64))
            ...     y = builder.func_arg("y", ir.ScalarType(DataType.FP32))
            ...     tuple_val = builder.make_tuple([x, y])
        """
        actual_span = span if span is not None else self.capture_call_span()
        return ir.MakeTuple(list(elements), actual_span)

    def emit(self, stmt: ir.Stmt) -> None:
        """Add a statement to the current context.

        Args:
            stmt: Statement to emit
        """
        self.builder.emit(stmt)

    def return_stmt(
        self,
        values: int | float | ir.Expr | Sequence[int | float | ir.Expr] | None = None,
        span: ir.Span | None = None,
    ) -> ir.ReturnStmt:
        """Create return statement and emit it.

        Args:
            values: Expression value(s) to return. Can be:
                   - None for empty return
                   - Single expression (int, float, or Expr)
                   - List of expressions (int, float, or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            ReturnStmt: The created return statement
        """
        actual_span = span if span is not None else self.capture_call_span()

        # Normalize values to list and convert each element
        if values is None:
            value_list = []
        elif isinstance(values, Sequence):
            value_list = [_normalize_expr(v, actual_span) for v in values]
        else:
            value_list = [_normalize_expr(values, actual_span)]

        return self.builder.return_(value_list, actual_span)

    def eval_stmt(
        self,
        expr: int | float | ir.Expr,
        span: ir.Span | None = None,
    ) -> ir.EvalStmt:
        """Create evaluation statement and emit it.

        Evaluation statements execute expressions for their side effects,
        discarding any return value. Useful for operations like barriers,
        synchronization primitives, or other side-effect-only operations.

        Args:
            expr: Expression to evaluate (int, float, or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            EvalStmt: The created evaluation statement
        """
        actual_span = span if span is not None else self.capture_call_span()
        expr_normalized = _normalize_expr(expr, actual_span)
        stmt = ir.EvalStmt(expr_normalized, actual_span)
        self.builder.emit(stmt)
        return stmt

    def break_stmt(self, span: ir.Span | None = None) -> ir.BreakStmt:
        """Create break statement and emit it.

        Break statement exits the nearest enclosing loop.

        Args:
            span: Optional explicit span. If None, captured from call site.

        Returns:
            BreakStmt: The created break statement
        """
        actual_span = span if span is not None else self.capture_call_span()
        stmt = ir.BreakStmt(actual_span)
        self.builder.emit(stmt)
        return stmt

    def continue_stmt(self, span: ir.Span | None = None) -> ir.ContinueStmt:
        """Create continue statement and emit it.

        Continue statement skips to the next iteration of the nearest enclosing loop.

        Args:
            span: Optional explicit span. If None, captured from call site.

        Returns:
            ContinueStmt: The created continue statement
        """
        actual_span = span if span is not None else self.capture_call_span()
        stmt = ir.ContinueStmt(actual_span)
        self.builder.emit(stmt)
        return stmt

    def in_function(self) -> bool:
        """Check if currently inside a function."""
        return self.builder.in_function()

    def in_loop(self) -> bool:
        """Check if currently inside a for loop."""
        return self.builder.in_loop()

    def in_if(self) -> bool:
        """Check if currently inside an if statement."""
        return self.builder.in_if()

    def memref(
        self,
        memory_space: ir.MemorySpace,
        addr: int | ir.Expr,
        size: int,
        memref_id: int,
        span: ir.Span | None = None,
    ) -> ir.MemRef:
        """Create a MemRef with normalized address expression.

        Args:
            memory_space: Memory space (DDR, Vec, Mat, Left, Right, Acc)
            addr: Address expression (int or Expr)
            size: Size in bytes
            memref_id: Unique identifier for this MemRef
            span: Optional explicit span. If None, captured from call site.

        Returns:
            MemRef: The created memory reference

        Example:
            >>> addr = ir.ConstInt(0x1000, DataType.INT64, ir.Span.unknown())
            >>> memref = ib.memref(ir.MemorySpace.DDR, addr, 1024, 0)
        """
        actual_span = span if span is not None else self.capture_call_span()
        addr_expr = _normalize_expr(addr, actual_span)
        return ir.MemRef(memory_space, addr_expr, size, memref_id, actual_span)

    def tile_view(
        self,
        valid_shape: Sequence[int | ir.Expr],
        stride: Sequence[int | ir.Expr],
        start_offset: int | ir.Expr,
        span: ir.Span | None = None,
    ) -> ir.TileView:
        """Create a TileView with normalized expressions.

        Args:
            valid_shape: Valid shape dimensions (list of int or Expr)
            stride: Stride for each dimension (list of int or Expr)
            start_offset: Starting offset (int or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TileView: The created tile view

        Example:
            >>> valid_shape = [16, 16]
            >>> stride = [1, 16]
            >>> start_offset = 0
            >>> tv = ib.tile_view(valid_shape, stride, start_offset)
        """
        actual_span = span if span is not None else self.capture_call_span()
        valid_shape_exprs = [_normalize_expr(dim, actual_span) for dim in valid_shape]
        stride_exprs = [_normalize_expr(s, actual_span) for s in stride]
        start_offset_expr = _normalize_expr(start_offset, actual_span)
        return ir.TileView(valid_shape_exprs, stride_exprs, start_offset_expr)

    def tensor_view(
        self,
        stride: Sequence[int | ir.Expr],
        layout: ir.TensorLayout,
        span: ir.Span | None = None,
    ) -> ir.TensorView:
        """Create a TensorView with normalized stride expressions.

        Args:
            stride: Stride for each dimension (list of int or Expr)
            layout: Tensor layout type (ND, DN, or NZ)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TensorView: The created tensor view

        Example:
            >>> stride = [1, 256]
            >>> tv = ib.tensor_view(stride, ir.TensorLayout.ND)
        """
        actual_span = span if span is not None else self.capture_call_span()
        stride_exprs = [_normalize_expr(s, actual_span) for s in stride]
        return ir.TensorView(stride_exprs, layout)

    def tensor_type(
        self,
        shape: Sequence[int | ir.Expr],
        dtype: DataType,
        memref: ir.MemRef | None = None,
        tensor_view: ir.TensorView | None = None,
        span: ir.Span | None = None,
    ) -> ir.TensorType:
        """Create a TensorType with normalized shape, optional memref and tensor_view.

        Args:
            shape: Shape dimensions (list of int or Expr)
            dtype: Element data type
            memref: Optional memory reference
            tensor_view: Optional tensor view information
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TensorType: The created tensor type

        Example:
            >>> # Simple tensor type
            >>> tensor_t = ib.tensor_type([64, 128], DataType.FP32)
            >>> # Tensor type with memref
            >>> memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 1024)
            >>> tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)
            >>> # Tensor type with tensor_view
            >>> tv = ib.tensor_view([1, 64], ir.TensorLayout.ND)
            >>> tensor_t = ib.tensor_type([64, 128], DataType.FP32, tensor_view=tv)
        """
        actual_span = span if span is not None else self.capture_call_span()
        shape_exprs = [_normalize_expr(dim, actual_span) for dim in shape]
        return ir.TensorType(shape_exprs, dtype, memref, tensor_view)

    def tile_type(
        self,
        shape: Sequence[int | ir.Expr],
        dtype: DataType,
        memref: ir.MemRef | None = None,
        tile_view: ir.TileView | None = None,
        span: ir.Span | None = None,
    ) -> ir.TileType:
        """Create a TileType with normalized shape, optional memref and tile_view.

        Args:
            shape: Shape dimensions (list of int or Expr)
            dtype: Element data type
            memref: Optional memory reference
            tile_view: Optional tile view information
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TileType: The created tile type

        Example:
            >>> # Simple tile type
            >>> tile_t = ib.tile_type([16, 16], DataType.FP16)
            >>> # Tile type with memref and tile_view
            >>> memref = ib.memref(ir.MemorySpace.Left, 0, 512)
            >>> tv = ib.tile_view([16, 16], [1, 16], 0)
            >>> tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tv)
        """
        actual_span = span if span is not None else self.capture_call_span()
        shape_exprs = [_normalize_expr(dim, actual_span) for dim in shape]
        return ir.TileType(shape_exprs, dtype, memref, tile_view)


class FunctionBuilder:
    """Helper for building functions within a function context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize function builder.

        Args:
            builder: Parent IR builder
        """
        self.builder = builder
        self.result: ir.Function | None = None

    def param(
        self,
        name: str,
        param_type: ir.Type,
        span: ir.Span | None = None,
    ) -> ir.Var:
        """Add function parameter.

        Args:
            name: Parameter name
            param_type: Parameter type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The parameter variable
        """
        actual_span = span if span is not None else self.builder.capture_call_span()
        return self.builder.builder.func_arg(name, param_type, actual_span)

    def return_type(self, ret_type: ir.Type) -> None:
        """Add return type to the function.

        Args:
            ret_type: Return type
        """
        self.builder.builder.return_type(ret_type)

    def get_result(self) -> ir.Function:
        """Get the built Function.

        Returns:
            Function: The completed function IR node (or None if not yet finalized)
        """
        if self.result is None:
            raise RuntimeError("Builder result is not available")
        return self.result


class _LoopBuilderBase:
    """Shared base for ForLoopBuilder and WhileLoopBuilder.

    Provides common iter_arg / return_var logic; subclasses implement
    _add_iter_arg / _add_return_var to call the appropriate IR builder method.
    """

    def __init__(self, builder: IRBuilder) -> None:
        self.builder = builder
        self._iter_args: list[ir.IterArg] = []
        self._return_var_count = 0

    def iter_arg(
        self,
        name: str,
        init_value: int | float | ir.Expr,
        iter_type: ir.Type | None = None,
        span: ir.Span | None = None,
    ) -> ir.IterArg:
        """Add iteration argument (loop-carried value).

        The type is automatically inferred from the init_value expression. If an explicit
        type is provided, it is used to validate that the inferred type matches.

        Args:
            name: Iteration argument name
            init_value: Initial value (int, float, or Expr)
            iter_type: Optional type for validation. If provided, must match the inferred type.
            span: Optional explicit span. If None, captured from call site.

        Returns:
            IterArg: The iteration argument variable

        Raises:
            ValueError: If explicit type is provided and doesn't match inferred type

        Example:
            >>> # Type is inferred from the initial value:
            >>> x_iter = loop.iter_arg("x", 0)
            >>> # Or with explicit type validation:
            >>> x_iter = loop.iter_arg("x", 0, iter_type=ir.ScalarType(ir.DataType.INT64))
        """
        actual_span = span if span is not None else self.builder.capture_call_span()
        init_expr = _normalize_expr(init_value, actual_span)

        inferred_type = init_expr.type

        if iter_type is not None and iter_type != inferred_type:
            raise ValueError(
                f"Type mismatch in iter_arg for '{name}':\n"
                f"  Inferred type: {inferred_type}\n"
                f"  Provided type: {iter_type}"
            )
        final_type = inferred_type

        iter_arg = ir.IterArg(name, final_type, init_expr, actual_span)
        self._add_iter_arg(iter_arg)
        self._iter_args.append(iter_arg)
        return iter_arg

    def return_var(self, name: str, var_type: ir.Type | None = None, span: ir.Span | None = None) -> ir.Var:
        """Add return variable to capture final iteration value.

        The type can be automatically inferred from the corresponding iter_arg (by index).
        If explicit type is provided, it is used to validate against the inferred type.

        Args:
            name: Return variable name
            var_type: Optional type. If None, inferred from corresponding iter_arg by index.
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The return variable

        Raises:
            ValueError: If type cannot be inferred or provided type doesn't match

        Example:
            >>> # Type is inferred from corresponding iter_arg:
            >>> x_final = loop.return_var("x_final")
            >>> # Or with explicit type validation:
            >>> x_final = loop.return_var("x_final", var_type=ir.ScalarType(ir.DataType.INT64))
        """
        actual_span = span if span is not None else self.builder.capture_call_span()

        inferred_type = None
        if self._return_var_count < len(self._iter_args):
            inferred_type = self._iter_args[self._return_var_count].iterVar.type

        if var_type is None:
            if inferred_type is None:
                raise ValueError(
                    f"Cannot infer type for return_var '{name}': "
                    f"no corresponding iter_arg found. Please provide explicit type."
                )
            final_type = inferred_type
        else:
            if inferred_type is not None and var_type != inferred_type:
                raise ValueError(
                    f"Type mismatch in return_var '{name}':\n"
                    f"  Inferred type (from iter_arg): {inferred_type}\n"
                    f"  Provided type: {var_type}"
                )
            final_type = var_type

        var = ir.Var(name, final_type, actual_span)
        self._add_return_var(var)
        self._return_var_count += 1
        return var

    def _add_iter_arg(self, iter_arg: ir.IterArg) -> None:
        raise NotImplementedError

    def _add_return_var(self, var: ir.Var) -> None:
        raise NotImplementedError


class ForLoopBuilder(_LoopBuilderBase):
    """Helper for building for loops within a loop context."""

    def __init__(self, builder: IRBuilder) -> None:
        super().__init__(builder)
        self.result: ir.ForStmt | None = None

    def output(self, index: int = 0) -> ir.Var:
        """Get a single output return variable from the for loop.

        This is a convenience method to access the return variables after the for
        loop is built. Use the index parameter to select which return variable.

        Args:
            index: Index of the return variable to get (default: 0)

        Returns:
            Var: The return variable at the specified index

        Raises:
            AssertionError: If called before for loop is complete
            IndexError: If index is out of range

        Example:
            >>> with ib.for_loop(i, 0, 10, 1) as loop:
            ...     sum_iter = loop.iter_arg("sum", 0)
            ...     loop.return_var("sum_final")
            ...     # ... loop body ...
            >>> result = loop.output()  # Get the first return variable
            >>> # Or for multiple return vars:
            >>> result1 = loop.output(0)
            >>> result2 = loop.output(1)
        """
        if self.result is None:
            raise RuntimeError("For loop not yet complete")
        if index >= len(self.result.return_vars):
            raise IndexError(
                f"Return variable index {index} out of range (for loop has {len(self.result.return_vars)} return vars)"
            )
        return self.result.return_vars[index]

    def outputs(self) -> list[ir.Var]:
        """Get all output return variables from the for loop.

        This is a convenience method to access all return variables at once after
        the for loop is built.

        Returns:
            List of all return variables

        Raises:
            AssertionError: If called before for loop is complete

        Example:
            >>> with ib.for_loop(i, 0, 10, 1) as loop:
            ...     sum_iter = loop.iter_arg("sum", 0)
            ...     prod_iter = loop.iter_arg("prod", 1)
            ...     loop.return_var("sum_final")
            ...     loop.return_var("prod_final")
            ...     # ... loop body ...
            >>> sum_result, prod_result = loop.outputs()  # Get all return variables
        """
        if self.result is None:
            raise RuntimeError("For loop not yet complete")
        return list(self.result.return_vars)

    def get_result(self) -> ir.ForStmt:
        """Get the built ForStmt.

        Returns:
            ForStmt: The completed for loop IR node
        """
        if self.result is None:
            raise RuntimeError("Builder result is not available")
        return self.result

    def _add_iter_arg(self, iter_arg: ir.IterArg) -> None:
        self.builder.builder.add_iter_arg(iter_arg)

    def _add_return_var(self, var: ir.Var) -> None:
        self.builder.builder.add_return_var(var)


class WhileLoopBuilder(_LoopBuilderBase):
    """Helper for building while loops within a loop context."""

    def __init__(self, builder: IRBuilder) -> None:
        super().__init__(builder)
        self.result: ir.WhileStmt | None = None

    def set_condition(self, condition: int | ir.Expr) -> None:
        """Set the condition for the while loop.

        Used to update the loop condition after setting up iter_args. This allows
        the condition to reference iter_arg variables.

        Args:
            condition: Condition expression (int or Expr)

        Example:
            >>> with ib.while_loop(ir.ConstBool(True, span), span) as loop:
            ...     x_iter = loop.iter_arg("x", x_0)
            ...     loop.set_condition(ir.Lt(x_iter, ir.ConstInt(10, ir.DataType.INT64, span), span))
        """
        actual_span = self.builder.capture_call_span()
        condition_expr = _normalize_expr(condition, actual_span)
        self.builder.builder.set_while_loop_condition(condition_expr)

    def output(self, index: int = 0) -> ir.Var:
        """Get a single output return variable from the while loop.

        This is a convenience method to access the return variables after the while
        loop is built. Use the index parameter to select which return variable.

        Args:
            index: Index of the return variable to get (default: 0)

        Returns:
            Var: The return variable at the specified index

        Raises:
            AssertionError: If called before while loop is complete
            IndexError: If index is out of range

        Example:
            >>> with ib.while_loop(condition) as loop:
            ...     x_iter = loop.iter_arg("x_iter", 0)
            ...     loop.return_var("x_final")
            ...     # ... loop body ...
            >>> result = loop.output()  # Get the first return variable
        """
        if self.result is None:
            raise RuntimeError("While loop not yet complete")
        if index >= len(self.result.return_vars):
            raise IndexError(
                f"Return variable index {index} out of range "
                f"(while loop has {len(self.result.return_vars)} return vars)"
            )
        return self.result.return_vars[index]

    def outputs(self) -> list[ir.Var]:
        """Get all output return variables from the while loop.

        This is a convenience method to access all return variables at once after
        the while loop is built.

        Returns:
            List of all return variables

        Raises:
            AssertionError: If called before while loop is complete

        Example:
            >>> with ib.while_loop(condition) as loop:
            ...     x_iter = loop.iter_arg("x_iter", 0)
            ...     y_iter = loop.iter_arg("y_iter", 1)
            ...     loop.return_var("x_final")
            ...     loop.return_var("y_final")
            ...     # ... loop body ...
            >>> x_result, y_result = loop.outputs()  # Get all return variables
        """
        if self.result is None:
            raise RuntimeError("While loop not yet complete")
        return list(self.result.return_vars)

    def get_result(self) -> ir.WhileStmt:
        """Get the built WhileStmt.

        Returns:
            WhileStmt: The completed while loop IR node
        """
        if self.result is None:
            raise RuntimeError("Builder result is not available")
        return self.result

    def _add_iter_arg(self, iter_arg: ir.IterArg) -> None:
        self.builder.builder.add_while_iter_arg(iter_arg)

    def _add_return_var(self, var: ir.Var) -> None:
        self.builder.builder.add_while_return_var(var)


class SectionBuilder:
    """Helper for building section statements within a section context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize section statement builder.

        Args:
            builder: Parent IR builder
        """
        self.builder = builder
        self.result: ir.SectionStmt | None = None

    def get_result(self) -> ir.SectionStmt:
        """Get the built SectionStmt.

        Returns:
            SectionStmt: The completed section statement IR node
        """
        if self.result is None:
            raise RuntimeError("Builder result is not available")
        return self.result


class IfStmtBuilder:
    """Helper for building if statements within an if context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize if statement builder.

        Args:
            builder: Parent IR builder
        """
        self.builder = builder
        self.result: ir.IfStmt | None = None

    def else_(self, span: ir.Span | None = None) -> None:
        """Begin else branch of the if statement.

        Args:
            span: Optional explicit span. If None, captured from call site.
        """
        actual_span = span if span is not None else self.builder.capture_call_span()
        self.builder.builder.begin_else(actual_span)

    def return_var(
        self,
        name: str,
        var_type: ir.Type,
        span: ir.Span | None = None,
    ) -> None:
        """Add return variable for SSA phi node.

        Note: Type must be provided explicitly. Type inference is not supported for
        if statement return_vars because they are declared before yield statements
        are executed. Type inference could be implemented in C++ EndIf logic.

        Args:
            name: Return variable name
            var_type: Variable type (required)
            span: Optional explicit span. If None, captured from call site.

        Example:
            >>> # Type must be provided explicitly:
            >>> if_builder.return_var("result", ir.ScalarType(ir.DataType.INT64))
        """
        actual_span = span if span is not None else self.builder.capture_call_span()
        var = ir.Var(name, var_type, actual_span)
        self.builder.builder.add_if_return_var(var)

    def output(self, index: int = 0) -> ir.Var:
        """Get a single output return variable from the if statement.

        This is a convenience method to access the return variables after the if
        statement is built. Use the index parameter to select which return variable.

        Args:
            index: Index of the return variable to get (default: 0)

        Returns:
            Var: The return variable at the specified index

        Raises:
            AssertionError: If called before if statement is complete
            IndexError: If index is out of range

        Example:
            >>> with ib.if_stmt(condition) as if_builder:
            ...     if_builder.return_var("result", ir.ScalarType(DataType.INT64))
            ...     # ... if/else branches ...
            >>> result = if_builder.output()  # Get the first return variable
            >>> # Or for multiple return vars:
            >>> result1 = if_builder.output(0)
            >>> result2 = if_builder.output(1)
        """
        if self.result is None:
            raise RuntimeError("If statement not yet complete")
        if index >= len(self.result.return_vars):
            raise IndexError(
                f"Return variable index {index} out of range "
                f"(if statement has {len(self.result.return_vars)} return vars)"
            )
        return self.result.return_vars[index]

    def outputs(self) -> list[ir.Var]:
        """Get all output return variables from the if statement.

        This is a convenience method to access all return variables at once after
        the if statement is built.

        Returns:
            List of all return variables

        Raises:
            AssertionError: If called before if statement is complete

        Example:
            >>> with ib.if_stmt(condition) as if_builder:
            ...     if_builder.return_var("x", ir.ScalarType(DataType.INT64))
            ...     if_builder.return_var("y", ir.ScalarType(DataType.INT64))
            ...     # ... if/else branches ...
            >>> x, y = if_builder.outputs()  # Get all return variables
        """
        if self.result is None:
            raise RuntimeError("If statement not yet complete")
        return list(self.result.return_vars)

    def get_result(self) -> ir.IfStmt:
        """Get the built IfStmt.

        Returns:
            IfStmt: The completed if statement IR node
        """
        if self.result is None:
            raise RuntimeError("Builder result is not available")
        return self.result


class ProgramBuilder:
    """Helper for building programs within a program context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize program builder.

        Args:
            builder: Parent IR builder
        """
        self.builder = builder
        self.result: ir.Program | None = None

    def add_function(self, func: ir.Function) -> None:
        """Add a function to the program.

        Args:
            func: Function to add
        """
        self.builder.builder.add_function(func)

    def get_result(self) -> ir.Program:
        """Get the built Program.

        Returns:
            Program: The completed program IR node

        Raises:
            AssertionError: If called before program is complete
        """
        if self.result is None:
            raise RuntimeError("Program not yet complete")
        return self.result
