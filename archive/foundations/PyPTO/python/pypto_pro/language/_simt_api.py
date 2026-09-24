#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro SIMT namespace (``pl.simt.*``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.simt.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.simt.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Any, Callable

from pypto.ir import RoundMode

from ._api import DType, Scalar, _api_decl


class Simt:
    """SIMT namespace (``pl.simt.*``).

    Provides SIMT context queries (thread_idx, block_dim, ...), launch, and
    the ``function`` decorator for defining SIMT VF functions.
    """

    @staticmethod
    def function(fn: Callable | None = None, *, max_threads: int | None = None) -> Callable:
        """Mark a callable for delayed SIMT parsing at its call site.

        Args:
            fn: Python function to decorate (when used without parentheses).
            max_threads: Launch bound for SIMT VF functions. When provided, the
                function is launchable via ``pl.simt.launch()``. When omitted,
                the function is a SIMT callee helper.
        """
        from .parser.decorator import simt_function

        return simt_function(fn, max_threads=max_threads)

    @staticmethod
    @_api_decl
    def thread_idx() -> Any:
        """Return the current thread coordinates within its thread block."""

    @staticmethod
    @_api_decl
    def block_dim() -> Any:
        """Return the dimensions of the current thread block."""

    @staticmethod
    @_api_decl
    def block_idx() -> Any:
        """Return the current block coordinates within the outer kernel grid."""

    @staticmethod
    @_api_decl
    def grid_dim() -> Any:
        """Return the dimensions of the outer kernel grid."""

    @staticmethod
    @_api_decl
    def linear_thread_idx() -> Scalar:
        """Return the x-major flattened thread index within the current block."""

    @staticmethod
    @_api_decl
    def warp_size() -> Scalar:
        """Return the target SIMT warp size."""

    @staticmethod
    @_api_decl
    def cast(
        value: Scalar,
        dtype: DType,
        *,
        mode: RoundMode = RoundMode.CAST_NONE,
    ) -> Scalar:
        """Convert one scalar expression to ``dtype`` inside a SIMT function.

        ``mode`` controls rounding when the target dtype cannot represent ``value``
        exactly. ``CAST_ODD`` is supported only for FP32-to-FP16 conversion.
        """

    @staticmethod
    @_api_decl
    def abs(value: Scalar) -> Scalar:
        """Return the absolute value of a FP16, BF16, FP32, or INT64 Scalar."""

    @staticmethod
    @_api_decl
    def min(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the minimum of two same-dtype floating-point or integer Scalars."""

    @staticmethod
    @_api_decl
    def max(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the maximum of two same-dtype floating-point or integer Scalars."""

    @staticmethod
    @_api_decl
    def sqrt(value: Scalar) -> Scalar:
        """Return the square root of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rsqrt(value: Scalar) -> Scalar:
        """Return the reciprocal square root of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def exp(value: Scalar) -> Scalar:
        """Return e raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def exp2(value: Scalar) -> Scalar:
        """Return two raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log(value: Scalar) -> Scalar:
        """Return the natural logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log2(value: Scalar) -> Scalar:
        """Return the base-two logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log1p(value: Scalar) -> Scalar:
        """Return the FP32 natural logarithm of one plus ``value``."""

    @staticmethod
    @_api_decl
    def sin(value: Scalar) -> Scalar:
        """Return the sine of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def cos(value: Scalar) -> Scalar:
        """Return the cosine of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def tanh(value: Scalar) -> Scalar:
        """Return the hyperbolic tangent of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rint(value: Scalar) -> Scalar:
        """Round a floating-point Scalar to the nearest integer value."""

    @staticmethod
    @_api_decl
    def round(value: Scalar) -> Scalar:
        """Round a floating-point Scalar halfway away from zero."""

    @staticmethod
    @_api_decl
    def floor(value: Scalar) -> Scalar:
        """Round a floating-point Scalar down to an integer value."""

    @staticmethod
    @_api_decl
    def ceil(value: Scalar) -> Scalar:
        """Round a floating-point Scalar up to an integer value."""

    @staticmethod
    @_api_decl
    def trunc(value: Scalar) -> Scalar:
        """Round a floating-point Scalar toward zero to an integer value."""

    @staticmethod
    @_api_decl
    def isnan(value: Scalar) -> Scalar:
        """Return a BOOL Scalar indicating whether a floating-point Scalar is NaN."""

    @staticmethod
    @_api_decl
    def isinf(value: Scalar) -> Scalar:
        """Return a BOOL Scalar indicating whether a floating-point Scalar is infinite."""

    @staticmethod
    @_api_decl
    def fma(lhs: Scalar, rhs: Scalar, addend: Scalar) -> Scalar:
        """Fused-multiply-add three same-dtype FP16, BF16, or FP32 Scalars."""

    @staticmethod
    @_api_decl
    def atomic_add(target: Scalar, value: Scalar) -> Scalar | None:
        """Atomically add ``value`` to one Tile or Tensor element.

        ``target`` must be written directly as a subscript expression such as
        ``tile[row, col]`` or ``tensor[index]``. FP16/BF16 targets return no value;
        other supported dtypes return the element value observed before the update.
        """

    @staticmethod
    @_api_decl
    def atomic_sub(target: Scalar, value: Scalar) -> Scalar:
        """Atomically subtract ``value`` from one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_exch(target: Scalar, value: Scalar) -> Scalar:
        """Atomically replace one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_max(target: Scalar, value: Scalar) -> Scalar | None:
        """Atomically update an element with its maximum; FP16/BF16 return no value."""

    @staticmethod
    @_api_decl
    def atomic_min(target: Scalar, value: Scalar) -> Scalar | None:
        """Atomically update an element with its minimum; FP16/BF16 return no value."""

    @staticmethod
    @_api_decl
    def atomic_inc(target: Scalar, limit: Scalar) -> Scalar:
        """Atomically increment and wrap one unsigned counter element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_dec(target: Scalar, limit: Scalar) -> Scalar:
        """Atomically decrement and wrap one unsigned counter element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_cas(target: Scalar, compare: Scalar, value: Scalar) -> Scalar:
        """Atomically compare and exchange one Tile or Tensor element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_and(target: Scalar, value: Scalar) -> Scalar:
        """Atomically apply bitwise AND to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_or(target: Scalar, value: Scalar) -> Scalar:
        """Atomically apply bitwise OR to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_xor(target: Scalar, value: Scalar) -> Scalar:
        """Atomically apply bitwise XOR to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def launch(callee: Any, *, threads: int | tuple[int, ...], args: tuple[Any, ...]) -> None:
        """Launch a SIMT function from a Vector section.

        Args:
            callee: Launchable function defined with ``@pl.simt.function(max_threads=...)``.
            threads: One- to three-dimensional compile-time thread configuration.
            args: Scalar, Tensor, and Tile arguments passed to ``callee``.
        """
