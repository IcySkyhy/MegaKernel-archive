# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar wrapper type for PyPTO Language DSL."""

from __future__ import annotations

__all__ = ["Scalar"]


from pypto.pypto_impl.ir import Expr


class Scalar:
    """Scalar type for PyPTO Language DSL.

    This class is the runtime wrapper around IR Expr/Call objects.

    Scalar annotations use DT_* dtype constants directly:
        x: pl.DT_FP32
        count: pl.DT_INT32

    Runtime mode (wraps IR expressions):
        scalar_value = pl.const(3.14, pl.DT_FP32)
        # Returns Scalar wrapping the Call expression

    Examples:
        >>> import pypto_pro.language as pl
        >>>
        >>> @pl.function
        ... def add_scalar(
        ...     x: pl.Tensor[[64], pl.DT_FP32],
        ...     scalar: pl.DT_FP32
        ... ) -> pl.Tensor[[64], pl.DT_FP32]:
        ...     result: pl.Tensor[[64], pl.DT_FP32] = pl.add(x, scalar)
        ...     return result
    """

    def __init__(
        self,
        expr: Expr | None = None,
    ):
        """Initialize Scalar.

        Args:
            expr: IR expression to wrap (for runtime mode)

        Raises:
            ValueError: If neither dtype nor expr is provided
        """
        if expr is not None:
            # Runtime mode: wrap IR expression
            self.expr = expr
            self.dtype = None
        else:
            raise ValueError("expr is required")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Scalar(expr={self.expr})"

    def unwrap(self) -> Expr:
        """Unwrap to get the underlying IR expression.

        Returns:
            The wrapped IR expression

        Raises:
            RuntimeError: If no expression is wrapped
        """
        if self.expr is None:
            raise RuntimeError("No expression to unwrap")
        return self.expr
