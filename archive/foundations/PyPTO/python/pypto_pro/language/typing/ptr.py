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

"""Ptr wrapper type for PyPTO Language DSL."""

from __future__ import annotations

__all__ = ["Ptr"]


from pypto.pypto_impl.ir import DataType, Expr


class Ptr:
    """Pointer type for PyPTO Language DSL.

    Represents a raw pointer to global memory of a specific element type,
    corresponding to ``!pto.ptr<dtype>`` in PTO MLIR.

    Used as the base-pointer argument to ``pl.make_tensor``:

    Example::

        @pl.function
        def kernel(ptr: pl.Ptr[pl.DT_FP32]):
            view: pl.Tensor[[32, 32], pl.DT_FP32] = pl.make_tensor(ptr, [32, 32], [32, 1])

    Annotation mode (used in type hints)::

        ptr: pl.Ptr[pl.DT_FP32]
        ptr: pl.Ptr[pl.DT_INT8]
    """

    def __init__(
        self,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Ptr.

        Args:
            dtype: Element data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Internal flag for annotation-only mode

        Raises:
            ValueError: If neither dtype nor expr is provided
        """
        if _annotation_only:
            if dtype is None:
                raise ValueError("dtype is required for annotation mode")
            self.dtype = dtype
            self.expr = None
            self._annotation_only = True
        elif expr is not None:
            self.expr = expr
            self.dtype = None
            self._annotation_only = False
        else:
            raise ValueError("Either dtype (for annotation) or expr (for runtime) must be provided")

    def __repr__(self) -> str:
        """Return string representation."""
        if self._annotation_only:
            return f"Ptr[{self.dtype}]"
        return f"Ptr(expr={self.expr})"

    @classmethod
    def __class_getitem__(cls, dtype: DataType) -> "Ptr":
        """Enable Ptr[dtype] subscript syntax."""
        return cls(dtype, _annotation_only=True)

    def unwrap(self) -> Expr:
        """Unwrap to get the underlying IR expression.

        Returns:
            The wrapped IR expression

        Raises:
            RuntimeError: If this is an annotation-only instance
        """
        if self._annotation_only:
            raise RuntimeError("Cannot unwrap annotation-only Ptr")
        if self.expr is None:
            raise RuntimeError("No expression to unwrap")
        return self.expr
