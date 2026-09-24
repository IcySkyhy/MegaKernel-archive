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


"""PTO Python Frontend - JIT Compilation and Parsing Interface.

This module provides the public API for the PTO frontend, which enables JIT
compilation of Python functions into optimized PTO kernels. The frontend parses
Python functions decorated with @jit or @function and converts them to PTO
intermediate representation (IR) for execution on NPU hardware or in simulation.

Public API
----------
jit : decorator
    JIT compilation decorator for converting Python functions to PTO kernels.
    Use @pypto.frontend.jit or @pypto.frontend.jit() to decorate functions.

function : decorator
    Nested function decorator for inline expansion. Functions decorated with
    @pypto.frontend.function are inlined when called from JIT-compiled kernels.

dynamic : function
    Create symbolic dimensions for dynamic tensor shapes. Returns a SymbolicScalar
    that represents a runtime-determined dimension value.

parser : module
    The parser submodule containing the core parsing implementation. Generally
    not accessed directly by users.

Usage Examples
--------------
Basic JIT compilation:
    >>> @pypto.frontend.jit()
    ... def my_kernel(x: pypto.Tensor([16,], pypto.DT_FP32)):
    ...     return pypto.add(x, x)

Dynamic dimensions:
    >>> N = pypto.DYNAMIC
    >>> @pypto.frontend.jit()
    ... def dynamic_kernel(x: pypto.Tensor([N, 128], pypto.DT_FP32)):
    ...     return x

Nested functions:
    >>> @pypto.frontend.function
    ... def helper(x):
    ...     return pypto.add(x, x)
    >>>
    >>> @pypto.frontend.jit
    ... def kernel(x):
    ...     return helper(x)  # Inlined during compilation

See Also
--------
developer_doc.md : Comprehensive developer documentation
parser.entry : Entry points and wrapper classes
parser.parser : Main parser implementation
"""

from ..symbolic_scalar import SymbolicScalar
from . import parser  # noqa: F401
from .parser import function, jit  # noqa: F401


def dynamic(name: str) -> SymbolicScalar:
    """Create a dynamic (symbolic) dimension for tensor shapes.

    This function creates a symbolic scalar that can be used to define
    dynamic dimensions in tensor shapes. The symbolic scalar represents
    a runtime value that is not known at compile time and will be bound
    to concrete values when the kernel is first invoked.

    Dynamic dimensions enable writing kernels that work with variable-sized
    inputs without recompilation. The parser validates that all uses of a
    symbolic dimension are consistent within a kernel.

    Parameters
    ----------
    name : str
        The name of the dynamic dimension. This name is used for binding
        concrete values and in error messages.

    Returns
    -------
    SymbolicScalar
        A symbolic scalar representing the dynamic dimension.

    Examples
    --------
    Create a batch-size dimension:
    >>> BS = pypto.DYNAMIC
    >>> @pypto.frontend.jit()
    ... def batch_kernel(x: pypto.Tensor((BS, 128), pypto.DT_FP16)):
    ...     return x

    Multiple dynamic dimensions:
    >>> N = pypto.DYNAMIC
    >>> M = pypto.DYNAMIC
    >>> @pypto.frontend.jit()
    ... def matmul_kernel(
    ...     a: pypto.Tensor((N, M), pypto.DT_FP32),
    ...     b: pypto.Tensor((M, N), pypto.DT_FP32),
    ... ):
    ...     # Kernel implementation
    ...     pass

    Notes
    -----
    - Symbolic dimensions are resolved on first kernel invocation
    - All uses of the same symbolic dimension must have consistent values
    - Symbolic dimensions can be used in shape calculations within kernels
    """
    return SymbolicScalar(name)
