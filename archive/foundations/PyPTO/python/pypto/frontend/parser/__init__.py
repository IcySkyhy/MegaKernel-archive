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

"""PTO Script Parser - Core Parsing Infrastructure.

This module provides the PTO Script Parser implementation, which parses Python
functions decorated with @pypto.frontend.jit or @pypto.frontend.function and converts them
to PTO intermediate representation (IR). The parser is inspired by TVM's Script
Parser and provides a robust, extensible system for compiling Python code to
optimized PTO operations.

Architecture
------------
The parser follows a multi-stage pipeline:
1. Source extraction (diagnostics.Source)
2. Python AST parsing (ast.parse)
4. Liveness analysis (liveness.LivenessAnalyzer)
5. IR generation (parser.Parser)
6. Runtime integration (entry.JitCallableWrapper)

Key Components
--------------
jit : decorator
    Main JIT compilation decorator exported from entry module.
    Supports both @jit and @jit() syntax with configuration options.

function : decorator
    Nested function decorator for inline expansion.
    Marks functions for inlining when called from JIT kernels.

error : module
    Exception classes and error handling utilities. Includes ParserError
    and RenderedParserError with backtrace control via PTO_BACKTRACE.

Submodules (Internal)
----------------------
- entry: Entry points and JitCallableWrapper for runtime integration
- parser: Main Parser class implementing the visitor pattern
- context: Variable scoping and lifetime management
- diagnostics: Rich error reporting with source context
- evaluator: Expression evaluation during parsing
- liveness: Automatic memory management through liveness analysis

Usage Example
-------------
```python
import pypto

@pypto.frontend.jit()
def my_kernel(
    x: pypto.Tensor((16,), pypto.DT_FP32)
) -> pypto.Tensor((16,), pypto.DT_FP32):
    result = pypto.tensor((16,), pypto.DT_FP32)
    result[:] = pypto.add(x, x)
    return result

# Call with torch tensors
import torch
x = torch.randn(16, dtype=torch.float32, device='npu:0')
y = my_kernel(x)
```

For detailed developer documentation, see developer_doc.md in the frontend directory.
"""

from ... import error  # noqa: F401
from .entry import function, jit  # noqa: F401
from .pil_parser import parse_stmts  # noqa: F401
