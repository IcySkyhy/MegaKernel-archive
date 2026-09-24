# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Language Parser module for converting high-level DSL code to IR structures.

This module (pypto_pro.language.parser) provides a decorator-based system for
parsing Python functions with DSL annotations and converting them to IR
builder programs.

Part of the pypto_pro.language package - use via:
    import pypto_pro.language as pl

    @pl.function
    def my_func(...):
        ...
"""

from __future__ import annotations

__all__ = [
    "function",
    "inline",
    "program",
    "parse",
    "loads",
    "parse_program",
    "loads_program",
    "Tensor",
    "Scalar",
]


from ..typing import Scalar, Tensor
from .decorator import function, inline, program
from .diagnostics import ParserError  # noqa: F401
from .text_parser import loads, loads_program, parse, parse_program
