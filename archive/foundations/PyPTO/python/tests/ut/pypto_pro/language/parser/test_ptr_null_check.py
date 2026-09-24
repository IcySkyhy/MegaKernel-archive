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
"""Parser tests for pointer null-check syntax on optional pl.Ptr parameters."""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest


def test_is_none_requires_simple_variable():
    """'is None' must be used on simple variable names, not complex expressions."""
    with pytest.raises(ParserSyntaxError, match="only supported on simple variable names"):
        @pl.function
        def bad_is_none():
            x: pl.DT_INT32 = 42
            if (x + 1) is None:  # Complex expression not allowed
                pass


def test_is_none_on_tensor_rejected():
    """'is None' on a pl.Tensor parameter is rejected at parse time.

    pl.Tensor is a descriptor (ptr + shape); null-checking it has no meaning
    and the generated code would reference an undeclared identifier. Users
    must declare optional inputs as pl.Ptr instead (reviewer guidance: a
    tensor with no argument has no meaningful shape).
    """
    with pytest.raises(ParserTypeError, match="pl.Ptr"):
        @pl.function
        def bad_tensor_none(
            src: pl.Tensor[[4], pl.DT_FP16],
            bias: pl.Tensor[[4], pl.DT_FP16],  # should be pl.Ptr[pl.DT_FP16]
        ):
            if bias is None:
                pass
