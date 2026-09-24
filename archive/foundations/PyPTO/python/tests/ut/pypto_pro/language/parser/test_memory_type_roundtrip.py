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
"""Comprehensive tests for MemRef, MemorySpace, and TileView."""

import textwrap

from pypto_pro import ir
import pypto_pro.language as pl


def test_parse_tensor_with_memref():
    """Parse pl.Tensor[[64], pl.DT_FP32, pl.MemRef(...)] annotation."""
    code = textwrap.dedent("""\
        @pl.program
        class TestProg:
            @pl.function
            def test_fn(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                y: pl.Tensor[[64], pl.DT_FP32, pl.MemRef(pl.MemorySpace.DDR, 0, 256, 1)] = pl.tensor.add(x, 1.0)
                return y
    """)
    program = pl.parse(code)
    assert isinstance(program, ir.Program)

    # Verify the parsed IR contains memref by re-printing
    printed = ir.python_print(program)
    assert "ir.MemRef" in printed
    assert "ir.MemorySpace.DDR" in printed
    assert "256" in printed


def test_backwards_compat_three_args_layout():
    """Existing 3-arg [shape, dtype, layout] still works for Tensor."""
    code = textwrap.dedent("""\
        @pl.program
        class TestProg:
            @pl.function
            def test_fn(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                y: pl.Tensor[[64], pl.DT_FP32, pl.NZ] = pl.tensor.add(x, 1.0)
                return y
    """)
    # Should parse without errors
    program = pl.parse(code)
    assert isinstance(program, ir.Program)
