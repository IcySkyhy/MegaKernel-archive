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

"""Parser tests for the public ``pl.simt.cast`` API.

Operation-level tests use eager SIMT IR parsing so diagnostics and generated IR
are observable at function definition time. Delayed ``pl.simt.function`` parsing
is covered by ``test_simt.py``.
"""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest

from pypto.pypto_impl import ir


def test_simt_cast_is_not_exported_as_shared_language_api():
    assert not hasattr(pl, "scalar_cast")


def test_simt_cast_rejects_ordinary_function():
    with pytest.raises(ParserSyntaxError, match="can only be used inside a SIMT function"):

        @pl.function(type=pl.FunctionType.Orchestration)
        def unsupported_context(value: pl.DT_INT64) -> pl.DT_INT32:
            return pl.simt.cast(value, pl.DT_INT32)


def test_simt_cast_is_available_inside_simt_helper():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def cast_value(value: pl.DT_FP32) -> pl.DT_FP16:
        return pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

    function_ir = str(cast_value)
    assert "simt.cast" in function_ir


def test_simt_cast_rejects_unsupported_dtype_pair():
    with pytest.raises(ParserTypeError, match=r"pl\.simt\.cast\(\) does not support"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def unsupported(value: pl.DT_FP16) -> pl.DT_INT32:
            return pl.simt.cast(value, pl.DT_INT32)


def test_simt_cast_rejects_odd_rounding_for_bfloat16():
    with pytest.raises(ParserTypeError, match=r"pl\.simt\.cast\(\) does not support"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def unsupported_mode(value: pl.DT_FP32) -> pl.DT_BF16:
            return pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_ODD)


def test_simt_cast_rejects_tile_operand():
    @pl.simt.function
    def tile_operand(value):
        _ = pl.simt.cast(value, pl.DT_FP16)

    @pl.simt.function(max_threads=1)
    def entry(value):
        tile_operand(value)

    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=[1, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        value = pl.make_tile(tile_type, addr=0, size=4)
        with pl.section_vector():
            pl.simt.launch(entry, threads=1, args=(value,))

    with pytest.raises(ParserTypeError, match="value must be a scalar expression"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_plain_integer_round_mode():
    with pytest.raises(ParserTypeError, match="expects an enum value"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def integer_mode(value: pl.DT_FP32) -> pl.DT_FP16:
            return pl.simt.cast(value, pl.DT_FP16, mode=1)


def test_simt_cast_rejects_wrong_positional_arity():
    with pytest.raises(ParserSyntaxError, match="requires exactly 2 positional arguments"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def missing_dtype(value: pl.DT_FP32) -> pl.DT_FP16:
            return pl.simt.cast(value)


def test_simt_cast_rejects_non_dtype_target():
    with pytest.raises(ParserTypeError, match=r"dtype must be a pl\.DT_\* value"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def invalid_dtype(value: pl.DT_FP32) -> pl.DT_FP16:
            return pl.simt.cast(value, 1)


def test_simt_cast_rejects_unexpected_keyword():
    with pytest.raises(ParserSyntaxError, match="only accepts one optional keyword argument"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def unexpected_keyword(value: pl.DT_FP32) -> pl.DT_FP16:
            return pl.simt.cast(value, pl.DT_FP16, saturate=True)
