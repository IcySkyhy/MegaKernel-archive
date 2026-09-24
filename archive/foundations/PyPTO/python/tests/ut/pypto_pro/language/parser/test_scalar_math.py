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

"""Parser tests for public ``pl.simt`` scalar math APIs.

Operation-level tests use eager SIMT IR parsing so diagnostics and generated IR
are observable at function definition time. Delayed ``pl.simt.function`` parsing
is covered by ``test_simt.py``.
"""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest

from pypto.pypto_impl import ir


def _parse_tile_function(function, tile_specs):
    if len(tile_specs) == 1:
        (shape0, dtype0) = tile_specs[0]

        @pl.simt.function(max_threads=1)
        def entry(tile0):
            function(tile0)

        @pl.kernel
        def kernel():
            tile0 = pl.make_tile(
                pl.TileType(shape=shape0, dtype=dtype0, target_memory=pl.MemorySpace.Vec),
                addr=0,
                size=4096,
            )
            with pl.section_vector():
                pl.simt.launch(entry, threads=1, args=(tile0,))

    elif len(tile_specs) == 2:
        (shape0, dtype0), (shape1, dtype1) = tile_specs

        @pl.simt.function(max_threads=1)
        def entry(tile0, tile1):
            function(tile0, tile1)

        @pl.kernel
        def kernel():
            tile0 = pl.make_tile(
                pl.TileType(shape=shape0, dtype=dtype0, target_memory=pl.MemorySpace.Vec),
                addr=0,
                size=4096,
            )
            tile1 = pl.make_tile(
                pl.TileType(shape=shape1, dtype=dtype1, target_memory=pl.MemorySpace.Vec),
                addr=4096,
                size=4096,
            )
            with pl.section_vector():
                pl.simt.launch(entry, threads=1, args=(tile0, tile1))

    elif len(tile_specs) == 3:
        (shape0, dtype0), (shape1, dtype1), (shape2, dtype2) = tile_specs

        @pl.simt.function(max_threads=1)
        def entry(tile0, tile1, tile2):
            function(tile0, tile1, tile2)

        @pl.kernel
        def kernel():
            tile0 = pl.make_tile(
                pl.TileType(shape=shape0, dtype=dtype0, target_memory=pl.MemorySpace.Vec),
                addr=0,
                size=4096,
            )
            tile1 = pl.make_tile(
                pl.TileType(shape=shape1, dtype=dtype1, target_memory=pl.MemorySpace.Vec),
                addr=4096,
                size=4096,
            )
            tile2 = pl.make_tile(
                pl.TileType(shape=shape2, dtype=dtype2, target_memory=pl.MemorySpace.Vec),
                addr=8192,
                size=4096,
            )
            with pl.section_vector():
                pl.simt.launch(entry, threads=1, args=(tile0, tile1, tile2))

    else:
        raise ValueError("Only one to three Tile parameters are supported")

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def test_scalar_math_is_available_in_simt_functions():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def scalar_math(value: pl.DT_FP32, rhs: pl.DT_FP32, addend: pl.DT_FP32) -> pl.DT_FP32:
        transformed = pl.simt.max(pl.simt.min(value, rhs), pl.simt.abs(addend))
        transformed = pl.simt.rsqrt(pl.simt.sqrt(transformed))
        transformed = pl.simt.exp2(pl.simt.exp(transformed))
        transformed = pl.simt.log2(pl.simt.log(pl.simt.log1p(transformed)))
        transformed = pl.simt.tanh(pl.simt.cos(pl.simt.sin(transformed)))
        transformed = pl.simt.trunc(
            pl.simt.floor(pl.simt.ceil(pl.simt.round(pl.simt.rint(transformed))))
        )
        classified = pl.simt.isnan(transformed) or pl.simt.isinf(transformed)
        if classified:
            transformed = value
        return pl.simt.fma(transformed, rhs, addend)

    function_ir = str(scalar_math)
    for name in (
        "abs",
        "min",
        "max",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "log",
        "log2",
        "log1p",
        "sin",
        "cos",
        "tanh",
        "rint",
        "round",
        "floor",
        "ceil",
        "trunc",
        "isnan",
        "isinf",
        "fma",
    ):
        assert f"simt.{name}(" in function_ir


def test_scalar_math_supports_fp16_and_bf16_scalars():
    @pl.simt.function
    def fp16_math(
        src,
        dst,
    ):
        value = src[0, 0]
        dst[0, 0] = pl.simt.fma(pl.simt.round(pl.simt.sin(value)), value, value)

    @pl.simt.function
    def bf16_math(
        src,
        dst,
        flags,
    ):
        value = src[0, 0]
        dst[0, 0] = pl.simt.tanh(pl.simt.exp2(value))
        flags[0, 0] = pl.simt.isinf(value)

    fp16_ir = str(_parse_tile_function(fp16_math, [([1, 1], pl.DT_FP16), ([1, 1], pl.DT_FP16)]))
    bf16_ir = str(
        _parse_tile_function(
            bf16_math,
            [([1, 1], pl.DT_BF16), ([1, 1], pl.DT_BF16), ([1, 1], pl.DT_BOOL)],
        )
    )
    assert "simt.fma(" in fp16_ir
    assert "simt.sin(" in fp16_ir
    assert "simt.tanh(" in bf16_ir
    assert "simt.isinf(" in bf16_ir


def test_scalar_math_supports_int64_abs_and_integer_min_max():
    @pl.simt.function
    def integer_math(
        src,
        dst,
    ):
        dst[0, 0] = pl.simt.abs(src[0, 0])
        dst[0, 1] = pl.simt.max(pl.simt.min(src[0, 0], src[0, 1]), src[0, 0])

    function_ir = str(
        _parse_tile_function(integer_math, [([1, 2], pl.DT_INT64), ([1, 2], pl.DT_INT64)])
    )
    assert "simt.abs(" in function_ir
    assert "simt.min(" in function_ir
    assert "simt.max(" in function_ir


def test_scalar_math_rejects_ordinary_function():
    with pytest.raises(ParserSyntaxError, match="can only be used inside a SIMT function"):

        @pl.function(type=pl.FunctionType.Orchestration)
        def unsupported(value: pl.DT_FP32) -> pl.DT_FP32:
            return pl.simt.exp(value)


def test_scalar_math_rejects_unsupported_dtype_and_mixed_operands():
    @pl.simt.function
    def unsupported_log1p(value):
        value[0, 0] = pl.simt.log1p(value[0, 0])

    with pytest.raises(ParserTypeError, match="supports only fp32"):
        _parse_tile_function(unsupported_log1p, [([1, 1], pl.DT_FP16)])

    @pl.simt.function
    def unsupported_exp(value):
        value[0, 0] = pl.simt.exp(value[0, 0])

    with pytest.raises(ParserTypeError, match="supports only fp16, bfloat16, fp32"):
        _parse_tile_function(unsupported_exp, [([1, 1], pl.DT_INT32)])

    @pl.simt.function
    def mixed_min(lhs, rhs):
        lhs[0, 0] = pl.simt.min(lhs[0, 0], rhs[0, 0])

    with pytest.raises(ParserTypeError, match="same dtype"):
        _parse_tile_function(mixed_min, [([1, 1], pl.DT_FP16), ([1, 1], pl.DT_FP32)])


def test_scalar_math_rejects_tile_operand():
    @pl.simt.function
    def unsupported(value):
        _ = pl.simt.sqrt(value)

    with pytest.raises(ParserTypeError, match="must be a scalar expression"):
        _parse_tile_function(unsupported, [([1, 1], pl.DT_FP32)])


def test_scalar_math_rejects_wrong_arity_and_keywords():
    with pytest.raises(ParserSyntaxError, match="requires exactly 3 positional arguments"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def missing_addend(value: pl.DT_FP32) -> pl.DT_FP32:
            return pl.simt.fma(value, value)

    with pytest.raises(ParserSyntaxError, match="requires exactly 1 positional argument"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def keyword_argument(value: pl.DT_FP32) -> pl.DT_FP32:
            return pl.simt.exp(value=value)
