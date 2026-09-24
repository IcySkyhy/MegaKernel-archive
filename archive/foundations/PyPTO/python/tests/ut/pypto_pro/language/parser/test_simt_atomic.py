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

"""Parser tests for PyPTO Pro SIMT atomic operations.

Scalar and Tensor operation-level tests use eager SIMT IR parsing. Tile tests use
delayed ``pl.simt.function`` parsing so their types are inferred from launch
arguments.
"""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest

from pypto.pypto_impl import ir


@pl.simt.function(max_threads=32)
def _atomic_add_tile(
    dst,
    old_values,
    value: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.simt.function(max_threads=32)
def _atomic_add_tensor(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.simt.function
def _atomic_rmw_ops(
    numeric,
    bitwise,
    counter,
    int_value: pl.DT_INT32,
    uint_value: pl.DT_UINT32,
):
    pl.simt.atomic_sub(numeric[0, 0], int_value)
    pl.simt.atomic_exch(numeric[0, 0], int_value)
    pl.simt.atomic_max(numeric[0, 0], int_value)
    pl.simt.atomic_min(numeric[0, 0], int_value)
    pl.simt.atomic_cas(numeric[0, 0], int_value, int_value)
    pl.simt.atomic_and(bitwise[0, 0], uint_value)
    pl.simt.atomic_or(bitwise[0, 0], uint_value)
    pl.simt.atomic_xor(bitwise[0, 0], uint_value)
    pl.simt.atomic_inc(counter[0, 0], uint_value)
    pl.simt.atomic_dec(counter[0, 0], uint_value)


@pl.simt.function(max_threads=1)
def _atomic_rmw_ops_entry(numeric, bitwise, counter, int_value, uint_value):
    _atomic_rmw_ops(numeric, bitwise, counter, int_value, uint_value)


@pl.kernel
def _atomic_add_tile_kernel(value: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=128)
    old_values = pl.make_tile(tile_type, addr=0x0080, size=128)
    with pl.section_vector():
        pl.simt.launch(_atomic_add_tile, threads=32, args=(dst, old_values, value))


@pl.kernel
def _atomic_add_tensor_kernel(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    with pl.section_vector():
        pl.simt.launch(_atomic_add_tensor, threads=32, args=(dst, old_values, value))


@pl.kernel
def _atomic_rmw_ops_kernel(int_value: pl.DT_INT32, uint_value: pl.DT_UINT32):
    int_type = pl.TileType(shape=[1, 1], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    uint_type = pl.TileType(shape=[1, 1], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    numeric = pl.make_tile(int_type, addr=0, size=4)
    bitwise = pl.make_tile(uint_type, addr=32, size=4)
    counter = pl.make_tile(uint_type, addr=64, size=4)
    with pl.section_vector():
        pl.simt.launch(
            _atomic_rmw_ops_entry,
            threads=1,
            args=(numeric, bitwise, counter, int_value, uint_value),
        )


def _parse_one_tile_function(function, dtype, shape=(1, 1)):
    @pl.simt.function(max_threads=1)
    def entry(tile):
        function(tile)

    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=shape, dtype=dtype, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=4096)
        with pl.section_vector():
            pl.simt.launch(entry, threads=1, args=(tile,))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def _parse_two_tile_function(function, first_dtype, second_dtype):
    @pl.simt.function(max_threads=1)
    def entry(first, second):
        function(first, second)

    @pl.kernel
    def kernel():
        first_type = pl.TileType(shape=[1, 1], dtype=first_dtype, target_memory=pl.MemorySpace.Vec)
        second_type = pl.TileType(shape=[1, 1], dtype=second_dtype, target_memory=pl.MemorySpace.Vec)
        first = pl.make_tile(first_type, addr=0, size=4096)
        second = pl.make_tile(second_type, addr=4096, size=4096)
        with pl.section_vector():
            pl.simt.launch(entry, threads=1, args=(first, second))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def _parse_tile_scalar_function(function, tile_dtype, scalar_dtype):
    @pl.simt.function(max_threads=1)
    def entry(tile, value):
        function(tile, value)

    if scalar_dtype == pl.DT_INT32:

        @pl.kernel
        def kernel(value: pl.DT_INT32):
            tile_type = pl.TileType(shape=[1, 1], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0, size=4096)
            with pl.section_vector():
                pl.simt.launch(entry, threads=1, args=(tile, value))

    elif scalar_dtype == pl.DT_UINT32:

        @pl.kernel
        def kernel(value: pl.DT_UINT32):
            tile_type = pl.TileType(shape=[1, 1], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0, size=4096)
            with pl.section_vector():
                pl.simt.launch(entry, threads=1, args=(tile, value))

    elif scalar_dtype == pl.DT_FP32:

        @pl.kernel
        def kernel(value: pl.DT_FP32):
            tile_type = pl.TileType(shape=[1, 1], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0, size=4096)
            with pl.section_vector():
                pl.simt.launch(entry, threads=1, args=(tile, value))

    else:
        raise ValueError(f"Unsupported scalar dtype: {scalar_dtype}")

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def _parse_tile_compare_function(function, tile_dtype):
    @pl.simt.function(max_threads=1)
    def entry(tile, compare, value):
        function(tile, compare, value)

    @pl.kernel
    def kernel(compare: pl.DT_UINT32, value: pl.DT_INT32):
        tile_type = pl.TileType(shape=[1, 1], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=4096)
        with pl.section_vector():
            pl.simt.launch(entry, threads=1, args=(tile, compare, value))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def test_atomic_add_preserves_tile_lvalue_and_returns_old_value():
    program, matched = _atomic_add_tile_kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_add_tile"))

    assert matched
    assert function_ir.count("simt.atomic_add") == 1
    assert "block.getval" not in function_ir
    assert "block.setval" in function_ir
    assert "simt.launch" in str(program)


def test_atomic_add_supports_gm_int64_tensor():
    program, matched = _atomic_add_tensor_kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_add_tensor"))

    assert matched
    assert function_ir.count("simt.atomic_add") == 1
    assert "block.getval" not in function_ir
    assert "block.setval" in function_ir
    assert "simt.launch" in str(program)


def test_atomic_add_contextually_types_numeric_literals():
    @pl.simt.function
    def literal_values(unsigned, signed):
        old = pl.simt.atomic_add(unsigned[0, 0], 1)
        signed[0, 0] = pl.simt.atomic_add(signed[0, 0], -1)
        unsigned[0, 0] = old

    function_ir = str(_parse_two_tile_function(literal_values, pl.DT_UINT32, pl.DT_INT32))
    assert function_ir.count("simt.atomic_add") == 2


def test_atomic_add_requires_direct_subscript_target():
    @pl.simt.function
    def scalar_alias(dst, value: pl.DT_INT32):
        current = dst[0, 0]
        pl.simt.atomic_add(current, value)

    with pytest.raises(ParserSyntaxError, match="direct Tile or Tensor subscript"):
        _parse_tile_scalar_function(scalar_alias, pl.DT_INT32, pl.DT_INT32)


def test_atomic_add_rejects_slice_target():
    @pl.simt.function
    def slice_target(dst, value: pl.DT_INT32):
        pl.simt.atomic_add(dst[0:1, 0:1], value)

    with pytest.raises(ParserSyntaxError, match="does not support slices"):
        _parse_tile_scalar_function(slice_target, pl.DT_INT32, pl.DT_INT32)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_add_max_min_are_void_on_ub_tile(dtype):
    @pl.simt.function
    def supported_tile(dst):
        pl.simt.atomic_add(dst[0, 0], 1.0)
        pl.simt.atomic_max(dst[0, 1], 2.0)
        pl.simt.atomic_min(dst[0, 2], 3.0)

    function = _parse_one_tile_function(supported_tile, dtype, shape=(1, 3))
    function_ir = str(function)
    for op_name in ("atomic_add", "atomic_max", "atomic_min"):
        assert function_ir.count(f"simt.{op_name}") == 1
    atomic_calls = [
        stmt.expr
        for stmt in function.body.stmts
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    assert len(atomic_calls) == 3
    assert all(isinstance(call.type, ir.NoneType) for call in atomic_calls)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_add_max_min_are_void_on_gm_tensor(dtype):
    @pl.function(type=pl.FunctionType.SimtCallee)
    def supported_tensor(dst: pl.Tensor[[1, 3], dtype]):
        pl.simt.atomic_add(dst[0, 0], 1.0)
        pl.simt.atomic_max(dst[0, 1], 2.0)
        pl.simt.atomic_min(dst[0, 2], 3.0)

    function_ir = str(supported_tensor)
    for op_name in ("atomic_add", "atomic_max", "atomic_min"):
        assert function_ir.count(f"simt.{op_name}") == 1
    atomic_calls = [
        stmt.expr
        for stmt in supported_tensor.body.stmts
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    assert len(atomic_calls) == 3
    assert all(isinstance(call.type, ir.NoneType) for call in atomic_calls)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_result_cannot_be_returned(dtype):
    @pl.simt.function
    def add_result(dst) -> None:
        return pl.simt.atomic_add(dst[0, 0], 1.0)

    with pytest.raises(ParserTypeError, match="must return None or one scalar value"):
        _parse_one_tile_function(add_result, dtype)


def test_atomic_add_requires_exact_value_dtype():
    @pl.simt.function
    def mismatched_value(dst, value: pl.DT_UINT32):
        pl.simt.atomic_add(dst[0, 0], value)

    with pytest.raises(ParserTypeError, match="operand 0 dtype must match target dtype"):
        _parse_tile_scalar_function(mismatched_value, pl.DT_INT32, pl.DT_UINT32)


def test_atomic_add_rejects_float_literal_for_integer_target():
    @pl.simt.function
    def float_literal(dst):
        pl.simt.atomic_add(dst[0, 0], 1.0)

    with pytest.raises(ParserTypeError, match="requires an integer value"):
        _parse_one_tile_function(float_literal, pl.DT_INT32)


def test_atomic_rmw_interfaces_preserve_lvalue_and_build_distinct_ir_ops():
    program, _ = _atomic_rmw_ops_kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_rmw_ops"))

    for op_name in (
        "atomic_sub",
        "atomic_exch",
        "atomic_max",
        "atomic_min",
        "atomic_cas",
        "atomic_and",
        "atomic_or",
        "atomic_xor",
        "atomic_inc",
        "atomic_dec",
    ):
        assert function_ir.count(f"simt.{op_name}") == 1
    assert "block.getval" not in function_ir


def test_atomic_cas_contextually_types_compare_and_value_literals():
    @pl.simt.function
    def literal_operands(dst):
        old = pl.simt.atomic_cas(dst[0, 0], 0, 1)
        dst[0, 0] = old

    assert "simt.atomic_cas" in str(_parse_one_tile_function(literal_operands, pl.DT_UINT32))


def test_atomic_cas_requires_exact_compare_dtype():
    @pl.simt.function
    def mismatched_compare(
        dst,
        compare: pl.DT_UINT32,
        value: pl.DT_INT32,
    ):
        pl.simt.atomic_cas(dst[0, 0], compare, value)

    with pytest.raises(ParserTypeError, match="operand 0 dtype must match target dtype"):
        _parse_tile_compare_function(mismatched_compare, pl.DT_INT32)


def test_atomic_bitwise_rejects_fp32_target():
    @pl.simt.function
    def float_bitwise(dst, value: pl.DT_FP32):
        pl.simt.atomic_or(dst[0, 0], value)

    with pytest.raises(ParserTypeError, match="atomic_or.*does not support dtype.*UB Tile"):
        _parse_tile_scalar_function(float_bitwise, pl.DT_FP32, pl.DT_FP32)


def test_atomic_counter_rejects_signed_target():
    @pl.simt.function
    def signed_counter(dst, limit: pl.DT_INT32):
        pl.simt.atomic_inc(dst[0, 0], limit)

    with pytest.raises(ParserTypeError, match="atomic_inc.*does not support dtype.*UB Tile"):
        _parse_tile_scalar_function(signed_counter, pl.DT_INT32, pl.DT_INT32)


def test_atomic_counter_supports_gm_uint64():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def uint64_counter(dst: pl.Tensor[[1], pl.DT_UINT64], limit: pl.DT_UINT64):
        pl.simt.atomic_dec(dst[0], limit)

    assert "simt.atomic_dec" in str(uint64_counter)
