# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ruff: noqa: F841

"""Constant-folding coverage for Python scalar expressions."""

import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _assignments(func: ir.Function) -> dict[str, ir.Expr]:
    return {
        stmt.var.name: stmt.value
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt)
    }


def _assert_constant(expr, expected, dtype):
    if dtype == ir.DataType.BOOL:
        expected_type = ir.ConstBool
    elif dtype.is_float():
        expected_type = ir.ConstFloat
    else:
        expected_type = ir.ConstInt
    assert isinstance(expr, expected_type)
    assert expr.type.dtype == dtype
    if dtype.is_float():
        assert expr.value == pytest.approx(expected)
    else:
        assert expr.value == expected


def test_all_binary_and_comparison_operators_fold():
    @pl.function(type=pl.FunctionType.Opaque)
    def folded_ops():
        add = 7 + 3
        sub = 7 - 3
        mul = 7 * 3
        truediv = 7 / 2
        floordiv = 7 // 3
        mod = 7 % 3
        bit_and = 7 & 3
        bit_or = 4 | 3
        bit_xor = 7 ^ 3
        shift_left = 3 << 2
        shift_right = 12 >> 2
        eq = 3 == 3
        ne = 3 != 4
        lt = 3 < 4
        le = 3 <= 3
        gt = 4 > 3
        ge = 4 >= 4

    values = _assignments(folded_ops)
    int_expected = {
        "add": 10,
        "sub": 4,
        "mul": 21,
        "floordiv": 2,
        "mod": 1,
        "bit_and": 3,
        "bit_or": 7,
        "bit_xor": 4,
        "shift_left": 12,
        "shift_right": 3,
    }
    for name, expected in int_expected.items():
        _assert_constant(values[name], expected, ir.DataType.INDEX)
    _assert_constant(values["truediv"], 3.5, ir.DataType.FP32)
    for name in ("eq", "ne", "lt", "le", "gt", "ge"):
        _assert_constant(values[name], True, ir.DataType.BOOL)


def test_mixed_numeric_constants_fold_to_fp32():
    @pl.function(type=pl.FunctionType.Opaque)
    def folded_numeric():
        float_add = 1.25 + 2.5
        mixed_add = pl.const(2, pl.DT_INT32) + 0.5
        mixed_mul = 2 * pl.const(1.5, pl.DT_FP16)
        mixed_compare = pl.const(2, pl.DT_INT32) < 2.5
        typed_float = pl.const(2, pl.DT_FP32)
        typed_bool = pl.const(0, pl.DT_BOOL)

    values = _assignments(folded_numeric)
    _assert_constant(values["float_add"], 3.75, ir.DataType.FP32)
    _assert_constant(values["mixed_add"], 2.5, ir.DataType.FP32)
    _assert_constant(values["mixed_mul"], 3.0, ir.DataType.FP32)
    _assert_constant(values["mixed_compare"], True, ir.DataType.BOOL)
    _assert_constant(values["typed_float"], 2.0, ir.DataType.FP32)
    _assert_constant(values["typed_bool"], False, ir.DataType.BOOL)


def test_bool_numeric_and_unary_result_dtypes():
    @pl.function(type=pl.FunctionType.Opaque)
    def folded_bool_numeric():
        add = True + True
        mixed_add = True + 0.5
        truediv = True / 2
        floordiv = True // True
        mod = True % True
        bit_bool = True & False
        bit_int = True | 2
        shift = True << True
        unary_plus = +True
        unary_minus = -True
        unary_invert = ~True
        unary_not = not 0.0
        float_plus = +1.5
        float_minus = -1.5
        int_not = not 3
        int_invert = ~3
        compare = False < 1

    values = _assignments(folded_bool_numeric)
    _assert_constant(values["add"], 2, ir.DataType.INT64)
    _assert_constant(values["mixed_add"], 1.5, ir.DataType.FP32)
    _assert_constant(values["truediv"], 0.5, ir.DataType.FP32)
    _assert_constant(values["floordiv"], 1, ir.DataType.INT64)
    _assert_constant(values["mod"], 0, ir.DataType.INT64)
    _assert_constant(values["bit_bool"], 0, ir.DataType.INT64)
    _assert_constant(values["bit_int"], 3, ir.DataType.INT64)
    _assert_constant(values["shift"], 2, ir.DataType.INT64)
    _assert_constant(values["unary_plus"], True, ir.DataType.BOOL)
    _assert_constant(values["unary_minus"], -1, ir.DataType.INT64)
    _assert_constant(values["unary_invert"], -2, ir.DataType.INT64)
    _assert_constant(values["unary_not"], True, ir.DataType.BOOL)
    _assert_constant(values["float_plus"], 1.5, ir.DataType.FP32)
    _assert_constant(values["float_minus"], -1.5, ir.DataType.FP32)
    _assert_constant(values["int_not"], False, ir.DataType.BOOL)
    _assert_constant(values["int_invert"], -4, ir.DataType.INDEX)
    _assert_constant(values["compare"], True, ir.DataType.BOOL)


def test_bool_ops_accept_bool_int_and_float_truthiness_but_return_bool():
    @pl.function(type=pl.FunctionType.Opaque)
    def folded_truth():
        int_and = 2 and 0
        int_or = 0 or 3
        float_and = 1.5 and 0.0
        float_or = 0.0 or -2.5
        short_and = False and missing_name  # noqa: F821
        short_or = True or missing_name  # noqa: F821

    values = _assignments(folded_truth)
    _assert_constant(values["int_and"], False, ir.DataType.BOOL)
    _assert_constant(values["int_or"], True, ir.DataType.BOOL)
    _assert_constant(values["float_and"], False, ir.DataType.BOOL)
    _assert_constant(values["float_or"], True, ir.DataType.BOOL)
    _assert_constant(values["short_and"], False, ir.DataType.BOOL)
    _assert_constant(values["short_or"], True, ir.DataType.BOOL)


def test_pl_and_builtin_min_max_fold_with_numeric_promotion():
    @pl.function(type=pl.FunctionType.Opaque)
    def folded_min_max():
        pl_min = pl.min(3, 2)
        pl_max = pl.max(1.5, 2)
        builtin_min = min(True, 2)
        builtin_max = max(pl.const(1, pl.DT_INT32), 2.5)

    values = _assignments(folded_min_max)
    _assert_constant(values["pl_min"], 2, ir.DataType.INDEX)
    _assert_constant(values["pl_max"], 2.0, ir.DataType.FP32)
    _assert_constant(values["builtin_min"], 1, ir.DataType.INT64)
    _assert_constant(values["builtin_max"], 2.5, ir.DataType.FP32)


def test_unsafe_folds_keep_validated_ir_nodes():
    @pl.function(type=pl.FunctionType.Opaque)
    def unsafe_constants():
        zero_div = 1 / 0
        zero_floordiv = 1 // 0
        negative_shift = 1 << -1
        float_floordiv = 3.0 // 2.0
        float_mod = 3.0 % 2.0

    values = _assignments(unsafe_constants)
    assert isinstance(values["zero_div"], ir.FloatDiv)
    assert isinstance(values["zero_floordiv"], ir.FloorDiv)
    assert isinstance(values["negative_shift"], ir.BitShiftLeft)
    _assert_constant(values["float_floordiv"], 1.0, ir.DataType.FP32)
    _assert_constant(values["float_mod"], 1.0, ir.DataType.FP32)


def test_nonconstant_operands_keep_runtime_ir():
    @pl.function(type=pl.FunctionType.Opaque)
    def runtime_ops(value: pl.DT_INT32, flag: pl.DT_BOOL):
        add = value + 1
        minimum = pl.min(value, 2)
        truth = value and 1.0
        positive = +value
        positive_bool = +flag

    values = _assignments(runtime_ops)
    assert isinstance(values["add"], ir.Add)
    assert isinstance(values["minimum"], ir.Min)
    assert isinstance(values["truth"], ir.And)
    assert values["truth"].type.dtype == ir.DataType.BOOL
    assert isinstance(values["positive"], ir.Var)
    assert values["positive"].type.dtype == ir.DataType.INT32
    assert isinstance(values["positive_bool"], ir.Var)
    assert values["positive_bool"].type.dtype == ir.DataType.BOOL


def test_unary_plus_is_identity_for_non_scalar_operand():
    @pl.function(type=pl.FunctionType.Opaque)
    def identity_pos(value: pl.Tensor[[1], pl.DT_FP32]):
        result = +value

    result = _assignments(identity_pos)["result"]
    assert isinstance(result, ir.Var)
    assert isinstance(result.type, ir.TensorType)


def test_invalid_constant_operand_types_raise_from_ir_builders():
    with pytest.raises(pl.parser.ParserError, match="bit_not.*integer dtype"):

        @pl.function(type=pl.FunctionType.Opaque)
        def invalid_invert():
            value = ~1.5

    with pytest.raises(pl.parser.ParserError, match="bit_and.*integer dtype"):

        @pl.function(type=pl.FunctionType.Opaque)
        def invalid_bit_and():
            value = 1.5 & 1
