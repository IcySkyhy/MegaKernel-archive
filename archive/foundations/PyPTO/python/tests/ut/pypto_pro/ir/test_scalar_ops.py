# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for scalar Make helper functions and their dtype promotion."""

from typing import cast

from pypto_pro import DataType, ir
import pytest


def _var(name: str, dtype: DataType) -> ir.Var:
    return ir.Var(name, ir.ScalarType(dtype), ir.Span.unknown())


def _bool(value: bool = True) -> ir.ConstBool:
    return ir.ConstBool(value, ir.Span.unknown())


def _assert_dtype(expr: ir.Expr, dtype: DataType) -> None:
    assert isinstance(expr.type, ir.ScalarType)
    assert expr.type.dtype == dtype


def _assert_cast_dtype(expr: ir.Expr, dtype: DataType) -> None:
    assert isinstance(expr, ir.Cast)
    _assert_dtype(expr, dtype)


class TestScalarMakeHelpers:
    """Tests for ir.min_ and ir.max_."""

    @classmethod
    def test_min_creation(cls):
        """Test ir.min_ creates a Min expression with type promotion."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        result = ir.min_(x, y, span)

        assert isinstance(result, ir.Min)
        assert cast(ir.Var, result.left).name == "x"
        assert cast(ir.Var, result.right).name == "y"

    @classmethod
    def test_max_creation(cls):
        """Test ir.max_ creates a Max expression with type promotion."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        result = ir.max_(x, y, span)

        assert isinstance(result, ir.Max)
        assert cast(ir.Var, result.left).name == "x"
        assert cast(ir.Var, result.right).name == "y"

    @classmethod
    def test_min_type_promotion(cls):
        """Test that min_ promotes operand types (e.g. INT32 + INT64 -> INT64)."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

        result = ir.min_(x, y, span)

        assert isinstance(result, ir.Min)
        assert result.type == ir.ScalarType(DataType.INT64)


class TestNumericPromotion:
    """Tests for Python-style scalar numeric promotion."""

    @pytest.mark.parametrize(
        ("make_op", "op_type"),
        [
            (ir.add, ir.Add),
            (ir.sub, ir.Sub),
            (ir.mul, ir.Mul),
            (ir.floordiv, ir.FloorDiv),
            (ir.mod, ir.FloorMod),
            (ir.pow, ir.Pow),
            (ir.min_, ir.Min),
            (ir.max_, ir.Max),
        ],
    )
    def test_bool_operands_promote_to_int64(self, make_op, op_type):
        """BOOL operands are normalized to the default integer dtype."""
        result = make_op(_bool(), _bool(False))

        assert isinstance(result, op_type)
        _assert_dtype(result, DataType.INT64)
        _assert_cast_dtype(result.left, DataType.INT64)
        _assert_cast_dtype(result.right, DataType.INT64)

    @pytest.mark.parametrize(
        ("make_op", "op_type"),
        [
            (ir.add, ir.Add),
            (ir.sub, ir.Sub),
            (ir.mul, ir.Mul),
            (ir.min_, ir.Min),
            (ir.max_, ir.Max),
        ],
    )
    @pytest.mark.parametrize("int_dtype", [DataType.INT32, DataType.INDEX])
    @pytest.mark.parametrize("bool_on_left", [True, False])
    def test_bool_and_int_promote_to_common_dtype(self, make_op, op_type, int_dtype, bool_on_left):
        """A bool first becomes INT64, then same-width promotion keeps the left dtype."""
        bool_operand = _bool()
        int_operand = _var("x", int_dtype)
        result = make_op(bool_operand, int_operand) if bool_on_left else make_op(int_operand, bool_operand)

        assert isinstance(result, op_type)
        expected_dtype = DataType.INDEX if int_dtype == DataType.INDEX and not bool_on_left else DataType.INT64
        _assert_dtype(result, expected_dtype)
        promoted_bool = result.left if bool_on_left else result.right
        promoted_int = result.right if bool_on_left else result.left
        _assert_cast_dtype(promoted_bool, expected_dtype)
        if int_dtype == DataType.INDEX:
            assert promoted_int is int_operand
        else:
            _assert_cast_dtype(promoted_int, expected_dtype)

    @pytest.mark.parametrize(
        ("make_op", "op_type"),
        [
            (ir.add, ir.Add),
            (ir.sub, ir.Sub),
            (ir.mul, ir.Mul),
            (ir.min_, ir.Min),
            (ir.max_, ir.Max),
        ],
    )
    def test_bool_and_float_promote_to_fp32(self, make_op, op_type):
        """Mixed bool/float operands are both promoted to FP32."""
        right = _var("x", DataType.FP16)
        result = make_op(_bool(), right)

        assert isinstance(result, op_type)
        _assert_dtype(result, DataType.FP32)
        _assert_cast_dtype(result.left, DataType.FP32)
        _assert_cast_dtype(result.right, DataType.FP32)

    @pytest.mark.parametrize(
        ("make_op", "op_type"),
        [
            (ir.add, ir.Add),
            (ir.sub, ir.Sub),
            (ir.mul, ir.Mul),
            (ir.min_, ir.Min),
            (ir.max_, ir.Max),
        ],
    )
    def test_mixed_int_float_promotes_to_fp32(self, make_op, op_type):
        """Mixed integer/float operands are both promoted to FP32."""
        left = _var("x", DataType.INT64)
        right = _var("y", DataType.FP16)
        result = make_op(left, right)

        assert isinstance(result, op_type)
        _assert_dtype(result, DataType.FP32)
        _assert_cast_dtype(result.left, DataType.FP32)
        _assert_cast_dtype(result.right, DataType.FP32)

    def test_float_operands_promote_to_wider_dtype(self):
        """Two float operands retain the wider floating dtype."""
        left = _var("x", DataType.FP16)
        right = _var("y", DataType.FP32)
        result = ir.add(left, right)

        _assert_dtype(result, DataType.FP32)
        _assert_cast_dtype(result.left, DataType.FP32)
        assert result.right is right

    @pytest.mark.parametrize("constant_on_left", [True, False])
    def test_equal_width_index_int64_keeps_left_dtype(self, constant_on_left):
        """Equal-width INDEX/INT64 promotion returns the left dtype without a redundant Cast."""
        span = ir.Span.unknown()
        constant = ir.ConstInt(2, DataType.INDEX, span)
        variable = _var("x", DataType.INT64)
        result = ir.add(constant, variable) if constant_on_left else ir.add(variable, constant)

        promoted_constant = result.left if constant_on_left else result.right
        expected_dtype = DataType.INDEX if constant_on_left else DataType.INT64
        _assert_dtype(result, expected_dtype)
        assert promoted_constant is constant
        _assert_dtype(promoted_constant, DataType.INDEX)

    @pytest.mark.parametrize("make_compare", [ir.eq, ir.ne, ir.lt, ir.le, ir.gt, ir.ge])
    def test_comparison_accepts_bool_as_numeric(self, make_compare):
        """Comparisons accept bool/int pairs and return BOOL."""
        result = make_compare(_bool(), _var("x", DataType.INT32))

        _assert_dtype(result, DataType.BOOL)
        _assert_cast_dtype(result.left, DataType.INT64)
        _assert_cast_dtype(result.right, DataType.INT64)

    @pytest.mark.parametrize("make_compare", [ir.eq, ir.ne, ir.lt, ir.le, ir.gt, ir.ge])
    def test_mixed_int_float_comparison_promotes_operands_to_fp32(self, make_compare):
        """Mixed-category comparisons use FP32 operands and return BOOL."""
        result = make_compare(_var("x", DataType.INT64), _var("y", DataType.FP16))

        _assert_dtype(result, DataType.BOOL)
        _assert_cast_dtype(result.left, DataType.FP32)
        _assert_cast_dtype(result.right, DataType.FP32)

    @pytest.mark.parametrize("make_op", [ir.floordiv, ir.mod, ir.pow])
    def test_other_numeric_ops_promote_bool_and_int_to_int64(self, make_op):
        """Floor division, modulo, and power use the same bool promotion."""
        result = make_op(_bool(), _var("x", DataType.INT32))

        _assert_dtype(result, DataType.INT64)
        _assert_cast_dtype(result.left, DataType.INT64)
        _assert_cast_dtype(result.right, DataType.INT64)

    @pytest.mark.parametrize(
        ("make_op", "op_type"),
        [
            (ir.floordiv, ir.FloorDiv),
            (ir.mod, ir.FloorMod),
            (ir.pow, ir.Pow),
        ],
    )
    def test_floor_div_mod_and_pow_mixed_int_float_promote_to_fp32(self, make_op, op_type):
        """All numeric binary helpers promote mixed categories to FP32."""
        right = _var("y", DataType.FP32)
        result = make_op(_var("x", DataType.INT32), right)

        assert isinstance(result, op_type)
        _assert_dtype(result, DataType.FP32)
        _assert_cast_dtype(result.left, DataType.FP32)
        assert result.right is right


class TestTrueDivisionPromotion:
    """Tests for Python true-division result types."""

    @pytest.mark.parametrize(
        ("left_dtype", "right_dtype"),
        [
            (DataType.INT32, DataType.INT64),
            (DataType.BOOL, DataType.BOOL),
            (DataType.BOOL, DataType.INT16),
        ],
    )
    def test_integer_like_operands_promote_to_fp32(self, left_dtype, right_dtype):
        """True division of integer-like operands casts both sides to FP32."""
        left = _bool() if left_dtype == DataType.BOOL else _var("x", left_dtype)
        right = _bool(False) if right_dtype == DataType.BOOL else _var("y", right_dtype)
        result = ir.truediv(left, right)

        assert isinstance(result, ir.FloatDiv)
        _assert_dtype(result, DataType.FP32)
        _assert_dtype(result.left, DataType.FP32)
        _assert_dtype(result.right, DataType.FP32)

    def test_mixed_int_float_promotes_to_fp32(self):
        """True division with mixed categories promotes both operands to FP32."""
        left = _var("x", DataType.INT64)
        right = _var("y", DataType.FP16)
        result = ir.truediv(left, right)

        _assert_dtype(result, DataType.FP32)
        _assert_cast_dtype(result.left, DataType.FP32)
        _assert_cast_dtype(result.right, DataType.FP32)

    def test_float_operands_promote_to_wider_dtype(self):
        """True division of two floats uses the wider float dtype."""
        result = ir.truediv(_var("x", DataType.FP16), _var("y", DataType.FP32))

        _assert_dtype(result, DataType.FP32)
        _assert_cast_dtype(result.left, DataType.FP32)


class TestBitwiseAndShiftPromotion:
    """Tests for Python-style bool behavior in bitwise and shift operations."""

    @pytest.mark.parametrize(
        "make_op",
        [ir.bit_and, ir.bit_or, ir.bit_xor, ir.bit_shift_left, ir.bit_shift_right],
    )
    def test_integer_operands_promote_to_wider_dtype(self, make_op):
        """Integer-only helpers promote both operands to one wider integer dtype."""
        right = _var("y", DataType.INT64)
        result = make_op(_var("x", DataType.INT32), right)

        _assert_dtype(result, DataType.INT64)
        _assert_cast_dtype(result.left, DataType.INT64)
        assert result.right is right

    @pytest.mark.parametrize(
        ("make_op", "op_type"),
        [
            (ir.bit_and, ir.BitAnd),
            (ir.bit_or, ir.BitOr),
            (ir.bit_xor, ir.BitXor),
        ],
    )
    def test_bool_bool_bitwise_promotes_to_int64(self, make_op, op_type):
        """Bitwise operations treat both BOOL operands as default integers."""
        result = make_op(_bool(), _bool(False))

        assert isinstance(result, op_type)
        _assert_dtype(result, DataType.INT64)
        _assert_cast_dtype(result.left, DataType.INT64)
        _assert_cast_dtype(result.right, DataType.INT64)

    @pytest.mark.parametrize("make_op", [ir.bit_and, ir.bit_or, ir.bit_xor])
    def test_bool_int_bitwise_returns_integer(self, make_op):
        """A bool/int bitwise operation promotes both operands to an integer dtype."""
        result = make_op(_bool(), _var("x", DataType.INT32))

        _assert_dtype(result, DataType.INT64)
        _assert_cast_dtype(result.left, DataType.INT64)
        _assert_cast_dtype(result.right, DataType.INT64)

    @pytest.mark.parametrize("make_op", [ir.bit_shift_left, ir.bit_shift_right])
    def test_shift_promotes_bool_operands_to_int64(self, make_op):
        """Shifts use the same BOOL-to-default-integer promotion."""
        result = make_op(_bool(), _bool(False))

        _assert_dtype(result, DataType.INT64)
        _assert_cast_dtype(result.left, DataType.INT64)
        _assert_cast_dtype(result.right, DataType.INT64)

    @pytest.mark.parametrize("make_op", [ir.bit_and, ir.bit_or, ir.bit_xor, ir.bit_shift_left])
    def test_bitwise_rejects_float(self, make_op):
        """Float operands remain invalid for bitwise and shift operations."""
        with pytest.raises(ValueError, match="requires integer dtype"):
            make_op(_bool(), _var("x", DataType.FP32))


class TestUnaryResultDtype:
    """Tests for unary operand validation and result dtype selection."""

    def test_pos_is_not_public_ir_helper(self):
        assert not hasattr(ir, "pos")

    def test_neg_bool_returns_int64(self):
        operand = _bool()
        result = ir.neg(operand)

        assert isinstance(result, ir.Neg)
        _assert_dtype(result, DataType.INT64)
        assert result.operand is operand
        _assert_dtype(result.operand, DataType.BOOL)

    def test_bit_not_bool_returns_int64(self):
        operand = _bool()
        result = ir.bit_not(operand)

        assert isinstance(result, ir.BitNot)
        _assert_dtype(result, DataType.INT64)
        assert result.operand is operand
        _assert_dtype(result.operand, DataType.BOOL)

    def test_not_bool_keeps_operand_and_returns_bool(self):
        operand = _bool()
        result = ir.not_(operand)

        assert isinstance(result, ir.Not)
        assert result.operand is operand
        _assert_dtype(result, DataType.BOOL)

    @pytest.mark.parametrize(
        ("make_op", "op_type", "dtype"),
        [
            (ir.neg, ir.Neg, DataType.INT32),
            (ir.neg, ir.Neg, DataType.FP16),
            (ir.bit_not, ir.BitNot, DataType.INT32),
        ],
    )
    def test_unary_numeric_preserves_operand_and_dtype(self, make_op, op_type, dtype):
        operand = _var("x", dtype)
        result = make_op(operand)

        assert isinstance(result, op_type)
        assert result.operand is operand
        _assert_dtype(result, dtype)

    @pytest.mark.parametrize("make_op", [ir.neg, ir.not_, ir.bit_not])
    def test_unary_rejects_non_scalar_operand(self, make_op):
        span = ir.Span.unknown()
        shape = [ir.ConstInt(1, DataType.INT32, span)]
        operand = ir.Var("x", ir.TensorType(shape, DataType.INT32), span)

        with pytest.raises(ValueError, match="Expression must be Var with ScalarType"):
            make_op(operand)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
