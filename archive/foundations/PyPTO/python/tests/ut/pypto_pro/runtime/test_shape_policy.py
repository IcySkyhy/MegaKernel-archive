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
"""Unit tests for normalized PyPTO Pro tensor shape policies."""

from pypto_pro.language.typing import DYNAMIC, STATIC
from pypto_pro.language.typing.shape import _ShapePolicy
from pypto_pro.runtime.shape_policy import KernelSignatureSpec, TensorShapeSpec
import pytest


def _spec(name, index, shape):
    return TensorShapeSpec.from_annotation(name, index, shape)


def test_mixed_shape_policy_binding():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC, STATIC, 128]),))

    bound = signature.bind_static_shapes({"x": [32, 64, 128]})
    dims = bound.tensor("x").dimensions

    assert dims[0].dynamic_name == "__pypto_dyn_x_0"
    assert dims[0].value is None
    assert dims[1].value == 64
    assert dims[1].is_static
    assert dims[2].value == 128
    assert not dims[2].is_static
    assert bound.static_signature == ((0, 1, 64),)


def test_bound_shape_lowers_dynamic_static_and_fixed_dimensions():
    from pypto.pypto_impl import ir

    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC, STATIC, 128]),))
    tensor_shape = signature.bind_static_shapes({"x": [32, 64, 128]}).tensor("x")

    ir_shape = tensor_shape.to_ir_shape(ir.Span.unknown())

    assert isinstance(ir_shape[0], ir.Var)
    assert ir_shape[0].name == "__pypto_dyn_x_0"
    assert isinstance(ir_shape[1], ir.ConstInt) and ir_shape[1].value == 64
    assert isinstance(ir_shape[2], ir.ConstInt) and ir_shape[2].value == 128


def test_empty_shape_is_exact_rank_zero():
    signature = KernelSignatureSpec((_spec("x", 0, []),))

    assert signature.bind_static_shapes({"x": []}).tensor("x").dimensions == ()
    with pytest.raises(ValueError, match=r"parameter 'x'.*rank"):
        signature.bind_static_shapes({"x": [1]})


def test_ellipsis_only_is_arbitrary_rank_with_static_specialization():
    signature = KernelSignatureSpec((_spec("x", 0, [...]),))

    bound = signature.bind_static_shapes({"x": [3, 5]})

    assert [dim.value for dim in bound.tensor("x").dimensions] == [3, 5]
    assert bound.static_signature == ((0, 0, 3), (0, 1, 5))


def test_dynamic_axes_are_independent_between_parameters():
    signature = KernelSignatureSpec(
        (
            _spec("x", 0, [DYNAMIC]),
            _spec("out", 1, [DYNAMIC]),
        )
    )

    bound = signature.bind_static_shapes({"x": [16], "out": [32]})

    assert bound.tensor("x").dimensions[0].dynamic_name == "__pypto_dyn_x_0"
    assert bound.tensor("out").dimensions[0].dynamic_name == "__pypto_dyn_out_0"


def test_final_ellipsis_expands_to_static_dimensions():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC, ...]),))

    bound = signature.bind_static_shapes({"x": [8, 16, 32]})

    assert [dim.value for dim in bound.tensor("x").dimensions] == [None, 16, 32]
    assert bound.static_signature == ((0, 1, 16), (0, 2, 32))


def test_static_signature_uses_parameter_then_axis_order():
    signature = KernelSignatureSpec(
        (
            _spec("lhs", 0, [STATIC, 4, STATIC]),
            _spec("rhs", 2, [STATIC]),
        )
    )

    bound = signature.bind_static_shapes({"rhs": [9], "lhs": [3, 4, 5]})

    assert bound.static_signature == ((0, 0, 3), (0, 2, 5), (2, 0, 9))


@pytest.mark.parametrize("bad_dim", [True, False, 0, -1, 1.5, "32", object()])
def test_invalid_fixed_dimension_is_rejected(bad_dim):
    with pytest.raises(TypeError, match=r"parameter 'x'.*axis 0"):
        _spec("x", 0, [bad_dim])


@pytest.mark.parametrize("shape", [[..., STATIC], [DYNAMIC, ..., ...]])
def test_invalid_ellipsis_placement_is_rejected(shape):
    with pytest.raises(TypeError, match="ellipsis"):
        _spec("x", 0, shape)


def test_fixed_dimension_mismatch_is_rejected():
    signature = KernelSignatureSpec((_spec("x", 0, [16, DYNAMIC]),))

    with pytest.raises(ValueError, match=r"parameter 'x'.*axis 0.*expected 16.*got 8"):
        signature.bind_static_shapes({"x": [8, 4]})


def test_rank_mismatch_is_rejected():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC, STATIC]),))

    with pytest.raises(ValueError, match=r"parameter 'x'.*rank"):
        signature.bind_static_shapes({"x": [8]})


def test_non_positive_runtime_dimension_is_rejected():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC]),))

    with pytest.raises(ValueError, match=r"parameter 'x'.*axis 0.*positive"):
        signature.bind_static_shapes({"x": [0]})


def test_binary_binding_requires_static_shape_source():
    signature = KernelSignatureSpec((_spec("x", 0, [STATIC]),))

    with pytest.raises(ValueError, match=r"static_shapes.*x"):
        signature.bind_static_shapes(None)


def test_binary_binding_allows_dynamic_only_without_shapes():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC, 128]),))

    bound = signature.bind_static_shapes(None)

    assert bound.static_signature == ()
    assert bound.tensor("x").dimensions[0].dynamic_name == "__pypto_dyn_x_0"


def test_binary_binding_rejects_unknown_shape_entry():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC]),))

    with pytest.raises(ValueError, match=r"unknown.*y"):
        signature.bind_static_shapes({"y": [8]})


def test_static_shapes_requires_mapping_type():
    signature = KernelSignatureSpec((_spec("x", 0, [DYNAMIC]),))

    with pytest.raises(TypeError, match="mapping"):
        signature.bind_static_shapes([("x", [8])])


def test_public_language_exports_shape_policies_without_frontend_var():
    import pypto_pro.language as pl
    import pypto_pro.language.typing as language_typing

    assert pl.DYNAMIC is DYNAMIC
    assert pl.STATIC is STATIC
    assert isinstance(pl.DYNAMIC, _ShapePolicy)
    assert isinstance(pl.STATIC, _ShapePolicy)
    assert not hasattr(pl, "DYN")
    assert not hasattr(pl, "StatusType")
    assert not hasattr(pl, "Var")
    assert not hasattr(language_typing, "Var")
