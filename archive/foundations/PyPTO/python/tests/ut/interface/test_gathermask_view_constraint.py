#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""UT for GatherMask viewshape constraint check."""

import pytest

import pypto
from pypto.error import FeError
from pypto.operation import (
    _gathermask_checks,
    _view_input_ids,
    _view_input_names,
    _view_original_shapes,
)


def _init_tensor(shape, dtype=pypto.DT_FP32, name="input"):
    return pypto.tensor(shape, dtype, name)


def _get_id(t):
    if hasattr(t, 'Id'):
        return t.Id()
    return t._base.Id()


def test_view_records_original_shape():
    a = _init_tensor((8, 32), name="test_input")
    _view_original_shapes.clear()
    _view_input_ids.clear()
    _view_input_names.clear()
    with pypto.function("MAIN", a):
        result = pypto.view(a, [8, 16], [0, 0])
    rid = _get_id(result)
    assert rid in _view_original_shapes
    assert _view_original_shapes[rid] == [8, 16]
    assert rid in _view_input_ids
    assert rid in _view_input_names


def test_view_with_valid_shape_records():
    a = _init_tensor((8, 32), name="vs_input")
    _view_original_shapes.clear()
    _view_input_ids.clear()
    _view_input_names.clear()
    with pypto.function("MAIN", a):
        result = pypto.view(a, [8, 16], valid_shape=[8, 16], offsets=[0, 0])
    rid = _get_id(result)
    assert rid in _view_original_shapes
    assert _view_original_shapes[rid] == [8, 16]


def test_gathermask_records_check():
    a = _init_tensor((8, 32), name="gm_input")
    _view_original_shapes.clear()
    _view_input_ids.clear()
    _view_input_names.clear()
    _gathermask_checks.clear()
    with pypto.function("MAIN", a):
        view_result = pypto.view(a, [8, 32], [0, 0])
        pypto.set_vec_tile_shapes(8, 32)
        pypto.gathermask(view_result, 1)
    assert len(_gathermask_checks) == 1
    check = _gathermask_checks[0]
    assert check["viewshape_last"] == 32
    assert check["pattern_mode"] == 1


def test_gathermask_mode7_no_check():
    a = _init_tensor((8, 32), name="mode7_input")
    _view_original_shapes.clear()
    _view_input_ids.clear()
    _view_input_names.clear()
    _gathermask_checks.clear()
    with pypto.function("MAIN", a):
        view_result = pypto.view(a, [8, 16], [0, 0])
        pypto.set_vec_tile_shapes(8, 16)
        pypto.gathermask(view_result, 7)
    assert len(_gathermask_checks) == 0


def test_gathermask_no_view_no_check():
    a = _init_tensor((8, 32), name="noview_input")
    _view_original_shapes.clear()
    _view_input_ids.clear()
    _view_input_names.clear()
    _gathermask_checks.clear()
    with pypto.function("MAIN", a):
        pypto.set_vec_tile_shapes(8, 32)
        pypto.gathermask(a, 1)
    assert len(_gathermask_checks) == 0


def test_check_gathermask_view_constraint_blocks_violation():
    from pypto.frontend.parser.entry import JitCallableWrapper

    class FakeTensorDef:
        def __init__(self, name, shape):
            self._base = type("FakeBase", (), {"GetName": lambda self: name, "Id": lambda self: 0})()

    class FakeWrapper:
        pass

    _gathermask_checks.clear()
    _gathermask_checks.append({
        "view_tensor_id": 1,
        "input_tensor_id": 0,
        "input_tensor_name": "violation_input",
        "viewshape_last": 16,
        "pattern_mode": 1,
    })

    tensor_defs = [FakeTensorDef("violation_input", (8, 32))]
    torch_tensors = [type("FakeTorch", (), {"shape": [8, 32]})()]

    wrapper = FakeWrapper()
    with pytest.raises(FeError, match="split by view"):
        JitCallableWrapper._check_gathermask_view_constraint(
            wrapper, torch_tensors, tensor_defs
        )


def test_check_gathermask_view_constraint_passes_normal():
    from pypto.frontend.parser.entry import JitCallableWrapper

    class FakeTensorDef:
        def __init__(self, name, shape):
            self._base = type("FakeBase", (), {"GetName": lambda self: name, "Id": lambda self: 0})()

    class FakeWrapper:
        pass

    _gathermask_checks.clear()
    _gathermask_checks.append({
        "view_tensor_id": 1,
        "input_tensor_id": 0,
        "input_tensor_name": "normal_input",
        "viewshape_last": 32,
        "pattern_mode": 1,
    })

    tensor_defs = [FakeTensorDef("normal_input", (8, 32))]
    torch_tensors = [type("FakeTorch", (), {"shape": [8, 32]})()]

    wrapper = FakeWrapper()
    JitCallableWrapper._check_gathermask_view_constraint(
        wrapper, torch_tensors, tensor_defs
    )


def test_check_gathermask_view_constraint_empty_checks():
    from pypto.frontend.parser.entry import JitCallableWrapper

    class FakeWrapper:
        pass

    _gathermask_checks.clear()
    wrapper = FakeWrapper()
    JitCallableWrapper._check_gathermask_view_constraint(wrapper, [], [])
