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
"""Tiling class detection and field extraction tests."""

from __future__ import annotations

from dataclasses import dataclass

from pypto_pro import DataType
from pypto_pro.language.typing._tiling import ArrayFieldInfo, ScalarFieldInfo, get_tiling_fields, is_tiling_class
import pytest


@dataclass
class ScalarTiling:
    n: int
    scale: float
    enabled: bool


@dataclass
class MixedTiling:
    n: int
    offsets: int[4]
    scales: float[2]


@dataclass
class InvalidStrField:
    name: str


@dataclass
class InvalidListField:
    items: list


class PlainClass:
    pass


@pytest.mark.parametrize("tiling_cls", [ScalarTiling, MixedTiling])
def test_is_tiling_class_accepts_valid_scalar_and_array_fields(tiling_cls):
    assert is_tiling_class(tiling_cls) is True


@pytest.mark.parametrize("value", [PlainClass, InvalidStrField, InvalidListField, 42, "string", None])
def test_is_tiling_class_rejects_invalid_inputs(value):
    assert is_tiling_class(value) is False


def test_is_tiling_class_rejects_empty_dataclass():
    @dataclass
    class EmptyTiling:
        pass

    assert is_tiling_class(EmptyTiling) is False


def test_get_tiling_fields_maps_scalar_fields_and_preserves_order():
    fields = get_tiling_fields(ScalarTiling)

    assert list(fields.keys()) == ["n", "scale", "enabled"]
    assert fields == {
        "n": ScalarFieldInfo(DataType.INDEX),
        "scale": ScalarFieldInfo(DataType.FP32),
        "enabled": ScalarFieldInfo(DataType.BOOL),
    }


def test_get_tiling_fields_maps_array_fields():
    fields = get_tiling_fields(MixedTiling)

    assert fields["n"] == ScalarFieldInfo(DataType.INDEX)
    assert fields["offsets"] == ArrayFieldInfo(DataType.INDEX, 4)
    assert fields["scales"] == ArrayFieldInfo(DataType.FP32, 2)


def test_duplicate_field_names_are_rejected():
    @dataclass
    class DuplicateField:
        value: int
        value: int

    with pytest.raises(ValueError):
        is_tiling_class(DuplicateField)
