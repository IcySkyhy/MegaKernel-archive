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
"""Unit tests for tiling parameter support in the PyPTO language DSL."""

from __future__ import annotations

from dataclasses import dataclass

from pypto_pro.language.typing._tiling import (
    ArrayFieldInfo,
    ScalarFieldInfo,
    get_tiling_fields,
    is_tiling_class,
)
import pytest

from pypto.pypto_impl.ir import DataType


def test_array_type_creation():
    @dataclass
    class TilingData:
        offsets: int[4]

    assert is_tiling_class(TilingData) is True
    fields = get_tiling_fields(TilingData)
    assert isinstance(fields["offsets"], ArrayFieldInfo)
    assert fields["offsets"].dtype == DataType.INDEX
    assert fields["offsets"].size == 4


def test_array_type_float():
    @dataclass
    class TilingData:
        scales: float[2]

    fields = get_tiling_fields(TilingData)
    assert isinstance(fields["scales"], ArrayFieldInfo)
    assert fields["scales"].dtype == DataType.FP32
    assert fields["scales"].size == 2


def test_array_type_bool():
    @dataclass
    class TilingData:
        flags: bool[3]

    fields = get_tiling_fields(TilingData)
    assert isinstance(fields["flags"], ArrayFieldInfo)
    assert fields["flags"].dtype == DataType.BOOL
    assert fields["flags"].size == 3


def test_array_type_invalid_dtype_raises():
    @dataclass
    class TilingData:
        invalid: str[4]

    assert is_tiling_class(TilingData) is False
    with pytest.raises(ValueError, match="invalid"):
        get_tiling_fields(TilingData)


def test_array_type_zero_size_raises():
    @dataclass
    class TilingData:
        invalid: int[0]

    assert is_tiling_class(TilingData) is False
    with pytest.raises(ValueError, match="invalid"):
        get_tiling_fields(TilingData)


def test_array_type_negative_size_raises():
    @dataclass
    class TilingData:
        invalid: int[-1]

    assert is_tiling_class(TilingData) is False
    with pytest.raises(ValueError, match="invalid"):
        get_tiling_fields(TilingData)


def test_array_bool_as_size_raises():
    @dataclass
    class TilingData:
        invalid: int[True]

    assert is_tiling_class(TilingData) is False
    with pytest.raises(ValueError, match="invalid"):
        get_tiling_fields(TilingData)


def test_is_tiling_class_with_array_field():
    @dataclass
    class TilingData:
        offsets: int[4]

    assert is_tiling_class(TilingData) is True


def test_is_tiling_class_mixed_scalar_and_array():
    @dataclass
    class TilingData:
        n: int
        offsets: int[3]
        scale: float

    assert is_tiling_class(TilingData) is True


def test_get_tiling_fields_returns_array_field_info():
    @dataclass
    class TilingData:
        offsets: int[3]

    fields = get_tiling_fields(TilingData)
    assert "offsets" in fields
    info = fields["offsets"]
    assert isinstance(info, ArrayFieldInfo)
    assert info.dtype == DataType.INDEX
    assert info.size == 3


def test_get_tiling_fields_mixed():
    @dataclass
    class TilingData:
        n: int
        offsets: float[2]

    fields = get_tiling_fields(TilingData)
    assert isinstance(fields["n"], ScalarFieldInfo)
    assert fields["n"].dtype == DataType.INDEX
    assert isinstance(fields["offsets"], ArrayFieldInfo)
    assert fields["offsets"].dtype == DataType.FP32
    assert fields["offsets"].size == 2
