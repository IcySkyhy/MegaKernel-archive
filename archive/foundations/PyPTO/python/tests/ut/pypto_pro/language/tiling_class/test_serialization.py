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
"""Tiling class serialization tests for mixed scalar and array layouts.

The cases verify ctypes struct size, field offsets, round-trip values, and
deterministic serialization for a representative complex tiling class.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypto_pro.language.typing._tiling import (
    get_tiling_ctype_struct,
    tiling_instance_to_bytes,
)


@dataclass
class ComplexLayout:
    pad_before: int[60]
    scalar_a: int
    shape_2d: int[2]
    scalar_b: float
    flags: bool[8]
    scalar_c: bool
    pad_after: int[32]


def _make_pad_before(fill_val=0):
    arr = [0] * 60
    for k in range(60):
        arr[k] = fill_val + k
    return arr


def _make_shape_2d(vals):
    return list(vals)


def _make_flags(pattern=None):
    arr = [0] * 8
    if pattern is None:
        for k in range(8):
            arr[k] = bool(k % 2)
    else:
        for k in range(8):
            arr[k] = pattern[k] if k < len(pattern) else False
    return arr


def _make_pad_after(fill_val=0):
    arr = [0] * 32
    for k in range(32):
        arr[k] = fill_val + k
    return arr


def test_struct_size_consistency():
    tiling = ComplexLayout(
        pad_before=_make_pad_before(0),
        scalar_a=42,
        shape_2d=_make_shape_2d([64, 128]),
        scalar_b=3.14,
        flags=_make_flags([True, False, True, False, True, False, True, False]),
        scalar_c=True,
        pad_after=_make_pad_after(100),
    )
    buf = tiling_instance_to_bytes(tiling)
    ctype_struct = get_tiling_ctype_struct(ComplexLayout)
    actual_size = len(buf)

    expected_min = 240 + 4 + 8 + 4 + 8 + 1 + 128
    assert actual_size >= expected_min, (
        f"Struct size {actual_size} < expected minimum {expected_min}"
    )

    decoded = ctype_struct.from_buffer_copy(buf)
    assert decoded.scalar_a == 42
    assert abs(decoded.scalar_b - 3.14) < 1e-5
    assert decoded.scalar_c == 1
    assert list(decoded.shape_2d) == [64, 128]
    assert list(decoded.flags) == [1, 0, 1, 0, 1, 0, 1, 0]
    assert decoded.pad_before[0] == 0
    assert decoded.pad_before[59] == 59
    assert decoded.pad_after[0] == 100
    assert decoded.pad_after[31] == 131


def test_round_trip_all_fields():
    tiling = ComplexLayout(
        pad_before=_make_pad_before(10),
        scalar_a=99,
        shape_2d=_make_shape_2d([512, 1024]),
        scalar_b=2.718,
        flags=_make_flags([True, True, False, False, True, False, True, True]),
        scalar_c=False,
        pad_after=_make_pad_after(200),
    )
    buf = tiling_instance_to_bytes(tiling)
    ctype_struct = get_tiling_ctype_struct(ComplexLayout)
    decoded = ctype_struct.from_buffer_copy(buf)

    assert decoded.scalar_a == 99
    assert abs(decoded.scalar_b - 2.718) < 1e-5
    assert decoded.scalar_c == 0
    assert list(decoded.shape_2d) == [512, 1024]
    assert list(decoded.flags) == [1, 1, 0, 0, 1, 0, 1, 1]
    assert decoded.pad_before[0] == 10
    assert decoded.pad_before[59] == 69
    assert decoded.pad_after[0] == 200
    assert decoded.pad_after[31] == 231

    buf2 = tiling_instance_to_bytes(tiling)
    assert buf == buf2, "Serialization must be deterministic"
