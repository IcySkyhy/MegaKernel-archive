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
"""Tiling class array serialization size checks."""

from __future__ import annotations

from dataclasses import dataclass

from pypto_pro.language.typing._tiling import tiling_instance_to_bytes
import pytest


@dataclass
class TilingSize4:
    arr: int[4]


@dataclass
class TilingSize8:
    arr: int[8]


@pytest.mark.parametrize(
    ("tiling_cls", "values", "expected_size", "actual_size"),
    [
        (TilingSize8, [0, 1, 2], 8, 3),
        (TilingSize8, [0] * 5, 8, 5),
        (TilingSize8, [0], 8, 1),
        (TilingSize8, [0] * 10, 8, 10),
        (TilingSize4, [0, 1], 4, 2),
        (TilingSize4, [], 4, 0),
    ],
)
def test_array_size_mismatch_raises(tiling_cls, values, expected_size, actual_size):
    tiling = tiling_cls(arr=values)
    with pytest.raises(ValueError, match=f"expected {expected_size} elements, got {actual_size}"):
        tiling_instance_to_bytes(tiling)
