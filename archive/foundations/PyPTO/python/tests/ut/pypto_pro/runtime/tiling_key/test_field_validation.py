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
"""TilingKeyField construction validation tests."""

from pypto_pro.runtime.tilingkey import TilingKeyField, TilingKeySchema
import pytest


def test_bits_zero_raises():
    class TkZero:
        OpType = TilingKeyField(bits=0, values=[0])

    with pytest.raises(ValueError, match="bits > 0"):
        TilingKeySchema(TkZero)


def test_values_empty_raises():
    class TkEmpty:
        OpType = TilingKeyField(bits=2, values=[])

    with pytest.raises(ValueError, match="non-empty"):
        TilingKeySchema(TkEmpty)


def test_values_non_int_raises():
    class TkNonInt:
        OpType = TilingKeyField(bits=2, values=["a"])

    with pytest.raises(ValueError, match="values must be ints"):
        TilingKeySchema(TkNonInt)


def test_valid_construction():
    tk = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    assert tk.bits == 2
    assert tk.values == (0, 1, 2, 3)
