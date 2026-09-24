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

import pytest


def test_serialize_list_round_trips():
    from pypto_pro.language.typing._tiling import (
        get_tiling_ctype_struct,
        tiling_instance_to_bytes,
    )

    @dataclass
    class TilingData:
        arr: int[3]
        n: int

    t = TilingData(arr=[10, 20, 30], n=5)
    buf = tiling_instance_to_bytes(t)
    decoded = get_tiling_ctype_struct(TilingData).from_buffer_copy(buf)
    assert list(decoded.arr) == [10, 20, 30]
    assert decoded.n == 5


def test_serialize_raises_type_error_for_scalar_in_array_field():
    from pypto_pro.language.typing._tiling import tiling_instance_to_bytes

    @dataclass
    class TilingData:
        arr: int[3]

    t = TilingData(arr=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected an indexable sequence"):
        tiling_instance_to_bytes(t)


def test_serialize_raises_value_error_for_wrong_size():
    from pypto_pro.language.typing._tiling import tiling_instance_to_bytes

    @dataclass
    class TilingData:
        arr: int[3]

    # The list length is checked while serializing the tiling value.
    t = TilingData(arr=[0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected 3 elements, got 2"):
        tiling_instance_to_bytes(t)
