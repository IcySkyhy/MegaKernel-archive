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
"""TilingKeyField value-count capacity tests."""

from pypto_pro.runtime.tilingkey import TilingKeyField, TilingKeySchema
import pytest


@pytest.mark.parametrize(
    "bits,values",
    [
        (1, [0, 2, 4]),
        (2, [0, 2, 4, 6, 8]),
    ],
)
def test_values_exceed_bit_capacity(bits, values):
    class Tk:
        OpType = TilingKeyField(bits=bits, values=values)

    with pytest.raises(ValueError, match="candidates"):
        TilingKeySchema(Tk)


@pytest.mark.parametrize(
    "bits,values",
    [
        (1, [0, 2]),
        (2, [0, 4, 16, 64]),
    ],
)
def test_non_contiguous_values_are_allowed(bits, values):
    class Tk:
        OpType = TilingKeyField(bits=bits, values=values)

    TilingKeySchema(Tk)
