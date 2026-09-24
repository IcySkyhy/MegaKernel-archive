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
"""TilingKey total bit-width limit tests."""

from pypto_pro.runtime.tilingkey import TilingKeyField, TilingKeySchema
import pytest


def test_total_66_bits_raises():
    class Tk:
        A = TilingKeyField(bits=32, values=[0])
        B = TilingKeyField(bits=32, values=[0])
        C = TilingKeyField(bits=2, values=[0])

    with pytest.raises(ValueError, match="64-bit limit"):
        TilingKeySchema(Tk)


def test_total_64_bits_passes():
    class Tk:
        A = TilingKeyField(bits=32, values=[0])
        B = TilingKeyField(bits=32, values=[0])

    TilingKeySchema(Tk)
