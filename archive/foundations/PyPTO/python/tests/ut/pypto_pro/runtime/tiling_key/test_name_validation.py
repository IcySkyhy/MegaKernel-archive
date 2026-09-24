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
"""TilingKey field-name conflict tests."""

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest

GLOBAL_KEY_NAME = 1


class TkNormal:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])


class TkConflictParam:
    x = TilingKeyField(bits=1, values=[0, 1])
    OpType = TilingKeyField(bits=1, values=[0, 1])


class TkConflictGlobal:
    GLOBAL_KEY_NAME = TilingKeyField(bits=1, values=[0, 1])
    OpType = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(auto_mutex=True, tiling_key=TkNormal)
def kernel_normal(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000, size=16384)
    tile_b = pl.make_tile(tile_type, addr=0x4000, size=16384)
    tile_c = pl.make_tile(tile_type, addr=0x8000, size=16384)

    with pl.section_vector():
        for i in pl.range(0, m, 64):
            for j in pl.range(0, n, 128):
                pl.load(tile_a, x, [i, j])
                pl.load(tile_b, y, [i, j])
                if OpType == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                elif OpType == 1:  # noqa: F821
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)
                pl.store(z, tile_c, [i, j])


def test_normal_no_conflict():
    kernel_normal[None, 1, {"OpType": 0}]


def test_param_name_conflict_raises():
    with pytest.raises(ValueError, match="kernel parameter"):
        @pl.jit(auto_mutex=True, tiling_key=TkConflictParam)
        def kernel_param_conflict(
            x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
            y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
            z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
        ):
            pl.store(z, x, [0, 0])


def test_global_name_conflict_raises():
    with pytest.raises(ValueError, match="module-level variable"):
        @pl.jit(auto_mutex=True, tiling_key=TkConflictGlobal)
        def kernel_global_conflict(
            x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
            y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
            z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
        ):
            pl.store(z, x, [0, 0])
