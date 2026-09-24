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
"""TilingKey launch-time validation tests."""

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest


class TkSingle:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])


class TkMulti:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])
    Mode = TilingKeyField(bits=1, values=[0, 1])


class TkWithValid:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])
    Mode = TilingKeyField(bits=1, values=[0, 1])

    @classmethod
    def is_valid(cls, key):
        return not (key[0] == 0 and key[1] == 1)


def _make_kernel(tiling_key_cls):
    @pl.jit(auto_mutex=True, tiling_key=tiling_key_cls)
    def _kernel(
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

    return _kernel


kernel_single = _make_kernel(TkSingle)
kernel_multi = _make_kernel(TkMulti)
kernel_with_valid = _make_kernel(TkWithValid)


@pytest.mark.parametrize(
    "kernel,key",
    [
        (kernel_single, {"OpType": 0}),
        (kernel_multi, {"OpType": 0, "Mode": 1}),
        (kernel_with_valid, {"OpType": 1, "Mode": 0}),
    ],
)
def test_valid_key_accepted(kernel, key):
    kernel[None, 1, key]


@pytest.mark.parametrize(
    "kernel,key",
    [
        (kernel_single, {"OpType": 99}),
        (kernel_multi, {"OpType": 0}),
        (kernel_single, {}),
        (kernel_single, {"OpType": 0, "ExtraField": 1}),
        (kernel_single, {"OpType": None}),
        (kernel_single, [0]),
        (kernel_with_valid, {"OpType": 0, "Mode": 1}),
    ],
)
def test_invalid_key_rejected(kernel, key):
    with pytest.raises(ValueError):
        kernel[None, 1, key]
