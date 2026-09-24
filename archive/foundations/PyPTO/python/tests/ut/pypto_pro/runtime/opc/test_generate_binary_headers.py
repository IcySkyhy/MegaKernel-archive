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
"""Tests for OPC binary-delivery header generation."""

from dataclasses import dataclass
from pathlib import Path

import pypto_pro.language as pl
from pypto_pro.runtime.opc.pypto_compile import generate_binary_headers
from pypto_pro.runtime.tilingkey import TilingKeyField


@dataclass
class HeaderTiling:
    rows: int
    columns: int


class HeaderTilingKey:
    Operation = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(auto_mutex=True, tiling_key=HeaderTilingKey)
def header_generation_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: HeaderTiling,
):
    tile_type = pl.TileType(
        shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=512)
    tile_y = pl.make_tile(tile_type, addr=0x0200, size=512)
    tile_z = pl.make_tile(tile_type, addr=0x0400, size=512)

    with pl.section_vector():
        for row in pl.range(0, tiling.rows, 16):
            for column in pl.range(0, tiling.columns, 16):
                pl.load(tile_x, x, [row, column])
                pl.load(tile_y, y, [row, column])
                if Operation == 0:  # noqa: F821
                    pl.add(tile_z, tile_x, tile_y)
                else:
                    pl.sub(tile_z, tile_x, tile_y)
                pl.store(z, tile_z, [row, column])


def test_generate_binary_headers_emits_tiling_and_tilingkey_headers():
    binary_dir = Path(generate_binary_headers(header_generation_kernel))
    assert binary_dir.is_dir(), f"binary dir not created: {binary_dir}"

    tiling_header = binary_dir / "HeaderTiling_tiling.h"
    tilingkey_header = binary_dir / "HeaderTilingKey_tilingkey.h"
    assert tiling_header.is_file(), f"missing tiling header: {tiling_header}"
    assert tilingkey_header.is_file(), f"missing tilingkey header: {tilingkey_header}"

    tiling_text = tiling_header.read_text(encoding="utf-8")
    assert "class HeaderTiling" in tiling_text
    assert "int64_t rows;" in tiling_text
    assert "int64_t columns;" in tiling_text

    tilingkey_text = tilingkey_header.read_text(encoding="utf-8")
    assert "ASCENDC_TPL_ARGS_DECL(header_generation_kernel" in tilingkey_text
    assert "ASCENDC_TPL_SEL(" in tilingkey_text
