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
"""CCE codegen tests for pl.select (TSEL / TSELS).

pto-isa intrinsics (include/pto/common/pto_instr.hpp):
- TSEL(dst, mask, src0, src1, tmp)        : dst = src0 if mask else src1
- TSELS(dst, mask, src, tmp, scalar)      : dst = src  if mask else scalar

pl.select with a scalar rhs follows the pto-isa TSELS form and is CCE-only (the PTO backend's
pto.tsels is a different, incompatible op, so it raises a clear error).
"""

import logging

import pypto_pro.language as pl


def _compile_to_cce(kernel_def) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel_def, "a5", "")
    return _assemble_cv_source(cube, vector).content


@pl.kernel
def _sel_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    lhs = pl.make_tile(tt, addr=0x0000, size=16384)
    rhs = pl.make_tile(tt, addr=0x4000, size=16384)
    tmp = pl.make_tile(tt, addr=0x8000, size=16384)
    out = pl.make_tile(tt, addr=0xC000, size=16384)
    mask_t = pl.TileType(shape=[64, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    mask = pl.make_tile(mask_t, addr=0x10000, size=4096)
    pl.load(lhs, x, [0, 0])
    pl.select(out, mask, lhs, rhs, tmp)


@pl.kernel
def _sels_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tt, addr=0x0000, size=16384)
    tmp = pl.make_tile(tt, addr=0x4000, size=16384)
    out = pl.make_tile(tt, addr=0x8000, size=16384)
    mask_t = pl.TileType(shape=[64, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    mask = pl.make_tile(mask_t, addr=0xC000, size=4096)
    pl.load(src, x, [0, 0])
    pl.select(out, mask, src, 0.0, tmp)


@pl.kernel(auto_mutex=True)
def _sel_auto_mutex_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    mask_tt = pl.TileType(shape=[64, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    out_group = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[0])
    mask_group = pl.make_tile_group(type=mask_tt, addrs=0x10000, mutex_ids=[1])
    lhs_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[2])
    rhs_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[3])
    tmp_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[4])
    out = out_group.current()
    mask = mask_group.current()
    lhs = lhs_group.current()
    rhs = rhs_group.current()
    tmp = tmp_group.current()
    pl.select(out, mask, lhs, rhs, tmp)
    pl.store(x, out, [0, 0])


def test_cce_sel_emits_tsel():
    cpp = _compile_to_cce(_sel_kernel)
    logging.info("\n=== test_cce_sel ===\n%s", cpp)
    assert "TSEL(out_0, mask_0, lhs_0, rhs_0, tmp_0);" in cpp


def test_cce_sels_emits_tsels():
    cpp = _compile_to_cce(_sels_kernel)
    logging.info("\n=== test_cce_sels ===\n%s", cpp)
    # TSELS(dst, mask, src, tmp, scalar) — pto-isa form.
    assert "TSELS(out_0, mask_0, src_0, tmp_0, " in cpp, "Expected TSELS(dst, mask, src, tmp, scalar)"
    # The scalar operand (0.0) is the last argument. CCE codegen emits float
    # literals with the C++ "f" suffix (e.g. 0.000000f) for correct float type.
    tsels_line = next(line for line in cpp.splitlines() if "TSELS(" in line)
    assert tsels_line.rstrip().endswith("0.000000f);"), f"scalar should be last arg: {tsels_line}"
