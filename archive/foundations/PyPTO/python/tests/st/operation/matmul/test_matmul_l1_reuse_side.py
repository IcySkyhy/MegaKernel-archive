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
"""
ST for cube_l1_reuse_setting matrix-side preference (issue: cube_l1_reuse_setting 支持指定合并
左矩阵或者右矩阵). cube_l1_reuse_setting value = int merge count, or (count, side) with
side="left"/"right"/"auto"; "left"/"right" are a hard restriction (a side-tagged subgraph only
merges on that matrix, no fall-back).

Which matrix is shareable across subgraphs is controlled by set_cube_tile_shapes (each dim is
[L0_tile, L1_tile]; a dim is "not tiled" when its L1 tile == the full size):
  - share the RIGHT(B) matrix: do NOT tile N -> every m-tile multiplies the full B;
  - share the LEFT(A) matrix : do NOT tile M -> every n-tile multiplies the full A.

Two scenarios (both check the result equals the torch golden -- side is only a perf/partition
knob -- and the merge axis is visible in the L1CopyInReuseMerge log):
  - test_matmul_l1_reuse_side       : 512x128x512, tiling picked per side so only the requested
                                      side is shareable.
  - test_matmul_l1_reuse_side_grid  : 1024x128x1024, M and N both tiled by 128 (8x8 = 64 tiles,
                                      both A and B shareable), l1reuse=4 -> side=left merges
                                      same-m tiles (group by M, "on the LEFT(L0A)"), side=right
                                      merges same-n tiles (group by N, "on the RIGHT(L0B)").

How to confirm the merge axis (grep each run's pass log; DEBUG for per-subgraph detail):
  "L1 reuse: subgraph X merged into subgraph Y on the LEFT(L0A)/RIGHT(L0B) matrix"
  and the summary "L1 reuse matrix-side outcome: N merged on the requested side into M group(s)".
"""

import os
from typing import List, NamedTuple

import pytest
import torch

import pypto


class _Case(NamedTuple):
    """A matmul-side test case: problem shape (m,k,n), per-dim [L0,L1] tiling, and l1reuse count."""

    m: int
    k: int
    n: int
    m_tile: List[int]
    k_tile: List[int]
    n_tile: List[int]
    merge_count: int


def _run_side_matmul(side: str, case: _Case):
    """Build C = A @ B with cube_l1_reuse_setting={-1: (case.merge_count, side)} and case's per-dim
    tiling, run it on device, and assert C matches the torch golden. Shared by the scenarios below
    so the jit/kernel/run boilerplate lives in one place."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    m, k, n = case.m, case.k, case.n

    @pypto.frontend.jit(
        pass_options={
            "cube_l1_reuse_setting": {-1: (case.merge_count, side)},
            "cube_nbuffer_setting": {-1: 1},
        },
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0},
    )
    def matmul_kernel(
        a: pypto.Tensor([m, k], pypto.DT_FP16),
        b: pypto.Tensor([k, n], pypto.DT_FP16),
        out: pypto.Tensor([m, n], pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes(case.m_tile, case.k_tile, case.n_tile)
        out[0:m, 0:n] = pypto.matmul(a, b, pypto.DT_FP16)

    a_cpu = torch.rand([m, k], dtype=torch.float16)
    b_cpu = torch.rand([k, n], dtype=torch.float16)
    golden = torch.matmul(a_cpu.to(torch.float32), b_cpu.to(torch.float32)).to(torch.float16)

    a = a_cpu.to(f"npu:{device_id}")
    b = b_cpu.to(f"npu:{device_id}")
    out = torch.zeros([m, n], dtype=torch.float16, device=f"npu:{device_id}")
    matmul_kernel(a, b, out)

    assert torch.allclose(out.cpu(), golden.cpu(), atol=1e-2, rtol=1e-2), (
        f"cube_l1_reuse_setting side={side!r} produced wrong matmul result"
    )


def _side_tiling_case(side: str) -> _Case:
    # Pick the tiling so only the requested side's matrix is shared (L1 tile == full size means
    # that dim is not tiled): left -> don't tile M (share full A); right/auto -> don't tile N.
    sz = 512
    if side == "left":
        return _Case(sz, 128, sz, [128, sz], [128, 128], [128, 128], 2)
    return _Case(sz, 128, sz, [128, 128], [128, 128], [128, sz], 2)


# 1024x128x1024 with M and N both tiled by 128 -> 8x8 = 64 tiles; both A (same m) and B (same n)
# are shareable. l1reuse=4 -> side=left forms same-m groups, side=right same-n.
_GRID_CASE = _Case(1024, 128, 1024, [128, 128], [128, 128], [128, 128], 4)


@pytest.mark.soc("950", "910")
@pytest.mark.parametrize("side", ["auto", "left", "right"])
def test_matmul_l1_reuse_side(side):
    if side != "left":
        _run_side_matmul(side, _side_tiling_case(side))
        return
    with pypto.options(pass_options={"enable_slice": False}):
        _run_side_matmul(side, _side_tiling_case(side))


@pytest.mark.soc("950", "910")
@pytest.mark.parametrize("side", ["left", "right"])
def test_matmul_l1_reuse_side_grid(side):
    if side != "right":
        _run_side_matmul(side, _GRID_CASE)
        return
    with pypto.options(pass_options={"enable_slice": False}):
        _run_side_matmul(side, _GRID_CASE)


if __name__ == "__main__":
    for s in ("auto", "left", "right"):
        _run_side_matmul(s, _side_tiling_case(s))
    for s in ("left", "right"):
        _run_side_matmul(s, _GRID_CASE)
