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
Tests for stitch write-read-write kernel (A@B*(1+scale) + C@D).
"""

from dataclasses import dataclass
import os

import numpy as np
from numpy.testing import assert_allclose
import torch

import pypto

FP32 = pypto.DT_FP32
INT32 = pypto.DT_INT32
INT8 = pypto.DT_INT8


@dataclass
class StitchConfig:
    m: int
    k: int
    n: int
    m_view: int
    n_view: int
    tile_m: list
    tile_k: list
    tile_n: list
    scale: float = 2
    seed: int = 42


@pypto.frontend.jit()
def write_read_write_kernel_2(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    scaled_buf: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    m: int,
    k: int,
    n: int,
    m_view: int,
    n_view: int,
    tile_m: list,
    tile_k: list,
    tile_n: list,
    scale: int,
):
    pypto.set_cube_tile_shapes(tile_m, tile_k, tile_n, enable_split_k=False)
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    pypto.set_vec_tile_shapes(m_view, n_view)

    # Loop1: A@B → out
    for mi in pypto.loop(0, m_loop, 1, name="LOOP_m", idx_name="mi"):
        for ni in pypto.loop(0, n_loop, 1, name="LOOP_n", idx_name="ni"):
            a_view = a[mi * m_view:mi * m_view + m_view, :]
            b_view = b[:, ni * n_view:ni * n_view + n_view]
            result1 = pypto.matmul(a_view, b_view, out_dtype=INT32)
            pypto.atomic_add(result1, [mi * m_view, ni * n_view], out)
            read_tile = out[mi * m_view:mi * m_view + m_view, ni * n_view:ni * n_view + n_view]
            read_view = pypto.view(read_tile, [m_view, n_view], [0, 0])
            scaled_tile = pypto.mul(read_view, scale)
            pypto.assemble(scaled_tile, [mi * m_view, ni * n_view], scaled_buf)

    # Loop3: scaled_buf → atomic_add → out, then C@D → atomic_add → out
    for mi in pypto.loop(0, m_loop, 1, name="LOOP_m", idx_name="mi"):
        for ni in pypto.loop(0, n_loop, 1, name="LOOP_n", idx_name="ni"):
            scaled_tile = scaled_buf[mi * m_view:mi * m_view + m_view, ni * n_view:ni * n_view + n_view]
            scaled_view = pypto.view(scaled_tile, [m_view, n_view], [0, 0])
            pypto.atomic_add(scaled_view, [mi * m_view, ni * n_view], out)
            c_view = c[mi * m_view:mi * m_view + m_view, :]
            d_view = d[:, ni * n_view:ni * n_view + n_view]
            result2 = pypto.matmul(c_view, d_view, out_dtype=INT32)
            pypto.atomic_add(result2, [mi * m_view, ni * n_view], out)


def _run(cfg: StitchConfig):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    np.random.seed(cfg.seed)
    a_cpu = torch.randint(-5, 6, (cfg.m, cfg.k), dtype=torch.int8)
    b_cpu = torch.randint(-5, 6, (cfg.k, cfg.n), dtype=torch.int8)
    c_cpu = torch.randint(-5, 6, (cfg.m, cfg.k), dtype=torch.int8)
    d_cpu = torch.randint(-5, 6, (cfg.k, cfg.n), dtype=torch.int8)

    ab = torch.matmul(a_cpu.int(), b_cpu.int()).numpy()
    cd = torch.matmul(c_cpu.int(), d_cpu.int()).numpy()
    golden = ab + (ab * cfg.scale).astype(np.int32) + cd

    out = torch.zeros(cfg.m, cfg.n, dtype=torch.int32, device=device)
    scaled_buf = torch.zeros(cfg.m, cfg.n, dtype=torch.int32, device=device)
    write_read_write_kernel_2(
        a_cpu.to(device),
        b_cpu.to(device),
        c_cpu.to(device),
        d_cpu.to(device),
        out,
        scaled_buf,
        cfg.m,
        cfg.k,
        cfg.n,
        cfg.m_view,
        cfg.n_view,
        cfg.tile_m,
        cfg.tile_k,
        cfg.tile_n,
        cfg.scale,
    )
    return out.cpu().numpy(), golden


@pypto.options(pass_options={"enable_slice": False})
def test_stitch_outcast_consumer():
    cfg = StitchConfig(128, 256, 128, 128, 128, [64, 64], [64, 128], [128, 128])
    result, golden = _run(cfg)
    assert_allclose(result, golden, atol=0, verbose=True)
