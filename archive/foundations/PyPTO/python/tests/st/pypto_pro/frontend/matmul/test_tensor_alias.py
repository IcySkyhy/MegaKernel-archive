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

"""Tensor alias assignment on Ascend A5.

Regression: tensor-to-tensor assignment (B = A) used to crash CCE codegen
with "tensor 'B' has no base pointer". This test uses a, tensor_b, tensor_c
simultaneously to verify multi-level alias propagation works end-to-end.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")


@pl.jit(auto_mutex=True)
def call_kernel_tensor_alias(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        tensor_b = a
        tensor_c = tensor_b
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.load(cur_a, tensor_b, [0, 0])
        pl.move(al, cur_a)
        pl.matmul(ac, al, br)
        pl.load(cur_a, tensor_c, [0, 0])
        pl.move(al, cur_a)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_tensor_alias():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([64, 64], device=device, dtype=torch.float16)
    b = torch.eye(64, device=device, dtype=torch.float16)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)
    call_kernel_tensor_alias(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("test_load_tensor_alias passed")


TILE = 64


@pl.jit(auto_mutex=True)
def call_kernel_tile_alias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        tensor_alias = a
        tile_alias = cur_a
        pl.load(tile_alias, tensor_alias, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, tile_alias)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_tile_alias():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([64, 64], device=device, dtype=torch.float16)
    b = torch.eye(64, device=device, dtype=torch.float16)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)
    call_kernel_tile_alias(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("test_load_tile_alias passed")


@pl.jit(auto_mutex=True)
def call_kernel_tile_alias_rebind(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        tile_alias = cur_a
        cur_a = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(tile_alias, a, [0, 0])
        pl.load(cur_a, b, [0, 0])
        pl.move(al, tile_alias)
        pl.move(br, cur_a)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_alias_after_source_rebind():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([TILE, TILE], device=device, dtype=torch.float16)
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_tile_alias_rebind(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("test_tile_alias_after_source_rebind passed")


logging.basicConfig(level=logging.INFO, format="%(message)s")
