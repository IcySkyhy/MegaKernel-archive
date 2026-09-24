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

"""pl.dump_data 统一调测接口测试 — 覆盖 Tensor/Tile Vec/Tile Acc(L0C) 自动分发场景。

dump_data 根据输入类型自动分发到 dump_tensor（GM Tensor）或 dump_tile（片上 Tile）。
合并自原 test_dump_data / test_dump_tile / test_dump_tile_l0c 三个文件，去重后保留 18 个用例。

控制流覆盖: for循环 / if-else / while循环
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


# =============================================================================
# Section A: Tensor 输入（自动分发到 dump_tensor）
# =============================================================================


@pl.jit()
def dump_data_tensor_full_kernel(
    out: pl.Tensor[[16], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i * 10
        pl.dump_data(out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tensor_full():
    _check_npu()
    logging.info("------------test_dump_data_tensor_full--------------")
    expected = torch.arange(16, device=ST_DEVICE, dtype=torch.int32) * 10
    out = torch.zeros(16, device=ST_DEVICE, dtype=torch.int32)
    dump_data_tensor_full_kernel(out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected)
    logging.info("dump_data_tensor_full passed!")


@pl.jit()
def dump_data_tensor_window_kernel(
    out: pl.Tensor[[32], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 32):
            out[i] = i * 2
        pl.dump_data(out, offsets=[8], shapes=[8])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tensor_window():
    _check_npu()
    logging.info("------------test_dump_data_tensor_window--------------")
    expected = torch.arange(32, device=ST_DEVICE, dtype=torch.int32) * 2
    out = torch.zeros(32, device=ST_DEVICE, dtype=torch.int32)
    dump_data_tensor_window_kernel(out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected)
    logging.info("dump_data_tensor_window passed!")


@pl.jit()
def dump_data_tensor_loc_kernel(
    out: pl.Tensor[[16], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i
        pl.dump_data(out, loc=True)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tensor_loc():
    _check_npu()
    logging.info("------------test_dump_data_tensor_loc--------------")
    expected = torch.arange(16, device=ST_DEVICE, dtype=torch.int32)
    out = torch.zeros(16, device=ST_DEVICE, dtype=torch.int32)
    dump_data_tensor_loc_kernel(out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected)
    logging.info("dump_data_tensor_loc passed!")


# =============================================================================
# Section B: Tile Vec 输入（自动分发到 dump_tile）
# =============================================================================


@pl.jit(auto_mutex=True)
def dump_data_tile_full_kernel(
    a: pl.Tensor[[32, 32], pl.DT_INT32],
    b: pl.Tensor[[32, 32], pl.DT_INT32],
    out: pl.Tensor[[32, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_full():
    _check_npu()
    logging.info("------------test_dump_data_tile_full--------------")
    a = torch.arange(32 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(32, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_tile_full_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("dump_data_tile_full passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_window_kernel(
    a: pl.Tensor[[32, 64], pl.DT_INT32],
    b: pl.Tensor[[32, 64], pl.DT_INT32],
    out: pl.Tensor[[32, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[32, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc, offsets=[4, 0], shapes=[8, 8])
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_window():
    _check_npu()
    logging.info("------------test_dump_data_tile_window--------------")
    a = torch.arange(32 * 64, device=ST_DEVICE, dtype=torch.int32).reshape(32, 64)
    b = a
    out = torch.empty_like(a)
    dump_data_tile_window_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("dump_data_tile_window passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_dynamic_offset_kernel(
    a: pl.Tensor[[32, 32], pl.DT_INT32],
    b: pl.Tensor[[32, 32], pl.DT_INT32],
    out: pl.Tensor[[32, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        for row_off in pl.range(0, 32, 8):
            pl.dump_data(tc, offsets=[row_off, 0], shapes=[8, 8])
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_dynamic_offset():
    _check_npu()
    logging.info("------------test_dump_data_tile_dynamic_offset--------------")
    a = torch.arange(32 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(32, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_tile_dynamic_offset_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("dump_data_tile_dynamic_offset passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_for_loop_kernel(
    a: pl.Tensor[[64, 32], pl.DT_INT32],
    b: pl.Tensor[[64, 32], pl.DT_INT32],
    out: pl.Tensor[[64, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[16, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        for offset in pl.range(0, 64, 16):
            pl.load(ta, a, [offset, 0])
            pl.load(tb, b, [offset, 0])
            pl.add(tc, ta, tb)
            pl.dump_data(tc)
            pl.store(out, tc, [offset, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_for_loop():
    _check_npu()
    logging.info("------------test_dump_data_tile_for_loop--------------")
    a = torch.arange(64 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(64, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_tile_for_loop_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("dump_data_tile_for_loop passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_if_else_kernel(
    a: pl.Tensor[[64, 32], pl.DT_INT32],
    b: pl.Tensor[[64, 32], pl.DT_INT32],
    out: pl.Tensor[[64, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[16, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        for offset in pl.range(0, 64, 16):
            pl.load(ta, a, [offset, 0])
            pl.load(tb, b, [offset, 0])
            pl.add(tc, ta, tb)
            if offset == 0:
                pl.dump_data(tc, offsets=[0, 0], shapes=[8, 16])
            else:
                pl.dump_data(tc)
            pl.store(out, tc, [offset, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_if_else():
    _check_npu()
    logging.info("------------test_dump_data_tile_if_else--------------")
    a = torch.arange(64 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(64, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_tile_if_else_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("dump_data_tile_if_else passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_while_kernel(
    a: pl.Tensor[[16, 32], pl.DT_FP32],
    b: pl.Tensor[[16, 32], pl.DT_FP32],
    out: pl.Tensor[[16, 32], pl.DT_FP32],
):
    tt = pl.TileType(shape=[16, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc)
        x: pl.DT_INT32 = 0
        while x < 3:
            pl.dump_data(tc, offsets=[x, 0], shapes=[4, 8])
            x = x + 1
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_while():
    _check_npu()
    logging.info("------------test_dump_data_tile_while--------------")
    a = torch.full((16, 32), 3.14, device=ST_DEVICE, dtype=torch.float32)
    b = torch.full((16, 32), 1.86, device=ST_DEVICE, dtype=torch.float32)
    out = torch.zeros(16, 32, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_while_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-3, atol=1e-3)
    logging.info("dump_data_tile_while passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_loc_kernel(
    a: pl.Tensor[[32, 32], pl.DT_INT32],
    b: pl.Tensor[[32, 32], pl.DT_INT32],
    out: pl.Tensor[[32, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc, loc=True)
        pl.dump_data(tc, offsets=[0, 0], shapes=[8, 8], loc=True)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_loc():
    _check_npu()
    logging.info("------------test_dump_data_tile_loc--------------")
    a = torch.arange(32 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(32, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_tile_loc_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("dump_data_tile_loc passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_large_fp16_kernel(
    a: pl.Tensor[[256, 320], pl.DT_FP16],
    out: pl.Tensor[[256, 320], pl.DT_FP16],
):
    tt = pl.TileType(shape=[256, 320], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        ta = ta_group.current()
        pl.load(ta, a, [0, 0])
        pl.dump_data(ta)
        pl.store(out, ta, [0, 0])


@pl.jit(auto_mutex=True)
def dump_data_tile_large_partitioned_kernel(
    a: pl.Tensor[[512, 256], pl.DT_INT32],
    b: pl.Tensor[[512, 256], pl.DT_INT32],
    out: pl.Tensor[[512, 256], pl.DT_INT32],
):
    tt = pl.TileType(shape=[128, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x20000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        for row_off in pl.range(0, 512, 128):
            pl.load(ta, a, [row_off, 0])
            pl.load(tb, b, [row_off, 0])
            pl.add(tc, ta, tb)
            if row_off == 0:
                pl.dump_data(tc)
            else:
                pl.dump_data(tc, offsets=[0, 0], shapes=[64, 64])
            pl.store(out, tc, [row_off, 0])


# =============================================================================
# Section C: Tile Acc(L0C) 输入（自动分发到 dump_tile + workspace）
# =============================================================================


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_fp16_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
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
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.dump_data(ac, workspace=workspace)
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8], workspace=workspace)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_acc_fp16():
    _check_npu()
    logging.info("------------test_dump_data_tile_acc_fp16--------------")
    a = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(64, 64, device=ST_DEVICE, dtype=torch.float32)
    workspace = torch.empty(64, 64, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_acc_fp16_kernel(a, b, out, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("dump_data_tile_acc_fp16 passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_large_offset_kernel(
    a: pl.Tensor[[256, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 256], pl.DT_FP16],
    out: pl.Tensor[[256, 256], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
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
        for i in pl.range(0, 256, 64):
            for j in pl.range(0, 256, 64):
                cur_a = a_l1.current()
                cur_b = b_l1.current()
                al = a_l0a.current()
                br = b_l0b.current()
                ac = c_l0c.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.dump_data(ac, offsets=[8, 8], shapes=[32, 32], workspace=workspace)
                pl.store(out, ac, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_acc_large_offset():
    _check_npu()
    logging.info("------------test_dump_data_tile_acc_large_offset--------------")
    a = torch.randn(256, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(64, 256, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(256, 256, device=ST_DEVICE, dtype=torch.float32)
    workspace = torch.empty(64, 64, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_acc_large_offset_kernel(a, b, out, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("dump_data_tile_acc_large_offset passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_bf16_kernel(
    a: pl.Tensor[[64, 64], pl.DT_BF16],
    b: pl.Tensor[[64, 64], pl.DT_BF16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
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
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.dump_data(ac, workspace=workspace)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_acc_bf16():
    _check_npu()
    logging.info("------------test_dump_data_tile_acc_bf16--------------")
    a = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.bfloat16)
    out = torch.zeros(64, 64, device=ST_DEVICE, dtype=torch.float32)
    workspace = torch.empty(64, 64, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_acc_bf16_kernel(a, b, out, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("dump_data_tile_acc_bf16 passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_256x256_kernel(
    a: pl.Tensor[[256, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 256], pl.DT_FP16],
    out: pl.Tensor[[256, 256], pl.DT_FP32],
    workspace: pl.Tensor[[256, 256], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[256, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 256], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x8000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[256, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 256], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[256, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.dump_data(ac, workspace=workspace)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_acc_256x256():
    _check_npu()
    logging.info("------------test_dump_data_tile_acc_256x256--------------")
    a = torch.randn(256, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(64, 256, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(256, 256, device=ST_DEVICE, dtype=torch.float32)
    workspace = torch.empty(256, 256, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_acc_256x256_kernel(a, b, out, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("dump_data_tile_acc_256x256 passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_control_flow_kernel(
    a: pl.Tensor[[256, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 256], pl.DT_FP16],
    out: pl.Tensor[[256, 256], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
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
        for i in pl.range(0, 256, 64):
            for j in pl.range(0, 256, 64):
                cur_a = a_l1.current()
                cur_b = b_l1.current()
                al = a_l0a.current()
                br = b_l0b.current()
                ac = c_l0c.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                if i == 0:
                    pl.dump_data(ac, workspace=workspace)
                elif i == 64:
                    pl.dump_data(ac, offsets=[0, 0], shapes=[32, 32], workspace=workspace)
                else:
                    x: pl.DT_INT32 = 0
                    while x < 2:
                        pl.dump_data(ac, offsets=[x, 0], shapes=[16, 32], workspace=workspace)
                        x = x + 1
                pl.store(out, ac, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_acc_control_flow():
    _check_npu()
    logging.info("------------test_dump_data_tile_acc_control_flow--------------")
    a = torch.randn(256, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(64, 256, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(256, 256, device=ST_DEVICE, dtype=torch.float32)
    workspace = torch.empty(64, 64, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_acc_control_flow_kernel(a, b, out, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("dump_data_tile_acc_control_flow passed!")


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_ptr_workspace_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace_ptr: pl.Ptr[pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
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

    ws = pl.make_tensor(workspace_ptr, [64, 64])

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.dump_data(ac, workspace=ws)
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8], workspace=ws)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tile_acc_ptr_workspace():
    _check_npu()
    logging.info("------------test_dump_data_tile_acc_ptr_workspace--------------")
    a = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(64, 64, device=ST_DEVICE, dtype=torch.float32)
    workspace = torch.empty(64, 64, device=ST_DEVICE, dtype=torch.float32)
    dump_data_tile_acc_ptr_workspace_kernel(a, b, out, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("dump_data_tile_acc_ptr_workspace passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_dump_data_tensor_full,
        test_dump_data_tensor_window,
        test_dump_data_tensor_loc,
        test_dump_data_tile_full,
        test_dump_data_tile_window,
        test_dump_data_tile_dynamic_offset,
        test_dump_data_tile_for_loop,
        test_dump_data_tile_if_else,
        test_dump_data_tile_while,
        test_dump_data_tile_loc,
        test_dump_data_tile_acc_fp16,
        test_dump_data_tile_acc_large_offset,
        test_dump_data_tile_acc_bf16,
        test_dump_data_tile_acc_256x256,
        test_dump_data_tile_acc_control_flow,
        test_dump_data_tile_acc_ptr_workspace,
    ]
    for t in tests:
        t()
        logging.info(f"{t.__name__} passed!")
    logging.info("\nAll pl.dump_data NPU tests passed!")
