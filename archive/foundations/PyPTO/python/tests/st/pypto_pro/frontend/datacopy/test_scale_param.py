# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Comprehensive test suite for the unified ``scale`` parameter in pl.store/pl.move.

Validates the new fixpipe quantization API where ``scale`` replaces the legacy
``pre_quant_scalar`` and ``fp_tile`` parameters.

Scenarios covered:
1. Per-tensor store with float scale (FP32 -> INT8)
2. Per-tensor store with random input (non-deterministic round alignment)
3. Per-tensor move with float scale (Acc -> Vec, FP32 -> INT8)
4. Dynamic (runtime) scale via Expr

Note: ReLU fusion is covered by test_scale_relu_fusion /
test_scale_precision_alignment::test_golden_per_tensor_relu (atol=0).
"""

import logging
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


FP_SCALE_VALUE = 2.0


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


@pl.jit()
def scale_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    raw_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    vm: pl.DT_INT32,
    vn: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        pl.set_validshape(q_mat, [vm, 64])
        pl.set_validshape(q_left, [vm, 64])
        pl.set_validshape(k_mat, [64, vn])
        pl.set_validshape(k_right, [64, vn])
        pl.set_validshape(acc, [vm, vn])

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(raw_out, acc, [0, 0])
        pl.store(quant_out, acc, [0, 0], scale=FP_SCALE_VALUE)
        pl.system.bar_all()


@pl.jit()
def scale_move_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    move_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    vm: pl.DT_INT32,
    vn: pl.DT_INT32,
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

    with pl.section_cube():
        mat_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        pl.set_validshape(q_mat, [vm, 64])
        pl.set_validshape(q_left, [vm, 64])
        pl.set_validshape(k_mat, [64, vn])
        pl.set_validshape(k_right, [64, vn])
        pl.set_validshape(acc, [vm, vn])

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.move(vec_tile, acc, scale=FP_SCALE_VALUE, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(move_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pl.jit()
def scale_dynamic_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT32,
    vm: pl.DT_INT32,
    vn: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        pl.set_validshape(q_mat, [vm, 64])
        pl.set_validshape(q_left, [vm, 64])
        pl.set_validshape(k_mat, [64, vn])
        pl.set_validshape(k_right, [64, vn])
        pl.set_validshape(acc, [vm, vn])

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale_bits)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pytest.mark.parametrize("m,n", [(64, 64), (48, 96), (96, 96)], ids=["full", "row_tail", "dual_tail"])
@pypto.options(pass_options={"enable_slice": False})
def test_scale_store_per_tensor(m, n):
    """编译期 float scale per-tensor store，覆盖完整/行尾/双尾块有效区。"""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    vm, vn = min(m, 64), min(n, 64)
    q = torch.randn(m, n, device=device, dtype=torch.float32)
    k = torch.eye(n, device=device, dtype=torch.float32)
    raw_out = torch.zeros((m, n), device=device, dtype=torch.float32)
    quant_out = torch.zeros((m, n), device=device, dtype=torch.int8)

    scale_store_kernel(q, k, raw_out, quant_out, vm, vn)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_quant = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(raw_out[:vm, :vn], raw_ref[:vm, :vn], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        quant_out[:vm, :vn].to(torch.int32), expected_quant[:vm, :vn].to(torch.int32), rtol=0, atol=1
    )
    logging.info("test_scale_store_per_tensor[%s,%s] passed.", m, n)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_scale_store_random_input():
    """随机输入 per-tensor 量化：golden = clamp(round(raw_ref * scale))。

    与 k=eye 的确定性用例互补：随机数据下 matmul 累加顺序与 torch 不同，
    且在 round 边界（x.5）处设备/torch 可能翻 1，故量化输出用 atol=1，
    原始输出（未量化）用相对容差验证 matmul 本身。
    """

    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    raw_out = torch.zeros((64, 64), device=device, dtype=torch.float32)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    scale_store_kernel(q, k, raw_out, quant_out, 64, 64)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_quant = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    # 随机数据下 matmul 累加误差（vs torch）远小于 1 LSB，但为稳健起见原始输出用相对容差
    torch.testing.assert_close(raw_out, raw_ref, rtol=1e-2, atol=1e-2)
    # 量化输出：round 边界（x.5）处允许 ±1 LSB 翻转
    torch.testing.assert_close(quant_out.to(torch.int32), expected_quant.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_store_random_input passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_scale_move_per_tensor():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    move_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    scale_move_kernel(q, k, move_out, 64, 64)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(move_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_scale_move_per_tensor passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_scale_dynamic():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    scale_bits = struct.unpack("!I", struct.pack("!f", FP_SCALE_VALUE))[0]

    scale_dynamic_kernel(q, k, quant_out, scale_bits, 64, 64)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_scale_dynamic passed.")


if __name__ == "__main__":
    test_scale_store_per_tensor()
    test_scale_store_random_input()
    test_scale_move_per_tensor()
    test_scale_dynamic()
    logging.info("\nAll tests passed!")
