# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Advanced per-channel quantization tests for the unified ``scale`` parameter.

Tests cover:
1. Dynamic scale Tensor (runtime kernel parameter)
2. Multiple per-channel stores in the same kernel (address allocation)
3. Per-channel move with dynamic scale
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
FP_SCALE_SIGNED_INT8_FLAG = 1 << 46
FP_SCALE_BITS = FP_SCALE_SIGNED_INT8_FLAG | 0x40000000


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


def _make_fp_params(device: str, scale_value: float = FP_SCALE_VALUE) -> torch.Tensor:
    scale_bits = FP_SCALE_SIGNED_INT8_FLAG | struct.unpack("!I", struct.pack("!f", scale_value))[0]
    return torch.full((1, 64), scale_bits, device=device, dtype=torch.int64)


# ============================================================================
# Test 1: Dynamic scale Tensor (runtime kernel parameter)
# ============================================================================


@pl.jit()
def dynamic_scale_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Per-channel store with a user-prepared Scaling tile (runtime kernel parameter data)."""
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)
        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # user-owned data flow: move scale into Scaling tile, sync MTE1->FIX
        pl.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_scale_per_channel_store():
    """Validate per-channel store with dynamic (runtime) scale tensor parameter."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device, scale_value=FP_SCALE_VALUE)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    dynamic_scale_store_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_dynamic_scale_per_channel_store passed.")


# ============================================================================
# Test 2: Multiple per-channel stores in the same kernel
# ============================================================================


@pl.jit()
def multi_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params_1: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    fp_params_2: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out_1: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    out_2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Two per-channel stores with different scale tiles in the same kernel."""
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat_1 = pl.make_tile(fp_mat_type, addr=0x8000, size=512)
        fp_mat_2 = pl.make_tile(fp_mat_type, addr=0x9000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile_1 = pl.make_tile(fp_type, addr=0x0000, size=512)
        fp_tile_2 = pl.make_tile(fp_type, addr=0x0200, size=512)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.load(fp_mat_1, fp_params_1, [0, 0])
        pl.load(fp_mat_2, fp_params_2, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # user-owned data flow: move both scales into Scaling tiles, sync MTE1->FIX
        pl.move(fp_tile_1, fp_mat_1)
        pl.move(fp_tile_2, fp_mat_2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        # First per-channel store with fp_tile_1
        pl.store(out_1, acc, [0, 0], scale=fp_tile_1)

        # Second per-channel store with fp_tile_2 (different scale, same acc)
        pl.store(out_2, acc, [0, 0], scale=fp_tile_2)

        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_multiple_per_channel_stores():
    """Validate multiple per-channel stores with different scales in the same kernel."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)

    scale_value_1 = 2.0
    scale_value_2 = 0.5
    fp_params_1 = _make_fp_params(device, scale_value=scale_value_1)
    fp_params_2 = _make_fp_params(device, scale_value=scale_value_2)

    out_1 = torch.zeros((64, 64), device=device, dtype=torch.int8)
    out_2 = torch.zeros((64, 64), device=device, dtype=torch.int8)

    multi_store_kernel(q, k, fp_params_1, fp_params_2, out_1, out_2)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_1 = torch.clamp(torch.round(raw_ref * scale_value_1), -128, 127).to(torch.int8)
    expected_2 = torch.clamp(torch.round(raw_ref * scale_value_2), -128, 127).to(torch.int8)

    torch.testing.assert_close(out_1.to(torch.int32), expected_1.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_2.to(torch.int32), expected_2.to(torch.int32), rtol=0, atol=0)
    logging.info("test_multiple_per_channel_stores passed.")


# ============================================================================
# Test 3: Per-channel move with dynamic scale
# ============================================================================


@pl.jit()
def dynamic_scale_move_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    move_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Per-channel move (Acc->Vec) with a user-prepared Scaling tile, then store to GM."""
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # user-owned data flow: move scale into Scaling tile, sync MTE1->FIX
        pl.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        # Per-channel move with user-prepared Scaling tile
        pl.move(vec_tile, acc, scale=fp_tile, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(move_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_scale_per_channel_move():
    """Validate per-channel move (Acc->Vec) with dynamic scale tensor."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device, scale_value=FP_SCALE_VALUE)
    move_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    dynamic_scale_move_kernel(q, k, fp_params, move_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(move_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_dynamic_scale_per_channel_move passed.")


# ============================================================================
# Test 4: make_tile_group Mat tiles + per-channel store (R3 group registration)
# ============================================================================


@pl.jit()
def per_channel_store_tile_group_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Per-channel store where user Mat tiles come from make_tile_group.

    The group occupies 0x0000..0xC000 (three 16KB tiles). The user's Mat
    intermediate tile for the scale data must be placed above the group range
    (0xD000) so the group tiles are never overwritten.
    """
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        qk_group = pl.make_tile_group(type=mat_type, addrs=[0x0000, 0x4000, 0x8000], mutex_ids=[0, 1, 2])

        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0xD000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)

        q_mat = qk_group.next()
        k_mat = qk_group.next()
        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # user-owned data flow: move scale into Scaling tile, sync MTE1->FIX
        pl.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_store_tile_group():
    """Validate per-channel store when user Mat tiles use make_tile_group.

    The user's Mat intermediate tile must be placed above the group's address
    range; output must still match the per-channel golden exactly.
    """
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device, scale_value=FP_SCALE_VALUE)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    per_channel_store_tile_group_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_per_channel_store_tile_group passed.")


# ============================================================================
# Test 5: user-prepared Scaling tile (scale=Tile) — no auto-allocation
# ============================================================================


@pl.jit()
def per_channel_store_user_tile_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Per-channel store with a user-prepared Scaling tile (scale=Tile).

    The user owns the whole deqTensor data flow: build a Mat intermediate and a
    Scaling tile, load -> move -> sync(MTE1->FIX), then pass the tile as scale.
    The framework reuses it directly (no auto-allocation of tiles/events).
    """
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)
        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)
        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)
        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # user-owned data flow: move scale into Scaling tile, sync MTE1->FIX
        pl.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_store_user_tile():
    """Validate per-channel store with a user-prepared Scaling tile (scale=Tile)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device, scale_value=FP_SCALE_VALUE)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    per_channel_store_user_tile_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_per_channel_store_user_tile passed.")


if __name__ == "__main__":
    test_dynamic_scale_per_channel_store()
    test_multiple_per_channel_stores()
    test_dynamic_scale_per_channel_move()
    test_per_channel_store_tile_group()
    test_per_channel_store_user_tile()
    logging.info("\nAll tests passed!")
