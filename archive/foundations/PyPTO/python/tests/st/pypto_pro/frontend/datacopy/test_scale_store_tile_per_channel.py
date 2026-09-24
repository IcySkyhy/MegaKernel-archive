# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P3: store_tile full scenarios + per-channel expansion tests (excluding NZ format).

This test file covers:
1. store_tile basic scenarios
2. store_tile multi-tile offsets
3. store_tile fusion scenarios (ReLU, phase)
4. per-channel store expansion (per-column [1, N] scales: varying, alternating, extreme, zero, negative)
5. per-channel move expansion
6. per-channel store_tile

NOTE: per-channel scale tensor must be [1, N] (row == 1). [N, 1] per-row scaling is
NOT supported by the hardware FixPipe deqTensor (see analysis in gravel/quant_design).
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


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


def _make_scale_tensor(device: str, scale_values: list, shape: tuple = (1, 64)) -> torch.Tensor:
    """Create INT64 scale tensor for per-channel quantization."""
    scale_bits_list = []
    for scale_value in scale_values:
        scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]
        scale_bits |= 1 << 46  # signed flag for INT8 output
        scale_bits_list.append(scale_bits)

    scale_tensor = torch.tensor(scale_bits_list, dtype=torch.int64, device=device)
    return scale_tensor.reshape(shape)


# ============================================================================
# Test 1: store_tile basic scenarios
# ============================================================================


@pl.jit()
def store_tile_basic_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

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

        # store_tile with tile offsets [0, 0]
        pl.store_tile(quant_out, acc, [0, 0], scale=scale_value)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_basic():
    """Test store_tile basic scenario."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    store_tile_basic_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_store_tile_basic passed.")


# ============================================================================
# Test 2: store_tile multi-tile offsets
# ============================================================================


@pl.jit()
def store_tile_offset_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
    tile_row: pl.DT_INT32,
    tile_col: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        # Load specific tile
        pl.load(q_mat, q, [tile_row * 64, 0])
        pl.load(k_mat, k, [0, tile_col * 64])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # store_tile with specified tile offsets
        pl.store_tile(quant_out, acc, [tile_row, tile_col], scale=scale_value)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pytest.mark.parametrize("tile_row,tile_col", [(0, 1), (1, 0), (1, 1)], ids=["0_1", "1_0", "1_1"])
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_offset(tile_row, tile_col):
    """store_tile with different tile offsets (single K=64 block product)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn((128, 128), dtype=torch.float32, device=device)
    k = torch.randn((128, 128), dtype=torch.float32, device=device)
    quant_out = torch.zeros((128, 128), device=device, dtype=torch.int8)
    scale_value = 0.1
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    store_tile_offset_kernel(q, k, quant_out, scale_bits, tile_row, tile_col)
    torch.npu.synchronize()

    # kernel loads q[row block][0:64] x k[0:64][col block] (single K=64 block)
    q_tile = q[tile_row * 64:tile_row * 64 + 64, 0:64]
    k_tile = k[0:64, tile_col * 64:tile_col * 64 + 64]
    raw_ref = torch.matmul(q_tile, k_tile)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    # Check only the [tile_row, tile_col] tile region
    actual_tile = quant_out[tile_row * 64:tile_row * 64 + 64, tile_col * 64:tile_col * 64 + 64]
    # matmul precision w/ random data, scale=0.1
    torch.testing.assert_close(actual_tile.to(torch.int32), expected.to(torch.int32), rtol=0, atol=4)
    logging.info("test_store_tile_offset[%d_%d] passed.", tile_row, tile_col)


@pl.jit()
def store_tile_relu_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

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

        # store_tile with ReLU
        pl.store_tile(quant_out, acc, [0, 0], scale=scale_value, relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_relu():
    """Test store_tile + ReLU fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    store_tile_relu_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    relu_ref = torch.relu(raw_ref)
    expected = torch.clamp(torch.round(relu_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_store_tile_relu passed.")


# ============================================================================
# Test 4: per-channel store expansion
# ============================================================================


@pl.jit()
def per_channel_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
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

        pl.store(quant_out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
def _per_channel_scale_values(pattern: str) -> list:
    """Generate a per-column [1, 64] scale pattern."""
    if pattern == "varying":
        return [0.5 + i * 0.5 for i in range(64)]
    if pattern == "alternating":
        return [1.0 if i % 2 == 0 else 2.0 for i in range(64)]
    if pattern == "extreme":
        return [1e-6 if i % 2 == 0 else 1e6 for i in range(64)]
    if pattern == "zero":
        return [0.0 if i % 2 == 0 else 2.0 for i in range(64)]
    if pattern == "negative":
        return [-1.0 - i * 0.1 for i in range(64)]
    raise ValueError(f"unknown pattern {pattern}")


@pytest.mark.soc("950")
@pytest.mark.parametrize("pattern", ["varying", "alternating", "extreme", "zero", "negative"])
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_scale_pattern(pattern):
    """Per-channel (scale=Tile) store with different per-column scale patterns."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    scale_values = _per_channel_scale_values(pattern)
    fp_params = _make_scale_tensor(device, scale_values)

    per_channel_store_kernel(q, k, fp_params, quant_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    scale_tensor_fp32 = torch.tensor(scale_values, dtype=torch.float32, device=device)
    scaled_ref = raw_ref * scale_tensor_fp32.unsqueeze(0)
    expected = torch.clamp(torch.round(scaled_ref), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_per_channel_scale_pattern[%s] passed.", pattern)


@pl.jit()
def per_channel_move_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    move_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
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

        # per-channel move with scale
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
def test_per_channel_move_varying_scale():
    """Test per-channel move with varying scale values."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    move_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    # Varying scale: [0.5, 1.0, 1.5, 2.0, ...]
    scale_values = [0.5 + i * 0.5 for i in range(64)]
    fp_params = _make_scale_tensor(device, scale_values)

    per_channel_move_kernel(q, k, fp_params, move_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    scale_tensor_fp32 = torch.tensor(scale_values, dtype=torch.float32, device=device)
    scaled_ref = raw_ref * scale_tensor_fp32.unsqueeze(0)
    expected = torch.clamp(torch.round(scaled_ref), -128, 127).to(torch.int8)

    torch.testing.assert_close(move_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_per_channel_move_varying_scale passed.")


# ============================================================================
# Test 6: per-channel store_tile
# ============================================================================


@pl.jit()
def per_channel_store_tile_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
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

        # per-channel store_tile
        pl.store_tile(quant_out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_store_tile():
    """Test per-channel store_tile basic scenario."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    # Uniform scale: all 2.0
    scale_values = [2.0] * 64
    fp_params = _make_scale_tensor(device, scale_values)

    per_channel_store_tile_kernel(q, k, fp_params, quant_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    scale_tensor_fp32 = torch.tensor(scale_values, dtype=torch.float32, device=device)
    scaled_ref = raw_ref * scale_tensor_fp32.unsqueeze(0)
    expected = torch.clamp(torch.round(scaled_ref), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_per_channel_store_tile passed.")


if __name__ == "__main__":
    # store_tile basic scenarios
    test_store_tile_basic()

    # store_tile multi-tile offsets
    for tile_row, tile_col in [(0, 1), (1, 0), (1, 1)]:
        test_store_tile_offset(tile_row, tile_col)

    # store_tile fusion scenarios
    test_store_tile_relu()

    # per-channel store expansion
    for pattern in ["varying", "alternating", "extreme", "zero", "negative"]:
        test_per_channel_scale_pattern(pattern)

    # per-channel move expansion
    test_per_channel_move_varying_scale()

    # per-channel store_tile
    test_per_channel_store_tile()

    logging.info("\nAll P3 tests passed!")
