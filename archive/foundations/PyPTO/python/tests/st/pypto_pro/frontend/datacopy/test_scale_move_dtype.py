# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Remaining test cases for scale parameter (45 tests).

Covers:
1. Per-tensor move (11): different dtypes, scales, AccToVecModes
2. Per-tensor store remaining (9): data types, scale values, input patterns, dynamic scale, ReLU/phase/atomic/order
3. Per-channel expansion (7): different scale shapes, dtypes
4. Scale=None baseline (4): no quantization baseline tests
5. Other (4): misc scenarios
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


# ============================================================================
# 1. Per-tensor move (11 tests)
# ============================================================================


@pl.jit()
def move_fp32_to_fp16_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    scale: pl.DT_INT32,
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=8192)

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

        pl.move(vec_tile, acc, scale=scale, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_fp32_to_fp16():
    """FP32 Acc -> FP16 Vec with scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.float16, device=device)

    # Convert float to INT32 bit representation
    scale_bits = struct.unpack("!I", struct.pack("!f", 0.5))[0]
    move_fp32_to_fp16_kernel(q, k, vec_out, scale=scale_bits)
    torch.npu.synchronize()

    assert vec_out.abs().sum() > 0
    logging.info("test_move_fp32_to_fp16 passed.")


@pl.jit()
def move_int32_to_int8_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=4096)
        k_mat = pl.make_tile(mat_type, addr=0x1000, size=4096)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=4096)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=4096)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
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

        pl.move(vec_tile, acc, scale=0.01, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_int32_to_int8():
    """INT32 Acc -> INT8 Vec with scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    move_int32_to_int8_kernel(q, k, vec_out)
    torch.npu.synchronize()

    assert vec_out.abs().sum() > 0
    logging.info("test_move_int32_to_int8 passed.")


# Note: test_move_dual_split_m and test_move_dual_split_n removed
# Hardware limitation: DualMode (SplitM/SplitN) does not support quantization (scale parameter)
# Error: "Quant is not support in dual Dst Mode."


# Note: test_move_dual_split_n removed (see comment above)


@pl.jit()
def move_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT32,
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

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

        pl.move(vec_tile, acc, scale=scale_bits, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


MOVE_SCALE_VALUES = [-2.0, 0.0, 0.5, 1e-6, 1e6]
MOVE_SCALE_IDS = ["negative", "zero", "fraction", "very_small", "very_large"]


@pytest.mark.soc("950")
@pytest.mark.parametrize("scale_value", MOVE_SCALE_VALUES, ids=MOVE_SCALE_IDS)
@pypto.options(pass_options={"enable_slice": False})
def test_move_scale_value(scale_value):
    """Acc->Vec move deqScalar value range (negative/zero/fraction/small/large)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.eye(64, dtype=torch.float32, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.int8, device=device)
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    move_scale_kernel(q, k, vec_out, scale_bits)
    torch.npu.synchronize()

    expected = torch.clamp(torch.round(q * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(vec_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_move_scale_value[%s] passed.", scale_value)


@pl.jit()
def move_relu_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

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

        pl.move(
            vec_tile,
            acc,
            scale=2.0,
            relu_pre_mode=pl.ReluPreMode.NormalRelu,
            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
        )
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_relu():
    """Move with ReLU fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    move_relu_kernel(q, k, vec_out)
    torch.npu.synchronize()

    # All outputs should be >= 0 due to ReLU
    assert (vec_out >= 0).all()
    assert vec_out.abs().sum() > 0
    logging.info("test_move_relu passed.")


@pl.jit()
def move_dynamic_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT32,
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

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

        pl.move(vec_tile, acc, scale=scale_bits, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_dynamic_scale():
    """Move with dynamic scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    move_dynamic_scale_kernel(q, k, vec_out, scale_bits)
    torch.npu.synchronize()

    assert vec_out.abs().sum() > 0
    logging.info("test_move_dynamic_scale passed.")


# ============================================================================
# 2. Per-tensor store remaining (9 tests)
# ============================================================================


@pl.jit()
def store_bf16_to_fp16_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=8192)
        k_mat = pl.make_tile(mat_type, addr=0x2000, size=8192)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=8192)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=8192)

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

        pl.store(out, acc, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_bf16_to_fp16():
    """BF16 matmul -> FP16 output (no scale)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    out = torch.zeros(64, 64, dtype=torch.float16, device=device)

    store_bf16_to_fp16_kernel(q, k, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_store_bf16_to_fp16 passed.")


@pl.jit()
def store_fp16_to_bf16_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=8192)
        k_mat = pl.make_tile(mat_type, addr=0x2000, size=8192)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=8192)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=8192)

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

        pl.store(out, acc, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_fp16_to_bf16():
    """FP16 matmul -> BF16 output (no scale)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float16, device=device)
    k = torch.randn(64, 64, dtype=torch.float16, device=device)
    out = torch.zeros(64, 64, dtype=torch.bfloat16, device=device)

    store_fp16_to_bf16_kernel(q, k, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_store_fp16_to_bf16 passed.")


@pl.jit()
def store_scale_boundary_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
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

        pl.store(out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_scale_boundary():
    """Store with scale=10.0 (moderate scale value)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device) * 0.1
    k = torch.randn(64, 64, dtype=torch.float32, device=device) * 0.1
    out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    # Convert float to INT32 bit representation
    scale_bits = struct.unpack("!I", struct.pack("!f", 10.0))[0]
    store_scale_boundary_kernel(q, k, out, scale=scale_bits)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_store_scale_boundary passed.")


@pl.jit()
def store_data_sparse_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
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

        pl.store(out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_data_sparse():
    """Store with sparse input data (90% zeros)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    # Create sparse data (90% zeros)
    q = torch.zeros(64, 64, dtype=torch.float32, device=device)
    k = torch.zeros(64, 64, dtype=torch.float32, device=device)

    # Properly index 2D tensor by flattening first
    q_flat = q.flatten()
    k_flat = k.flatten()
    q_indices = torch.randperm(64 * 64)[:410]
    k_indices = torch.randperm(64 * 64)[:410]
    q_flat[q_indices] = torch.randn(410, dtype=torch.float32, device=device)
    k_flat[k_indices] = torch.randn(410, dtype=torch.float32, device=device)
    q = q_flat.reshape(64, 64)
    k = k_flat.reshape(64, 64)

    out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    # Convert float to INT32 bit representation
    scale_bits = struct.unpack("!I", struct.pack("!f", 2.0))[0]
    store_data_sparse_kernel(q, k, out, scale=scale_bits)
    torch.npu.synchronize()

    # Most outputs should be zero due to sparse input
    zero_ratio = (out == 0).sum().item() / out.numel()
    assert zero_ratio > 0.5
    logging.info(f"test_store_data_sparse passed: {zero_ratio:.2%} zeros.")


@pl.jit()
def store_data_periodic_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
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

        pl.store(out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_data_periodic():
    """Store with periodic input data (sine wave)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    # Create periodic data (sine wave)
    x = torch.linspace(0, 4 * 3.14159, 64, device=device)
    q = torch.sin(x).unsqueeze(0).repeat(64, 1)
    k = torch.eye(64, dtype=torch.float32, device=device)
    out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    # Convert float to INT32 bit representation
    scale_bits = struct.unpack("!I", struct.pack("!f", 50.0))[0]
    store_data_periodic_kernel(q, k, out, scale=scale_bits)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_store_data_periodic passed.")


@pl.jit()
def store_dynamic_scale_int64_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT64,
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

        pl.store(out, acc, [0, 0], scale=scale_bits)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_dynamic_scale_int64():
    """Store with INT64 dynamic scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]
    # Pass as Python int, not torch tensor
    store_dynamic_scale_int64_kernel(q, k, out, scale_bits)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_store_dynamic_scale_int64 passed.")


@pl.jit()
# Note: test_store_dynamic_scale_from_gm removed
# Current implementation does not support reading scalar values from tensors at runtime
# The scale parameter must be a scalar value passed as a kernel parameter


@pl.jit()
def store_multiple_calls_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out1: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    out2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale1: pl.DT_INT32,
    scale2: pl.DT_INT32,
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

        pl.store(out1, acc, [0, 0], scale=scale1)
        pl.store(out2, acc, [0, 0], scale=scale2)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_multiple_calls():
    """Multiple store calls with different scales."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    out1 = torch.zeros(64, 64, dtype=torch.int8, device=device)
    out2 = torch.zeros(64, 64, dtype=torch.int8, device=device)

    # Convert float to INT32 bit representation
    scale1_bits = struct.unpack("!I", struct.pack("!f", 1.0))[0]
    scale2_bits = struct.unpack("!I", struct.pack("!f", 2.0))[0]
    store_multiple_calls_kernel(q, k, out1, out2, scale1=scale1_bits, scale2=scale2_bits)
    torch.npu.synchronize()

    # Verify different scales produce different results
    assert not torch.equal(out1, out2)
    assert out1.abs().sum() > 0
    assert out2.abs().sum() > 0
    logging.info("test_store_multiple_calls passed.")


# ============================================================================
# 3. Per-channel expansion (7 tests)
# ============================================================================


@pl.jit()
# Note: test_per_channel_row_scale removed
# Hardware limitation: When TileType is Scaling, row must be 1 and col * sizeof(Dtype) must be aligned to 128 bytes
# For INT64 (8 bytes), col must be a multiple of 16 (128/8=16)
# Row-wise scaling with [64, 1] shape is not supported by hardware


@pl.jit()
def per_channel_int32_to_fp16_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=4096)
        k_mat = pl.make_tile(mat_type, addr=0x1000, size=4096)

        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=4096)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=4096)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
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
def test_per_channel_int32_to_fp16():
    """Per-channel INT32 -> FP16 dequantization."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    import torch_npu

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    out = torch.zeros(64, 64, dtype=torch.float16, device=device)

    scale_fp32 = torch.ones(1, 64, dtype=torch.float32, device=device) * 0.01
    fp_params = torch_npu.npu_trans_quant_param(scale_fp32)

    per_channel_int32_to_fp16_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_per_channel_int32_to_fp16 passed.")


@pl.jit()
# Note: test_per_channel_move_row_scale removed
# Hardware limitation: When TileType is Scaling, row must be 1 and col * sizeof(Dtype) must be aligned to 128 bytes
# For INT64 (8 bytes), col must be a multiple of 16 (128/8=16)
# Row-wise scaling with [64, 1] shape is not supported by hardware


@pl.jit()
def per_channel_move_int32_to_fp16_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=8192)

    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=4096)
        k_mat = pl.make_tile(mat_type, addr=0x1000, size=4096)

        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=4096)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=4096)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
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

        pl.move(vec_tile, acc, scale=fp_tile, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_move_int32_to_fp16():
    """Per-channel move INT32 -> FP16 dequantization."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    import torch_npu

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.float16, device=device)

    scale_fp32 = torch.ones(1, 64, dtype=torch.float32, device=device) * 0.01
    fp_params = torch_npu.npu_trans_quant_param(scale_fp32)

    per_channel_move_int32_to_fp16_kernel(q, k, fp_params, vec_out)
    torch.npu.synchronize()

    assert vec_out.abs().sum() > 0
    logging.info("test_per_channel_move_int32_to_fp16 passed.")


@pl.jit()
def per_channel_store_tile_single_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    tile_row: pl.DT_INT32,
    tile_col: pl.DT_INT32,
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

        pl.load(fp_mat, fp_params, [0, 0])
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

        # user-owned data flow: move scale into Scaling tile, sync MTE1->FIX
        pl.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        pl.store_tile(out, acc, [tile_row, tile_col], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_store_tile_offset():
    """Per-channel store_tile with multiple tile offsets, region-level golden check.

    One kernel invocation writes one tile at [tile_row, tile_col] (a single
    K=64 block product, no K accumulation). Two invocations cover [0, 0] and
    [0, 1] so each tile's offset is strictly validated:
      tile [0, 0] -> q[0:64,0:64] @ k[0:64,0:64]  at out[0:64,0:64]
      tile [0, 1] -> q[0:64,0:64] @ k[0:64,64:128] at out[0:64,64:128]
    Per-channel scale is uniformly 2.0 here, so golden = clamp(round(raw * 2.0)).
    """
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    import torch_npu

    q = torch.randn(128, 128, dtype=torch.float32, device=device)
    k = torch.randn(128, 128, dtype=torch.float32, device=device)
    out = torch.zeros(128, 128, dtype=torch.int8, device=device)

    scale_value = 2.0
    scale_fp32 = torch.ones(1, 64, dtype=torch.float32, device=device) * scale_value
    fp_params = torch_npu.npu_trans_quant_param(scale_fp32)

    # Tile [0, 0]: out[0:64, 0:64] = q[0:64,0:64] @ k[0:64,0:64]
    per_channel_store_tile_single_kernel(q, k, fp_params, out, 0, 0)
    torch.npu.synchronize()
    raw_ref_00 = torch.matmul(q[0:64, 0:64].cpu(), k[0:64, 0:64].cpu())
    expected_00 = torch.clamp(torch.round(raw_ref_00 * scale_value), -128, 127)
    torch.testing.assert_close(out[0:64, 0:64].cpu().to(torch.int32), expected_00.to(torch.int32), rtol=0, atol=4)

    # Tile [0, 1]: out[0:64, 64:128] = q[0:64,64:128] @ k[64:128,0:64]
    per_channel_store_tile_single_kernel(q, k, fp_params, out, 0, 1)
    torch.npu.synchronize()
    raw_ref_01 = torch.matmul(q[0:64, 0:64].cpu(), k[0:64, 64:128].cpu())
    expected_01 = torch.clamp(torch.round(raw_ref_01 * scale_value), -128, 127)
    torch.testing.assert_close(out[0:64, 64:128].cpu().to(torch.int32), expected_01.to(torch.int32), rtol=0, atol=4)
    logging.info("test_per_channel_store_tile_offset passed.")

addr_offset = 0x0000
@pl.jit()
def per_channel_store_tile_dynamic_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    with pl.section_cube():
        addr1 = 0x0000 + 16 - 16
        addr2 = addr1 + addr_offset - addr_offset * 1
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=addr2, size=16384)
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

        pl.store_tile(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_per_channel_store_tile_dynamic_scale():
    """Per-channel store_tile with dynamic scale tensor."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    import torch_npu

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_fp32 = torch.ones(1, 64, dtype=torch.float32, device=device) * 2.0
    fp_params = torch_npu.npu_trans_quant_param(scale_fp32)

    per_channel_store_tile_dynamic_scale_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_per_channel_store_tile_dynamic_scale passed.")


# ============================================================================
# 4. Scale=None baseline (4 tests)
# ============================================================================


@pl.jit()
def no_quant_store_fp32_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
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

        pl.store(out, acc, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_no_quant_store_fp32():
    """Baseline: FP32 store without quantization."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    out = torch.zeros(64, 64, dtype=torch.float32, device=device)

    no_quant_store_fp32_kernel(q, k, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_no_quant_store_fp32 passed.")


@pl.jit()
def no_quant_store_int32_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=4096)
        k_mat = pl.make_tile(mat_type, addr=0x1000, size=4096)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=4096)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=4096)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
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

        pl.store(out, acc, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_no_quant_store_int32():
    """Baseline: INT32 store without quantization."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    out = torch.zeros(64, 64, dtype=torch.int32, device=device)

    no_quant_store_int32_kernel(q, k, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_no_quant_store_int32 passed.")


@pl.jit()
def no_quant_move_fp32_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=16384)

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

        pl.move(vec_tile, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(vec_out, vec_tile, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_no_quant_move_fp32():
    """Baseline: FP32 move without quantization."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    vec_out = torch.zeros(64, 64, dtype=torch.float32, device=device)

    no_quant_move_fp32_kernel(q, k, vec_out)
    torch.npu.synchronize()

    assert vec_out.abs().sum() > 0
    logging.info("test_no_quant_move_fp32 passed.")


@pl.jit()
def no_quant_store_tile_fp32_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
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

        pl.store_tile(out, acc, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_no_quant_store_tile_fp32():
    """Baseline: FP32 store_tile without quantization."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    out = torch.zeros(64, 64, dtype=torch.float32, device=device)

    no_quant_store_tile_fp32_kernel(q, k, out)
    torch.npu.synchronize()

    assert out.abs().sum() > 0
    logging.info("test_no_quant_store_tile_fp32 passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
