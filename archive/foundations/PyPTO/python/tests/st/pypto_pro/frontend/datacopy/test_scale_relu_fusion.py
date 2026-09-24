# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P1: scale + relu_pre_mode fusion tests for per-tensor and per-channel quantization."""

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
    base = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    return (base.remainder(9) - 4.0).to(device)


def _make_k(device: str) -> torch.Tensor:
    base = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    return ((base * 3).remainder(7) - 3.0).to(device)


# ============================================================================
# Test 1: Per-tensor scale + relu fusion
# ============================================================================


@pl.jit()
def per_tensor_scale_relu_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
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

        # scale + relu fusion
        pl.store(out, acc, [0, 0], scale=scale_value, relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pytest.mark.parametrize("m,n", [(64, 64), (48, 96), (96, 96)], ids=["full", "row_tail", "dual_tail"])
@pypto.options(pass_options={"enable_slice": False})
def test_per_tensor_scale_relu_fusion(m, n):
    """Test per-tensor scale + relu fusion, covering tail blocks."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    vm, vn = min(m, 64), min(n, 64)
    q = torch.randn(m, n, device=device, dtype=torch.float32)
    k = torch.eye(n, device=device, dtype=torch.float32)
    out = torch.zeros((m, n), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    per_tensor_scale_relu_kernel(q, k, out, scale_bits, vm, vn)
    torch.npu.synchronize()

    # Expected: relu(q) * scale (k=eye), then quantize to INT8
    relu_ref = torch.relu(q)
    expected = torch.clamp(torch.round(relu_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(out[:vm, :vn].to(torch.int32), expected[:vm, :vn].to(torch.int32), rtol=0, atol=1)
    logging.info("test_per_tensor_scale_relu_fusion[%s,%s] passed!", m, n)


# ============================================================================
# Test 2: per-channel + relu fusion
# NOTE: per-channel scale (Tensor) is mutually exclusive with relu_pre_mode —
# the combination "scale=Tensor + relu_pre_mode=NormalRelu" is rejected at parse
# time (see test_scale_error_handling.py::test_err_per_channel_with_relu).
# Per-channel quantization MUST be fused with ReLU by the user after the store.
# ============================================================================


# ============================================================================
# Test 3: Dynamic scale + relu fusion
# ============================================================================


@pl.jit()
def dynamic_scale_relu_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
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

        # dynamic scale + relu fusion
        pl.store(out, acc, [0, 0], scale=scale_bits, relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_scale_relu_fusion():
    """Test dynamic scale + relu fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    dynamic_scale_relu_kernel(q, k, out, scale_bits, 64, 64)
    torch.npu.synchronize()

    # Expected: relu(matmul(q, k)) * scale, then quantize to INT8
    raw_ref = torch.matmul(q, k)
    relu_ref = torch.relu(raw_ref)
    scaled_ref = relu_ref * scale_value
    expected = torch.clamp(torch.round(scaled_ref), -128, 127).to(torch.int8)

    logging.info("***********dynamic scale + relu fusion***********")
    logging.info("raw output (top-left 8x8): %s", raw_ref[:8, :8])
    logging.info("relu output (top-left 8x8): %s", relu_ref[:8, :8])
    logging.info("scaled output (top-left 8x8): %s", scaled_ref[:8, :8])
    logging.info("quantized output (top-left 8x8): %s", out[:8, :8])
    logging.info("expected output (top-left 8x8): %s", expected[:8, :8])

    torch.testing.assert_close(out, expected, rtol=0, atol=1)
    logging.info("test_dynamic_scale_relu_fusion passed!")


# ============================================================================
# Test 4: Multiple scale values with relu
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_multiple_scale_values_with_relu():
    """Test different scale values with relu fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    raw_ref = torch.matmul(q, k)
    relu_ref = torch.relu(raw_ref)

    scale_values = [0.5, 1.0, 2.0, 4.0]

    for scale_value in scale_values:
        out = torch.zeros((64, 64), device=device, dtype=torch.int8)
        # Convert float to INT32 bit representation
        scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]
        per_tensor_scale_relu_kernel(q, k, out, scale_bits, 64, 64)
        torch.npu.synchronize()

        scaled_ref = relu_ref * scale_value
        expected = torch.clamp(torch.round(scaled_ref), -128, 127).to(torch.int8)

        logging.info(f"***********scale={scale_value} + relu fusion***********")
        logging.info("quantized output (top-left 4x4): %s", out[:4, :4])
        logging.info("expected output (top-left 4x4): %s", expected[:4, :4])

        torch.testing.assert_close(out, expected, rtol=0, atol=1)
        logging.info(f"scale={scale_value} passed!")

    logging.info("test_multiple_scale_values_with_relu passed!")


if __name__ == "__main__":
    test_per_tensor_scale_relu_fusion()
    test_dynamic_scale_relu_fusion()
    test_multiple_scale_values_with_relu()
    logging.info("\nAll P1 scale + relu fusion tests passed!")
