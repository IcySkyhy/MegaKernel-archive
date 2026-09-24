# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scale dtype validation tests for per-channel quantization (scale=Tile).

Per-channel quantization requires a user-prepared Scaling Tile (INT64). Tests
verify that:
1. FP32 / FP16 / INT32 Scaling tiles raise clear errors with conversion guidance
2. An INT64 Scaling tile works correctly (baseline)

Note: the GM Tensor scale rejection is covered by
test_scale_error_handling::test_err_scale_tensor_rejected.
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics._exceptions import ParserSyntaxError
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


FP_SCALE_VALUE = 2.0


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


# ============================================================================
# Test 1: FP32 Scaling tile should raise clear error
# ============================================================================


@pl.jit()
def fp32_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Kernel with FP32 Scaling tile (should fail at parse time)."""
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

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=256)

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

        # This should raise ValueError at parse time
        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_tile_fp32():
    """Validate that FP32 Scaling tile raises clear error with conversion guidance."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    with pytest.raises(ParserSyntaxError, match=r"scale tile dtype FP32 is not supported.*npu_trans_quant_param"):
        fp32_scale_kernel(q, k, out)

    logging.info("test_err_scale_tile_fp32 passed: FP32 Scaling tile raises clear error.")


# ============================================================================
# Test 3: FP16 Scaling tile should raise error
# ============================================================================


@pl.jit()
def fp16_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Kernel with FP16 Scaling tile (should fail at parse time)."""
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

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=128)

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

        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_tile_fp16():
    """Validate that FP16 Scaling tile raises appropriate error."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    with pytest.raises(ParserSyntaxError, match=r"scale tile dtype must be INT64"):
        fp16_scale_kernel(q, k, out)

    logging.info("test_err_scale_tile_fp16 passed: FP16 Scaling tile raises error.")


# ============================================================================
# Test 4: INT32 Scaling tile should raise error
# ============================================================================


@pl.jit()
def int32_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Kernel with INT32 Scaling tile (should fail at parse time)."""
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

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=256)

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

        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_tile_int32():
    """Validate that INT32 Scaling tile raises appropriate error."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    with pytest.raises(ParserSyntaxError, match=r"scale tile dtype must be INT64"):
        int32_scale_kernel(q, k, out)

    logging.info("test_err_scale_tile_int32 passed: INT32 Scaling tile raises error.")


# ============================================================================
# Test 5: FP32 scale manually converted to INT64 tile works correctly
# ============================================================================


@pl.jit()
def int64_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    """Kernel with a user-prepared INT64 Scaling tile (correct usage)."""
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
def test_per_channel_fp32_manual_conversion():
    """Validate that a manually converted FP32→INT64 scale tile works correctly."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)

    # Manual conversion: FP32 → INT64 using npu_trans_quant_param
    scale_fp32 = torch.full((1, 64), FP_SCALE_VALUE, device=device, dtype=torch.float32)
    scale_int64 = torch_npu.npu_trans_quant_param(scale_fp32)

    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    int64_scale_kernel(q, k, scale_int64, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_per_channel_fp32_manual_conversion passed: FP32→INT64 manual conversion works.")


if __name__ == "__main__":
    test_err_scale_tile_fp32()
    test_err_scale_tile_fp16()
    test_err_scale_tile_int32()
    test_per_channel_fp32_manual_conversion()
    logging.info("\nAll tests passed!")
