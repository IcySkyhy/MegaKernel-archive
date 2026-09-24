# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P4: 精度对齐测试（golden 对比）

验证 scale 参数的量化结果与 Torch golden 模拟的精度对齐：
1. per-tensor 编译期 float scale（量化 / 反量化 / ReLU / 饱和）
2. 动态 scale（运行时位编码）
3. per-channel scale tensor（量化 / 反量化 / 变尺度）

说明：原 review 版本中与 Ascend C ``extend_params`` 的 bitwise 对齐依赖一个
不存在的上层 API，已在本次修复中移除，统一改为与 Torch golden（CPU 模拟）对比。
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


def _make_scale_tensor(device: str, scale_values: list) -> torch.Tensor:
    scale_bits_list = []
    for scale_value in scale_values:
        scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]
        scale_bits |= 1 << 46  # signed INT8 flag
        scale_bits_list.append(scale_bits)
    return torch.tensor(scale_bits_list, dtype=torch.int64, device=device).reshape(1, 64)


def _make_per_tensor_store_kernel(scale_value: float):
    """Factory: per-tensor FP32->INT8 store kernel with a compile-time float scale."""

    @pl.jit(name=f"p4_per_tensor_store_{scale_value}")
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
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

            pl.store(out, acc, [0, 0], scale=scale_value)
            pl.system.bar_all()

    kernel.__name__ = f"per_tensor_store_scale_{scale_value}"
    return kernel


def _make_per_tensor_relu_kernel(scale_value: float):
    """Factory: per-tensor FP32->INT8 store kernel with scale + ReLU fusion."""

    @pl.jit(name=f"p4_per_tensor_relu_{scale_value}")
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
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

            pl.store(out, acc, [0, 0], scale=scale_value, relu_pre_mode=pl.ReluPreMode.NormalRelu)
            pl.system.bar_all()

    kernel.__name__ = f"per_tensor_relu_scale_{scale_value}"
    return kernel


@pl.jit()
def per_tensor_dynamic_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT32,
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


@pl.jit()
def per_channel_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
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

        pl.store(out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()


@pl.jit()
def per_channel_dequant_store_kernel(
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


# ============================================================================
# 1. per-tensor 编译期 float scale：golden 对比
# ============================================================================


@pytest.mark.soc("950")
@pytest.mark.parametrize("scale_value", [2.0, -1.0, 0.5])
@pypto.options(pass_options={"enable_slice": False})
def test_golden_per_tensor_int8(scale_value):
    """FP32->INT8 编译期 float scale，与 Torch golden 精确对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    kernel = _make_per_tensor_store_kernel(scale_value)
    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    kernel(q, k, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_golden_per_tensor_int8(scale=%s) passed.", scale_value)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_golden_per_tensor_int8_saturation():
    """FP32->INT8 大 scale 饱和路径"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    kernel = _make_per_tensor_store_kernel(1e6)
    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    kernel(q, k, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * 1e6), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_golden_per_tensor_int8_saturation passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_golden_per_tensor_relu():
    """FP32->INT8 编译期 float scale + ReLU 融合，与 Torch golden 对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    kernel = _make_per_tensor_relu_kernel(2.0)
    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    kernel(q, k, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(torch.relu(raw_ref) * 2.0), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_golden_per_tensor_relu passed.")


# ============================================================================
# 2. 动态 scale（运行时位编码）
# ============================================================================


@pytest.mark.soc("950")
@pytest.mark.parametrize("scale_value", [2.0, 0.5])
@pypto.options(pass_options={"enable_slice": False})
def test_golden_dynamic_scale(scale_value):
    """动态 scale（运行时 INT32 位编码），与 Torch golden 对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    per_tensor_dynamic_store_kernel(q, k, out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_golden_dynamic_scale(scale=%s) passed.", scale_value)


# ============================================================================
# 3. per-channel scale tensor（量化 / 反量化）
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_golden_per_channel_int8():
    """per-channel 量化（uniform scale tensor），与 Torch golden 对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_values = [2.0] * 64
    fp_params = _make_scale_tensor(device, scale_values)

    per_channel_store_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * 2.0), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_golden_per_channel_int8 passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_golden_per_channel_varying():
    """per-channel 量化（逐列不同 scale），与 Torch golden 对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    q = _make_q(device)
    k = _make_k(device)
    out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_values = [0.5 + i * 0.1 for i in range(64)]
    fp_params = _make_scale_tensor(device, scale_values)

    per_channel_store_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q.cpu(), k.cpu())
    scale_fp32 = torch.tensor(scale_values, dtype=torch.float32)
    expected = torch.clamp(torch.round(raw_ref * scale_fp32.unsqueeze(0)), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.cpu().to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_golden_per_channel_varying passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_golden_per_channel_dequant_fp16():
    """per-channel 反量化（INT32 Acc -> FP16），与 Torch golden 对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    q = torch.randint(-4, 4, (64, 64), dtype=torch.int8, device=device)
    k = torch.eye(64, dtype=torch.int8, device=device)
    out = torch.zeros((64, 64), device=device, dtype=torch.float16)
    scale_values = [0.1] * 64
    fp_params = _make_scale_tensor(device, scale_values)

    per_channel_dequant_store_kernel(q, k, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q.cpu().float(), k.cpu().float())
    expected = (raw_ref * 0.1).to(torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)
    logging.info("test_golden_per_channel_dequant_fp16 passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
