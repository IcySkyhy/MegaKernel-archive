# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P2 supplement: 编译期 float scale 路径的值域测试

设计文档的核心接口是 ``scale=2.0``（编译期 float，走 ``_encode_deq_scalar`` 位编码）。
原有 P2 值域测试全部走运行时 Expr 路径，本文件补充编译期 float scale 的
正 / 负 / 零 / 单位 / 分数 / 极小 / 极大值测试，直接验证 ``_encode_deq_scalar``。
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _make_q(device: str, pattern: str) -> torch.Tensor:
    if pattern == "all_positive":
        return torch.arange(1, 65, dtype=torch.float32, device=device).unsqueeze(0).repeat(64, 1)
    if pattern == "all_negative":
        return torch.arange(-64, 0, dtype=torch.float32, device=device).unsqueeze(0).repeat(64, 1)
    if pattern == "all_zeros":
        return torch.zeros((64, 64), dtype=torch.float32, device=device)
    if pattern == "large_magnitude":
        base = torch.tensor([1000.0, -1000.0, 5000.0, -5000.0], dtype=torch.float32, device=device)
        return base.unsqueeze(0).repeat(64, 16)
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


def _make_ct_kernel(scale_value: float):
    """Factory: FP32->INT8 store kernel with a compile-time float scale literal."""

    @pl.jit(name=f"ct_float_scale_{scale_value}")
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
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

            pl.store(quant_out, acc, [0, 0], scale=scale_value)
            pl.system.bar_all()

    return kernel


# scale 值域：编译期 float
# 输入通过 matmul(q, k)（k=eye），因此输出 == q，便于 golden 计算


@pytest.mark.soc("950")
@pytest.mark.parametrize("m,n", [(64, 64), (48, 96), (96, 96)], ids=["full", "row_tail", "dual_tail"])
@pytest.mark.parametrize(
    "scale_value,pattern",
    [
        (2.0, "mixed"),
        (-1.0, "mixed"),
        (0.0, "mixed"),
        (1.0, "mixed"),
        (0.5, "mixed"),
        (1e-6, "large_magnitude"),
        (1e6, "mixed"),
    ],
    ids=["positive", "negative", "zero", "unit", "fraction", "very_small", "very_large"],
)
@pypto.options(pass_options={"enable_slice": False})
def test_ct_float_scale_value_range(scale_value, pattern, m, n):
    """编译期 float scale 值域：与 clamp(round(x * scale)) golden 对比"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip(f"Current device is {torch.npu.get_device_name()}")

    kernel = _make_ct_kernel(scale_value)
    vm, vn = min(m, 64), min(n, 64)
    if pattern == "large_magnitude":
        q = torch.randn(m, n, device=device, dtype=torch.float32) * 1000.0
    else:
        q = torch.randn(m, n, device=device, dtype=torch.float32)
    k = torch.eye(n, device=device, dtype=torch.float32)
    quant_out = torch.zeros((m, n), device=device, dtype=torch.int8)

    kernel(q, k, quant_out, vm, vn)
    torch.npu.synchronize()

    expected = torch.clamp(torch.round(q * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(quant_out[:vm, :vn].to(torch.int32), expected[:vm, :vn].to(torch.int32), rtol=0, atol=1)
    logging.info("test_ct_float_scale_value_range(scale=%s, %sx%s) passed.", scale_value, m, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
