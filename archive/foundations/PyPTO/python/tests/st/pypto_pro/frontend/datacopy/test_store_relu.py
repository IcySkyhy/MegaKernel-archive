# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Simplified frontend runtime example for pl.store(..., relu_pre_mode=...) on A5 CCE."""

import logging
import os

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


@pl.jit()
def store_relu_cce_kernel(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    raw_out: pl.Tensor[[64, 64], pl.DT_FP32],
    relu_out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
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

        pl.store(raw_out, acc, [0, 0])
        pl.store(relu_out, acc, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.bar_all()
        pl.dump_data(raw_out, offsets=[0, 0], shapes=[8, 8])
        pl.dump_data(relu_out, offsets=[0, 0], shapes=[8, 8])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_relu_cce():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    raw_out = torch.zeros((64, 64), device=device, dtype=torch.float32)
    relu_out = torch.zeros((64, 64), device=device, dtype=torch.float32)

    store_relu_cce_kernel(q, k, raw_out, relu_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    relu_ref = torch.relu(raw_ref)

    logging.info("***********cce raw acc->gm output (top-left 8x8)***********")
    logging.info("%s", raw_out[:8, :8])
    logging.info("***********cce relu store output (top-left 8x8)***********")
    logging.info("%s", relu_out[:8, :8])
    logging.info("***********cce golden raw output (top-left 8x8)***********")
    logging.info("%s", raw_ref[:8, :8])
    logging.info("***********cce golden relu output (top-left 8x8)***********")
    logging.info("%s", relu_ref[:8, :8])

    torch.testing.assert_close(raw_out, raw_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(relu_out, relu_ref, rtol=1e-2, atol=1e-2)
    logging.info("result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_store_relu_cce()
    logging.info("\nAll tests passed!")
