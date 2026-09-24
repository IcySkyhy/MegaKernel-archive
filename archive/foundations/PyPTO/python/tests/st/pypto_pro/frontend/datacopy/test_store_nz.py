# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for pl.store(...)->GM(NZ) with per-channel quantization on A5 CCE.

This test validates the unified ``scale`` parameter API for per-channel quantization
with NZ format output. The user prepares a Scaling tile (scale=Tile) and owns the
data flow (load -> move -> sync MTE1->FIX).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


FP_SCALE_VALUE = 2.0
FP_SCALE_SIGNED_INT8_FLAG = 1 << 46
FP_SCALE_BITS = FP_SCALE_SIGNED_INT8_FLAG | 0x40000000  # signed-int8 + float32 bit-pattern for 2.0f


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


def _make_fp_params(device: str) -> torch.Tensor:
    return torch.full((1, 64), FP_SCALE_BITS, device=device, dtype=torch.int64)


def _make_nz_output(device: str) -> torch.Tensor:
    base = torch.zeros((64, 64), device=device, dtype=torch.float32)
    nz = torch_npu.npu_format_cast(base, 29)
    assert torch_npu.get_npu_format(nz) == 29
    return nz


def _make_nz_int8_output(device: str) -> torch.Tensor:
    base = torch.zeros((64, 64), device=device, dtype=torch.int8)
    nz = torch_npu.npu_format_cast(base, 29)
    assert torch_npu.get_npu_format(nz) == 29
    return nz


@pl.jit()
def store_nz_cce_kernel(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    fp_params: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
    nd_out: pl.Tensor[[64, 64], pl.DT_FP32],
    nz_out: pl.Tensor[[64, 64], pl.DT_FP32, pl.NZ],
    fp_nd_out: pl.Tensor[[64, 64], pl.DT_INT8],
    fp_nz_out: pl.Tensor[[64, 64], pl.DT_INT8, pl.NZ],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        fp_mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

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

        pl.store(nd_out, acc, [0, 0])
        pl.store(nz_out, acc, [0, 0])
        pl.store(fp_nd_out, acc, [0, 0], scale=fp_tile)
        pl.store(fp_nz_out, acc, [0, 0], scale=fp_tile)
        pl.system.bar_all()
        pl.dump_data(nd_out, offsets=[0, 0], shapes=[16, 8])
        pl.dump_data(nz_out, offsets=[0, 0], shapes=[16, 8])
        pl.dump_data(fp_nz_out, offsets=[0, 0], shapes=[16, 32])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_nz_cce():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device)
    nd_out = torch.zeros((64, 64), device=device, dtype=torch.float32)
    nz_out = _make_nz_output(device)
    fp_nd_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    fp_nz_out = _make_nz_int8_output(device)

    store_nz_cce_kernel(q, k, fp_params, nd_out, nz_out, fp_nd_out, fp_nz_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_fp = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    torch.testing.assert_close(nd_out.cpu(), raw_ref.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fp_nd_out.to(torch.int32), expected_fp.to(torch.int32), rtol=0, atol=0)
    assert torch_npu.get_npu_format(nz_out) == 29
    assert torch_npu.get_npu_format(fp_nz_out) == 29


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_store_nz_cce()
    logging.info("\nAll tests passed!")
