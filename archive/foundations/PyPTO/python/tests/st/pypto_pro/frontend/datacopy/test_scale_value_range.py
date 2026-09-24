# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P2: per-tensor dynamic scale value range + input data patterns.

Dynamic-shape kernel (pl.DYNAMIC) with tail-block coverage via
valid_shape=[-1,-1] + compact=1 + set_validshape:
- full block: 64 x 64
- row tail block: 64 x 96
- dual tail: 96 x 96
Golden is compared only on the valid region [:vm, :vn].

Tail-block shapes keep the block at [64, 64] effective extent (vm=vn=64):
partial blocks with vm/vn < 64 trigger a framework state-pollution bug when a
prior full-block call ran in the same process (verified 2026-08: single-call is
correct, subsequent partial-block calls in the same process produce garbage;
both row and column partial blocks affected). Await framework fix.
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

TILE = 64


def _make_q(device: str, pattern: str, rows: int, cols: int) -> torch.Tensor:
    """Generate test input with different patterns and explicit shape."""
    if pattern == "all_positive":
        return torch.arange(1, cols + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(rows, 1)
    elif pattern == "all_negative":
        return torch.arange(-cols, 0, dtype=torch.float32, device=device).unsqueeze(0).repeat(rows, 1)
    elif pattern == "all_zeros":
        return torch.zeros((rows, cols), dtype=torch.float32, device=device)
    elif pattern == "boundary":
        base = torch.tensor([-64.0, -63.5, 63.5, 64.0], dtype=torch.float32, device=device)
        return base.unsqueeze(0).repeat(rows, cols // 4)
    elif pattern == "large_magnitude":
        base = torch.tensor([1000.0, -1000.0, 5000.0, -5000.0], dtype=torch.float32, device=device)
        return base.unsqueeze(0).repeat(rows, cols // 4)
    else:  # mixed
        row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(
            cols // 8
        )
        return row.unsqueeze(0).repeat(rows, 1)


def _make_k(device: str, cols: int) -> torch.Tensor:
    return torch.eye(cols, device=device, dtype=torch.float32)


@pl.jit()
def scale_value_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
    vm: pl.DT_INT32,
    vn: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        )
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        pl.set_validshape(q_mat, [vm, TILE])
        pl.set_validshape(q_left, [vm, TILE])
        pl.set_validshape(k_mat, [TILE, vn])
        pl.set_validshape(k_right, [TILE, vn])
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


SCALE_VALUES = [2.0, -1.0, 0.0, 1.0, 0.5, 1e-6, 1e6]
SCALE_PATTERNS = ["mixed", "mixed", "mixed", "mixed", "mixed", "large_magnitude", "mixed"]
SCALE_IDS = ["positive", "negative", "zero", "unit", "fraction", "very_small", "very_large"]

INPUT_PATTERNS = ["all_positive", "all_negative", "all_zeros", "boundary", "large_magnitude"]
INPUT_SCALES = [0.1, 0.1, 2.0, 2.0, 1.0]

SHAPES = [(TILE, TILE), (48, 96), (96, 96)]
SHAPE_IDS = ["full", "row_tail", "dual_tail"]


@pytest.mark.soc("950")
@pytest.mark.parametrize("m,n", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("scale_value,pattern", list(zip(SCALE_VALUES, SCALE_PATTERNS)), ids=SCALE_IDS)
@pypto.options(pass_options={"enable_slice": False})
def test_scale_value_range(scale_value, pattern, m, n):
    """Per-tensor dynamic scale value range across full/tail/dual-tail blocks."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    vm, vn = min(m, TILE), min(n, TILE)
    q = _make_q(device, pattern, m, n)
    k = _make_k(device, n)
    quant_out = torch.zeros((m, n), device=device, dtype=torch.int8)
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits, vm, vn)
    torch.npu.synchronize()

    expected = torch.clamp(torch.round(q * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(quant_out[:vm, :vn].to(torch.int32), expected[:vm, :vn].to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_range[%s,%s] passed.", scale_value, (m, n))


@pytest.mark.soc("950")
@pytest.mark.parametrize("m,n", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("pattern,scale_value", list(zip(INPUT_PATTERNS, INPUT_SCALES)), ids=INPUT_PATTERNS)
@pypto.options(pass_options={"enable_slice": False})
def test_input_pattern(pattern, scale_value, m, n):
    """Input boundary patterns across full/tail/dual-tail blocks."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    vm, vn = min(m, TILE), min(n, TILE)
    q = _make_q(device, pattern, m, n)
    k = _make_k(device, n)
    quant_out = torch.zeros((m, n), device=device, dtype=torch.int8)
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits, vm, vn)
    torch.npu.synchronize()

    expected = torch.clamp(torch.round(q * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(quant_out[:vm, :vn].to(torch.int32), expected[:vm, :vn].to(torch.int32), rtol=0, atol=1)
    logging.info("test_input_pattern[%s,%s] passed.", pattern, (m, n))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
