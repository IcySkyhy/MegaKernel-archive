#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test gather compare form (GT + EQ) for TGather.

TGather compare form collects indices where:
- GT mode (cmp_mode=4): src[i] > k_value
- EQ mode (cmp_mode=0): src[i] == k_value

Test design:
1. Create src = [0, 10, 20, 30, 20, 10, 40, 20, ...] repeated
2. k_value = 20
3. GT: collect indices where value > 20 (values 30, 40) into indices_gt_tile
4. EQ: collect indices where value == 20 into indices_eq_tile (separate tile)
5. Pack all results into single output tensor:
   out[0, 0:COLS]          = GT indices
   out[0, COLS:2*COLS]     = EQ indices
   out[0, 2*COLS:2*COLS+16]   = GT cdst tile (count at [0])
   out[0, 2*COLS+16:2*COLS+32] = EQ cdst tile (count at [0])
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

ROWS = 1
COLS = 128
K_VALUE = 20

# UB tile sizes
SRC_SIZE = 256  # [1, 128] INT16 = 256 bytes
K_VALUE_SIZE = 32  # [1, 16]  UINT16 = 32 bytes
TMP_SIZE = 1024  # [1, 256] UINT32 = 1024 bytes
DST_SIZE = 512  # [1, 128] UINT32 = 512 bytes
CDST_SIZE = 64  # [1, 16]  UINT32 = 64 bytes

# Output tensor layout: [GT indices | EQ indices | GT cdst (16) | EQ cdst (16)]
OUT_COLS = COLS * 2 + 32  # 128 + 128 + 16 + 16 = 288 UINT32 elements
GT_OFF = 0
EQ_OFF = COLS
CNT_OFF = COLS * 2  # GT count tile starts here (16 elements)
CNT_EQ_OFF = COLS * 2 + 16  # EQ count tile starts here (16 elements)


@pl.jit()
def gather_cmp_kernel(
    src: pl.Tensor[[ROWS, COLS], pl.DT_INT16],
    out: pl.Tensor[[ROWS, OUT_COLS], pl.DT_UINT32],
):
    src_tile = pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=SRC_SIZE
    )
    k_value_tile = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec), addr=0x1000, size=K_VALUE_SIZE
    )
    tmp_tile = pl.make_tile(
        pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec), addr=0x2000, size=TMP_SIZE
    )
    indices_gt_tile = pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x3000,
        size=DST_SIZE,
    )
    indices_eq_tile = pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x3200,
        size=DST_SIZE,
    )
    cdst_gt_tile = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec), addr=0x4000, size=CDST_SIZE
    )
    cdst_eq_tile = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec), addr=0x4040, size=CDST_SIZE
    )

    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        pl.expands(k_value_tile, K_VALUE)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        pl.gather(indices_gt_tile, src_tile, k_value_tile, cdst_gt_tile, tmp_tile, cmp_mode=4, offset=0)

        pl.gather(indices_eq_tile, src_tile, k_value_tile, cdst_eq_tile, tmp_tile, cmp_mode=0, offset=0)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)

        pl.store(out, indices_gt_tile, [0, GT_OFF])
        pl.store(out, indices_eq_tile, [0, EQ_OFF])
        pl.store(out, cdst_gt_tile, [0, CNT_OFF])
        pl.store(out, cdst_eq_tile, [0, CNT_EQ_OFF])


def create_test_data():
    device = ST_DEVICE
    torch.npu.set_device(device)

    pattern = torch.tensor([0, 10, 20, 30, 20, 10, 40, 20], dtype=torch.int32)
    repeats = COLS // len(pattern) + 1
    src_data = pattern.repeat(repeats)[:COLS]
    src_int16 = src_data.to(torch.int16).unsqueeze(0).to(device).contiguous()

    return src_int16


def compute_expected(src_data, k_value):
    flat = src_data.flatten().tolist()
    gt_indices = [i for i, v in enumerate(flat) if v > k_value]
    eq_indices = [i for i, v in enumerate(flat) if v == k_value]
    return gt_indices, eq_indices


@pl.jit()
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gather_cmp():
    device = ST_DEVICE
    torch.npu.set_device(device)

    logging.info("\n=== Test TGather GT + EQ ===")

    src = create_test_data()
    gt_expected, eq_expected = compute_expected(src, K_VALUE)

    logging.info("src pattern: [0, 10, 20, 30, 20, 10, 40, 20]")
    logging.info("k_value: %s", K_VALUE)
    logging.info("Expected GT (value > %s): indices %s... count=%s", K_VALUE, gt_expected[:10], len(gt_expected))
    logging.info("Expected EQ (value == %s): indices %s... count=%s", K_VALUE, eq_expected[:10], len(eq_expected))

    out = torch.zeros([ROWS, OUT_COLS], device=device, dtype=torch.int32)

    gather_cmp_kernel(src, out)
    torch.npu.synchronize()

    # cdst[0] stores byte count written to dst; divide by sizeof(uint32_t)=4 for element count
    gt_count = out[0, CNT_OFF].item() // 4
    eq_count = out[0, CNT_EQ_OFF].item() // 4
    gt_result = out[0, GT_OFF:GT_OFF + gt_count].tolist()
    eq_result = out[0, EQ_OFF:EQ_OFF + eq_count].tolist()

    logging.info("NPU GT count: %s, indices: %s...", gt_count, gt_result[:10])
    logging.info("NPU EQ count: %s, indices: %s...", eq_count, eq_result[:10])

    assert gt_count == len(gt_expected), f"GT count mismatch: {gt_count} vs {len(gt_expected)}"
    assert eq_count == len(eq_expected), f"EQ count mismatch: {eq_count} vs {len(eq_expected)}"
    assert gt_result == gt_expected, "GT indices mismatch"
    assert eq_result == eq_expected, "EQ indices mismatch"

    logging.info("PASSED: GT + EQ verified!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_gather_cmp()
    logging.info("\nAll tests passed!")
