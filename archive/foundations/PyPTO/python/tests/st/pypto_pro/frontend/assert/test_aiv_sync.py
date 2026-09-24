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

"""AIV 间同步（INTER_SUBBLOCK）功能验证 — 验证同一 AI Core 内部两个 AIV 子核之间的 barrier 同步。

A5 架构中一个 AI Core 含 1 AIC + 2 AIV。sync_mode=INTER_SUBBLOCK 的 set_cross_core 实现
AIV-to-AIV 的 barrier：两个 AIV 都 set 后，wait 才能通过。

通过 enable_sync=True 验证同步生效。
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

AIV_SYNC_IDS = [0, 1]
AIV_SYNC_MAX_EID = 2
TARGET_VAL = 5654
INITAL_VALUE = 2341


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


@pl.jit()
def aiv_sync_kernel(
    buf: pl.Tensor[[8, 8], pl.DT_INT32],
    out: pl.Tensor[[1], pl.DT_INT32],
    slot: pl.DT_INT64,
    enable_sync: pl.DT_BOOL,
):
    tile_buf = pl.make_tile(
        pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=256,
    )
    with pl.section_cube():
        pass
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()

        if sub_id == 0:
            for _ in pl.range(0, 10000):
                tile_buf[0, 0] = 41
            tile_buf[0, 0] = TARGET_VAL
            pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=2)
            pl.store(buf, tile_buf, [0, 0])

        if enable_sync:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=AIV_SYNC_IDS[slot],
                sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE2,
                event_id=AIV_SYNC_IDS[slot],
                sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
            )

        if sub_id == 1:
            pl.load(tile_buf, buf, [0, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=5)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=5)
            val = tile_buf[0, 0]
            out[0] = val

    return


def _run(enable_sync):
    _check_npu()
    buf = torch.zeros(8, 8, device=ST_DEVICE, dtype=torch.int32) + INITAL_VALUE
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32) + 2
    aiv_sync_kernel(buf, out, 0, enable_sync)
    torch.npu.synchronize()
    return out.tolist()[0]


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_aiv_barrier_sync():
    """正对照：有同步时 AIV1 能读到 AIV0 写入的 TARGET_VAL。"""
    result = _run(True)
    assert result == TARGET_VAL, f"AIV sync failed: out[0]={result}, expected {TARGET_VAL}"
    logging.info("aiv_barrier_sync passed! out[0]=%d", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_aiv_barrier_sync()
    logging.info("\nAIV barrier sync tests passed!")
