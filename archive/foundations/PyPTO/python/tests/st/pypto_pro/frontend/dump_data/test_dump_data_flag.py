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

"""pl.dump_data flag 参数测试 — 简单 dump / 窗口 dump / tensor dump 的标记行输出。"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


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
def dump_data_flag_simple_kernel(out: pl.Tensor[[16], pl.DT_INT32]):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i
        pl.dump_data(out, flag="checkpoint_A")


@pytest.mark.soc("950")
def test_dump_data_flag_tensor_simple():
    _check_npu()
    logging.info("------------test_dump_data_flag_tensor_simple--------------")
    out = torch.zeros(16, device=ST_DEVICE, dtype=torch.int32)
    dump_data_flag_simple_kernel(out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.arange(16, device=ST_DEVICE, dtype=torch.int32))
    logging.info("dump_data_flag_tensor_simple passed!")


@pl.jit(auto_mutex=True)
def dump_data_flag_window_kernel(
    a: pl.Tensor[[32, 32], pl.DT_INT32],
    out: pl.Tensor[[32, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        tc = tc_group.current()
        pl.load(tc, a, [0, 0])
        pl.add(tc, tc, tc)
        pl.dump_data(tc, offsets=[4, 4], shapes=[8, 8], flag="checkpoint_B")
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_dump_data_flag_tile_window():
    _check_npu()
    logging.info("------------test_dump_data_flag_tile_window--------------")
    a = torch.arange(32 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(32, 32)
    out = torch.empty_like(a)
    dump_data_flag_window_kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + a)
    logging.info("dump_data_flag_tile_window passed!")
