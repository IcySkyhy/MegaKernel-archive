# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.jit()
def ssbuf_copy_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    message = pl.struct("Message", batch=0, block=0, offset=0)
    with pl.section_vector():
        message.batch = 8
        message.block = 1
        message.offset = 32768
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            pl.ssbuf_store(message, 0)
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.ssbuf_load(message, 0)
        pl.printf("Get ssbuf mssage: batch=%d, block=%d, offset=%d", message.batch, message.block, message.offset)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ssbuf_kernel():
    device = ST_DEVICE
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Currrent device is not Ascend950, skip.")
        return
    torch.manual_seed(42)

    x = torch.tensor([3], dtype=torch.int32).to(device)
    ssbuf_copy_kernel(x)


logging.basicConfig(level=logging.INFO, format="%(message)s")
