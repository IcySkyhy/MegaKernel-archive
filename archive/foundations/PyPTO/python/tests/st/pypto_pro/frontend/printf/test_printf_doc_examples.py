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

# UT for printf.md documentation examples — covers all 5 call patterns.
# Run on NPU and check device log for actual output.

from dataclasses import dataclass
import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

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


@dataclass
class OpTiling:
    valid_size: int


@pl.jit()
def printf_doc_examples_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
    offset: pl.DT_INT32,
    value_f32: pl.DT_FP32,
    addr_u32: pl.DT_UINT32,
    i: pl.DT_INT32,
    tiling: OpTiling,
):
    with pl.section_vector():
        # Example 1: %d integer
        pl.printf("flag=%d, offset=%d\n", flag, offset)

        # Example 2: %+08.3f float
        pl.printf("value=%+08.3f\n", value_f32)

        # Example 3: %#08x hex
        pl.printf("addr=%#08x\n", addr_u32)

        # Example 4: plain text
        pl.printf("reached checkpoint A\n")

        pl.printf("debug: i=%d\n", i, loc=True)

        out[0] = 1
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        if core_id > 0:
            tmp = tiling.valid_size * 100
            if tmp > 10000:
                tmp = tiling.valid_size * 300
            else:
                tmp = (tiling.valid_size + 1) * 1000
        else:
            tmp = tiling.valid_size * 200
        pl.printf("tmp = %d", tmp)

        tmp1 = 112
        if core_id > 0:
            tmp1 = tiling.valid_size * 111
            if tmp1 > 10000:
                tmp1 = tiling.valid_size * 333
            else:
                tmp1 = (tiling.valid_size + 1) * 1111
        else:
            tmp = tiling.valid_size * 222
            tmp1 = tiling.valid_size * 222
        pl.printf("tmp = %d\n", tmp)
        pl.printf("tmp1 = %d\n", tmp1)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_printf_doc_examples():
    """Verify all printf.md documentation examples produce correct device log output.

    Inputs: flag=True, offset=32, value_f32=3.14, addr_u32=0x1234, i=5
    Expected device log:
        flag=1, offset=32
        value=+003.140
        addr=0x001234
        reached checkpoint A
        <source_location> debug: i=5
    """
    _check_npu()
    logging.info("------------test_printf_doc_examples--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    tiling = OpTiling(valid_size=128)
    printf_doc_examples_kernel[None, 2](out, True, 32, 3.14, 0x1234, 5, tiling)
    torch.npu.synchronize()
    assert out.tolist()[0] == 1, f"got {out.tolist()}"
    logging.info("printf_doc_examples passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_printf_doc_examples()
    logging.info("\nAll printf doc example tests passed!")
