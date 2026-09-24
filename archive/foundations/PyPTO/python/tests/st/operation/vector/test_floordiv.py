#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test floor_div operation on board
"""

import os
from typing import List

import torch
import torch_npu

import pypto
from pypto import SymInt, ceildiv, loop
from pypto import Tensor as PTensor
from pypto.frontend import jit


@jit()
def floor_div_2d(
    x: PTensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    y: PTensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    out: PTensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    view_shape: List[SymInt],
    tile_shape: List[int],
):
    b, s = x.shape
    pypto.set_vec_tile_shapes(*tile_shape)
    for i in loop(ceildiv(b, view_shape[0])):
        for j in loop(ceildiv(s, view_shape[1])):
            tile_x = pypto.view(x, view_shape, [i * view_shape[0], j * view_shape[1]])
            tile_y = pypto.view(y, view_shape, [i * view_shape[0], j * view_shape[1]])
            result = pypto.floor_div(tile_x, tile_y)
            pypto.assemble(result, [i * view_shape[0], j * view_shape[1]], out)
            del tile_x, tile_y


@pypto.options(pass_options={"enable_slice": True})
def test_floor_div():
    view_shape = [32, 128]
    tile_shape = [32, 32]
    x_pt = torch.randint(0, 100, (32, 128), dtype=torch.int32)
    y_pt = torch.randint(1, 100, (32, 128), dtype=torch.int32)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch_npu.npu.set_device(device_id)
    out = torch.zeros((32, 128), dtype=torch.int32, device=f"npu:{device_id}")
    floor_div_2d(x_pt.npu(), y_pt.npu(), out.npu(), view_shape, tile_shape)
    assert out.shape == (32, 128)
    golden = torch.floor_divide(x_pt.npu(), y_pt.npu()).to(torch.int32)
    assert torch.allclose(golden.cpu(), out.cpu())


if __name__ == "__main__":
    test_floor_div()
