#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import os
from typing import List

import torch
import torch_npu

import pypto
from pypto import SymInt, ceildiv, loop
from pypto import Tensor as PTensor
from pypto.frontend import jit


@jit()
def isnan_2d_fp32(
    x: PTensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out: PTensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BOOL),
    view_shape: List[SymInt],
    tile_shape: List[int],
):
    b, s = x.shape
    pypto.set_vec_tile_shapes(*tile_shape)
    for i in loop(ceildiv(b, view_shape[0])):
        for j in loop(ceildiv(s, view_shape[1])):
            tile = pypto.view(x, view_shape, [i * view_shape[0], j * view_shape[1]])
            result = pypto.isnan(tile)
            pypto.assemble(result, [i * view_shape[0], j * view_shape[1]], out)
            del tile


def _inject_mixed_special(x_pt: torch.Tensor) -> torch.Tensor:
    """Explicitly inject a mix of NaN/+Inf/-Inf/+0/-0 into a normal-valued tensor."""
    flat = x_pt.view(-1, 1)
    n = flat.shape[0]
    torch.manual_seed(1234)
    for value, count in [(torch.nan, 30), (torch.inf, 20), (-torch.inf, 20), (0.0, 10), (-0.0, 10)]:
        ids = torch.randint(n, (count,))
        flat[ids] = value
    return x_pt


def _run_and_check(fn, x_pt, shape, view_shape, tile_shape):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch_npu.npu.set_device(device_id)
    x = x_pt.npu()
    golden = torch.isnan(x_pt.float())
    out = torch.zeros(shape, dtype=torch.bool, device=f"npu:{device_id}")
    fn(x, out, view_shape, tile_shape)
    # BOOL output is compared element-wise and exactly, no float tolerance.
    assert torch.equal(golden, out.cpu())


@pypto.options(pass_options={"enable_slice": True})
def test_isnan_fp32_mixed():
    shape = (32, 128)
    x_pt = _inject_mixed_special(torch.rand(*shape, dtype=torch.float32))
    _run_and_check(isnan_2d_fp32, x_pt, shape, [32, 128], [32, 32])


if __name__ == "__main__":
    test_isnan_fp32_mixed()
