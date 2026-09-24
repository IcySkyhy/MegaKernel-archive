#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE. See LICENSE in the root of the software repository for the full text
# of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test out_tensor_1 usage: loop1 full produces it, loop2 slice-assigns into it, then consumed."""

import os

import torch
import torch_npu

import pypto

SHAPE = [16, 8]
TILE_S = 4
TILE_B = 8


@pypto.frontend.jit(create_new_logical_tensor=True)
def loop_full_assemble_kernel(
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(TILE_S, TILE_B)

    for b_idx in pypto.loop(2, name="Loop_b", idx_name="b_idx"):
        out_tensor_1 = pypto.full([TILE_B, TILE_B], 0.0, pypto.DT_FP32)
        for s_idx in pypto.loop(2, name="Loop_S", idx_name="s_idx"):
            out_tmp = pypto.full([TILE_S, TILE_B], 1.0, pypto.DT_FP32)
            out_tensor_1[s_idx * TILE_S:(s_idx + 1) * TILE_S, :] = out_tmp
        out[b_idx * TILE_B:(b_idx + 1) * TILE_B, :] = out_tensor_1


@pypto.options(pass_options={"enable_slice": True})
def test_loop_full_then_assemble():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_result = torch.zeros(SHAPE, dtype=torch.float32, device=f'npu:{device_id}')
    loop_full_assemble_kernel(output_result)
    torch_npu.npu.synchronize()
    output_cpu = output_result.cpu()
    golden = torch.ones(SHAPE, dtype=torch.float32)
    assert torch.allclose(output_cpu, golden, atol=1e-5, rtol=1e-5)
