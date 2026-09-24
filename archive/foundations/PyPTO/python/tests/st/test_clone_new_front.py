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
""" """

import os

import torch
import torch_npu

import pypto

B = 3
S = 4
N1 = 64
D = 64


@pypto.frontend.jit()
def clone_kernel_func(
    in_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 64, 64)
    for b_idx in pypto.loop(B, name="b_loop", idx_name="b_idx"):
        for s_idx in pypto.loop(S, name="s_loop", idx_name="s_idx"):
            a0 = pypto.view(in_tensor, [1, 1, N1, D], [b_idx, s_idx, 0, 0])
            a1 = pypto.add(a0, 1.0)
            a2 = pypto.reshape(a1, [1, 1, N1 * D])
            a3 = a2.clone()
            pypto.assemble(a3, [b_idx, s_idx, 0], out_tensor)


@pypto.options(pass_options={"enable_slice": True})
def test_clone():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    # prepare data
    input_data = torch.rand((B, S, N1, D), dtype=torch.float32, device=f'npu:{device_id}')

    output_shape = (B, S, N1 * D)
    output_result = torch.zeros(output_shape, dtype=torch.float32, device=f'npu:{device_id}')
    # compute on npu
    clone_kernel_func(input_data, output_result)
    torch_npu.npu.synchronize()

    output_cpu = output_result.cpu()

    ## golden
    golden = input_data.cpu().reshape((B, S, N1 * D)) + 1

    assert torch.allclose(output_cpu, golden, atol=1e-3, rtol=1e-3)
