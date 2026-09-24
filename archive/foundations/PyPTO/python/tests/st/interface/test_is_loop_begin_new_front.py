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
def dyn_loop_with_loop_begin(
    in_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 64, 64)

    for b_idx in pypto.loop(B, name="b_loop", idx_name="b_idx"):
        for s_idx in pypto.loop(S, name="s_loop", idx_name="s_idx"):
            a0 = pypto.view(in_tensor, [1, 1, N1, D], [b_idx, s_idx, 0, 0])
            if pypto.is_loop_begin(b_idx):
                a1 = pypto.add(a0, 1.0)
                pypto.assemble(a1, [b_idx, s_idx, 0, 0], out_tensor)
            else:
                a1 = pypto.mul(a0, 1.0)
                pypto.assemble(a1, [b_idx, s_idx, 0, 0], out_tensor)


@pypto.options(pass_options={"enable_slice": True})
def test_is_loop_begin():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)

    # Define shape parameters
    shape_in = (B, S, N1, D)
    shape_out = (B, S, N1, D)

    # Prepare data
    input_torch = torch.rand(shape_in, dtype=torch.float32, device=f'npu:{device_id}')
    output_torch = torch.ones(shape_out, dtype=torch.float32, device=f'npu:{device_id}')

    # Execute kernel
    dyn_loop_with_loop_begin(input_torch, output_torch)

    # Synchronize device
    torch_npu.npu.synchronize()

    # Get results and verify
    output_result = output_torch.cpu()

    # Golden reference: first batch gets +1, rest stay same
    output_golden = input_torch.clone().cpu()
    output_golden[0:1, :, :, :] = output_golden[0:1, :, :, :] + 1

    assert torch.allclose(output_result, output_golden, atol=1e-5)


@pypto.frontend.jit()
def dyn_loop_with_loop_end(
    in_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 64, 64)

    for b_idx in pypto.loop(B, name="b_loop", idx_name="b_idx"):
        for s_idx in pypto.loop(S, name="s_loop", idx_name="s_idx"):
            a0 = pypto.view(in_tensor, [1, 1, N1, D], [b_idx, s_idx, 0, 0])
            if b_idx == B - 1:
                a1 = pypto.add(a0, 1.0)
                pypto.assemble(a1, [b_idx, s_idx, 0, 0], out_tensor)
            else:
                a1 = pypto.mul(a0, 1.0)
                pypto.assemble(a1, [b_idx, s_idx, 0, 0], out_tensor)


@pypto.options(pass_options={"enable_slice": True})
def test_is_loop_end():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)

    # Define shape parameters

    shape_in = (B, S, N1, D)
    shape_out = (B, S, N1, D)

    # Prepare data on NPU
    input_torch = torch.rand(shape_in, dtype=torch.float32, device=f'npu:{device_id}')
    output_torch = torch.ones(shape_out, dtype=torch.float32, device=f'npu:{device_id}')

    # Execute kernel
    dyn_loop_with_loop_end(input_torch, output_torch)

    # Synchronize device
    torch_npu.npu.synchronize()

    # Move results to CPU for comparison
    output_result = output_torch.cpu()
    output_golden = input_torch.clone().cpu()
    # is_loop_end 返回True时是最后一个batch (B-1=2，所以是[2:3])
    output_golden[B - 1:B, :, :, :] = output_golden[B - 1:B, :, :, :] + 1

    # Verify with torch.allclose
    assert torch.allclose(output_result, output_golden, atol=1e-5)


if __name__ == "__main__":
    test_is_loop_begin()
    test_is_loop_end()
