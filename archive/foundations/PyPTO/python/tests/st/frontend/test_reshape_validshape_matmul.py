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
""" """

import logging
import os

import torch
import torch_npu

import pypto

M = 256
N = 64
TILE_B = 5
TILE_S = 5


def _reshape_matmul_only_torch(input_tensor_a, input_tensor_b):
    batch_size, seq_len = input_tensor_b.shape[:2]
    b_loop = (batch_size + TILE_B - 1) // TILE_B
    s_loop = (seq_len + TILE_S - 1) // TILE_S
    output = torch.empty((b_loop, s_loop, M, N), dtype=input_tensor_b.dtype, device=input_tensor_b.device)

    for b_idx in range(b_loop):
        for s_idx in range(s_loop):
            input_a_view = input_tensor_a[
                :, b_idx * TILE_B:(b_idx + 1) * TILE_B, s_idx * TILE_S:(s_idx + 1) * TILE_S
            ]
            input_b_view = input_tensor_b[
                b_idx * TILE_B:(b_idx + 1) * TILE_B, s_idx * TILE_S:(s_idx + 1) * TILE_S, :
            ]
            input_a_view_2d = input_a_view.reshape([input_a_view.shape[0], -1])
            input_b_view_2d = input_b_view.reshape([-1, input_b_view.shape[-1]])
            output[b_idx, s_idx, :, :] = torch.matmul(input_a_view_2d, input_b_view_2d)

    return output


@pypto.options(pass_options={"enable_slice": False})
def test_reshape_validshape_matmul_pypto():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch_npu.npu.set_device(device_id)
    torch.manual_seed(20260608)

    batch_size = 3
    seq_len = 5
    input_tensor_a = torch.randn((M, batch_size, seq_len), dtype=torch.float32).npu()
    input_tensor_b = torch.randn((batch_size, seq_len, N), dtype=torch.float32).npu()
    output_tensor = torch.empty(
        ((batch_size + TILE_B - 1) // TILE_B, (seq_len + TILE_S - 1) // TILE_S, M, N),
        dtype=torch.float32,
    ).npu()

    golden = _reshape_matmul_only_torch(input_tensor_a.cpu(), input_tensor_b.cpu())
    pypto.set_verify_golden_data(goldens=[None, None, golden])
    reshape_matmul_only(input_tensor_a, input_tensor_b, output_tensor)

    output_cpu = output_tensor.cpu()
    not_close = (~torch.isclose(output_cpu, golden, rtol=1e-3, atol=1e-3)).sum()
    logging.info(f"not close count: {not_close} / {output_cpu.numel()}")
    assert torch.allclose(output_cpu, golden, rtol=1e-3, atol=1e-3)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU}, debug_options={"compile_debug_mode": 0})
def reshape_matmul_only(
    input_tensor_a: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_tensor_b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
):
    batch_size, seq_len = input_tensor_b.shape[:2]
    b_loop = (batch_size + TILE_B - 1) // TILE_B
    s_loop = (seq_len + TILE_S - 1) // TILE_S

    for b_idx in pypto.loop(b_loop, name="FirstMatmul_Loop_B", idx_name="b_idx", unroll_list=[1]):
        for s_idx in pypto.loop(s_loop, name="FirstMatmul_Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
            pypto.set_vec_tile_shapes(32, 32, 32)

            input_a_view = input_tensor_a[
                :, b_idx * TILE_B:(b_idx + 1) * TILE_B, s_idx * TILE_S:(s_idx + 1) * TILE_S
            ]
            input_b_view = input_tensor_b[
                b_idx * TILE_B:(b_idx + 1) * TILE_B, s_idx * TILE_S:(s_idx + 1) * TILE_S, :
            ]
            valid_b = pypto.min(TILE_B, input_tensor_b.shape[0] - b_idx * TILE_B)
            valid_s = pypto.min(TILE_S, input_tensor_b.shape[1] - s_idx * TILE_S)

            input_a_view_2d = pypto.reshape(
                input_a_view,
                [input_a_view.shape[0], TILE_B * TILE_S],
                valid_shape=[input_a_view.shape[0], valid_b * valid_s],
            )
            input_b_view_2d = pypto.reshape(
                input_b_view,
                [TILE_B * TILE_S, input_b_view.shape[-1]],
                valid_shape=[valid_b * valid_s, input_b_view.shape[-1]],
            )

            output_tensor[b_idx, s_idx, :, :] = pypto.matmul(
                input_a_view_2d,
                input_b_view_2d,
                out_dtype=input_tensor_b.dtype,
            )


if __name__ == "__main__":
    test_reshape_validshape_matmul_pypto()
