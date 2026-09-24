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


@pypto.options(pass_options={"enable_slice": True})
def test_attention_residuals_pypto():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch_npu.npu.set_device(device_id)
    t = 4
    l = 8  # noqa: E741
    d = 512
    k = torch.randn((t, l, d), dtype=torch.bfloat16).npu()
    k_out = torch.zeros((t, l, d), dtype=torch.float32).npu()

    k_golden = k.to(torch.float32).cpu()
    k_golden = k_golden**2

    pypto.set_verify_golden_data(goldens=[None, k_golden])
    attention_residuals(k, k_out)

    k_out_cpu = k_out.cpu()

    not_close = (~torch.isclose(k_out_cpu, k_golden, rtol=3e-3, atol=3e-3)).sum()
    logging.info(f"不相等数量：{not_close} / {k_out_cpu.numel()}")
    assert torch.allclose(k_out_cpu, k_golden, rtol=3e-3, atol=3e-3)


@pypto.frontend.jit(debug_options={"compile_debug_mode": 0})
def attention_residuals(
    k_in: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    k_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
):
    t, l, d = k_in.shape  # noqa: E741
    l_max = 32
    unroll_list = [4]
    for t_idx, unroll_length in pypto.loop_unroll(0, t, 1, name="Loop_t", idx_name="tIdx", unroll_list=unroll_list):
        t_tile = unroll_length
        k = pypto.view(k_in, [t_tile, l_max, d], [t_idx, 0, 0], valid_shape=[t_tile, l, d])
        pypto.set_vec_tile_shapes(8, 16, 128)
        k_2d = pypto.reshape(k, [t_tile * l_max, d], valid_shape=[t_tile * l, d])
        pypto.set_vec_tile_shapes(128, 128)
        k_fp32 = pypto.cast(k_2d, pypto.DT_FP32)
        k = pypto.mul(k_fp32, k_fp32)
        pypto.set_vec_tile_shapes(8, 16, 128)
        k_3d = pypto.reshape(k, [t_tile, l_max, d], valid_shape=[t_tile, l, d])
        pypto.assemble(k_3d, [t_idx, 0, 0], k_out)


if __name__ == "__main__":
    test_attention_residuals_pypto()
