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
import os
import sys

import torch
import torch_npu

import pypto

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../models/deepseek_v32_exp/utils'))
from compare import compare  # noqa: E402

num, d, eps = 4, 512, 1e-6
num2 = (2 + num) * num


def gen_data(t=16):
    x_ori = torch.empty((t, num2), dtype=torch.bfloat16).uniform_(-1, 1)
    scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1)
    hc_base_ori = torch.empty((num2,), dtype=torch.float32).uniform_(-1, 1)

    base = hc_base_ori.reshape(1, num2)
    x = x_ori.to(torch.float32)
    pre = x[:, :num] * scale[0] + base[:, :num]  # (t, 4)
    pre = x = 1 / (1 + pre) + eps  # (t, 4)
    res = pre.to(torch.bfloat16)

    return x_ori, scale, hc_base_ori, res


@pypto.frontend.jit()
def kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    scale: pypto.Tensor([3], pypto.DT_FP32),
    base_: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    result: pypto.Tensor([pypto.STATIC, num], pypto.DT_BF16),
):
    pypto.set_vec_tile_shapes(64, 64)
    pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])

    tile_t = 16
    real_t = x.shape[0]
    loop_t_times = (real_t + tile_t - 1) // tile_t

    for t_idx in pypto.loop(loop_t_times, name="t_loop", idx_name="t_idx"):
        x_2d = pypto.reshape(x, [real_t, num2], inplace=True)
        base = pypto.reshape(base_, [1, num2], inplace=True)
        x_view = pypto.view(x_2d, [tile_t, num * d], [t_idx * tile_t, 0])
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        rms_res = x_fp32
        pre = rms_res[:, :num] * (scale[0:1].reshape([1, 1]).expand_clone([tile_t, 1])) + base[:, :num]
        ones = pypto.full(pre.shape, 1.0, pre.dtype, valid_shape=pre.shape)
        pre = pypto.div(ones, pre + 1.0)
        result[t_idx * tile_t:, :] = pypto.cast(pre, pypto.DT_BF16)


@pypto.options(pass_options={"enable_slice": True})
def test_main(t=16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    x, scale, base, y_gd = gen_data(t)

    _shape = (t, num2)

    # Move tensors to NPU device
    x_npu = x.to(device=f'npu:{device_id}')
    scale_npu = scale.to(device=f'npu:{device_id}')
    base_npu = base.to(device=f'npu:{device_id}')
    result = torch.zeros((t, num), dtype=torch.bfloat16, device=f'npu:{device_id}')
    pypto.set_debug_options(runtime_debug_mode=1)
    kernel(x_npu, scale_npu, base_npu, result)
    torch_npu.npu.synchronize()

    y = result.cpu()

    compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)


if __name__ == "__main__":
    test_main(16)
