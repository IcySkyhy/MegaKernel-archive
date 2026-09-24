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
import os

from numpy.testing import assert_allclose
import torch

import pypto


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_kernel_range(
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32), c: pypto.Tensor([16, 16], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(16, 16)
    c.move(b[0:16, 0:16])
    for i in range(1, 4):
        c.move(pypto.add(c, b[i * 16:(i + 1) * 16, i * 16:(i + 1) * 16]))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_kernel_list(
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32), c: pypto.Tensor([16, 16], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(16, 16)
    c.move(b[0:16, 0:16])
    for i in [1, 2, 3]:
        c.move(pypto.add(c, b[i * 16:(i + 1) * 16, i * 16:(i + 1) * 16]))


@pypto.options(pass_options={"enable_slice": True})
def test_range_list_iterate():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)

    b = torch.rand((64, 64), dtype=torch.float32, device=device_id)
    expected = b[:16, :16] + b[16:32, 16:32] + b[32:48, 32:48] + b[48:, 48:]

    out_range = torch.zeros((16, 16), dtype=torch.float32, device=device_id)
    out_list = torch.zeros((16, 16), dtype=torch.float32, device=device_id)

    add_kernel_range(b, out_range)
    add_kernel_list(b, out_list)

    assert_allclose(out_range.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    assert_allclose(out_list.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
