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

import os

import torch
import torch_npu

import pypto


@pypto.frontend.jit()
def subscript_assign_ok(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
):
    for _ in pypto.loop(1):
        pypto.set_vec_tile_shapes(32, 32)
        c[:] = a + b


@pypto.options(pass_options={"enable_slice": True})
def test_subscript_assign_allowed():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    a = torch.rand((32, 32), dtype=torch.float16, device=device)
    b = torch.rand((32, 32), dtype=torch.float16, device=device)
    c = torch.zeros((32, 32), dtype=torch.float16, device=device)
    golden = a + b
    subscript_assign_ok(a, b, c)
    torch_npu.npu.synchronize()
    assert torch.allclose(c.cpu(), golden.cpu())


@pypto.frontend.jit()
def move_assign_ok(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
):
    for _ in pypto.loop(1):
        pypto.set_vec_tile_shapes(32, 32)
        c.move(a + b)


@pypto.options(pass_options={"enable_slice": True})
def test_move_assign_allowed():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    a = torch.rand((32, 32), dtype=torch.float16, device=device)
    b = torch.rand((32, 32), dtype=torch.float16, device=device)
    c = torch.zeros((32, 32), dtype=torch.float16, device=device)
    golden = a + b
    move_assign_ok(a, b, c)
    torch_npu.npu.synchronize()
    assert torch.allclose(c.cpu(), golden.cpu())


@pypto.frontend.jit()
def var_reassign_ok(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
):
    for _ in pypto.loop(1):
        pypto.set_vec_tile_shapes(32, 32)
        x = a + b
        x = x + 1.0
        c[:] = x


@pypto.options(pass_options={"enable_slice": True})
def test_non_input_var_reassign_allowed():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    a = torch.rand((32, 32), dtype=torch.float16, device=device)
    b = torch.rand((32, 32), dtype=torch.float16, device=device)
    c = torch.zeros((32, 32), dtype=torch.float16, device=device)
    golden = a + b + 1.0
    var_reassign_ok(a, b, c)
    torch_npu.npu.synchronize()
    assert torch.allclose(c.cpu(), golden.cpu())
