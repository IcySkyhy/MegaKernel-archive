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

import torch
import torch_npu

import pypto


def add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=None,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c.move(a + b)


@pypto.options(pass_options={"enable_slice": True})
def test_sched_degrade_launch_aicpu_2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    jit_kernel = pypto.frontend.jit(runtime_options={"device_sched_mode": 1, "launch_sched_aicpu_num": 2})(add_kernel)

    tiling = 32
    a_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}') * 5
    b_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')

    jit_kernel(a_data, b_data, c_data, tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32) * 6
    assert torch.allclose(golden, c_data.cpu(), atol=1e-5)


@pypto.options(pass_options={"enable_slice": True})
def test_sched_degrade_launch_aicpu_3():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    jit_kernel = pypto.frontend.jit(runtime_options={"device_sched_mode": 0, "launch_sched_aicpu_num": 3})(add_kernel)

    tiling = 32
    a_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}') * 5
    b_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')

    jit_kernel(a_data, b_data, c_data, tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32) * 6
    assert torch.allclose(golden, c_data.cpu(), atol=1e-5)


@pypto.options(pass_options={"enable_slice": True})
def test_sched_degrade_launch_aicpu_4():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    jit_kernel = pypto.frontend.jit(runtime_options={"device_sched_mode": 1, "launch_sched_aicpu_num": 4})(add_kernel)

    tiling = 32
    a_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}') * 5
    b_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')

    jit_kernel(a_data, b_data, c_data, tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32) * 6
    assert torch.allclose(golden, c_data.cpu(), atol=1e-5)


@pypto.options(pass_options={"enable_slice": True})
def test_sched_degrade_disable_early_launch():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    jit_kernel = pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1, "launch_sched_aicpu_num": 3, "launch_early_mode": 1},
    )(add_kernel)

    tiling = 32
    a_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}') * 5
    b_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')

    jit_kernel(a_data, b_data, c_data, tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32) * 6
    assert torch.allclose(golden, c_data.cpu(), atol=1e-5)


@pypto.options(pass_options={"enable_slice": True})
def test_sched_degrade_allow_cross_cluster():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    jit_kernel = pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1, "launch_sched_aicpu_num": 2, "launch_early_mode": 2},
    )(add_kernel)

    tiling = 32
    a_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}') * 5
    b_data = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((tiling * 2, tiling * 2), dtype=torch.int32, device=f'npu:{device_id}')

    jit_kernel(a_data, b_data, c_data, tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((tiling * 2, tiling * 2), dtype=torch.int32) * 6
    assert torch.allclose(golden, c_data.cpu(), atol=1e-5)
