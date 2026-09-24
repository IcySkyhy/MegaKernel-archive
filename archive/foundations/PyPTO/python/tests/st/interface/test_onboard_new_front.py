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


@pypto.frontend.jit()
def cust_dyn_func(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(32, 32)

    for k in pypto.loop(10, name="s0", idx_name="k"):
        if k == 0:
            b.move(pypto.add(a, a))
        else:
            b.move(pypto.add(a, b))


@pypto.options(pass_options={"enable_slice": True})
def test_device_run_data_from_device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)

    # prepare data
    a_rawdata = torch.tensor([[k * 100 + v for v in range(m)] for k in range(n)], dtype=torch.int32)
    a_data = a_rawdata.to(device=f'npu:{device_id}')
    b_data = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')

    cust_dyn_func(a_data, b_data)

    torch_npu.npu.synchronize()
    # get data and compare result
    a_data_cpu = a_data.cpu()
    b_data_cpu = b_data.cpu()
    # verify
    a_data_list = [c for r in a_data_cpu.tolist() for c in r]
    b_data_list = [c for r in b_data_cpu.tolist() for c in r]
    assert b_data_list == [v * 11 for v in a_data_list]

    c_rawdata = torch.tensor([[k * 1000 + v for v in range(m)] for k in range(n)], dtype=torch.int32)
    c_data = c_rawdata.to(device=f'npu:{device_id}')
    d_data = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')

    cust_dyn_func(c_data, d_data)

    torch_npu.npu.synchronize()

    c_data_list = [c for r in c_data.cpu().tolist() for c in r]
    d_data_list = [c for r in d_data.cpu().tolist() for c in r]
    assert d_data_list == [v * 11 for v in c_data_list]


@pypto.frontend.jit()
def matmul_add(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [a.shape[0], a.shape[1]], [0, 0])
        b0 = pypto.view(b, [b.shape[0], b.shape[1]], [0, 0])
        d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


@pypto.options(pass_options={"enable_slice": False})
def test_device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    d_data_list = []

    count = 16

    a_data = torch.tensor([[1] * k] * n, dtype=torch.int8, device=f'npu:{device_id}')
    b_data = torch.tensor([[1] * m] * k, dtype=torch.int8, device=f'npu:{device_id}')

    for idx in range(count):
        c_data = torch.tensor([[idx] * m] * n, dtype=torch.int32, device=f'npu:{device_id}')
        d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')

        matmul_add(a_data, b_data, c_data, d_data)
        d_data_list.append(d_data)

    torch_npu.npu.synchronize()

    for idx in range(count):
        # get data and compare result
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)
