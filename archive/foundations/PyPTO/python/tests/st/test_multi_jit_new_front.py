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

import torch
import torch_npu

import pypto


@pypto.frontend.jit()
def cust_dyn_func_add(
    a: pypto.Tensor[[], pypto.DT_INT32], b: pypto.Tensor[[], pypto.DT_INT32], c: pypto.Tensor[[], pypto.DT_INT32]
):
    pypto.set_vec_tile_shapes(32, 32)
    c.move(a + b)


@pypto.frontend.jit()
def cust_dyn_func_sub(
    a: pypto.Tensor[[...], pypto.DT_INT32],
    b: pypto.Tensor[[...], pypto.DT_INT32],
    c: pypto.Tensor[[...], pypto.DT_INT32],
):
    pypto.set_vec_tile_shapes(32, 32)
    c.move(a - b)


def device_run(is_run_add):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)

    # prepare data
    a_rawdata = torch.ones((n, m)) * 2
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    b_rawdata = torch.ones((n, m))
    b_data = b_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    if is_run_add:
        add_result = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')
        cust_dyn_func_add(a_data, b_data, add_result)
        torch_npu.npu.synchronize()

        golden = torch.ones((n, m), dtype=torch.int32) * 3
        assert torch.allclose(golden.int(), add_result.cpu(), atol=1e-5)
    else:
        sub_result = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')
        cust_dyn_func_sub(a_data, b_data, sub_result)
        torch_npu.npu.synchronize()

        golden = torch.ones((n, m))
        assert torch.allclose(golden.int(), sub_result.cpu(), atol=1e-5)


@pypto.options(pass_options={"enable_slice": True})
def test_run_multi_jit():
    device_run(True)
    device_run(False)
    device_run(True)
