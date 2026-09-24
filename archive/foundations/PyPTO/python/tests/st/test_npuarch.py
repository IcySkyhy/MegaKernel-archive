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


def runtime_options_list():
    # 910
    if pypto.platform.npuarch == 'DAV_1001':
        return {"stitch_function_max_num": 128, "device_sched_mode": 3}
    # 910B/910C
    elif pypto.platform.npuarch == 'DAV_2201':
        return {"stitch_function_max_num": 128, "device_sched_mode": 3}
    # 950
    elif pypto.platform.npuarch == 'DAV_3510':
        return {"stitch_function_max_num": 128, "device_sched_mode": 1}
    else:
        return {"stitch_function_max_num": 128, "device_sched_mode": 1}


@pypto.frontend.jit(runtime_options=runtime_options_list())
def add(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=None,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    assert isinstance(pypto.platform.npuarch, str)
    assert pypto.platform.npuarch in ['DAV_1001', 'DAV_2201', 'DAV_3510']
    c.move(a + b)


@pypto.options(pass_options={"enable_slice": True})
def test_npuarch_config():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    tiling = 32
    n, m = tiling * 1, tiling * 1

    a_rawdata = torch.ones((n, m)) * 2
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    b_rawdata = torch.ones((n, m))
    b_data = b_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    c_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')

    add(a_data, b_data, c_data, tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((n, m)) * 3
    assert torch.allclose(golden.int(), c_data.cpu(), atol=1e-5)
