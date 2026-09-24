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

from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_range():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    _dtype = pypto.DT_FP32
    size = 32
    view_shape = (16,)
    tile_shape = (8,)
    start_data = 1.0
    end_data = 32.1
    step_data = 1.0

    pypto.runtime._device_init()

    a = pypto.tensor((1, 1, 1), pypto.DT_FP32, "Range_TENSOR_a")
    b = pypto.tensor((size,), pypto.DT_FP32, "Range_TENSOR_b")
    start = 1.0
    end = 32.1
    step = 1.0

    with pypto.function("RANGE", a, b):
        for b_idx in pypto.loop(1, name="LOOP_L0_b_idex", idx_name="b_idx"):
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.tensor()
            res.move(pypto.arange(start, end, step))
            pypto.assemble(res, [b_idx * view_shape[0]], b)

    a_tensor = torch.rand([1, 1, 1], dtype=torch.float32) * 99.999 + 0.001
    res_tensor = torch.zeros(size, dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_res_tensor)

    expected = torch.arange(start_data, end_data, step_data)
    assert_allclose(res_tensor.flatten(), expected.flatten(), rtol=1e-6, atol=1e-7)

    pypto.runtime._device_fini()
