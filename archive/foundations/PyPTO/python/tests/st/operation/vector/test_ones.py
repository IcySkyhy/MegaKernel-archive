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

import numpy as np
from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_ones():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor((n, m), dtype, "VEC_DUP_TENSOR_a")
    with pypto.function("VEC_DUP", a):
        pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_VEC_DUP_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_VEC_DUP_L1", idx_name="s_idx"):
                tile_a = pypto.tensor()
                tile_a.move(pypto.ones(view_shape, dtype=dtype))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], a)
                del tile_a
    a_tensor = torch.zeros(n, m, dtype=torch.float32)
    pto_result_tensor = pypto.from_torch(a_tensor, "output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_result_tensor)
    expected = torch.ones(n, m, dtype=torch.float32)
    assert_allclose(a_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_zeros():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 64
    n, m = tiling * 1, tiling * 1
    view_shape = (32, 32)
    tile_shape = (8, 8)
    pypto.runtime._device_init()

    a = pypto.tensor((n, m), dtype, "VEC_DUP_TENSOR_a")

    with pypto.function("VEC_DUP", a):
        pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_VEC_DUP_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_VEC_DUP_L1", idx_name="s_idx"):
                tile_a = pypto.tensor()
                tile_a.move(pypto.zeros(view_shape[0], view_shape[1], dtype=dtype))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], a)
                del tile_a
    a_tensor = torch.ones(n, m, dtype=torch.float32)
    pto_result_tensor = pypto.from_torch(a_tensor, "output_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_result_tensor)
    expected = torch.zeros(n, m, dtype=torch.float32)
    assert_allclose(a_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()
