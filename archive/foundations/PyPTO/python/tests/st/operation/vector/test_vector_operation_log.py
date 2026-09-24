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
import pytest
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_log():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    _dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_FP32, "LOG_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_FP32, "LOG_TENSOR_b")

    with pypto.function("LOG", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOG_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOG_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(n)),
                        pypto.min(pypto.symbolic_scalar(m) - b_idx * view_shape[1], pypto.symbolic_scalar(m)),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.log(tile_a))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 99.999 + 0.001
    b_tensor = torch.zeros(n, m, dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)
    golden_data = torch.log(a_tensor)
    assert torch.allclose(b_tensor, golden_data, rtol=1e-6, atol=1e-7)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="Dep operation interface")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_log2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    _dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_FP32, "LOG2_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_FP32, "LOG2_TENSOR_b")

    with pypto.function("LOG2", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOG2_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_LOG2_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(n)),
                        pypto.min(pypto.symbolic_scalar(m) - b_idx * view_shape[1], pypto.symbolic_scalar(m)),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.log(tile_a, pypto.LogBaseType.LOG_2))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 99.999 + 0.001
    b_tensor = torch.zeros(n, m, dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)
    golden_data = torch.log2(a_tensor)
    assert torch.allclose(b_tensor, golden_data, rtol=1e-6, atol=1e-7)
    pypto.runtime._device_fini()


@pytest.mark.skip(reason="Dep operation interface")
@pypto.options(pass_options={"enable_slice": True})
def test_vector_operation_log10():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    _dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_FP32, "LOG10_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_FP32, "LOG10_TENSOR_b")

    with pypto.function("LOG10", a, b):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOG2_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_LOG2_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(pypto.symbolic_scalar(n) - b_idx * view_shape[0], pypto.symbolic_scalar(n)),
                        pypto.min(pypto.symbolic_scalar(m) - b_idx * view_shape[1], pypto.symbolic_scalar(m)),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pypto.log(tile_a, pypto.LogBaseType.LOG_10))
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 99.999 + 0.001
    b_tensor = torch.zeros(n, m, dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)
    golden_data = torch.log10(a_tensor)
    assert torch.allclose(b_tensor, golden_data, rtol=1e-6, atol=1e-7)
    pypto.runtime._device_fini()
