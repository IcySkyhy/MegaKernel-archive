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

import math
import os

from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_cumsum_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (10, 10)
    view_shape = (10, 4)
    tile_shape = (10, 4)
    pypto.runtime._device_init()

    input1 = pypto.tensor(shape, pypto.DT_INT32, "pypto_TENSOR_input1")
    dim = 0
    output = pypto.tensor(shape, pypto.DT_INT64, "pypto_TENSOR_output")

    b_loop_num = math.ceil(shape[1] / view_shape[1])

    with pypto.function("MAIN", input1, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            view_tensor_a = pypto.view(
                input1,
                view_shape,
                [0, b_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(view_shape[0]),
                    pypto.min(
                        pypto.symbolic_scalar(shape[1]) - b_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                    ),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            view_tensor_a.move(pypto.cumsum(view_tensor_a, dim))
            pypto.assemble(view_tensor_a, [0, b_idx * view_shape[1]], output)
            del view_tensor_a
    assert isinstance(output, pypto.tensor)

    a_tensor = torch.randint(low=-10, high=10, size=[shape[0], shape[1]], dtype=torch.int32)
    b_tensor = torch.zeros(shape[0], shape[1], dtype=torch.int64)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = torch.cumsum(a_tensor, dim)
    assert_allclose(b_tensor.flatten(), golden.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_cumprod_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (10, 10)
    view_shape = (10, 4)
    tile_shape = (10, 4)
    pypto.runtime._device_init()

    input1 = pypto.tensor(shape, pypto.DT_FP32, "pypto_TENSOR_input1")
    dim = 0
    output = pypto.tensor(shape, pypto.DT_FP32, "pypto_TENSOR_output")

    b_loop_num = math.ceil(shape[1] / view_shape[1])

    with pypto.function("MAIN", input1, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            view_tensor_a = pypto.view(
                input1,
                view_shape,
                [0, b_idx * view_shape[1]],
                valid_shape=[
                    pypto.symbolic_scalar(view_shape[0]),
                    pypto.min(
                        pypto.symbolic_scalar(shape[1]) - b_idx * view_shape[1], pypto.symbolic_scalar(view_shape[1])
                    ),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            view_tensor_a.move(pypto.cumprod(view_tensor_a, dim))
            pypto.assemble(view_tensor_a, [0, b_idx * view_shape[1]], output)
            del view_tensor_a
    assert isinstance(output, pypto.tensor)

    a_tensor = torch.randn(shape[0], shape[1], dtype=torch.float32)
    b_tensor = torch.zeros(shape[0], shape[1], dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = torch.cumprod(a_tensor, dim)
    assert_allclose(b_tensor.flatten(), golden.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()


if __name__ == "__main__":
    test_cumprod_onboard()
