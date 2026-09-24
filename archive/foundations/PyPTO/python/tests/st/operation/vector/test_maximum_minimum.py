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

import math
import os
from typing import List

import torch

import pypto


def pto_dtype_to_torch_dtype(pto_type: pypto.DataType):
    mapping = {
        pypto.DT_BF16: torch.bfloat16,
        pypto.DT_INT8: torch.int8,
        pypto.DT_INT16: torch.int16,
        pypto.DT_INT32: torch.int32,
        pypto.DT_FP16: torch.float16,
        pypto.DT_FP32: torch.float32,
    }
    return mapping[pto_type]


def test_with_tensor_scalar_minimum(
    scalar: float = 5,
    shape: List[int] = None,
    data_type: pypto.DataType = pypto.DT_INT32,
    data_range: List[int] = None,
    tile_shape: List[int] = None,
    view_shape: List[int] = None,
    function_name: str = "TensorScalarMinimumTest",
):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    data_range = data_range or [-10, 10]
    shape = shape or [16, 16]
    view_shape = view_shape or ([8] * len(shape))
    tile_shape = tile_shape or ([8] * len(shape))

    pypto.runtime._device_init()

    x = pypto.tensor(shape, data_type)
    y = pypto.tensor(shape, data_type)

    with pypto.function(function_name, x, y):
        for b_idx in pypto.loop(math.ceil(shape[0] / view_shape[0])):
            for s_idx in pypto.loop(math.ceil(shape[1] / view_shape[1])):
                tile_tensor_0 = pypto.view(
                    x,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(*tile_shape)
                res = pypto.tensor()
                res.move(pypto.minimum(tile_tensor_0, scalar))
                pypto.assemble(
                    res,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    y,
                )

    nx_tensor = torch.randint(*data_range, shape, dtype=pto_dtype_to_torch_dtype(data_type))
    ny_tensor = torch.zeros(shape, dtype=pto_dtype_to_torch_dtype(data_type))
    pto_nx_tensor = pypto.from_torch(nx_tensor, "nx_tensor")
    pto_ny_tensor = pypto.from_torch(ny_tensor, "ny_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_nx_tensor, pto_ny_tensor)

    golden_data = torch.minimum(nx_tensor, torch.tensor(scalar, dtype=pto_dtype_to_torch_dtype(data_type)))
    assert torch.allclose(ny_tensor, golden_data, rtol=1e-9, atol=1e-10)
    pypto.runtime._device_fini()


def test_with_tensor_scalar_maximum(
    scalar: float = 5,
    shape: List[int] = None,
    data_type: pypto.DataType = pypto.DT_INT32,
    data_range: List[int] = None,
    tile_shape: List[int] = None,
    view_shape: List[int] = None,
    function_name: str = "TensorScalarMaximumTest",
):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    data_range = data_range or [-10, 10]
    shape = shape or [16, 16]
    view_shape = view_shape or ([8] * len(shape))
    tile_shape = tile_shape or ([8] * len(shape))

    pypto.runtime._device_init()

    x = pypto.tensor(shape, data_type)
    y = pypto.tensor(shape, data_type)

    with pypto.function(function_name, x, y):
        for b_idx in pypto.loop(math.ceil(shape[0] / view_shape[0])):
            for s_idx in pypto.loop(math.ceil(shape[1] / view_shape[1])):
                tile_tensor_0 = pypto.view(
                    x,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(*tile_shape)
                res = pypto.tensor()
                res.move(pypto.maximum(tile_tensor_0, scalar))
                pypto.assemble(
                    res,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    y,
                )

    nx_tensor = torch.randint(*data_range, shape, dtype=pto_dtype_to_torch_dtype(data_type))
    ny_tensor = torch.zeros(shape, dtype=pto_dtype_to_torch_dtype(data_type))
    pto_nx_tensor = pypto.from_torch(nx_tensor, "nx_tensor")
    pto_ny_tensor = pypto.from_torch(ny_tensor, "ny_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_nx_tensor, pto_ny_tensor)

    golden_data = torch.maximum(nx_tensor, torch.tensor(scalar, dtype=pto_dtype_to_torch_dtype(data_type)))
    assert torch.allclose(ny_tensor, golden_data, rtol=1e-9, atol=1e-10)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": False})
def test_with_tensor_tensor_minimum(
    shape: List[int] = None,
    data_type: pypto.DataType = pypto.DT_INT32,
    data_range: List[int] = None,
    tile_shape: List[int] = None,
    view_shape: List[int] = None,
    function_name: str = "TensorTensorMinimumTest",
):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    data_range = data_range or [-10, 10]
    shape = shape or [16, 16]
    view_shape = view_shape or ([8] * len(shape))
    tile_shape = tile_shape or ([8] * len(shape))

    pypto.runtime._device_init()

    x = pypto.tensor(shape, data_type)
    y = pypto.tensor(shape, data_type)
    z = pypto.tensor(shape, data_type)

    with pypto.function(function_name, x, y, z):
        for b_idx in pypto.loop(math.ceil(shape[0] / view_shape[0])):
            for s_idx in pypto.loop(math.ceil(shape[1] / view_shape[1])):
                tile_tensor_0 = pypto.view(
                    x,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                tile_tensor_1 = pypto.view(
                    y,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(*tile_shape)
                res = pypto.tensor()
                res.move(pypto.minimum(tile_tensor_0, tile_tensor_1))
                pypto.assemble(
                    res,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    z,
                )

    nx_tensor = torch.randint(*data_range, shape, dtype=pto_dtype_to_torch_dtype(data_type))
    ny_tensor = torch.randint(*data_range, shape, dtype=pto_dtype_to_torch_dtype(data_type))
    nz_tensor = torch.zeros(shape, dtype=pto_dtype_to_torch_dtype(data_type))

    pto_nx_tensor = pypto.from_torch(nx_tensor, "nx_tensor")
    pto_ny_tensor = pypto.from_torch(ny_tensor, "ny_tensor")
    pto_nz_tensor = pypto.from_torch(nz_tensor, "nz_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_nx_tensor, pto_ny_tensor, pto_nz_tensor)

    golden_data = torch.minimum(nx_tensor, ny_tensor)
    assert torch.allclose(nz_tensor, golden_data, rtol=1e-9, atol=1e-10)
    pypto.runtime._device_fini()


def test_with_tensor_tensor_maximum(
    shape: List[int] = None,
    data_type: pypto.DataType = pypto.DT_INT32,
    data_range: List[int] = None,
    tile_shape: List[int] = None,
    view_shape: List[int] = None,
    function_name: str = "TensorTensorMaximumTest",
):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    data_range = data_range or [-10, 10]
    shape = shape or [16, 16]
    view_shape = view_shape or ([8] * len(shape))
    tile_shape = tile_shape or ([8] * len(shape))

    pypto.runtime._device_init()

    x = pypto.tensor(shape, data_type)
    y = pypto.tensor(shape, data_type)
    z = pypto.tensor(shape, data_type)

    with pypto.function(function_name, x, y, z):
        for b_idx in pypto.loop(math.ceil(shape[0] / view_shape[0])):
            for s_idx in pypto.loop(math.ceil(shape[1] / view_shape[1])):
                tile_tensor_0 = pypto.view(
                    x,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                tile_tensor_1 = pypto.view(
                    y,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(*tile_shape)
                res = pypto.tensor()
                res.move(pypto.maximum(tile_tensor_0, tile_tensor_1))
                pypto.assemble(
                    res,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    z,
                )

    nx_tensor = torch.randint(*data_range, shape, dtype=pto_dtype_to_torch_dtype(data_type))
    ny_tensor = torch.randint(*data_range, shape, dtype=pto_dtype_to_torch_dtype(data_type))
    nz_tensor = torch.zeros(shape, dtype=pto_dtype_to_torch_dtype(data_type))

    pto_nx_tensor = pypto.from_torch(nx_tensor, "nx_tensor")
    pto_ny_tensor = pypto.from_torch(ny_tensor, "ny_tensor")
    pto_nz_tensor = pypto.from_torch(nz_tensor, "nz_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_nx_tensor, pto_ny_tensor, pto_nz_tensor)

    golden_data = torch.maximum(nx_tensor, ny_tensor)
    assert torch.allclose(nz_tensor, golden_data, rtol=1e-9, atol=1e-10)
    pypto.runtime._device_fini()
