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
import torch

import pypto
from pypto import function, set_vec_tile_shapes, symbolic_scalar, tensor, view


@pypto.options(pass_options={"enable_slice": True})
def test_maximum():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    _scalar_data = 5
    first_dim, second_dim = 90, 90
    view_shape, tile_shape = (64, 64), (32, 32)
    pypto.runtime._device_init()
    x = tensor((first_dim, second_dim), pypto.DT_INT32, "Operand1")
    y = tensor((first_dim, second_dim), pypto.DT_INT32, "Operand2")
    out = tensor((first_dim, second_dim), pypto.DT_INT32, "Operand2")

    first_view_shape, second_view_shape = view_shape

    with function("Maximum", x, y, out):
        for b_idx in pypto.loop(int(np.ceil(first_dim / view_shape[0])), name="LOOP_ADD_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(second_dim / view_shape[1])), name="LOOP_ADD_L1", idx_name="s_idx"):
                tile_tensor_0 = view(
                    x,
                    view_shape,
                    [b_idx * first_view_shape, s_idx * second_view_shape],
                    valid_shape=[
                        pypto.min(
                            symbolic_scalar(first_dim) - b_idx * first_view_shape, symbolic_scalar(first_view_shape)
                        ),
                        pypto.min(
                            symbolic_scalar(second_dim) - s_idx * second_view_shape, symbolic_scalar(second_view_shape)
                        ),
                    ],
                )
                tile_tensor_1 = view(
                    y,
                    view_shape,
                    [b_idx * first_view_shape, s_idx * second_view_shape],
                    valid_shape=[
                        pypto.min(
                            symbolic_scalar(first_dim) - b_idx * first_view_shape, symbolic_scalar(first_view_shape)
                        ),
                        pypto.min(
                            symbolic_scalar(second_dim) - s_idx * second_view_shape, symbolic_scalar(second_view_shape)
                        ),
                    ],
                )
                set_vec_tile_shapes(*tile_shape)
                res = tensor()
                res.move(pypto.maximum(tile_tensor_0, tile_tensor_1))
                pypto.assemble(
                    res,
                    [b_idx * first_view_shape, s_idx * second_view_shape],
                    out,
                )

    nx_tensor = torch.randint(-100, 100, [first_dim, second_dim], dtype=torch.int32)
    ny_tensor = torch.randint(-100, 100, [first_dim, second_dim], dtype=torch.int32)
    nout_tensor = torch.zeros([first_dim, second_dim], dtype=torch.int32)

    pto_nx_tensor = pypto.from_torch(nx_tensor, "nx_tensor")
    pto_ny_tensor = pypto.from_torch(ny_tensor, "ny_tensor")
    pto_nout_tensor = pypto.from_torch(nout_tensor, "nout_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_nx_tensor, pto_ny_tensor, pto_nout_tensor)
    golden_data = torch.maximum(nx_tensor, ny_tensor)
    assert torch.allclose(nout_tensor, golden_data, rtol=1e-9, atol=1e-10)
    pypto.runtime._device_fini()
