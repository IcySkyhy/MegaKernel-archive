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

import math
import os

import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_unpack_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dst_dtype = pypto.DT_INT32
    dst_byte = 4
    input_shape = (256,)
    output_shape = (input_shape[0] // dst_byte,)
    tile_shape = (256,)
    view_shape = (64,)
    pypto.runtime._device_init()

    input1 = pypto.tensor(input_shape, pypto.DT_UINT8, "PTO_TENSOR_input1")
    output = pypto.tensor(output_shape, dst_dtype, "PTO_TENSOR_output")

    loop_num = math.ceil(input_shape[0] / view_shape[0])
    with pypto.function("MAIN", input1, output):
        for idx in pypto.loop(loop_num, name="b0", idx_name="bidx"):
            view_tensor = pypto.view(
                input1,
                view_shape,
                [idx * view_shape[0]],
                valid_shape=[
                    pypto.min(
                        pypto.symbolic_scalar(input_shape[0]) - idx * view_shape[0],
                        pypto.symbolic_scalar(view_shape[0]),
                    ),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0])
            view_tensor = pypto.unpack(view_tensor, dst_dtype)
            pypto.assemble(view_tensor, [idx * view_shape[0] // dst_byte], output)

    assert isinstance(output, pypto.tensor)

    a_tensor = torch.randint(0, 256, input_shape, dtype=torch.uint8)
    b_tensor = torch.zeros(output_shape[0], dtype=torch.int32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = a_tensor.view(torch.int32)
    assert torch.equal(b_tensor, golden)
    pypto.runtime._device_fini()
