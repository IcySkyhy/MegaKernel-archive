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

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_pack_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    input_shape = (67, 69)
    output_shape = (67 * 69 * 4,)
    tile_shape = (32, 32)
    pypto.runtime._device_init()

    input1 = pypto.tensor(input_shape, pypto.DT_INT32, "PTO_TENSOR_input1")
    output = pypto.tensor(output_shape, pypto.DT_UINT8, "PTO_TENSOR_output")

    with pypto.function("MAIN", input1, output):
        for _ in pypto.loop(1, name="b0", idx_name="bidx"):
            for _ in pypto.loop(1, name="s0", idx_name="sidx"):
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                result = pypto.pack(input1)
                pypto.assemble(result, [0], output)

    assert isinstance(output, pypto.tensor)

    a_tensor = torch.randint(-2147483648, 2147483647, input_shape, dtype=torch.int32)
    b_tensor = torch.zeros(output_shape[0], dtype=torch.uint8)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = a_tensor.flatten().view(torch.uint8)
    assert torch.equal(b_tensor, golden)
    pypto.runtime._device_fini()
