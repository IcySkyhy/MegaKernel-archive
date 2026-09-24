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

import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_gather_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 23
    s = 29
    axis = 0
    idx0 = 4
    idx1 = 4
    src_shape = (b, s)
    index_shape = (idx0, idx1)
    view_shape = (b, 4)
    tile_shape = (b, 4)

    pypto.runtime._device_init()

    src_tensor = pypto.tensor(src_shape, pypto.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pypto.tensor(index_shape, pypto.DT_INT32, "PTO_TENSOR_INDEX")
    dst_tensor = pypto.tensor(index_shape, pypto.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(index_shape[0] / view_shape[0])
    s_loop_num = math.ceil(index_shape[1] / view_shape[1])
    with pypto.function("GATHER", src_tensor, index_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_DIV_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="LOOP_SIV_L0", idx_name="s_idx"):
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])

                view_tensor_src = pypto.view(
                    src_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                view_tensor_index = pypto.view(index_tensor, view_shape, [b_idx * view_shape[0], s_idx * view_shape[1]])
                tmp_dst_tensor = pypto.tensor()
                tmp_dst_tensor.move(pypto.gather(view_tensor_src, axis, view_tensor_index))
                pypto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], 0], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.randint(1, 100, src_shape, dtype=torch.int32)
    input1_tensor = torch.randint(0, src_shape[axis], index_shape, dtype=torch.int32)
    result_tensor = torch.zeros(index_shape, dtype=torch.int32)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_input1_tensor = pypto.from_torch(input1_tensor, "input1_tensor")
    pto_result_tensor = pypto.from_torch(result_tensor, "result_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_input1_tensor, pto_result_tensor)

    result = torch.zeros(index_shape, dtype=torch.int32)
    for i in range(index_shape[0]):
        for j in range(index_shape[1]):
            result[i][j] = input0_tensor[input1_tensor[i][j]][j]

    assert torch.equal(result_tensor, result)
    pypto.runtime._device_fini
