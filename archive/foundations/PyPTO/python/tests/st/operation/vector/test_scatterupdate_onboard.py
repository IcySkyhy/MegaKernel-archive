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

import copy
import math
import os

import numpy as np
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_scatterupdate_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 1
    s = 1
    n = 1
    d = 8
    block_num = 1
    block_size = 1
    src_shape = (b, s, n, d)
    index_shape = (b, s)
    dst_shape = (block_num, block_size, n, d)

    view_shape = (b, s, n, d)
    tile_shape = (b, s, n, d)
    pypto.runtime._device_init()

    src_tensor = pypto.tensor(src_shape, pypto.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pypto.tensor(index_shape, pypto.DT_INT32, "PTO_TENSOR_INDEX")
    update_tensor = pypto.tensor(dst_shape, pypto.DT_INT32, "PTO_TENSOR_DST")
    dst_tensor = pypto.tensor(dst_shape, pypto.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(src_shape[0] / view_shape[0])
    s_loop_num = math.ceil(src_shape[1] / view_shape[1])
    with pypto.function("MAIN", src_tensor, index_tensor, update_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pypto.tensor(dst_shape, pypto.DT_INT32, "PTO_TENSOR_TMP")
                view_tensor_src = pypto.view(
                    src_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1], 0, 0],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                        n,
                        d,
                    ],
                )
                view_tensor_index = pypto.view(
                    index_tensor,
                    [view_shape[0], view_shape[1]],
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(index_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(index_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                view_tensor_dst = pypto.view(update_tensor, dst_shape, [0, 0, 0, 0])
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1], tile_shape[2], tile_shape[3])
                tmp_dst_tensor.move(pypto.scatter_update(view_tensor_dst, -2, view_tensor_index, view_tensor_src))
                pypto.set_vec_tile_shapes(1, 64, n, d)
                pypto.assemble(tmp_dst_tensor, [0, 0, 0, 0], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = np.random.uniform(2, 3, src_shape).astype(np.int32)
    input1_tensor = np.random.choice(range(0, dst_shape[0] * dst_shape[1]), index_shape, replace=False).astype(np.int32)
    input2_tensor = np.random.uniform(1, 2, dst_shape).astype(np.int32)
    result = copy.copy(input2_tensor)
    d_data = np.zeros(dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3]).astype(np.int32)

    a_tensor = torch.from_numpy(input0_tensor)
    b_tensor = torch.from_numpy(input1_tensor)
    c_tensor = torch.from_numpy(input2_tensor)
    d_tensor = torch.from_numpy(d_data)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")
    pto_d_tensor = pypto.from_torch(d_tensor, "d_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor, pto_d_tensor)

    for _b in range(b):
        for _s in range(s):
            result[input1_tensor[_b][_s] // block_size][input1_tensor[_b][_s] % block_size][:] = input0_tensor[_b][_s][
                :
            ]
    result_t = torch.from_numpy(result).to(d_tensor.device).to(d_tensor.dtype)
    result_t = result_t.reshape_as(d_tensor)
    assert (d_tensor == result_t).all().item()
    pypto.runtime._device_fini()
