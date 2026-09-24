#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import os

import numpy as np
import torch

import pypto


class IndexaPutParamInfo:
    def __init__(self, accumulate, b1, s1, b2, vs, ts):
        self.self_shape = (b1, s1)
        self.values_shape = (b2, s1)
        self.indices_shape = (b2,)
        self.view_shape = (vs,)
        self.tile_shape = (ts,)
        self.accumulate = accumulate


def indexput_comm_test_body(indexput_para, test_func):
    self_shape = indexput_para.self_shape
    values_shape = indexput_para.values_shape
    indices_shape = indexput_para.indices_shape
    view_shape = indexput_para.view_shape
    tile_shape = indexput_para.tile_shape
    accumulate = indexput_para.accumulate
    pypto.runtime._device_init()

    self_tensor = pypto.tensor(self_shape, pypto.DataType.DT_FP32, "PTO_TENSOR_SELF")
    values_tensor = pypto.tensor(values_shape, pypto.DataType.DT_FP32, "PTO_TENSOR_VALUES")
    indices_tensor0 = pypto.tensor(indices_shape, pypto.DataType.DT_INT32, "PTO_TENSOR_INDEX0")
    dst_tensor = pypto.tensor(self_shape, pypto.DataType.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(values_shape[0] / view_shape[0])
    with pypto.function("INDEXPUT_", self_tensor, indices_tensor0, values_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_B0", idx_name="b_idx"):
            pypto.set_vec_tile_shapes(tile_shape[0])
            view_values = pypto.view(
                values_tensor,
                [view_shape[0], values_shape[1]],
                [b_idx * view_shape[0], 0],
                valid_shape=[
                    pypto.min(
                        pypto.symbolic_scalar(values_shape[0]) - b_idx * view_shape[0],
                        pypto.symbolic_scalar(view_shape[0]),
                    ),
                    pypto.symbolic_scalar(values_shape[1]),
                ],
            )
            view_indices0 = pypto.view(
                indices_tensor0,
                [view_shape[0]],
                [b_idx * view_shape[0]],
                valid_shape=[
                    pypto.min(
                        pypto.symbolic_scalar(indices_shape[0]) - b_idx * view_shape[0],
                        pypto.symbolic_scalar(view_shape[0]),
                    )
                ],
            )
            test_func(self_tensor, (view_indices0,), view_values, accumulate=accumulate)
            del view_values, view_indices0
    assert isinstance(dst_tensor, pypto.tensor)

    self_input = torch.ones(self_shape, dtype=torch.float32) * (-1)
    self_copy = self_input.clone()
    values_input = torch.ones(values_shape, dtype=torch.float32)
    random_indices = np.random.choice(range(0, self_shape[0]), indices_shape, False).astype(np.int32)
    indices_input0 = torch.from_numpy(random_indices)
    result_tensor = torch.zeros(self_shape, dtype=torch.float32)

    pto_x1_tensor = pypto.from_torch(self_input, "x1_tensor")
    pto_x2_tensor = pypto.from_torch(indices_input0, "x2_tensor")
    pto_x3_tensor = pypto.from_torch(values_input, "x3_tensor")
    pto_res_tensor = pypto.from_torch(result_tensor, "res_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_x1_tensor, pto_x2_tensor, pto_x3_tensor, pto_res_tensor)

    expect = self_copy.index_put_((indices_input0,), values_input, accumulate=False)
    assert torch.allclose(self_input, expect, rtol=1e-4, atol=1e-5)
    pypto.runtime._device_fini


@pypto.options(pass_options={"enable_slice": True})
def test_index_put__onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    accumulate = False
    b1 = 8
    s1 = 2
    b2 = 4
    vs = 4
    ts = 4
    indexput_para = IndexaPutParamInfo(accumulate, b1, s1, b2, vs, ts)

    indexput_comm_test_body(indexput_para, pypto.index_put_)
