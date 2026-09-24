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
from typing import List

import torch

import pypto

TORCH_TO_PTO_TYPES = {
    torch.int8: pypto.DT_INT8,
    torch.int16: pypto.DT_INT16,
    torch.int32: pypto.DT_INT32,
    torch.float16: pypto.DT_FP16,
    torch.float32: pypto.DT_FP32,
    torch.bfloat16: pypto.DT_BF16,
}


class IndexAddArgs:
    def __init__(self, axis: int, alpha, view_shape, tile_shape):
        self.view_shape = view_shape
        self.tile_shape = tile_shape
        self.value = alpha
        self.axis = axis


def indexadd_2dim_build(inputs: List[pypto.Tensor], args: IndexAddArgs):
    src_shape = inputs[1].shape
    view_shape = args.view_shape
    tile_shape = args.tile_shape
    axis = args.axis
    value = args.value

    b_loop_num = math.ceil(src_shape[0] / view_shape[0])
    s_loop_num = math.ceil(src_shape[1] / view_shape[1])
    with pypto.function("INDEXADD", inputs[0], inputs[1], inputs[2]):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_B0", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="LOOP_S0", idx_name="s_idx"):
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]
                src_valid_shape = [
                    pypto.min(src_shape[0] - b_idx * view_shape[0], view_shape[0]),
                    pypto.min(src_shape[1] - s_idx * view_shape[1], view_shape[1]),
                ]
                view_src = pypto.view(inputs[1], view_shape, offsets, valid_shape=src_valid_shape)
                view_index = pypto.view(
                    inputs[2], [view_shape[axis]], [offsets[axis]], valid_shape=[src_valid_shape[axis]]
                )

                pypto.index_add_(inputs[0], axis, view_index, view_src, alpha=value)
                del view_src, view_index


def run_indexadd(inputs: List[torch.Tensor], args: IndexAddArgs) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 3))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    inputs_tensors = [pypto.tensor(x.shape, TORCH_TO_PTO_TYPES[x.dtype]) for x in inputs]
    indexadd_2dim_build(inputs_tensors, args)

    pto_x1_tensor = pypto.from_torch(inputs[0], "x1_tensor")
    pto_x2_tensor = pypto.from_torch(inputs[1], "x2_tensor")
    pto_x3_tensor = pypto.from_torch(inputs[2], "x3_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_x1_tensor, pto_x2_tensor, pto_x3_tensor)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": False})
def test_indexadd__onboard():
    axis = 0
    alpha = 1.3
    self_shape = [7, 28]
    src_shape = [13, 28]
    index_shape = [src_shape[axis]]
    view_shape = [7, 28]
    tile_shape = [5, 8]
    args = IndexAddArgs(axis, alpha, view_shape, tile_shape)

    inputs = [
        torch.rand(self_shape, dtype=torch.float32) * 200 - 100,
        torch.rand(src_shape, dtype=torch.float32) * 200 - 100,
        torch.randint(0, self_shape[axis], index_shape, dtype=torch.int32),
    ]
    golden = inputs[0].index_add(axis, inputs[2], inputs[1], alpha=alpha)
    run_indexadd(inputs, args)
    pypto_out = inputs[0]
    assert torch.allclose(pypto_out.flatten(), golden.flatten(), rtol=1e-4, atol=1e-5)
