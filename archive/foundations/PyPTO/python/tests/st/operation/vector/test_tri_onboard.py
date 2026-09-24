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
from pypto.symbolic_scalar import SymInt

TORCH_TO_PTO_TYPES = {
    torch.int8: pypto.DT_INT8,
    torch.int16: pypto.DT_INT16,
    torch.int32: pypto.DT_INT32,
    torch.float16: pypto.DT_FP16,
    torch.float32: pypto.DT_FP32,
    torch.bfloat16: pypto.DT_BF16,
}


class TriArgs:
    def __init__(self, diagonal: SymInt, is_upper: bool, view_shape, tile_shape) -> None:
        self.view_shape = view_shape
        self.tile_shape = tile_shape
        self.diagonal = diagonal
        self.is_upper = is_upper


def build_tri_2d(inputs_tensors, outputs_tensors, args: TriArgs):
    shape = inputs_tensors[0].shape
    view_shape = args.view_shape
    tile_shape = args.tile_shape
    is_upper = args.is_upper

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    with pypto.function("TRI", inputs_tensors[0], outputs_tensors[0]):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]
                view_tensor = pypto.view(
                    inputs_tensors[0],
                    view_shape,
                    offsets,
                    valid_shape=[
                        pypto.min(shape[0] - b_idx * view_shape[0], view_shape[0]),
                        pypto.min(shape[1] - s_idx * view_shape[1], view_shape[1]),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                diagonal = args.diagonal + offsets[0] - offsets[1]
                res = pypto.triu(view_tensor, diagonal) if is_upper else pypto.tril(view_tensor, diagonal)
                view_tensor.move(res)
                pypto.assemble(view_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], outputs_tensors[0])
                del view_tensor, res


def run_tri(inputs: List[torch.Tensor], outputs: List[torch.Tensor], args: TriArgs) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    inputs_tensors = [pypto.tensor(x.shape, TORCH_TO_PTO_TYPES[x.dtype]) for x in inputs]
    outputs_tensors = [pypto.tensor(y.shape, TORCH_TO_PTO_TYPES[y.dtype]) for y in outputs]
    build_tri_2d(inputs_tensors, outputs_tensors, args)

    pto_x_tensor = pypto.from_torch(inputs[0], "x_tensor")
    pto_y_tensor = pypto.from_torch(outputs[0], "y_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_x_tensor, pto_y_tensor)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_triu_onboard():
    diagonal = 1
    shape = (12, 12)
    view_shape = (8, 4)
    tile_shape = (5, 16)
    args = TriArgs(diagonal, True, view_shape, tile_shape)
    inputs = [torch.randint(low=-10, high=10, size=shape, dtype=torch.int32)]
    outputs = [torch.zeros(shape, dtype=torch.int32)]
    run_tri(inputs, outputs, args)
    golden = torch.triu(inputs[0], diagonal)
    pypto_out = outputs[0]
    assert torch.allclose(pypto_out.flatten(), golden.flatten(), rtol=1e-4, atol=1e-5)


@pypto.options(pass_options={"enable_slice": True})
def test_tril_onboard():
    diagonal = -1
    shape = (12, 13)
    view_shape = (8, 4)
    tile_shape = (5, 16)
    args = TriArgs(diagonal, False, view_shape, tile_shape)
    inputs = [torch.randint(low=-1, high=1, size=shape, dtype=torch.int32)]
    outputs = [torch.zeros(shape, dtype=torch.int32)]
    run_tri(inputs, outputs, args)
    golden = torch.tril(inputs[0], diagonal)
    pypto_out = outputs[0]
    assert torch.allclose(pypto_out.flatten(), golden.flatten(), rtol=1e-4, atol=1e-5)
