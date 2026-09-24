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

import enum
import math
import os
from typing import List, Optional

import torch

import pypto
from pypto import (
    function,
    set_vec_tile_shapes,
    tensor,
    view,
)
from pypto.symbolic_scalar import SymInt

TORCH_TO_PTO_TYPES = {
    torch.int8: pypto.DT_INT8,
    torch.int16: pypto.DT_INT16,
    torch.int32: pypto.DT_INT32,
    torch.float16: pypto.DT_FP16,
    torch.float32: pypto.DT_FP32,
}


class ClipMode(int, enum.Enum):
    NotDefault2D = 0
    NotDefault3D = 1
    NotDefault4D = 2
    ElementDefaultMinDefaultMax = 3
    ElementDefaultMinNotDefaultMax = 4
    ElementNotDefaultMinDefaultMax = 5
    TensorDefaultMinDefaultMax = 6
    TensorNotDefaultMinDefaultMax = 7
    TensorDefaultMinNotDefaultMax = 8
    NoValue = 10


class ClipArgs:
    tile_shape = None
    view_shape = None
    min = None
    max = None
    is_element = False

    def __init__(
        self,
        tile_shape: List[int],
        view_shape: List[int],
        mode: ClipMode,
        min=None,
        max=None,
        is_element: bool = False,
    ) -> None:
        self.tile_shape = tile_shape
        self.view_shape = view_shape
        self.mode = mode
        self.min = min
        self.max = max
        self.is_element = is_element


def get_broadcast_view_shape(
    self: pypto.Tensor,
    other: pypto.Tensor,
    view_shape: List[int],
) -> List[int]:
    results = []
    for i, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
        if self_dim != other_dim and self_dim == 1 and other_dim != 1:
            results.append(1)
        else:
            results.append(min(self_dim, view_shape[i]))
    return results


def get_broadcast_offset_ratio(
    self: pypto.Tensor,
    other: pypto.Tensor,
) -> List[int]:
    results = []
    for _, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
        if self_dim != other_dim and self_dim == 1 and other_dim != 1:
            results.append(0)
        else:
            results.append(1)
    return results


def get_valid_shape(origin_shapes: List[SymInt], view_shapes: List[int], loop_vars: List[SymInt]) -> List[SymInt]:
    if len(loop_vars) != len(origin_shapes) or len(origin_shapes) != len(view_shapes):
        raise ValueError("Length of `origin_shapes`/`view_shapes` should be the same as `loop_vars`")
    valid_shapes = []
    for origin_shape, view_shape, loop_var in zip(origin_shapes, view_shapes, loop_vars):
        valid_shape = pypto.min(origin_shape - loop_var * view_shape, view_shape)
        valid_shapes.append(valid_shape)
    return valid_shapes


def get_offsets(
    view_shapes: List[int],
    loop_vars: List[SymInt],
    ratios: Optional[List[int]] = None,
) -> List[SymInt]:
    if len(loop_vars) != len(view_shapes):
        raise ValueError("Length of `view_shapes` should be the same as `loop_vars`")
    ratios = ratios or [1] * len(view_shapes)

    offsets = []
    for loop_var, view_shape, ratio in zip(loop_vars, view_shapes, ratios):
        offsets.append(loop_var * view_shape * ratio)
    return offsets


def broadcast_view(
    need_broadcast: pypto.Tensor, broadcasted: pypto.Tensor, view_shapes: List[int], loop_vars: List[SymInt]
) -> pypto.Tensor:
    tile_view_shape = get_broadcast_view_shape(need_broadcast, broadcasted, view_shapes)
    tile_offset_ratio = get_broadcast_offset_ratio(need_broadcast, broadcasted)
    valid_shapes = get_valid_shape(broadcasted.shape, tile_view_shape, loop_vars)
    offsets = get_offsets(tile_view_shape, loop_vars, tile_offset_ratio)
    result = view(need_broadcast, tile_view_shape, offsets, valid_shape=valid_shapes)
    return result


def process_element_mode(tile_tensor_0, args):
    result = tensor()
    if args.mode in [ClipMode.NotDefault2D, ClipMode.NotDefault3D, ClipMode.NotDefault4D]:
        result = pypto.clip(tile_tensor_0, args.min, args.max)
    elif args.mode == ClipMode.ElementDefaultMinDefaultMax:
        result = pypto.clip(tile_tensor_0)
    elif args.mode == ClipMode.ElementDefaultMinNotDefaultMax:
        result = pypto.clip(tile_tensor_0, max=args.max)
    elif args.mode == ClipMode.ElementNotDefaultMinDefaultMax:
        result = pypto.clip(tile_tensor_0, min=args.min)
    return result


def process_tensor_mode(tile_tensor_0, inputs, args, loop_vars):
    result = tensor()
    if args.mode in [ClipMode.NotDefault2D, ClipMode.NotDefault3D, ClipMode.NotDefault4D]:
        min = broadcast_view(inputs[1], inputs[0], args.view_shape, loop_vars)
        max = broadcast_view(inputs[2], inputs[0], args.view_shape, loop_vars)
        result = pypto.clip(tile_tensor_0, min, max)
    elif args.mode == ClipMode.TensorDefaultMinDefaultMax:
        result = pypto.clip(tile_tensor_0)
    elif args.mode == ClipMode.TensorDefaultMinNotDefaultMax:
        max = broadcast_view(inputs[2], inputs[0], args.view_shape, loop_vars)
        result = pypto.clip(tile_tensor_0, max=max)
    elif args.mode == ClipMode.TensorNotDefaultMinDefaultMax:
        min = broadcast_view(inputs[1], inputs[0], args.view_shape, loop_vars)
        result = pypto.clip(tile_tensor_0, min=min)
    return result


def build_clip_2d(inputs, outputs, view_shape, tile_shape, args):
    shape = inputs[0].shape
    view_shape = [min(v, self_dim) for v, self_dim in zip(view_shape, shape)]
    with function("Clip", inputs[0], inputs[1], inputs[2], outputs[0]):
        for b_idx in pypto.loop(math.ceil(shape[0] / view_shape[0])):
            for s_idx in pypto.loop(math.ceil(shape[1] / view_shape[1])):
                loop_vars = [b_idx, s_idx]
                offsets = get_offsets(view_shape, loop_vars)
                valid_shape = get_valid_shape(inputs[0].shape, view_shape, loop_vars)
                tile_tensor_0 = view(inputs[0], view_shape, offsets, valid_shape=valid_shape)
                set_vec_tile_shapes(*tile_shape)

                res = tensor()
                if args.is_element:
                    res.move(process_element_mode(tile_tensor_0, args))
                else:
                    res.move(process_tensor_mode(tile_tensor_0, inputs, args, loop_vars))
                pypto.assemble(res, offsets, outputs[0])


def run_clip(inputs: List[torch.Tensor], outputs: List[torch.Tensor], args: ClipArgs):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()
    input_tensors = [tensor(x.shape, TORCH_TO_PTO_TYPES[x.dtype]) for x in inputs]
    output_tensors = [tensor(x.shape, TORCH_TO_PTO_TYPES[x.dtype]) for x in outputs]
    build_clip_2d(input_tensors, output_tensors, args.view_shape, args.tile_shape, args)
    pto_input_tensors = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_output_tensors = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(outputs)]
    pypto.runtime._device_run_once_data_from_host(*pto_input_tensors, *pto_output_tensors)
    pypto.runtime._device_fini()
    return outputs


@pypto.options(pass_options={"enable_slice": True})
def test_clip_1():
    inputs = [torch.rand(128, 128), torch.rand(128, 1), torch.rand(1, 128)]
    outputs = [torch.rand(128, 128)]
    args = ClipArgs(
        view_shape=[32, 32],
        tile_shape=[17, 8],
        mode=ClipMode.NotDefault2D,
    )

    outputs = run_clip(inputs, outputs, args)
    golden = torch.clip(inputs[0], inputs[1], inputs[2])
    assert torch.allclose(outputs[0], golden, rtol=1e-9, atol=1e-10)
