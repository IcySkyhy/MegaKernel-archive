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
import math
import os

from numpy.testing import assert_allclose
import torch

import pypto

TORCH_TO_PTO_TYPES = {
    torch.int8: pypto.DT_INT8,
    torch.int16: pypto.DT_INT16,
    torch.int32: pypto.DT_INT32,
    torch.float16: pypto.DT_FP16,
    torch.float32: pypto.DT_FP32,
    torch.bfloat16: pypto.DT_BF16,
    torch.uint8: pypto.DT_UINT8,
}


def dequantize_golden(input_tensor, scale, axis, zero_points=None):
    """Golden reference: matches dequantize formula."""
    normalized_axis = axis if axis >= 0 else input_tensor.dim() + axis

    # Broadcast scale
    if normalized_axis == 1:  # axis=-1 for 2D: per-row scale
        scale_bc = scale.unsqueeze(1)
    else:  # axis=-2 for 2D: per-col scale
        scale_bc = scale.unsqueeze(0)

    result = input_tensor.to(torch.float32)

    if zero_points is not None:
        if normalized_axis == 1:
            zp_bc = zero_points.unsqueeze(1)
        else:
            zp_bc = zero_points.unsqueeze(0)
        result = result - zp_bc

    result = result * scale_bc

    return result


@pypto.options(pass_options={"enable_slice": True})
def test_dequantize_sym_axis_neg1_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    input_shape = [4, 16]
    scale_shape = [4]
    axis = -1
    view_shape = [4, 16]
    tile_shape = [4, 16]

    pypto.runtime._device_init()

    input1 = pypto.tensor(input_shape, pypto.DT_INT8, "PTO_TENSOR_input1")
    scale1 = pypto.tensor(scale_shape, pypto.DT_FP32, "PTO_TENSOR_scale1")
    output = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])
    s_loop_num = math.ceil(input_shape[1] / view_shape[1])

    output_dtype = pypto.DT_FP32

    with pypto.function("MAIN", input1, scale1, output):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_B0", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="LOOP_S0", idx_name="s_idx"):
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]

                # View input (2D)
                view_input = pypto.view(
                    input1,
                    view_shape,
                    offsets,
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(input_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(input_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )

                # View scale (1D) - axis=-1, so scale is per-row, shape=[4], view along axis=0
                view_scale = pypto.view(
                    scale1,
                    [view_shape[0]],
                    [offsets[0]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(scale_shape[0]) - offsets[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                    ],
                )

                res = pypto.dequantize(view_input, view_scale, output_dtype, axis)
                pypto.assemble(res, offsets, output)

    input_tensor = torch.randint(-128, 127, input_shape, dtype=torch.int8)
    scale_tensor = torch.rand(scale_shape, dtype=torch.float32) * 0.14 + 0.01
    out_tensor = torch.zeros(input_shape, dtype=torch.float32)

    pto_input1 = pypto.from_torch(input_tensor, "input1")
    pto_scale1 = pypto.from_torch(scale_tensor, "scale1")
    pto_output = pypto.from_torch(out_tensor, "output")

    pypto.runtime._device_run_once_data_from_host(pto_input1, pto_scale1, pto_output)

    golden = dequantize_golden(input_tensor, scale_tensor, axis)
    assert_allclose(out_tensor.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_dequantize_sym_axis_neg1_aligned_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    input_shape = [32, 64]
    scale_shape = [32]
    axis = -1
    view_shape = [32, 64]
    tile_shape = [32, 64]

    pypto.runtime._device_init()

    input1 = pypto.tensor(input_shape, pypto.DT_INT8, "PTO_TENSOR_input1")
    scale1 = pypto.tensor(scale_shape, pypto.DT_FP32, "PTO_TENSOR_scale1")
    output = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])
    s_loop_num = math.ceil(input_shape[1] / view_shape[1])

    output_dtype = pypto.DT_FP32

    with pypto.function("MAIN", input1, scale1, output):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_B0", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="LOOP_S0", idx_name="s_idx"):
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]

                # View input (2D)
                view_input = pypto.view(
                    input1,
                    view_shape,
                    offsets,
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(input_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(input_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )

                # View scale (1D)
                view_scale = pypto.view(
                    scale1,
                    [view_shape[0]],
                    [offsets[0]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(scale_shape[0]) - offsets[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                    ],
                )

                res = pypto.dequantize(view_input, view_scale, output_dtype, axis)
                pypto.assemble(res, offsets, output)

    input_tensor = torch.randint(-128, 127, input_shape, dtype=torch.int8)
    scale_tensor = torch.rand(scale_shape, dtype=torch.float32)
    out_tensor = torch.zeros(input_shape, dtype=torch.float32)

    pto_input1 = pypto.from_torch(input_tensor, "input1")
    pto_scale1 = pypto.from_torch(scale_tensor, "scale1")
    pto_output = pypto.from_torch(out_tensor, "output")

    pypto.runtime._device_run_once_data_from_host(pto_input1, pto_scale1, pto_output)

    golden = dequantize_golden(input_tensor, scale_tensor, axis)
    assert_allclose(out_tensor.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_dequantize_asym_axis_neg1_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    input_shape = [4, 16]
    scale_shape = [4]
    axis = -1
    view_shape = [4, 16]
    tile_shape = [4, 16]

    pypto.runtime._device_init()

    input1 = pypto.tensor(input_shape, pypto.DT_INT8, "PTO_TENSOR_input1")
    scale1 = pypto.tensor(scale_shape, pypto.DT_FP32, "PTO_TENSOR_scale1")
    zp1 = pypto.tensor(scale_shape, pypto.DT_FP32, "PTO_TENSOR_zp1")
    output = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])
    s_loop_num = math.ceil(input_shape[1] / view_shape[1])

    output_dtype = pypto.DT_FP32

    with pypto.function("MAIN", input1, scale1, zp1, output):
        loop_count = 0
        for b_idx in pypto.loop(b_loop_num, name="LOOP_B0", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="LOOP_S0", idx_name="s_idx"):
                loop_count += 1
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                offsets = [b_idx * view_shape[0], s_idx * view_shape[1]]

                # View input (2D)
                view_input = pypto.view(
                    input1,
                    view_shape,
                    offsets,
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(input_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(input_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )

                # View scale (1D)
                view_scale = pypto.view(
                    scale1,
                    [view_shape[0]],
                    [offsets[0]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(scale_shape[0]) - offsets[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                    ],
                )

                # View zero_points (1D)
                view_zp = pypto.view(
                    zp1,
                    [view_shape[0]],
                    [offsets[0]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(scale_shape[0]) - offsets[0], pypto.symbolic_scalar(view_shape[0])
                        ),
                    ],
                )

                res = pypto.dequantize(view_input, view_scale, output_dtype, axis, view_zp)
                pypto.assemble(res, offsets, output)

    input_tensor = torch.randint(-128, 127, input_shape, dtype=torch.int8)
    scale_tensor = torch.rand(scale_shape, dtype=torch.float32) * 0.14 + 0.01
    zero_points = torch.rand(scale_shape, dtype=torch.float32) * 10
    out_tensor = torch.zeros(input_shape, dtype=torch.float32)

    pto_input1 = pypto.from_torch(input_tensor, "input1")
    pto_scale1 = pypto.from_torch(scale_tensor, "scale1")
    pto_zp1 = pypto.from_torch(zero_points, "zp1")
    pto_output = pypto.from_torch(out_tensor, "output")

    pypto.runtime._device_run_once_data_from_host(pto_input1, pto_scale1, pto_zp1, pto_output)

    golden = dequantize_golden(input_tensor, scale_tensor, axis, zero_points)
    assert_allclose(out_tensor.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()
