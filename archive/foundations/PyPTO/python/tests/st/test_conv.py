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
""" """

import os

import pytest
import torch

import pypto
from pypto import pypto_impl


def create_conv_kernel(
    fmap_shape,
    weight_shape,
    bias_shape,
    out_shape,
    dtype,
    tile_l1_info,
    tile_l0_info,
    strides,
    pads,
    dilations,
    groups=1,
):
    @pypto.frontend.jit(pass_options={"enable_slice": False})
    def conv_kernel(
        fmap: pypto.Tensor(fmap_shape, dtype),
        weight: pypto.Tensor(weight_shape, dtype),
        bias: pypto.Tensor(bias_shape, dtype),
        out: pypto.Tensor(out_shape, dtype),
    ):
        pypto.set_conv_tile_shapes(tile_l1_info, tile_l0_info)
        extend_params = {'bias_tensor': bias}
        output = pypto.conv(fmap, weight, dtype, strides, pads, dilations, extend_params=extend_params, groups=groups)
        out.move(output)

    return conv_kernel


@pytest.mark.soc("910")
def test_conv1d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 32)
    weight_shape = (16, 8, 3)
    bias_shape = (16,)
    out_shape = (1, 16, 15)
    groups = 2
    dtype = pypto.DT_FP16
    tile_l1_info = pypto_impl.TileL1Info(
        tileHin=1, tileHout=1, tileWin=32, tileWout=16, tileCinFmap=16, tileCinWeight=16, tileN=16, tileBatch=1
    )
    tile_l0_info = pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=16, tileN=16)
    strides = [2]
    pads = [1, 1]
    dilations = [2]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')
    d = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    @pypto.frontend.jit(pass_options={"enable_slice": False})
    def conv_kernel(
        fmap: pypto.Tensor(fmap_shape, dtype),
        weight: pypto.Tensor(weight_shape, dtype),
        bias: pypto.Tensor(bias_shape, dtype),
        out: pypto.Tensor(out_shape, dtype),
    ):
        pypto.set_conv_tile_shapes(tile_l1_info, tile_l0_info)
        extend_params = {'bias_tensor': bias}
        output = pypto.conv(fmap, weight, dtype, strides, pads, dilations, extend_params=extend_params, groups=groups)
        out.move(output)

    conv_kernel(a, b, c, d)
    golden = torch.nn.functional.conv1d(a, b, c, stride=(2), padding=(1), dilation=(2), groups=2)

    assert torch.allclose(d.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950")
def test_conv3d_bf16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 5, 5, 32)
    weight_shape = (16, 8, 3, 3, 3)
    bias_shape = (16,)
    out_shape = (1, 16, 2, 2, 15)
    dtype = pypto.DT_BF16
    tile_l1_info = pypto_impl.TileL1Info(
        tileHin=1, tileHout=1, tileWin=32, tileWout=16, tileCinFmap=16, tileCinWeight=16, tileN=16, tileBatch=1
    )
    tile_l0_info = pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=16, tileN=16)
    strides = [2, 2, 2]
    pads = [1, 1, 1, 1, 1, 1]
    dilations = [2, 2, 2]
    dtype_torch = torch.bfloat16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')
    d = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    create_conv_kernel(
        fmap_shape, weight_shape, bias_shape, out_shape, dtype, tile_l1_info, tile_l0_info, strides, pads, dilations, 2
    )(a, b, c, d)
    golden = torch.nn.functional.conv3d(a, b, c, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(2, 2, 2), groups=2)

    assert torch.allclose(d.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)


# ============================================================================
# Dynamic Axis Tests with Dtype Cross and Parameter Variations
# ============================================================================


@pytest.mark.soc("950")
def test_conv2d_dynamic_batch_stride():
    """Conv2D dynamic batch with stride=2, dilation=2, pad=1, dtype=FP16"""
    dtype_torch = torch.float16
    dtype = pypto.DT_FP16
    stride = 2
    dilation = 2
    pad = 1
    fmap_shape = (2, 16, 32, 32)
    weight_shape = (64, 16, 3, 3)
    out_shape = (2, 64, 15, 15)

    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c_out = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    @pypto.frontend.jit(pass_options={"enable_slice": False})
    def conv2d_dynamic_batch_stride_kernel(
        input_a: pypto.Tensor([pypto.DYNAMIC, 16, 32, 32]),
        input_b: pypto.Tensor([64, 16, 3, 3]),
        output_c: pypto.Tensor([pypto.DYNAMIC, 64, 15, 15]),
        params,
    ):
        batch = params["batch"]
        tile_batch = pypto.symbolic_scalar(1)
        batch_loop = (batch + tile_batch - 1) // tile_batch

        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=1, tileHout=1, tileWin=16, tileWout=16, tileCinFmap=16, tileCinWeight=16, tileN=64, tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=144, tileN=64),
        )

        for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
            batch_offset = batch_idx * tile_batch
            input_a_view = pypto.view(input_a, [tile_batch, 16, 32, 32], [batch_offset, 0, 0, 0])
            out = pypto.conv(
                input_a_view,
                input_b,
                dtype,
                [stride, stride],
                [pad, pad, pad, pad],
                [dilation, dilation],
                extend_params={},
                groups=1,
            )
            pypto.assemble(out, [batch_offset, 0, 0, 0], output_c)

    params = {"batch": 2}
    conv2d_dynamic_batch_stride_kernel(a, b, c_out, params)

    golden = torch.nn.functional.conv2d(
        a, b, stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), groups=1
    )
    assert torch.allclose(c_out.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950")
def test_conv2d_dynamic_hout_relu():
    """Conv2D dynamic hout axis with pad=0 (constraint), stride=1, dilation=1, dtype=FP32"""
    dtype_torch = torch.float32
    dtype = pypto.DT_FP32
    stride = 1
    dilation = 1
    pad = 0  # dhw 动态轴切分不支持 pad 非 0
    fmap_shape = (1, 16, 34, 34)
    weight_shape = (64, 16, 3, 3)
    out_shape = (1, 64, 32, 32)

    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c_out = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    @pypto.frontend.jit(pass_options={"enable_slice": False})
    def conv2d_dynamic_hout_kernel(
        input_a: pypto.Tensor([1, 16, pypto.DYNAMIC, 34]),
        input_b: pypto.Tensor([64, 16, 3, 3]),
        output_c: pypto.Tensor([1, 64, pypto.DYNAMIC, 32]),
    ):
        hin = input_a.shape[2]
        ho = output_c.shape[2]
        tile_hout = 8
        hout_loop = (ho + tile_hout - 1) // tile_hout
        tile_hin = tile_hout + 2  # stride=1, kernel=3

        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=8, tileHout=8, tileWin=32, tileWout=32, tileCinFmap=16, tileCinWeight=16, tileN=64, tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(tileH=8, tileW=32, tileK=48, tileN=64),
        )

        for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_hout"):
            hout_offset = hout_idx * tile_hout
            hin_offset = hout_idx * tile_hout
            hin_current = (hin - hin_offset).min(tile_hin)
            input_a_view = pypto.view(
                input_a, [1, 16, tile_hin, 34], [0, 0, hin_offset, 0], valid_shape=[1, 16, hin_current, 34]
            )
            out = pypto.conv(
                input_a_view,
                input_b,
                dtype,
                [stride, stride],
                [pad, pad, pad, pad],
                [dilation, dilation],
                extend_params={'relu_type': pypto_impl.ConvReLuType.RELU},
                groups=1,
            )
            pypto.assemble(out, [0, 0, hout_offset, 0], output_c)

    conv2d_dynamic_hout_kernel(a, b, c_out)

    golden = torch.nn.functional.conv2d(
        a, b, stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), groups=1
    )
    golden = torch.relu(golden)
    assert torch.allclose(c_out.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950")
def test_conv1d_dynamic_cout():
    """Conv1D dynamic cout axis with stride=1, dilation=1, pad=1, dtype=BF16"""
    dtype_torch = torch.bfloat16
    dtype = pypto.DT_BF16
    stride = 1
    dilation = 1
    pad = 1
    fmap_shape = (1, 16, 64)
    weight_shape = (64, 16, 3)
    out_shape = (1, 64, 64)

    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c_out = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    @pypto.frontend.jit(pass_options={"enable_slice": False})
    def conv1d_dynamic_cout_kernel(
        input_a: pypto.Tensor([1, 16, 64]),
        input_b: pypto.Tensor([pypto.DYNAMIC, 16, 3]),
        output_c: pypto.Tensor([1, pypto.DYNAMIC, 64]),
        params,
    ):
        cout = params["cout"]
        tile_cout = pypto.symbolic_scalar(32)
        cout_loop = (cout + tile_cout - 1) // tile_cout

        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=1,
                tileHout=1,
                tileWin=64,
                tileWout=64,
                tileCinFmap=16,
                tileCinWeight=16,
                tileN=tile_cout,
                tileBatch=1,
            ),
            pypto.pypto_impl.TileL0Info(tileH=1, tileW=64, tileK=48, tileN=tile_cout),
        )

        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_cout"):
            cout_offset = cout_idx * tile_cout
            input_b_view = input_b[cout_offset:cout_offset + tile_cout, 0:16, 0:3]
            out = pypto.conv(input_a, input_b_view, dtype, [stride], [pad, pad], [dilation], extend_params={}, groups=1)
            pypto.assemble(out, [0, cout_offset, 0], output_c)

    params = {"cout": 64}
    conv1d_dynamic_cout_kernel(a, b, c_out, params)

    golden = torch.nn.functional.conv1d(a, b, stride=stride, padding=pad, dilation=dilation, groups=1)
    assert torch.allclose(c_out.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)
