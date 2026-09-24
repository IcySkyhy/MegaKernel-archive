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

from dataclasses import dataclass, field
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
import torch_npu

import pypto

FP32 = pypto.DT_FP32
FP16 = pypto.DT_FP16
INT32 = pypto.DT_INT32
INT8 = pypto.DT_INT8
UINT64 = pypto.DT_UINT64
UINT32 = pypto.DT_UINT32


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    gm_acc: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None


def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = (0,) * batch_num
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        data = functional.pad(data, (batch_padding + (0, padding_n, 0, padding_m)), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        data = functional.pad(data, (batch_padding + (0, padding_n, 0, 0)), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_kernel_with_mn_split(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    m = shape_info.ori_shape[0]
    n = shape_info.ori_shape[2]
    m_view = shape_info.view_shape[0]
    n_view = shape_info.view_shape[1]
    pypto.set_cube_tile_shapes(
        shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape, enable_split_k=shape_info.gm_acc
    )
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if shape_info.a_trans:
                a_view = a_tensor[:, m_idx * m_view:m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view:m_idx * m_view + m_view, :]
            if shape_info.b_trans:
                b_view = b_tensor[n_idx * n_view:n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, n_idx * n_view:n_idx * n_view + n_view]
            out_view = pypto.matmul(
                a_view, b_view, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans, out_dtype=shape_info.out_dtype
            )
            out_tensor[m_idx * m_view:m_idx * m_view + m_view, n_idx * n_view:n_idx * n_view + n_view] = out_view


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def bmm_kernel_with_no_mn_split(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    pypto.set_cube_tile_shapes(
        shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape, enable_split_k=shape_info.gm_acc
    )
    result = pypto.matmul(
        a_tensor, b_tensor, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans, out_dtype=shape_info.out_dtype
    )
    out_tensor.move(result)


@pytest.mark.soc("950", "910")
def test_mm_with_mn_split():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    m = 69
    k = 99
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    shape_info = ShapeConfig(
        [m, k, n],
        [tile_m, tile_m],
        [tile_k, tile_k],
        [tile_n, tile_n],
        [m_view, n_view],
        FP16,
        FP32,
        True,
        True,
        False,
        False,
        False,
        False,
    )
    a1_tensor = torch.rand([k, m], dtype=torch.float16, device=f"npu:{device_id}")
    b1_tensor = torch.rand([n, k], dtype=torch.float16, device=f"npu:{device_id}")
    c1_tensor = torch.zeros([m, n], dtype=torch.float32, device=f"npu:{device_id}")
    golden = torch.matmul(a1_tensor.to(torch.float32).T, b1_tensor.to(torch.float32).T)
    matmul_kernel_with_mn_split(a1_tensor, b1_tensor, c1_tensor, shape_info)
    assert torch.allclose(c1_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
def test_mm_with_mn_split_nz():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(int(device_id))
    m = 64
    k = 128
    n = 128
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    shape_info = ShapeConfig(
        [m, k, n],
        [tile_m, tile_m],
        [tile_k, tile_k],
        [tile_n, tile_n],
        [m_view, n_view],
        FP16,
        FP32,
        True,
        True,
        True,
        True,
        False,
        False,
    )
    a1_tensor = torch.rand([k, m], dtype=torch.float16, device=f'npu:{device_id}')
    b1_tensor = torch.rand([n, k], dtype=torch.float16, device=f'npu:{device_id}')
    c1_tensor = torch.zeros([m, n], dtype=torch.float32, device=f'npu:{device_id}')
    a1_tensor_nz = torch_npu.npu_format_cast(a1_tensor, 29) if shape_info.a_format_nz else a1_tensor
    b1_tensor_nz = torch_npu.npu_format_cast(b1_tensor, 29) if shape_info.b_format_nz else b1_tensor
    golden = torch.matmul(a1_tensor.to(torch.float32).T, b1_tensor.to(torch.float32).T)
    matmul_kernel_with_mn_split(a1_tensor_nz, b1_tensor_nz, c1_tensor, shape_info)
    assert torch.allclose(c1_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)
    torch.npu.synchronize()


@pytest.mark.soc("950", "910")
@pypto.options(pass_options={"enable_slice": False})
def test_bmm_with_mn_split():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    b = 3
    m = 63
    k = 127
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    shape_info = ShapeConfig(
        [b, m, k, n],
        [tile_m, tile_m],
        [tile_k, tile_k],
        [tile_n, tile_n],
        [-1, -1],
        FP16,
        FP32,
        True,
        False,
        False,
        False,
        False,
        False,
    )
    a1_tensor = torch.rand([b, k, m], dtype=torch.float16, device=f'npu:{device_id}')
    b1_tensor = torch.rand([b, k, n], dtype=torch.float16, device=f'npu:{device_id}')
    c1_tensor = torch.zeros([b, m, n], dtype=torch.float32, device=f'npu:{device_id}')
    golden = torch.matmul(a1_tensor.to(torch.float32).transpose(-2, -1), b1_tensor.to(torch.float32))
    bmm_kernel_with_no_mn_split(a1_tensor, b1_tensor, c1_tensor, shape_info)
    assert torch.allclose(c1_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)
