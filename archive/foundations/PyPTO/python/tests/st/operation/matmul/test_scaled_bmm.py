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
"""
Scaled BMM (3D/4D) ST test script.
Supports both pytest and direct execution modes.
"""

from dataclasses import dataclass
import os

import pytest
from testcase.scaled_bmm_test_case import SCALED_BMM_TESTS, ScaledBmmConfig
import torch
import torch.nn.functional as functional
import torch_npu

import pypto

K_BLOCK_SIZE_64 = 64
K_BLOCK_SIZE_32 = 32
SHAPE_DIM_2 = 2


@dataclass
class ScaledBmmInputs:
    a_npu: torch.Tensor
    b_npu: torch.Tensor
    scale_a_npu: torch.Tensor
    scale_b_npu: torch.Tensor
    bias_npu: torch.Tensor
    scale_tensor_npu: torch.Tensor
    golden: torch.Tensor
    out_shape: list


@dataclass
class ScaledBmmKernelParams:
    a_tensor: pypto.Tensor
    b_tensor: pypto.Tensor
    out_tensor: pypto.Tensor
    scale_a_tensor: pypto.Tensor
    scale_b_tensor: pypto.Tensor
    bias_tensor: pypto.Tensor
    scale_tensor: pypto.Tensor


@dataclass
class TensorViewParams3D:
    a_tensor: object
    b_tensor: object
    scale_a_tensor: object
    scale_b_tensor: object
    config: object
    batch_offset: int
    m_offset: int
    n_offset: int
    tile_b: int
    vm: int
    vn: int
    scale_k: int
    k: int


@dataclass
class TensorViewParams4D:
    a_tensor: object
    b_tensor: object
    scale_a_tensor: object
    scale_b_tensor: object
    config: object
    batch_outer_offset: int
    batch_inner_offset: int
    m_offset: int
    n_offset: int
    tile_b0: int
    tile_b1: int
    vm: int
    vn: int
    scale_k: int
    k: int


@dataclass
class BiasViewParams3D:
    bias_tensor: object
    batch_offset: int
    n_offset: int
    tile_b: int
    batch_a: int
    batch_b: int
    vn: int


@dataclass(frozen=True)
class BiasViewParams4D:
    bias_tensor: object
    n_offset: int
    vn: int


@dataclass
class Kernel3dInnerParams:
    a_tensor: object
    b_tensor: object
    out_tensor: object
    scale_a_tensor: object
    scale_b_tensor: object
    bias_tensor: object
    scale_tensor: object
    config: object
    tile_b: int
    vm: int
    vn: int
    scale_k: int
    k: int
    batch_a: int
    batch_b: int


@dataclass(frozen=True)
class Process4dLoopParams:
    a_tensor: object
    b_tensor: object
    out_tensor: object
    scale_a_tensor: object
    scale_b_tensor: object
    bias_tensor: object
    scale_tensor: object
    config: object
    vm: int
    vn: int
    scale_k: int
    k: int
    tile_b0: int
    tile_b1: int
    batch_outer_offset: int
    batch_inner_offset: int
    m_loop: int
    n_loop: int


@dataclass
class GoldenComputeParams:
    config: ScaledBmmConfig
    mat_a_cpu: torch.Tensor
    mat_b_cpu: torch.Tensor
    scale_a_cpu: torch.Tensor
    scale_b_cpu: torch.Tensor
    bias_cpu: torch.Tensor
    scale_tensor_cpu: torch.Tensor
    m: int
    n: int


def get_tensor_views_3d(params: TensorViewParams3D):
    batch_a = params.config.a_shape[0]
    batch_b = params.config.b_shape[0]

    if batch_a == 1 and batch_a != batch_b:
        batch_a_start = 0
        batch_a_end = 1
    else:
        batch_a_start = params.batch_offset
        batch_a_end = params.batch_offset + params.tile_b

    if batch_b == 1 and batch_a != batch_b:
        batch_b_start = 0
        batch_b_end = 1
    else:
        batch_b_start = params.batch_offset
        batch_b_end = params.batch_offset + params.tile_b

    if params.config.a_trans:
        a_view = params.a_tensor[batch_a_start:batch_a_end, 0:params.k, params.m_offset:params.m_offset + params.vm]
    else:
        a_view = params.a_tensor[batch_a_start:batch_a_end, params.m_offset:params.m_offset + params.vm, 0:params.k]

    if params.config.b_trans:
        b_view = params.b_tensor[batch_b_start:batch_b_end, params.n_offset:params.n_offset + params.vn, 0:params.k]
    else:
        b_view = params.b_tensor[batch_b_start:batch_b_end, 0:params.k, params.n_offset:params.n_offset + params.vn]

    if params.config.scale_a_trans:
        scale_a_view = params.scale_a_tensor[0:params.scale_k, params.m_offset:params.m_offset + params.vm, :]
    else:
        scale_a_view = params.scale_a_tensor[params.m_offset:params.m_offset + params.vm, 0:params.scale_k, :]

    if params.config.scale_b_trans:
        scale_b_view = params.scale_b_tensor[params.n_offset:params.n_offset + params.vn, 0:params.scale_k, :]
    else:
        scale_b_view = params.scale_b_tensor[0:params.scale_k, params.n_offset:params.n_offset + params.vn, :]

    return a_view, b_view, scale_a_view, scale_b_view


def get_bias_view_3d(params: BiasViewParams3D, config):
    bias_tensor = params.bias_tensor
    batch_offset = params.batch_offset
    n_offset = params.n_offset
    tile_b = params.tile_b
    _batch_a = params.batch_a
    _batch_b = params.batch_b
    vn = params.vn
    if config.bias_shape_type == "b_1_n":
        if config.bias_batch == 1 and params.batch_a != params.batch_b:
            bias_batch_start = 0
            bias_batch_end = 1
        else:
            bias_batch_start = batch_offset
            bias_batch_end = batch_offset + tile_b
        return bias_tensor[bias_batch_start:bias_batch_end, :, n_offset:n_offset + vn]
    else:
        return bias_tensor[:, n_offset:n_offset + vn]


def process_3d_inner_loop(params: Kernel3dInnerParams, batch_offset: int, m_offset: int, n_offset: int):
    view_params = TensorViewParams3D(
        a_tensor=params.a_tensor,
        b_tensor=params.b_tensor,
        scale_a_tensor=params.scale_a_tensor,
        scale_b_tensor=params.scale_b_tensor,
        config=params.config,
        batch_offset=batch_offset,
        m_offset=m_offset,
        n_offset=n_offset,
        tile_b=params.tile_b,
        vm=params.vm,
        vn=params.vn,
        scale_k=params.scale_k,
        k=params.k,
    )
    a_view, b_view, scale_a_view, scale_b_view = get_tensor_views_3d(view_params)

    pypto.set_vec_tile_shapes(params.config.m_tile_shape[0], params.config.n_tile_shape[0], 32)
    tile_shape = (params.config.m_tile_shape, params.config.k_tile_shape, params.config.n_tile_shape)
    pypto.set_cube_tile_shapes(*tile_shape, params.config.enable_ksplit)

    extend_params = {'relu_type': params.config.relu_type}
    if params.bias_tensor is not None:
        bias_params = BiasViewParams3D(
            bias_tensor=params.bias_tensor,
            batch_offset=batch_offset,
            n_offset=n_offset,
            tile_b=params.tile_b,
            batch_a=params.batch_a,
            batch_b=params.batch_b,
            vn=params.vn,
        )
        bias_view = get_bias_view_3d(bias_params, params.config)
        extend_params['bias_tensor'] = bias_view
    if params.config.scale != 0:
        extend_params['scale'] = params.config.scale
    elif params.config.has_scale_tensor and params.scale_tensor is not None:
        scale_tensor_view = params.scale_tensor[0:1, n_offset:n_offset + params.vn]
        extend_params['scale_tensor'] = scale_tensor_view

    out_view = pypto.scaled_mm(
        a_view,
        b_view,
        params.config.out_dtype,
        scale_a_view,
        scale_b_view,
        a_trans=params.config.a_trans,
        b_trans=params.config.b_trans,
        scale_a_trans=params.config.scale_a_trans,
        scale_b_trans=params.config.scale_b_trans,
        c_matrix_nz=params.config.c_format == "NZ",
        extend_params=extend_params,
    )
    pypto.assemble(out_view, [batch_offset, m_offset, n_offset], params.out_tensor)


def d3_kernel_common(params: ScaledBmmKernelParams, config):
    a_tensor, b_tensor, out_tensor = params.a_tensor, params.b_tensor, params.out_tensor
    scale_a_tensor, scale_b_tensor = params.scale_a_tensor, params.scale_b_tensor
    bias_tensor = params.bias_tensor
    scale_tensor = params.scale_tensor
    output_m = config.out_shape[-2]
    output_n = config.out_shape[-1]
    batch = max(config.a_shape[0], config.b_shape[0])
    batch_a, batch_b = config.a_shape[0], config.b_shape[0]
    _, m, k, n = config.get_logical_dims_3d()
    tile_b, vm, vn = config.view_shape
    m_loop, n_loop = (output_m + vm - 1) // vm, (output_n + vn - 1) // vn
    batch_loop = (batch + tile_b - 1) // tile_b
    scale_k = k // K_BLOCK_SIZE_64

    pypto.set_vec_tile_shapes(config.m_tile_shape[0], config.n_tile_shape[0])
    pypto.set_matrix_size([output_m, k, output_n])

    inner_params = Kernel3dInnerParams(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        out_tensor=out_tensor,
        scale_a_tensor=scale_a_tensor,
        scale_b_tensor=scale_b_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
        config=config,
        tile_b=tile_b,
        vm=vm,
        vn=vn,
        scale_k=scale_k,
        k=k,
        batch_a=batch_a,
        batch_b=batch_b,
    )

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_LO_batchIdx", idx_name="batch_idx"):
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                m_offset = m_idx * vm
                n_offset = n_idx * vn
                batch_offset = batch_idx * tile_b
                process_3d_inner_loop(inner_params, batch_offset, m_offset, n_offset)


def process_4d_mn_loops(params: Process4dLoopParams):
    for m_idx in pypto.loop(0, params.m_loop, 1, name="LOOP_L2_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, params.n_loop, 1, name="LOOP_L3_nIdx", idx_name="n_idx"):
            m_offset = m_idx * params.vm
            n_offset = n_idx * params.vn

            view_params = TensorViewParams4D(
                a_tensor=params.a_tensor,
                b_tensor=params.b_tensor,
                scale_a_tensor=params.scale_a_tensor,
                scale_b_tensor=params.scale_b_tensor,
                config=params.config,
                batch_outer_offset=params.batch_outer_offset,
                batch_inner_offset=params.batch_inner_offset,
                m_offset=m_offset,
                n_offset=n_offset,
                tile_b0=params.tile_b0,
                tile_b1=params.tile_b1,
                vm=params.vm,
                vn=params.vn,
                scale_k=params.scale_k,
                k=params.k,
            )
            a_view, b_view, scale_a_view, scale_b_view = get_4d_tensor_views(view_params)

            pypto.set_vec_tile_shapes(params.config.m_tile_shape[0], params.config.n_tile_shape[0], 32)
            tile_shape = (params.config.m_tile_shape, params.config.k_tile_shape, params.config.n_tile_shape)
            pypto.set_cube_tile_shapes(*tile_shape, params.config.enable_ksplit)

            extend_params = {'relu_type': params.config.relu_type}
            if params.bias_tensor is not None:
                bias_view = params.bias_tensor[:, n_offset:n_offset + params.vn]
                extend_params['bias_tensor'] = bias_view
            if params.config.scale != 0:
                extend_params['scale'] = params.config.scale
            elif params.config.has_scale_tensor and params.scale_tensor is not None:
                scale_tensor_view = params.scale_tensor[0:1, n_offset:n_offset + params.vn]
                extend_params['scale_tensor'] = scale_tensor_view

            out_view = pypto.scaled_mm(
                a_view,
                b_view,
                params.config.out_dtype,
                scale_a_view,
                scale_b_view,
                a_trans=params.config.a_trans,
                b_trans=params.config.b_trans,
                scale_a_trans=params.config.scale_a_trans,
                scale_b_trans=params.config.scale_b_trans,
                c_matrix_nz=params.config.c_format == "NZ",
                extend_params=extend_params,
            )
            pypto.assemble(
                out_view, [params.batch_outer_offset, params.batch_inner_offset, m_offset, n_offset], params.out_tensor
            )


def get_4d_tensor_views(params: TensorViewParams4D):
    def get_batch_slice(batch_self, batch_other, offset, tile_size):
        if batch_self == 1 and batch_self != batch_other:
            return 0, 1
        return offset, offset + tile_size

    batch_a_outer = params.config.a_shape[0]
    batch_a_inner = params.config.a_shape[1]
    batch_b_outer = params.config.b_shape[0]
    batch_b_inner = params.config.b_shape[1]

    batch_a_outer_start, batch_a_outer_end = get_batch_slice(
        batch_a_outer, batch_b_outer, params.batch_outer_offset, params.tile_b0
    )
    batch_a_inner_start, batch_a_inner_end = get_batch_slice(
        batch_a_inner, batch_b_inner, params.batch_inner_offset, params.tile_b1
    )
    batch_b_outer_start, batch_b_outer_end = get_batch_slice(
        batch_b_outer, batch_a_outer, params.batch_outer_offset, params.tile_b0
    )
    batch_b_inner_start, batch_b_inner_end = get_batch_slice(
        batch_b_inner, batch_a_inner, params.batch_inner_offset, params.tile_b1
    )

    if params.config.a_trans:
        a_view = params.a_tensor[
            batch_a_outer_start:batch_a_outer_end,
            batch_a_inner_start:batch_a_inner_end,
            0:params.k,
            params.m_offset:params.m_offset + params.vm,
        ]
    else:
        a_view = params.a_tensor[
            batch_a_outer_start:batch_a_outer_end,
            batch_a_inner_start:batch_a_inner_end,
            params.m_offset:params.m_offset + params.vm,
            0:params.k,
        ]
    if params.config.b_trans:
        b_view = params.b_tensor[
            batch_b_outer_start:batch_b_outer_end,
            batch_b_inner_start:batch_b_inner_end,
            params.n_offset:params.n_offset + params.vn,
            0:params.k,
        ]
    else:
        b_view = params.b_tensor[
            batch_b_outer_start:batch_b_outer_end,
            batch_b_inner_start:batch_b_inner_end,
            0:params.k,
            params.n_offset:params.n_offset + params.vn,
        ]
    if params.config.scale_a_trans:
        scale_a_view = params.scale_a_tensor[0:params.scale_k, params.m_offset:params.m_offset + params.vm, :]
    else:
        scale_a_view = params.scale_a_tensor[params.m_offset:params.m_offset + params.vm, 0:params.scale_k, :]
    if params.config.scale_b_trans:
        scale_b_view = params.scale_b_tensor[params.n_offset:params.n_offset + params.vn, 0:params.scale_k, :]
    else:
        scale_b_view = params.scale_b_tensor[0:params.scale_k, params.n_offset:params.n_offset + params.vn, :]
    return a_view, b_view, scale_a_view, scale_b_view


def d4_kernel_common(params: ScaledBmmKernelParams, config):
    a_tensor = params.a_tensor
    b_tensor = params.b_tensor
    out_tensor = params.out_tensor
    scale_a_tensor = params.scale_a_tensor
    scale_b_tensor = params.scale_b_tensor
    bias_tensor = params.bias_tensor
    scale_tensor = params.scale_tensor
    output_m = config.out_shape[-2]
    output_n = config.out_shape[-1]
    b0, b1, m, k, n = config.get_logical_dims_4d()
    tile_b0, tile_b1, vm, vn = config.view_shape
    m_loop, n_loop = (output_m + vm - 1) // vm, (output_n + vn - 1) // vn
    batch_outer_loop = (b0 + tile_b0 - 1) // tile_b0
    batch_inner_loop = (b1 + tile_b1 - 1) // tile_b1
    scale_k = k // K_BLOCK_SIZE_64

    pypto.set_vec_tile_shapes(config.m_tile_shape[0], config.n_tile_shape[0])
    pypto.set_matrix_size([output_m, k, output_n])

    for batch_outer_idx in pypto.loop(0, batch_outer_loop, 1, name="LO_batchOuterIdx", idx_name="batch_outer_idx"):
        for batch_inner_idx in pypto.loop(0, batch_inner_loop, 1, name="L1_batchInnerIdx", idx_name="batch_inner_idx"):
            batch_outer_offset = batch_outer_idx * tile_b0
            batch_inner_offset = batch_inner_idx * tile_b1

            loop_params = Process4dLoopParams(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                out_tensor=out_tensor,
                scale_a_tensor=scale_a_tensor,
                scale_b_tensor=scale_b_tensor,
                bias_tensor=bias_tensor,
                scale_tensor=scale_tensor,
                config=config,
                vm=vm,
                vn=vn,
                scale_k=scale_k,
                k=k,
                tile_b0=tile_b0,
                tile_b1=tile_b1,
                batch_outer_offset=batch_outer_offset,
                batch_inner_offset=batch_inner_offset,
                m_loop=m_loop,
                n_loop=n_loop,
            )
            process_4d_mn_loops(loop_params)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_bmm_kernel_3d_no_bias(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    config: ScaledBmmConfig,
):
    params = ScaledBmmKernelParams(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        out_tensor=out_tensor,
        scale_a_tensor=scale_a_tensor,
        scale_b_tensor=scale_b_tensor,
        bias_tensor=None,
        scale_tensor=scale_tensor,
    )
    d3_kernel_common(params, config)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_bmm_kernel_3d_bias_1n(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    bias_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    scale_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    config: ScaledBmmConfig,
):
    params = ScaledBmmKernelParams(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        out_tensor=out_tensor,
        scale_a_tensor=scale_a_tensor,
        scale_b_tensor=scale_b_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )
    d3_kernel_common(params, config)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_bmm_kernel_3d_bias_b1n(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    bias_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    scale_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    config: ScaledBmmConfig,
):
    params = ScaledBmmKernelParams(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        out_tensor=out_tensor,
        scale_a_tensor=scale_a_tensor,
        scale_b_tensor=scale_b_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )
    d3_kernel_common(params, config)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_bmm_kernel_4d_no_bias(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    config: ScaledBmmConfig,
):
    params = ScaledBmmKernelParams(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        out_tensor=out_tensor,
        scale_a_tensor=scale_a_tensor,
        scale_b_tensor=scale_b_tensor,
        bias_tensor=None,
        scale_tensor=scale_tensor,
    )
    d4_kernel_common(params, config)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_bmm_kernel_4d_bias_1n(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    bias_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    scale_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    config: ScaledBmmConfig,
):
    params = ScaledBmmKernelParams(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        out_tensor=out_tensor,
        scale_a_tensor=scale_a_tensor,
        scale_b_tensor=scale_b_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )
    d4_kernel_common(params, config)


def _process_scale_tensors(scale_a_cpu, scale_b_cpu, config):
    if len(config.a_shape) == 3:
        b, m, k, n = config.get_logical_dims_3d()
    else:
        b0, b1, m, k, n = config.get_logical_dims_4d()

    scale_k_32 = k // K_BLOCK_SIZE_32

    if config.scale_b_trans:
        scale_b_tmp = scale_b_cpu.view(n, scale_k_32).T
    else:
        scale_b_tmp = torch.transpose(scale_b_cpu, -2, -1).reshape(scale_k_32, n)

    if config.scale_a_trans:
        scale_a_tmp = torch.transpose(scale_a_cpu, -2, -1).reshape(scale_k_32, m).T
    else:
        scale_a_tmp = scale_a_cpu.view(m, scale_k_32)

    scale_a_tmp = scale_a_tmp.to(torch.float32).repeat_interleave(32, dim=1)
    scale_b_tmp = scale_b_tmp.to(torch.float32).repeat_interleave(32, dim=0)

    return scale_a_tmp, scale_b_tmp


def _compute_golden(params: GoldenComputeParams):
    config = params.config
    k = config.get_logical_dims_3d()[2] if len(config.a_shape) == 3 else config.get_logical_dims_4d()[3]
    scale_k_32 = k // K_BLOCK_SIZE_32
    output_m = config.out_shape[-2]
    output_n = config.out_shape[-1]
    padding_m = abs(output_m - params.m)
    padding_n = abs(output_n - params.n)
    if config.scale_b_trans:
        scale_b_tmp = params.scale_b_cpu.view(params.n, scale_k_32).T
    else:
        scale_b_tmp = torch.transpose(params.scale_b_cpu, -2, -1).reshape(scale_k_32, params.n)

    if config.scale_a_trans:
        scale_a_tmp = torch.transpose(params.scale_a_cpu, -2, -1).reshape(scale_k_32, params.m).T
    else:
        scale_a_tmp = params.scale_a_cpu.view(params.m, scale_k_32)

    scale_a_tmp = scale_a_tmp.to(torch.float32).repeat_interleave(32, dim=1)
    scale_b_tmp = scale_b_tmp.to(torch.float32).repeat_interleave(32, dim=0)

    mat_a_tmp = (
        params.mat_a_cpu.to(torch.float32).transpose(-2, -1) if config.a_trans else params.mat_a_cpu.to(torch.float32)
    )
    mat_a_tmp *= scale_a_tmp
    mat_b_tmp = (
        params.mat_b_cpu.to(torch.float32).transpose(-2, -1) if config.b_trans else params.mat_b_cpu.to(torch.float32)
    )
    mat_b_tmp = scale_b_tmp * mat_b_tmp
    golden = torch.matmul(mat_a_tmp, mat_b_tmp)
    golden = functional.pad(golden, ((0, padding_n, 0, padding_m)), "constant")
    if params.bias_cpu is not None:
        if len(config.a_shape) == 3 and config.bias_shape_type == "b_1_n":
            golden += params.bias_cpu
        else:
            golden += params.bias_cpu.repeat_interleave(output_m, dim=0)

    if config.relu_type == pypto.ReLuType.RELU:
        golden = torch.relu(golden)

    if config.scale != 0:
        golden = golden * config.scale
    elif config.has_scale_tensor and params.scale_tensor_cpu is not None:
        golden = golden * params.scale_tensor_cpu

    out_dtype = ScaledBmmConfig.pto_to_torch(params.config.out_dtype)
    if out_dtype == torch.int8:
        golden = torch.round(golden.clamp(-128, 127))
    return golden.to(out_dtype)


def prepare_inputs(config: ScaledBmmConfig, device_id: int):
    if len(config.a_shape) == 3:
        b, m, k, n = config.get_logical_dims_3d()
        out_shape = config.out_shape
    else:
        b0, b1, m, k, n = config.get_logical_dims_4d()
        out_shape = config.out_shape
        b = max(b0, b1)

    output_n = out_shape[-1]
    scale_k = k // K_BLOCK_SIZE_64
    scale_a_shape = [scale_k, m, SHAPE_DIM_2] if config.scale_a_trans else [m, scale_k, SHAPE_DIM_2]
    scale_b_shape = [n, scale_k, SHAPE_DIM_2] if config.scale_b_trans else [scale_k, n, SHAPE_DIM_2]

    torch_in_dtype = ScaledBmmConfig.pto_to_torch(config.in_dtype)
    mat_a_cpu = torch.rand(list(config.a_shape), dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    mat_b_cpu = torch.rand(list(config.b_shape), dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_a_cpu = torch.rand(scale_a_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    scale_b_cpu = torch.rand(scale_b_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    bias_cpu = None
    if config.has_bias:
        bias_shape = (
            [b, 1, output_n] if (len(config.a_shape) == 3 and config.bias_shape_type == "b_1_n") else [1, output_n]
        )
        bias_cpu = torch.rand(bias_shape, dtype=torch.float32).uniform_(-3, 3)

    scale_tensor_cpu = None
    if config.has_scale_tensor:
        scale_tensor_cpu = torch.rand([1, output_n], dtype=torch.float32).uniform_(0.01, 0.15)

    golden_params = GoldenComputeParams(
        config=config,
        mat_a_cpu=mat_a_cpu,
        mat_b_cpu=mat_b_cpu,
        scale_a_cpu=scale_a_cpu,
        scale_b_cpu=scale_b_cpu,
        bias_cpu=bias_cpu,
        scale_tensor_cpu=scale_tensor_cpu,
        m=m,
        n=n,
    )
    golden = _compute_golden(golden_params)

    device = f"npu:{device_id}"
    a_npu = mat_a_cpu.to(device)
    b_npu = mat_b_cpu.to(device)
    scale_a_npu = scale_a_cpu.to(device)
    scale_b_npu = scale_b_cpu.to(device)
    bias_npu = bias_cpu.to(device) if bias_cpu is not None else None

    if config.has_scale_tensor:
        scale_tensor_npu = torch_npu.npu_trans_quant_param(scale_tensor_cpu.to(device))
    else:
        scale_tensor_npu = torch.zeros([1, n], dtype=torch.int64, device=device)

    if config.a_format == "NZ":
        a_npu = torch_npu.npu_format_cast(a_npu, 29)
    if config.b_format == "NZ":
        b_npu = torch_npu.npu_format_cast(b_npu, 29)

    return ScaledBmmInputs(a_npu, b_npu, scale_a_npu, scale_b_npu, bias_npu, scale_tensor_npu, golden, out_shape)


def run_scaled_bmm_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = ScaledBmmConfig.from_test_case(case)
    inputs = prepare_inputs(config, device_id)

    out_torch_dtype = ScaledBmmConfig.pto_to_torch(config.out_dtype)
    out_npu = torch.zeros(inputs.out_shape, dtype=out_torch_dtype, device=f"npu:{device_id}")

    if len(config.a_shape) == 3:
        if not config.has_bias:
            scaled_bmm_kernel_3d_no_bias(
                inputs.a_npu,
                inputs.b_npu,
                out_npu,
                inputs.scale_a_npu,
                inputs.scale_b_npu,
                inputs.scale_tensor_npu,
                config,
            )
        else:
            if config.bias_shape_type == "b_1_n":
                scaled_bmm_kernel_3d_bias_b1n(
                    inputs.a_npu,
                    inputs.b_npu,
                    out_npu,
                    inputs.scale_a_npu,
                    inputs.scale_b_npu,
                    inputs.bias_npu,
                    inputs.scale_tensor_npu,
                    config,
                )
            else:
                scaled_bmm_kernel_3d_bias_1n(
                    inputs.a_npu,
                    inputs.b_npu,
                    out_npu,
                    inputs.scale_a_npu,
                    inputs.scale_b_npu,
                    inputs.bias_npu,
                    inputs.scale_tensor_npu,
                    config,
                )
    else:
        if not config.has_bias:
            scaled_bmm_kernel_4d_no_bias(
                inputs.a_npu,
                inputs.b_npu,
                out_npu,
                inputs.scale_a_npu,
                inputs.scale_b_npu,
                inputs.scale_tensor_npu,
                config,
            )
        else:
            scaled_bmm_kernel_4d_bias_1n(
                inputs.a_npu,
                inputs.b_npu,
                out_npu,
                inputs.scale_a_npu,
                inputs.scale_b_npu,
                inputs.bias_npu,
                inputs.scale_tensor_npu,
                config,
            )

    atol, rtol = ScaledBmmConfig.get_tolerance(case["out_dtype"])
    assert torch.allclose(out_npu.cpu(), inputs.golden, atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in SCALED_BMM_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_scaled_bmm(case: dict):
    run_scaled_bmm_test(case)


def run_scaled_bmm_demo(run_mode):
    b_size, m_size, k_size, n_size = 3, 256, 128, 64
    vm_view_size, vn_view_size = 128, 64
    b_view_size = 3

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}, runtime_options={"run_mode": mode}
    )
    def scaled_bmm_demo_kernel(
        a_tensor: pypto.Tensor([], pypto.DT_FP8E4M3),
        b_tensor: pypto.Tensor([], pypto.DT_FP8E4M3),
        out_tensor: pypto.Tensor([], pypto.DT_FP16),
        scale_a_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
        scale_b_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
    ):
        pypto.set_cube_tile_shapes([64, 64], [64, 128], [64, 64])
        pypto.set_vec_tile_shapes(64, 64)
        pypto.set_matrix_size([m_size, k_size, n_size])

        m_loop = (m_size + vm_view_size - 1) // vm_view_size
        n_loop = (n_size + vn_view_size - 1) // vn_view_size
        b_loop = (b_size + b_view_size - 1) // b_view_size

        for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_LO_bIdx", idx_name="b_idx"):
            for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
                for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                    m_offset = m_idx * vm_view_size
                    n_offset = n_idx * vn_view_size
                    b_offset = b_idx * b_view_size

                    a_view = a_tensor[b_offset:b_offset + b_view_size, m_offset:m_offset + vm_view_size, :]
                    b_view = b_tensor[b_offset:b_offset + b_view_size, :, n_offset:n_offset + vn_view_size]
                    scale_a_view = scale_a_tensor[m_offset:m_offset + vm_view_size, :, :]
                    scale_b_view = scale_b_tensor[:, n_offset:n_offset + vn_view_size, :]

                    out_view = pypto.scaled_mm(
                        a_view,
                        b_view,
                        pypto.DT_FP16,
                        scale_a_view,
                        scale_b_view,
                        a_trans=False,
                        b_trans=False,
                        scale_a_trans=False,
                        scale_b_trans=False,
                        c_matrix_nz=False,
                    )
                    out_tensor[
                        b_offset:b_offset + b_view_size,
                        m_offset:m_offset + vm_view_size,
                        n_offset:n_offset + vn_view_size,
                    ] = out_view

    scale_k = k_size // 64
    device = "npu:0" if run_mode == "npu" else "cpu"
    a = torch.randn([b_size, m_size, k_size], dtype=torch.float32).uniform_(-3, 3).to(torch.float8_e4m3fn).to(device)
    b = torch.randn([b_size, k_size, n_size], dtype=torch.float32).uniform_(-3, 3).to(torch.float8_e4m3fn).to(device)
    scale_a = torch.randn([m_size, scale_k, 2], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu).to(device)
    scale_b = torch.randn([scale_k, n_size, 2], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu).to(device)
    out = torch.zeros([b_size, m_size, n_size], dtype=torch.float16).to(device)
    scaled_bmm_demo_kernel(a, b, out, scale_a, scale_b)


if __name__ == "__main__":
    run_scaled_bmm_demo("npu")
