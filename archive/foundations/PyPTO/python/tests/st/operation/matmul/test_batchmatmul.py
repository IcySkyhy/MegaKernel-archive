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
BatchMatmul BASIC_TESTS test script.
Supports both pytest and direct execution modes.
"""

from dataclasses import dataclass
import os
from typing import Tuple

import pytest
from testcase.batchmatmul_test_case import BASIC_3D_TESTS, BASIC_4D_TESTS, BatchMatmulConfig
import torch
import torch.nn.functional as functional
import torch_npu

import pypto


# ========== 参数封装类 ==========
@dataclass
class SliceContext3D:
    """3D 切片与分块参数"""

    batch_slice: Tuple[int, int]
    offset: int
    k: int
    tile_size: int


@dataclass
class SliceContext4D:
    """4D 切片与分块参数"""

    b0_slice: Tuple[int, int]
    b1_slice: Tuple[int, int]
    offset: int
    k: int
    tile_size: int


@dataclass
class TileProcessContext4D:
    """4D tile处理所需的统一参数封装"""

    a_tensor: any
    b_tensor: any
    out_tensor: any
    config: BatchMatmulConfig
    b0_idx: int
    b1_idx: int
    m_idx: int = 0
    n_idx: int = 0
    tile_b0: int = 0
    tile_b1: int = 0
    tile_m: int = 0
    tile_n: int = 0
    k: int = 0
    b0_a: int = 0
    b1_a: int = 0
    b0_b: int = 0
    b1_b: int = 0
    m_loop: int = 0
    n_loop: int = 0


def get_batch_slice(batch_size, tile_size, offset, other_batch_size):
    if batch_size == 1 and batch_size != other_batch_size:
        return 0, 1
    return offset, offset + tile_size


def get_a_view_3d(a_tensor, config, ctx: SliceContext3D):
    start, end = ctx.batch_slice
    if config.a_trans:
        return a_tensor[start:end, 0:ctx.k, ctx.offset:ctx.offset + ctx.tile_size]
    return a_tensor[start:end, ctx.offset:ctx.offset + ctx.tile_size, 0:ctx.k]


def get_b_view_3d(b_tensor, config, ctx: SliceContext3D):
    start, end = ctx.batch_slice
    if config.b_trans:
        return b_tensor[start:end, ctx.offset:ctx.offset + ctx.tile_size, 0:ctx.k]
    return b_tensor[start:end, 0:ctx.k, ctx.offset:ctx.offset + ctx.tile_size]


def get_a_view_4d(a_tensor, config, ctx: SliceContext4D):
    b0_start, b0_end = ctx.b0_slice
    b1_start, b1_end = ctx.b1_slice
    if config.a_trans:
        return a_tensor[b0_start:b0_end, b1_start:b1_end, 0:ctx.k, ctx.offset:ctx.offset + ctx.tile_size]
    return a_tensor[b0_start:b0_end, b1_start:b1_end, ctx.offset:ctx.offset + ctx.tile_size, 0:ctx.k]


def get_b_view_4d(b_tensor, config, ctx: SliceContext4D):
    b0_start, b0_end = ctx.b0_slice
    b1_start, b1_end = ctx.b1_slice
    if config.b_trans:
        return b_tensor[b0_start:b0_end, b1_start:b1_end, ctx.offset:ctx.offset + ctx.tile_size, 0:ctx.k]
    return b_tensor[b0_start:b0_end, b1_start:b1_end, 0:ctx.k, ctx.offset:ctx.offset + ctx.tile_size]


def process_tile_4d(ctx: TileProcessContext4D):
    """处理单个4D tile的矩阵乘计算（参数已封装）"""
    m_offset = ctx.m_idx * ctx.tile_m
    n_offset = ctx.n_idx * ctx.tile_n
    b0_offset = ctx.b0_idx * ctx.tile_b0
    b1_offset = ctx.b1_idx * ctx.tile_b1

    b0_a_slice = get_batch_slice(ctx.b0_a, ctx.tile_b0, b0_offset, ctx.b0_b)
    b1_a_slice = get_batch_slice(ctx.b1_a, ctx.tile_b1, b1_offset, ctx.b1_b)
    b0_b_slice = get_batch_slice(ctx.b0_b, ctx.tile_b0, b0_offset, ctx.b0_a)
    b1_b_slice = get_batch_slice(ctx.b1_b, ctx.tile_b1, b1_offset, ctx.b1_a)

    ctx_a = SliceContext4D(b0_slice=b0_a_slice, b1_slice=b1_a_slice, offset=m_offset, k=ctx.k, tile_size=ctx.tile_m)
    ctx_b = SliceContext4D(b0_slice=b0_b_slice, b1_slice=b1_b_slice, offset=n_offset, k=ctx.k, tile_size=ctx.tile_n)

    a_view = get_a_view_4d(ctx.a_tensor, ctx.config, ctx_a)
    b_view = get_b_view_4d(ctx.b_tensor, ctx.config, ctx_b)

    out_view = pypto.matmul(
        a_view, b_view, out_dtype=ctx.config.out_dtype, a_trans=ctx.config.a_trans, b_trans=ctx.config.b_trans
    )

    ctx.out_tensor[
        b0_offset:b0_offset + ctx.tile_b0,
        b1_offset:b1_offset + ctx.tile_b1,
        m_offset:m_offset + ctx.tile_m,
        n_offset:n_offset + ctx.tile_n,
    ] = out_view


def process_mn_loops_4d(ctx: TileProcessContext4D):
    """处理单个batch pair的MN循环（降低主函数嵌套深度）"""
    for m_idx in pypto.loop(0, ctx.m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, ctx.n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            ctx.m_idx = m_idx
            ctx.n_idx = n_idx
            process_tile_4d(ctx)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def batch_matmul_kernel_3d(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    config: BatchMatmulConfig,
):
    b, m, k, n = config.get_logical_dims_3d()
    output_m = config.out_shape[-2]
    output_n = config.out_shape[-1]
    pypto.set_cube_tile_shapes(*config.tile_shape, config.is_acc)
    pypto.set_vec_tile_shapes(128, 128)
    tile_b = config.view_shape[0]
    tile_m = config.view_shape[1]
    tile_n = config.view_shape[2]

    batch_a = config.a_shape[0]
    batch_b = config.b_shape[0]

    m_loop = (output_m + tile_m - 1) // tile_m
    n_loop = (output_n + tile_n - 1) // tile_n
    b_loop = (b + tile_b - 1) // tile_b
    pypto.set_matrix_size([output_m, k, output_n])

    for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                m_offset = m_idx * tile_m
                n_offset = n_idx * tile_n
                b_offset = b_idx * tile_b

                batch_a_slice = get_batch_slice(batch_a, tile_b, b_offset, batch_b)
                batch_b_slice = get_batch_slice(batch_b, tile_b, b_offset, batch_a)

                a_ctx = SliceContext3D(batch_slice=batch_a_slice, offset=m_offset, k=k, tile_size=tile_m)
                b_ctx = SliceContext3D(batch_slice=batch_b_slice, offset=n_offset, k=k, tile_size=tile_n)

                a_view = get_a_view_3d(a_tensor, config, a_ctx)
                b_view = get_b_view_3d(b_tensor, config, b_ctx)

                out_view = pypto.matmul(
                    a_view, b_view, out_dtype=config.out_dtype, a_trans=config.a_trans, b_trans=config.b_trans
                )
                out_tensor[b_offset:b_offset + tile_b, m_offset:m_offset + tile_m, n_offset:n_offset + tile_n] = (
                    out_view
                )


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def batch_matmul_kernel_4d(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    config: BatchMatmulConfig,
):
    b0, b1, m, k, n = config.get_logical_dims_4d()
    b0_a = config.a_shape[0]
    b1_a = config.a_shape[1]
    b0_b = config.b_shape[0]
    b1_b = config.b_shape[1]
    output_m = config.out_shape[-2]
    output_n = config.out_shape[-1]
    pypto.set_cube_tile_shapes(*config.tile_shape, config.is_acc)
    pypto.set_vec_tile_shapes(1, 128, 128)
    tile_b0 = config.view_shape[0]
    tile_b1 = config.view_shape[1]
    tile_m = config.view_shape[2]
    tile_n = config.view_shape[3]

    m_loop = (output_m + tile_m - 1) // tile_m
    n_loop = (output_n + tile_n - 1) // tile_n
    b0_loop = (b0 + tile_b0 - 1) // tile_b0
    b1_loop = (b1 + tile_b1 - 1) // tile_b1
    pypto.set_matrix_size([output_m, k, output_n])

    for b0_idx in pypto.loop(0, b0_loop, 1, name="LOOP_L0_b0Idx", idx_name="b0_idx"):
        for b1_idx in pypto.loop(0, b1_loop, 1, name="LOOP_L0_b1Idx", idx_name="b1_idx"):
            ctx = TileProcessContext4D(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                out_tensor=out_tensor,
                config=config,
                b0_idx=b0_idx,
                b1_idx=b1_idx,
                tile_b0=tile_b0,
                tile_b1=tile_b1,
                tile_m=tile_m,
                tile_n=tile_n,
                k=k,
                b0_a=b0_a,
                b1_a=b1_a,
                b0_b=b0_b,
                b1_b=b1_b,
                m_loop=m_loop,
                n_loop=n_loop,
            )
            process_mn_loops_4d(ctx)


def prepare_tensors_3d(config, a_dtype, b_dtype, c_dtype, device_id):
    b, m, k, n = config.get_logical_dims_3d()
    output_shape = config.out_shape
    output_m = output_shape[-2]
    output_n = output_shape[-1]
    padding_m = abs(output_m - m)
    padding_n = abs(output_n - n)
    a_shape = config.a_shape
    b_shape = config.b_shape
    c_shape = output_shape

    if a_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-2, 3, a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.randint(-2, 3, b_shape, dtype=b_dtype)
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype)

    a_cpu = a_tensor_cpu.transpose(1, 2) if config.a_trans else a_tensor_cpu
    b_cpu = b_tensor_cpu.transpose(1, 2) if config.b_trans else b_tensor_cpu
    accum_dtype = torch.int32 if a_dtype == torch.int8 else torch.float32
    golden = torch.matmul(a_cpu.to(accum_dtype), b_cpu.to(accum_dtype)).to(c_dtype)
    golden = functional.pad(golden, ((0, padding_n, 0, padding_m)), "constant")
    a_tensor = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = b_tensor_cpu.to(f"npu:{device_id}")
    if config.a_format == "NZ":
        a_tensor = torch_npu.npu_format_cast(a_tensor, 29)
    if config.b_format == "NZ":
        b_tensor = torch_npu.npu_format_cast(b_tensor, 29)
    c_tensor = torch.zeros(c_shape, dtype=c_dtype, device=f"npu:{device_id}")

    return a_tensor, b_tensor, c_tensor, golden


def prepare_tensors_4d(config, a_dtype, b_dtype, c_dtype, device_id):
    b0, b1, m, k, n = config.get_logical_dims_4d()
    output_shape = config.out_shape
    output_m = output_shape[-2]
    output_n = output_shape[-1]
    padding_m = abs(m - output_m)
    padding_n = abs(n - output_n)
    a_shape = config.a_shape
    b_shape = config.b_shape
    c_shape = output_shape

    if a_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-2, 3, a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.randint(-2, 3, b_shape, dtype=b_dtype)
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype)

    a_cpu = a_tensor_cpu.transpose(2, 3) if config.a_trans else a_tensor_cpu
    b_cpu = b_tensor_cpu.transpose(2, 3) if config.b_trans else b_tensor_cpu
    accum_dtype = torch.int32 if a_dtype == torch.int8 else torch.float32
    golden = torch.matmul(a_cpu.to(accum_dtype), b_cpu.to(accum_dtype)).to(c_dtype)
    golden = functional.pad(golden, ((0, padding_n, 0, padding_m)), "constant")
    a_tensor = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = b_tensor_cpu.to(f"npu:{device_id}")
    c_tensor = torch.zeros(c_shape, dtype=c_dtype, device=f"npu:{device_id}")

    return a_tensor, b_tensor, c_tensor, golden


def run_batch_matmul_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = BatchMatmulConfig.from_test_case(case)
    a_dtype = BatchMatmulConfig.get_torch_dtype(case["a_dtype"])
    b_dtype = BatchMatmulConfig.get_torch_dtype(case["b_dtype"])
    c_dtype = BatchMatmulConfig.get_torch_dtype(case["c_dtype"])

    if config.dim == 3:
        a_tensor, b_tensor, c_tensor, golden = prepare_tensors_3d(config, a_dtype, b_dtype, c_dtype, device_id)
        batch_matmul_kernel_3d(a_tensor, b_tensor, c_tensor, config)
    else:
        a_tensor, b_tensor, c_tensor, golden = prepare_tensors_4d(config, a_dtype, b_dtype, c_dtype, device_id)
        batch_matmul_kernel_4d(a_tensor, b_tensor, c_tensor, config)

    atol, rtol = BatchMatmulConfig.get_tolerance(case["c_dtype"])
    assert torch.allclose(c_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in BASIC_3D_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_batch_matmul_3d_nd(case: dict):
    run_batch_matmul_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in BASIC_4D_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_batch_matmul_4d_nd(case: dict):
    run_batch_matmul_test(case)


def run_batch_matmul_demo():
    b_size, m_size, k_size, n_size = 4, 256, 256, 256
    m_view_size, n_view_size = 128, 128
    b_view_size = 2

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def batch_matmul_demo_kernel(
        a: pypto.Tensor([], pypto.DT_FP16),
        b: pypto.Tensor([], pypto.DT_FP16),
        out: pypto.Tensor([], pypto.DT_FP32),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        pypto.set_vec_tile_shapes(128, 128)
        pypto.set_matrix_size([m_size, k_size, n_size])

        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size
        b_loop = (b_size + b_view_size - 1) // b_view_size

        for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
            for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
                for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                    a_view = a[
                        b_idx * b_view_size:b_idx * b_view_size + b_view_size,
                        m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                        :,
                    ]
                    b_view = b[
                        b_idx * b_view_size:b_idx * b_view_size + b_view_size,
                        :,
                        n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                    ]
                    out_view = pypto.matmul(a_view, b_view, pypto.DT_FP32)
                    out[
                        b_idx * b_view_size:b_idx * b_view_size + b_view_size,
                        m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                        n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                    ] = out_view

    a = torch.randn([b_size, m_size, k_size], dtype=torch.float16, device="npu:0")
    b = torch.randn([b_size, k_size, n_size], dtype=torch.float16, device="npu:0")
    out = torch.empty(b_size, m_size, n_size, dtype=torch.float32, device="npu:0")
    batch_matmul_demo_kernel(a, b, out)


if __name__ == "__main__":
    run_batch_matmul_demo()
