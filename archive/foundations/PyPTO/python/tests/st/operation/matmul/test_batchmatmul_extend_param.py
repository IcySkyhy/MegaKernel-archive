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
BatchMatmul bias and fixpipe test script.
Supports both pytest and direct execution modes.
"""

from dataclasses import dataclass
import os
from typing import Optional

import pytest
from testcase.batchmatmul_extend_param_test_case import BIAS_FIXPIPE_TESTS, BiasFixpipeMatmulConfig
import torch
import torch_npu

import pypto


@dataclass
class BatchOffsetParams:
    batch_size: int
    batch_idx: int
    tile_size: int
    is_broadcast: bool
    other_batch: int


def _get_batch_offsets(params: BatchOffsetParams):
    if params.is_broadcast:
        return 0, 1
    else:
        offset = params.batch_idx * params.tile_size
        return offset, offset + params.tile_size


@dataclass
class TensorViewParams:
    tensor: torch.Tensor
    config: BiasFixpipeMatmulConfig
    batch_starts: list
    batch_ends: list
    offset: int
    tile_size: int
    k: int


def _get_a_view(view_params: TensorViewParams) -> torch.Tensor:
    if view_params.config.a_trans:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            0:view_params.k,
            view_params.offset:view_params.offset + view_params.tile_size,
        ]
    else:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            view_params.offset:view_params.offset + view_params.tile_size,
            0:view_params.k,
        ]


def _get_a_view_4d(view_params: TensorViewParams) -> torch.Tensor:
    if view_params.config.a_trans:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            view_params.batch_starts[1]:view_params.batch_ends[1],
            0:view_params.k,
            view_params.offset:view_params.offset + view_params.tile_size,
        ]
    else:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            view_params.batch_starts[1]:view_params.batch_ends[1],
            view_params.offset:view_params.offset + view_params.tile_size,
            0:view_params.k,
        ]


def _get_b_view(view_params: TensorViewParams) -> torch.Tensor:
    if view_params.config.b_trans:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            view_params.offset:view_params.offset + view_params.tile_size,
            0:view_params.k,
        ]
    else:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            0:view_params.k,
            view_params.offset:view_params.offset + view_params.tile_size,
        ]


def _get_b_view_4d(view_params: TensorViewParams) -> torch.Tensor:
    if view_params.config.b_trans:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            view_params.batch_starts[1]:view_params.batch_ends[1],
            view_params.offset:view_params.offset + view_params.tile_size,
            0:view_params.k,
        ]
    else:
        return view_params.tensor[
            view_params.batch_starts[0]:view_params.batch_ends[0],
            view_params.batch_starts[1]:view_params.batch_ends[1],
            0:view_params.k,
            view_params.offset:view_params.offset + view_params.tile_size,
        ]


@dataclass
class BiasViewParams:
    bias_tensor: torch.Tensor
    batch_sizes: list
    indices: list
    tile_sizes: list
    n_offset: int
    tile_n: int
    reference_batch: list


def _get_bias_view_3d(params: BiasViewParams) -> torch.Tensor:
    if params.bias_tensor.dim == 2:
        return params.bias_tensor[0:1, params.n_offset:params.n_offset + params.tile_n]
    offset_params = BatchOffsetParams(
        params.batch_sizes[0],
        params.indices[0],
        params.tile_sizes[0],
        params.batch_sizes[0] == 1,
        params.reference_batch[0],
    )
    bias_start, bias_end = _get_batch_offsets(offset_params)
    return params.bias_tensor[bias_start:bias_end, 0:1, params.n_offset:params.n_offset + params.tile_n]


def _get_bias_view_4d(params: BiasViewParams) -> torch.Tensor:
    return params.bias_tensor[0:1, params.n_offset:params.n_offset + params.tile_n]


@dataclass
class ScaleViewParams:
    scale_tensor: torch.Tensor
    batch_starts: list
    batch_ends: list
    n_offset: int
    tile_n: int


def _get_scale_view_3d(params: ScaleViewParams) -> torch.Tensor:
    return params.scale_tensor[
        params.batch_starts[0]:params.batch_ends[0], 0:1, params.n_offset:params.n_offset + params.tile_n
    ]


def _get_scale_view_4d(params: ScaleViewParams) -> torch.Tensor:
    return params.scale_tensor[
        params.batch_starts[0]:params.batch_ends[0],
        params.batch_starts[1]:params.batch_ends[1],
        0:1,
        params.n_offset:params.n_offset + params.tile_n,
    ]


def _compute_matmul_out(a_view, b_view, config: BiasFixpipeMatmulConfig, extend_params):
    return pypto.matmul(
        a_view,
        b_view,
        out_dtype=config.get_c_pto_dtype(),
        a_trans=config.a_trans,
        b_trans=config.b_trans,
        extend_params=extend_params,
    )


@dataclass
class WriteTensorParams:
    out_tensor: torch.Tensor
    out_view: torch.Tensor
    offsets: list
    tile_sizes: list


def _write_out_tensor_3d(params: WriteTensorParams) -> None:
    b_offset, m_offset, n_offset = params.offsets
    tile_b, tile_m, tile_n = params.tile_sizes
    params.out_tensor[
        b_offset:b_offset + tile_b,
        m_offset:m_offset + tile_m,
        n_offset:n_offset + tile_n,
    ] = params.out_view


@dataclass
class WriteTensor4DParams:
    out_tensor: torch.Tensor
    out_view: torch.Tensor
    b0_offset: int
    b1_offset: int
    tile_b0: int
    tile_b1: int
    m_offset: int
    tile_m: int
    n_offset: int
    tile_n: int


def _write_out_tensor_4d(params: WriteTensor4DParams) -> None:
    params.out_tensor[
        params.b0_offset:params.b0_offset + params.tile_b0,
        params.b1_offset:params.b1_offset + params.tile_b1,
        params.m_offset:params.m_offset + params.tile_m,
        params.n_offset:params.n_offset + params.tile_n,
    ] = params.out_view


@dataclass
class MatmulKernelContext:
    a_tensor: torch.Tensor
    b_tensor: torch.Tensor
    bias_tensor: torch.Tensor
    scale_tensor: torch.Tensor
    out_tensor: torch.Tensor
    config: BiasFixpipeMatmulConfig


@dataclass
class LoopParams:
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    batch_sizes: list
    tile_batch: list


@dataclass
class LoopIndices:
    batch_indices: list
    m_idx: int
    n_idx: int


@dataclass
class BatchOffsets:
    a_starts: list
    a_ends: list
    b_starts: list
    b_ends: list


def _calculate_batch_offsets_3d(ctx: MatmulKernelContext, lp: LoopParams, b_idx: int) -> BatchOffsets:
    batch_a = lp.batch_sizes[0]
    batch_b = lp.batch_sizes[1]
    tile_b = lp.tile_batch[0]

    a_params = BatchOffsetParams(batch_a, b_idx, tile_b, batch_a == 1, batch_b)
    a_start, a_end = _get_batch_offsets(a_params)
    b_params = BatchOffsetParams(batch_b, b_idx, tile_b, batch_b == 1, batch_a)
    b_start, b_end = _get_batch_offsets(b_params)

    return BatchOffsets([a_start], [a_end], [b_start], [b_end])


def _calculate_batch_offsets_4d(ctx: MatmulKernelContext, lp: LoopParams, indices: list) -> BatchOffsets:
    b0_a, b1_a = lp.batch_sizes[0], lp.batch_sizes[1]
    b0_b, b1_b = lp.batch_sizes[2], lp.batch_sizes[3]
    tile_b0, tile_b1 = lp.tile_batch[0], lp.tile_batch[1]
    b0_idx, b1_idx = indices[0], indices[1]

    a0_params = BatchOffsetParams(b0_a, b0_idx, tile_b0, b0_a == 1, b0_b)
    a0_start, a0_end = _get_batch_offsets(a0_params)
    a1_params = BatchOffsetParams(b1_a, b1_idx, tile_b1, b1_a == 1, b1_b)
    a1_start, a1_end = _get_batch_offsets(a1_params)
    b0_params = BatchOffsetParams(b0_b, b0_idx, tile_b0, b0_b == 1, b0_a)
    b0_start, b0_end = _get_batch_offsets(b0_params)
    b1_params = BatchOffsetParams(b1_b, b1_idx, tile_b1, b1_b == 1, b1_a)
    b1_start, b1_end = _get_batch_offsets(b1_params)

    return BatchOffsets([a0_start, a1_start], [a0_end, a1_end], [b0_start, b1_start], [b0_end, b1_end])


@dataclass
class ExtendParamsBuilder:
    config: BiasFixpipeMatmulConfig
    bias_tensor: torch.Tensor
    scale_tensor: torch.Tensor
    batch_offsets: BatchOffsets
    indices: list
    n_offset: int
    tile_batch: list
    tile_n: int
    reference_batch: list


def _build_extend_params(builder: ExtendParamsBuilder) -> dict:
    extend_params = {"relu_type": builder.config.relu_mode}

    if builder.config.mode == "bias":
        bias_params = BiasViewParams(
            builder.bias_tensor,
            builder.config.bias_shape,
            builder.indices,
            builder.tile_batch,
            builder.n_offset,
            builder.tile_n,
            builder.reference_batch,
        )
        if len(builder.indices) == 1:
            extend_params["bias_tensor"] = _get_bias_view_3d(bias_params)
        else:
            extend_params["bias_tensor"] = _get_bias_view_4d(bias_params)
    elif builder.config.mode == "pertensor":
        extend_params["scale"] = builder.config.scale
    elif builder.config.mode == "perchannel":
        scale_params = ScaleViewParams(
            builder.scale_tensor,
            builder.batch_offsets.b_starts,
            builder.batch_offsets.b_ends,
            builder.n_offset,
            builder.tile_n,
        )
        if len(builder.indices) == 1:
            extend_params["scale_tensor"] = _get_scale_view_3d(scale_params)
        else:
            extend_params["scale_tensor"] = _get_scale_view_4d(scale_params)

    return extend_params


def _process_tile_3d(ctx: MatmulKernelContext, lp: LoopParams, indices: LoopIndices, offsets: BatchOffsets):
    m_offset = indices.m_idx * lp.tile_m
    n_offset = indices.n_idx * lp.tile_n
    b_offset = indices.batch_indices[0] * lp.tile_batch[0]

    a_view_params = TensorViewParams(
        ctx.a_tensor, ctx.config, offsets.a_starts, offsets.a_ends, m_offset, lp.tile_m, lp.k
    )
    b_view_params = TensorViewParams(
        ctx.b_tensor, ctx.config, offsets.b_starts, offsets.b_ends, n_offset, lp.tile_n, lp.k
    )

    a_view = _get_a_view(a_view_params)
    b_view = _get_b_view(b_view_params)

    builder = ExtendParamsBuilder(
        ctx.config,
        ctx.bias_tensor,
        ctx.scale_tensor,
        offsets,
        indices.batch_indices,
        n_offset,
        lp.tile_batch,
        lp.tile_n,
        lp.batch_sizes[:2],
    )
    extend_params = _build_extend_params(builder)

    out_view = _compute_matmul_out(a_view, b_view, ctx.config, extend_params)
    write_params = WriteTensorParams(
        ctx.out_tensor, out_view, [b_offset, m_offset, n_offset], [lp.tile_batch[0], lp.tile_m, lp.tile_n]
    )
    _write_out_tensor_3d(write_params)


@dataclass
class ProcessBatchParams:
    ctx: MatmulKernelContext
    lp: LoopParams
    batch_indices: list
    loop_counts: list


def _process_batch_3d_inner(params: ProcessBatchParams):
    b_idx = params.batch_indices[0]
    m_loop, n_loop = params.loop_counts
    offsets = _calculate_batch_offsets_3d(params.ctx, params.lp, b_idx)
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            indices = LoopIndices([b_idx], m_idx, n_idx)
            _process_tile_3d(params.ctx, params.lp, indices, offsets)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def batch_matmul_kernel_3d(
    a_tensor: pypto.Tensor(),
    b_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    bias_tensor: pypto.Tensor(),
    scale_tensor: pypto.Tensor(),
    config: BiasFixpipeMatmulConfig,
):
    k = config.get_k()
    m = config.get_m()
    n = config.get_n()
    batch_a = config.input_shape_a[0]
    batch_b = config.input_shape_b[0]
    batch = config.output_shape[0]

    pypto.set_cube_tile_shapes(*config.tile_shape)
    pypto.set_vec_tile_shapes(config.tile_shape[0][0], config.tile_shape[2][0])
    tile_b = config.get_tile_batch()[0]
    tile_m = config.get_tile_m()
    tile_n = config.get_tile_n()

    m_loop = (m + tile_m - 1) // tile_m
    n_loop = (n + tile_n - 1) // tile_n
    b_loop = (batch + tile_b - 1) // tile_b
    pypto.set_matrix_size([m, k, n])

    ctx = MatmulKernelContext(a_tensor, b_tensor, bias_tensor, scale_tensor, out_tensor, config)
    lp = LoopParams(m, n, k, tile_m, tile_n, [batch_a, batch_b], [tile_b])

    for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
        batch_params = ProcessBatchParams(ctx, lp, [b_idx], [m_loop, n_loop])
        _process_batch_3d_inner(batch_params)


def _process_tile_4d(ctx: MatmulKernelContext, lp: LoopParams, indices: LoopIndices, offsets: BatchOffsets):
    m_offset = indices.m_idx * lp.tile_m
    n_offset = indices.n_idx * lp.tile_n
    b0_offset = indices.batch_indices[0] * lp.tile_batch[0]
    b1_offset = indices.batch_indices[1] * lp.tile_batch[1]

    a_view_params = TensorViewParams(
        ctx.a_tensor, ctx.config, offsets.a_starts, offsets.a_ends, m_offset, lp.tile_m, lp.k
    )
    b_view_params = TensorViewParams(
        ctx.b_tensor, ctx.config, offsets.b_starts, offsets.b_ends, n_offset, lp.tile_n, lp.k
    )

    a_view = _get_a_view_4d(a_view_params)
    b_view = _get_b_view_4d(b_view_params)

    builder = ExtendParamsBuilder(
        ctx.config,
        ctx.bias_tensor,
        ctx.scale_tensor,
        offsets,
        indices.batch_indices,
        n_offset,
        lp.tile_batch,
        lp.tile_n,
        [lp.batch_sizes[0], lp.batch_sizes[1]],
    )
    extend_params = _build_extend_params(builder)

    out_view = _compute_matmul_out(a_view, b_view, ctx.config, extend_params)
    write_params = WriteTensor4DParams(
        ctx.out_tensor,
        out_view,
        b0_offset,
        b1_offset,
        lp.tile_batch[0],
        lp.tile_batch[1],
        m_offset,
        lp.tile_m,
        n_offset,
        lp.tile_n,
    )
    _write_out_tensor_4d(write_params)


def _process_batch_4d_inner(params: ProcessBatchParams):
    b0_idx, b1_idx = params.batch_indices
    m_loop, n_loop = params.loop_counts
    offsets = _calculate_batch_offsets_4d(params.ctx, params.lp, [b0_idx, b1_idx])
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            indices = LoopIndices([b0_idx, b1_idx], m_idx, n_idx)
            _process_tile_4d(params.ctx, params.lp, indices, offsets)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def batch_matmul_kernel_4d(
    a_tensor: pypto.Tensor(),
    b_tensor: pypto.Tensor(),
    bias_tensor: pypto.Tensor(),
    scale_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    config: BiasFixpipeMatmulConfig,
):
    m = config.get_m()
    n = config.get_n()
    k = config.get_k()
    b0_a, b1_a = config.input_shape_a[0], config.input_shape_a[1]
    b0_b, b1_b = config.input_shape_b[0], config.input_shape_b[1]

    pypto.set_cube_tile_shapes(*config.tile_shape)
    pypto.set_vec_tile_shapes(config.tile_shape[0][0], config.tile_shape[2][0])
    tile_b0, tile_b1 = config.get_tile_batch()[0], config.get_tile_batch()[1]
    tile_m = config.get_tile_m()
    tile_n = config.get_tile_n()

    m_loop = (m + tile_m - 1) // tile_m
    n_loop = (n + tile_n - 1) // tile_n
    b0_loop = (b0_a + tile_b0 - 1) // tile_b0
    b1_loop = (b1_a + tile_b1 - 1) // tile_b1
    pypto.set_matrix_size([m, k, n])

    ctx = MatmulKernelContext(a_tensor, b_tensor, bias_tensor, scale_tensor, out_tensor, config)
    lp = LoopParams(m, n, k, tile_m, tile_n, [b0_a, b1_a, b0_b, b1_b], [tile_b0, tile_b1])

    for b0_idx in pypto.loop(0, b0_loop, 1, name="LOOP_L0_b0Idx", idx_name="b0_idx"):
        for b1_idx in pypto.loop(0, b1_loop, 1, name="LOOP_L0_b1Idx", idx_name="b1_idx"):
            batch_params = ProcessBatchParams(ctx, lp, [b0_idx, b1_idx], [m_loop, n_loop])
            _process_batch_4d_inner(batch_params)


@dataclass
class GoldenComputeResult:
    a_tensor_cpu: torch.Tensor
    b_tensor_cpu: torch.Tensor
    bias_tensor_cpu: Optional[torch.Tensor]
    scale_tensor_cpu: Optional[torch.Tensor]
    golden: torch.Tensor
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    c_dtype: torch.dtype


def _compute_golden_tensors(a_shape: list, b_shape: list, n: int, config: BiasFixpipeMatmulConfig):
    a_dtype = config.get_a_torch_dtype()
    b_dtype = config.get_b_torch_dtype()
    c_dtype = config.get_c_torch_dtype()

    if a_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-5, 6, a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.randint(-5, 6, b_shape, dtype=b_dtype)
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype)

    a_cpu = a_tensor_cpu.transpose(-2, -1) if config.a_trans else a_tensor_cpu
    b_cpu = b_tensor_cpu.transpose(-2, -1) if config.b_trans else b_tensor_cpu

    accum_dtype = torch.int32 if a_dtype == torch.int8 else torch.float32
    matmul_result = torch.matmul(a_cpu.to(accum_dtype), b_cpu.to(accum_dtype))

    bias_tensor_cpu = None
    scale_tensor_cpu = None
    flattened = None
    golden = matmul_result.to(c_dtype)

    if config.mode == "bias":
        bias_shape = config.bias_shape
        bias_dtype = config.get_torch_dtype(config.bias_dtype)

        if bias_dtype == torch.int32:
            bias_tensor_cpu = torch.randint(-5, 6, bias_shape, dtype=bias_dtype)
        else:
            bias_tensor_cpu = torch.rand(bias_shape, dtype=bias_dtype)

        golden = (matmul_result + bias_tensor_cpu.to(accum_dtype)).to(c_dtype)

    if config.relu_mode == pypto.ReLuType.RELU:
        golden = torch.relu(golden)

    if config.mode == "pertensor":
        golden = golden * config.scale
    elif config.mode == "perchannel":
        batch_sizes = config.input_shape_b[:-2]
        scale_shape = batch_sizes + [1, n]
        scale_tensor_cpu = torch.rand(scale_shape, dtype=torch.float16).to(torch.float32)
        flattened = scale_tensor_cpu.to("npu").view(-1, scale_shape[-2], scale_shape[-1])
        flattened_int64 = torch.empty(flattened.shape, dtype=torch.int64)
        for i in range(flattened.shape[0]):
            flattened_int64[i] = torch_npu.npu_trans_quant_param(flattened[i])
        flattened = flattened_int64.view(scale_shape)
        golden = (golden * scale_tensor_cpu).to(torch.float16)

    return GoldenComputeResult(
        a_tensor_cpu, b_tensor_cpu, bias_tensor_cpu, flattened, golden, a_dtype, b_dtype, c_dtype
    )


def run_fixpipe_bias_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = BiasFixpipeMatmulConfig.from_test_case(case)

    m = config.get_m()
    k = config.get_k()
    n = config.get_n()

    if config.get_batch_dims() == 1:
        batch_a = config.input_shape_a[0]
        batch_b = config.input_shape_b[0]
        a_shape = [batch_a, k, m] if config.a_trans else [batch_a, m, k]
        b_shape = [batch_b, n, k] if config.b_trans else [batch_b, k, n]
        c_shape = config.output_shape
    else:
        b0_a, b1_a = config.input_shape_a[0], config.input_shape_a[1]
        b0_b, b1_b = config.input_shape_b[0], config.input_shape_b[1]
        a_shape = [b0_a, b1_a, k, m] if config.a_trans else [b0_a, b1_a, m, k]
        b_shape = [b0_b, b1_b, n, k] if config.b_trans else [b0_b, b1_b, k, n]
        c_shape = config.output_shape

    golden_result = _compute_golden_tensors(a_shape, b_shape, n, config)

    a_tensor = golden_result.a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = golden_result.b_tensor_cpu.to(f"npu:{device_id}")
    c_tensor = torch.zeros(c_shape, dtype=golden_result.c_dtype, device=f"npu:{device_id}")

    if config.mode == "bias":
        bias_tensor = golden_result.bias_tensor_cpu.to(f"npu:{device_id}")
        dummy_scale = torch.zeros([1], dtype=torch.float16, device=f"npu:{device_id}")
    else:
        bias_tensor = torch.zeros([1], dtype=torch.float32, device=f"npu:{device_id}")
        if golden_result.scale_tensor_cpu is not None:
            dummy_scale = golden_result.scale_tensor_cpu.to(torch.uint64).to(f"npu:{device_id}")
        else:
            dummy_scale = torch.zeros([1], dtype=torch.float16, device=f"npu:{device_id}")

    if config.get_batch_dims() == 1:
        batch_matmul_kernel_3d(a_tensor, b_tensor, c_tensor, bias_tensor, dummy_scale, config)
    else:
        batch_matmul_kernel_4d(a_tensor, b_tensor, bias_tensor, dummy_scale, c_tensor, config)

    atol, rtol = config.get_tolerance(config.c_dtype)
    assert torch.allclose(c_tensor.cpu(), golden_result.golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in BIAS_FIXPIPE_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_fixpipe_bias(case: dict):
    run_fixpipe_bias_test(case)


def run_batch_matmul_demo(run_mode):
    b_size, m_size, k_size, n_size = 3, 256, 256, 256
    b_view_size, m_view_size, n_view_size = 3, 128, 128

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
        runtime_options={"run_mode": mode},
    )
    def batch_matmul_demo_kernel(
        a: pypto.Tensor([], pypto.DT_INT8),
        b: pypto.Tensor([], pypto.DT_INT8),
        out: pypto.Tensor([], pypto.DT_FP16),
        bias: pypto.Tensor([], pypto.DT_INT32),
        scale: pypto.Tensor([], pypto.DT_UINT64),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size
        b_loop = (b_size + b_view_size - 1) // b_view_size

        for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
            for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
                for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
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
                    bias_view = bias[
                        b_idx * b_view_size:b_idx * b_view_size + b_view_size,
                        :,
                        n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                    ]
                    scale_view = scale[
                        b_idx * b_view_size:b_idx * b_view_size + b_view_size,
                        :,
                        n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                    ]
                    out_view = pypto.matmul(
                        a_view,
                        b_view,
                        pypto.DT_FP16,
                        extend_params={"bias_tensor": bias_view, "scale_tensor": scale_view},
                    )
                    out[
                        b_idx * b_view_size:b_idx * b_view_size + b_view_size,
                        m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                        n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                    ] = out_view

    device = "npu:0" if run_mode == "npu" else "cpu"
    a = torch.randint(0, 10, [b_size, m_size, k_size], dtype=torch.int8, device=device)
    b = torch.randint(0, 10, [b_size, k_size, n_size], dtype=torch.int8, device=device)
    out = torch.empty([b_size, m_size, n_size], dtype=torch.float16, device=device)
    bias = torch.randint(0, 10, [b_size, 1, n_size], dtype=torch.int32, device=device)
    scale = torch.empty([b_size, 1, n_size], dtype=torch.int64, device=device)
    for i in range(b_size):
        scale[i] = torch_npu.npu_trans_quant_param(torch.rand([1, n_size], dtype=torch.float32, device=device))
    batch_matmul_demo_kernel(a, b, out, bias, scale.to(torch.uint64))


if __name__ == "__main__":
    run_batch_matmul_demo("npu")
