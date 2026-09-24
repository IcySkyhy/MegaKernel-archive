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

"""FlashAttention graph mode test -- wraps fa_perf_tkv_preload_dn_kernel with
@allow_in_graph + torch.library for aclgraph (NPUGraph capture/replay) integration.

Demonstrates multi-op graph capture: FA -> Exp -> ReLU pipeline.

Supports two execution modes:
  - Single-op mode: direct call via launch functions
  - Graph mode: NPUGraph capture/replay with torch.compile(backend="eager")

Usage:
    python test_fa_graph_mode.py                     # single-op mode
    python test_fa_graph_mode.py -g                  # graph mode (NPUGraph capture)
"""

import logging
import os
import sys

import pypto_pro.language as pl
import torch
from torch._dynamo import allow_in_graph

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fa_perf_tkv_preload_dn_vf_tail_tilegroup import (  # noqa: E402
    FIFO_SIZE,
    PV_CORE_STRIDE,
    fa_perf_tkv_preload_dn_kernel,
    flash_attention_ref,
)


# ================================================================
#  FA launch / meta functions
# ================================================================
def fa_launch_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_cores: int) -> torch.Tensor:
    """Allocate workspace buffers and launch the compiled FA kernel."""
    sq, d = q.shape
    skv = k.shape[0]

    o = torch.zeros((sq, d), device=q.device, dtype=torch.float16)
    qk_buf = torch.zeros((sq * FIFO_SIZE, skv), device=q.device, dtype=torch.float32)
    p_buf = torch.zeros((sq * FIFO_SIZE, skv), device=q.device, dtype=torch.float16)
    pv_buf = torch.zeros((48 * PV_CORE_STRIDE, d), device=q.device, dtype=torch.float32)

    fa_perf_tkv_preload_dn_kernel[None, num_cores](q, k, v, o, qk_buf, p_buf, pv_buf)

    return o


def fa_meta_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_cores: int) -> torch.Tensor:
    """Shape inference: output has same shape/dtype as q."""
    return torch.zeros(q.shape, dtype=q.dtype, device=f'{q.device}')


# ================================================================
#  Exp kernel: element-wise exp using pl.exp
# ================================================================


@pl.jit()
def exp_kernel(
    inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_inp = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x00000, size=16384,
    )
    tile_f32 = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x04000, size=32768,
    )
    tile_exp = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0C000, size=32768,
    )
    tile_out = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x14000, size=16384,
    )
    with pl.section_vector():
        m_dim = inp.shape[0]
        for i in pl.range(0, m_dim, 64):
            pl.load(tile_inp, inp, [i, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.cast(tile_f32, tile_inp, mode=pl.RoundMode.CAST_NONE)
            pl.exp(tile_exp, tile_f32)
            pl.cast(tile_out, tile_exp, mode=pl.RoundMode.CAST_NONE)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [i, 0])


def exp_launch_fn(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    exp_kernel(x, out)
    return out


def exp_meta_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=f'{x.device}')


# ================================================================
#  ReLU kernel: element-wise relu using pl.relu
# ================================================================


@pl.jit()
def relu_kernel(
    inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_inp = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x00000, size=16384,
    )
    tile_f32 = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x04000, size=32768,
    )
    tile_relu = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0C000, size=32768,
    )
    tile_out = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x14000, size=16384,
    )
    with pl.section_vector():
        m_dim = inp.shape[0]
        for i in pl.range(0, m_dim, 64):
            pl.load(tile_inp, inp, [i, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.cast(tile_f32, tile_inp, mode=pl.RoundMode.CAST_NONE)
            pl.relu(tile_relu, tile_f32)
            pl.cast(tile_out, tile_relu, mode=pl.RoundMode.CAST_NONE)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [i, 0])


def relu_launch_fn(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    relu_kernel(x, out)
    return out


def relu_meta_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=f'{x.device}')


# ================================================================
#  @allow_in_graph wrapped functions for NPUGraph capture
# ================================================================
@allow_in_graph
def fa_op(q, k, v, num_cores):
    """FA operator with FakeTensor detection for graph tracing."""
    for a in (q, k, v):
        if isinstance(a, torch.Tensor):
            try:
                a.data_ptr()
            except RuntimeError:
                return fa_meta_fn(q, k, v, num_cores)
    return fa_launch_fn(q, k, v, num_cores)


@allow_in_graph
def exp_op(x):
    """Exp operator with FakeTensor detection for graph tracing."""
    if isinstance(x, torch.Tensor):
        try:
            x.data_ptr()
        except RuntimeError:
            return exp_meta_fn(x)
    return exp_launch_fn(x)


@allow_in_graph
def relu_op(x):
    """ReLU operator with FakeTensor detection for graph tracing."""
    if isinstance(x, torch.Tensor):
        try:
            x.data_ptr()
        except RuntimeError:
            return relu_meta_fn(x)
    return relu_launch_fn(x)


# ================================================================
#  torch.library registration for torch.compile path
# ================================================================
_PYPTO_LIB = torch.library.Library("pypto", "FRAGMENT")

_PYPTO_LIB.define("fa_flash_attention_a5(Tensor q, Tensor k, Tensor v, int num_cores) -> Tensor")
_PYPTO_LIB.define("elem_exp_a5(Tensor x) -> Tensor")
_PYPTO_LIB.define("elem_relu_a5(Tensor x) -> Tensor")


@torch.library.impl(_PYPTO_LIB, "fa_flash_attention_a5", "Meta")
def _fa_meta(q, k, v, num_cores):
    return fa_meta_fn(q, k, v, num_cores)


@torch.library.impl(_PYPTO_LIB, "elem_exp_a5", "Meta")
def _exp_meta(x):
    return exp_meta_fn(x)


@torch.library.impl(_PYPTO_LIB, "elem_relu_a5", "Meta")
def _relu_meta(x):
    return relu_meta_fn(x)


try:
    @torch.library.impl(_PYPTO_LIB, "fa_flash_attention_a5", "NPU")
    def _fa_npu(q, k, v, num_cores):
        return fa_launch_fn(q, k, v, num_cores)

    @torch.library.impl(_PYPTO_LIB, "elem_exp_a5", "NPU")
    def _exp_npu(x):
        return exp_launch_fn(x)

    @torch.library.impl(_PYPTO_LIB, "elem_relu_a5", "NPU")
    def _relu_npu(x):
        return relu_launch_fn(x)
except Exception as e:
    logging.warning("Failed to register NPU dispatch: %s. torch.compile mode may not be available.", e)


# ================================================================
#  nn.Module wrappers for graph capture
# ================================================================
class FAModel(torch.nn.Module):
    """Model wrapper: FA -> Exp -> ReLU pipeline for NPUGraph capture path."""
    def __init__(self, num_cores: int):
        super().__init__()
        self.num_cores = num_cores

    def forward(self, q, k, v):
        o = fa_op(q, k, v, self.num_cores)
        o = exp_op(o)
        o = relu_op(o)
        return o


class FAModelCompile(torch.nn.Module):
    """Model wrapper: FA -> Exp -> ReLU pipeline for torch.compile path."""
    def __init__(self, num_cores: int):
        super().__init__()
        self.num_cores = num_cores

    def forward(self, q, k, v):
        o = torch.ops.pypto.fa_flash_attention_a5(q, k, v, self.num_cores)
        o = torch.ops.pypto.elem_exp_a5(o)
        o = torch.ops.pypto.elem_relu_a5(o)
        return o


# ================================================================
#  Reference: FA -> exp -> relu
# ================================================================
def fa_exp_relu_ref(q, k, v, d):
    """Golden reference: flash_attention -> exp -> relu.

    Matches NPU kernel behavior: cast to FP32 for exp/relu, then back to FP16.
    """
    _, _, fa_out = flash_attention_ref(q, k, v, d)
    exp_out = torch.exp(fa_out.float()).half()
    relu_out = torch.relu(exp_out.float()).half()
    return relu_out

logging.basicConfig(level=logging.INFO, format="%(message)s")
