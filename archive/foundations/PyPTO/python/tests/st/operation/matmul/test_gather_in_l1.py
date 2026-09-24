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
Matmul GATHER_IN_L1_TESTS test script.
Supports both pytest and direct execution modes.
"""

import os

import pytest
from testcase.gather_in_l1_test_case import GATHER_IN_L1_TESTS, GatherInL1Config
import torch

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gather_matmul_pto_kernel(
    src: pypto.Tensor(),
    indices: pypto.Tensor(),
    page_table: pypto.Tensor(),
    mat2_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    config: GatherInL1Config,
):
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dyn_src = pypto.view(src, src.shape, [0, 0], valid_shape=src.shape)
    dyn_offsets = pypto.view(indices, indices.shape, [0, 0], valid_shape=indices.shape)
    dyn_page_table = pypto.view(page_table, page_table.shape, [0, 0], valid_shape=page_table.shape)
    input_tensor = pypto.experimental.gather_in_l1(
        dyn_src,
        dyn_offsets,
        dyn_page_table,
        config.block_size,
        config.token_dim,
        is_b_matrix=config.is_gather_b_matrix,
        is_trans=config.is_gather_trans,
    )
    if not config.is_gather_b_matrix:
        out_tensor[:] = pypto.matmul(
            input_tensor, mat2_tensor, config.out_dtype, a_trans=config.is_gather_trans, b_trans=False
        )
    else:
        out_tensor[:] = pypto.matmul(
            mat2_tensor, input_tensor, config.out_dtype, a_trans=False, b_trans=config.is_gather_trans
        )


def run_gather_in_l1_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)

    config = GatherInL1Config.from_test_case(case)

    in_dtype = GatherInL1Config.get_torch_dtype(case["in_dtype"])
    out_dtype = GatherInL1Config.get_torch_dtype(case["out_dtype"])
    if case["in_dtype"] == "DT_INT8":
        all_buffer = torch.randint(-5, 5, (config.num_buffer_tokens, config.token_dim), dtype=in_dtype)
    else:
        all_buffer = torch.randn(config.num_buffer_tokens, config.token_dim, dtype=in_dtype)
    if config.is_gather_b_matrix == config.is_gather_trans:
        unit_tensor = torch.ones((config.token_dim, config.token_dim), dtype=in_dtype)
    else:
        unit_tensor = torch.ones((config.topk, config.topk), dtype=in_dtype)
    page_table = (
        torch.randperm(config.num_buffer_tokens // config.block_size)[:config.num_logical_blocks]
        .to(torch.int32)
        .view(1, -1)
    )
    indices = torch.randint(0, config.block_size * page_table.shape[1], (config.topk,)).to(torch.int32).view(1, -1)
    extracted = (
        page_table.view(-1)[indices.view(-1) // config.block_size] * config.block_size
        + indices.view(-1) % config.block_size
    )
    extracted_tokens = all_buffer[extracted]
    if config.is_gather_b_matrix and not config.is_gather_trans:
        golden = torch.matmul(unit_tensor, extracted_tokens).to(out_dtype)
    elif config.is_gather_b_matrix and config.is_gather_trans:
        golden = torch.matmul(unit_tensor, extracted_tokens.T).to(out_dtype)
    elif not config.is_gather_b_matrix and not config.is_gather_trans:
        golden = torch.matmul(extracted_tokens, unit_tensor).to(out_dtype)
    else:
        golden = torch.matmul(extracted_tokens.T, unit_tensor).to(out_dtype)
    out_tensor = torch.zeros(golden.shape, dtype=out_dtype, device=f"npu:{device_id}")
    gather_matmul_pto_kernel(
        all_buffer.npu(), indices.npu(), page_table.npu(), unit_tensor.npu(), out_tensor.npu(), config
    )
    atol, rtol = GatherInL1Config.get_tolerance(case["out_dtype"])
    assert torch.allclose(out_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in GATHER_IN_L1_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_gather_in_l1_basic(case: dict):
    run_gather_in_l1_test(case)


def create_gather_demo_inputs(device: str, config: GatherInL1Config):
    all_buffer = torch.randn(config.num_buffer_tokens, config.token_dim, dtype=torch.float16, device=device)

    if config.is_gather_b_matrix == config.is_gather_trans:
        unit_tensor = torch.ones((config.token_dim, config.token_dim), dtype=torch.float16, device=device)
    else:
        unit_tensor = torch.ones((config.topk, config.topk), dtype=torch.float16, device=device)

    page_table = (
        torch.randperm(config.num_buffer_tokens // config.block_size)[:config.num_logical_blocks]
        .to(torch.int32)
        .view(1, -1)
    )
    page_table = page_table.to(device)

    indices = torch.randint(0, config.block_size * page_table.shape[1], (config.topk,)).to(torch.int32).view(1, -1)
    indices = indices.to(device)

    return all_buffer, indices, page_table, unit_tensor


def run_gather_in_l1_demo(run_mode):
    config = GatherInL1Config(
        topk=8,
        num_logical_blocks=3,
        num_buffer_tokens=32,
        token_dim=30,
        block_size=10,
        is_gather_b_matrix=False,
        is_gather_trans=False,
        out_dtype=pypto.DT_FP16,
    )

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
    def gather_demo_kernel(
        src: pypto.Tensor([], pypto.DT_FP16),
        indices: pypto.Tensor([], pypto.DT_INT32),
        page_table: pypto.Tensor([], pypto.DT_INT32),
        mat2: pypto.Tensor([], pypto.DT_FP16),
        out: pypto.Tensor([], pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dyn_offsets = pypto.view(indices, indices.shape, [0, 0], valid_shape=indices.shape)
        dyn_src = pypto.view(src, src.shape, [0, 0], valid_shape=src.shape)
        dyn_page_table = pypto.view(page_table, page_table.shape, [0, 0], valid_shape=page_table.shape)
        input_tensor = pypto.experimental.gather_in_l1(
            dyn_src,
            dyn_offsets,
            dyn_page_table,
            config.block_size,
            config.token_dim,
            is_b_matrix=config.is_gather_b_matrix,
            is_trans=config.is_gather_trans,
        )
        if not config.is_gather_b_matrix:
            out[:] = pypto.matmul(input_tensor, mat2, pypto.DT_FP16, a_trans=config.is_gather_trans, b_trans=False)
        else:
            out[:] = pypto.matmul(mat2, input_tensor, pypto.DT_FP16, a_trans=False, b_trans=config.is_gather_trans)

    device = "npu:0" if run_mode == "npu" else "cpu"
    all_buffer, indices, page_table, unit_tensor = create_gather_demo_inputs(device, config)
    out_shape = (config.topk, config.token_dim) if not config.is_gather_b_matrix else (config.token_dim, config.topk)
    if config.is_gather_trans:
        out_shape = (out_shape[1], out_shape[0])
    out_tensor = torch.empty(out_shape, dtype=torch.float16, device=device)
    gather_demo_kernel(all_buffer, indices, page_table, unit_tensor, out_tensor)


if __name__ == "__main__":
    run_gather_in_l1_demo("npu")
