#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test attention codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto

TEST_CASES_4INPUT = [
    # kernel_name: name of the kernel
    # shape_a: first input tensor shape
    # shape_b: second input tensor shape
    # shape_c: third input tensor shape
    # shape_out: output tensor shape
    # vec_tile_shape: vector tile shape
    # cube_tile_shapes: cube tile shapes for matmul
    # marks: pytest marks
    pytest.param(
        "mul_matmul",
        (16, 16),
        (16, 16),
        (16, 16),
        (16, 16),
        (16, 16),
        ([16, 16], [16, 16], [16, 16]),
        marks=[pytest.mark.skip()],
        id="mul_matmul",
    ),
    pytest.param(
        "matmul_mul",
        (16, 16),
        (16, 16),
        (16, 16),
        (16, 16),
        (16, 16),
        ([16, 16], [16, 16], [16, 16]),
        marks=[pytest.mark.skip()],
        id="matmul_mul",
    ),
    pytest.param(
        "matmul_mul_with_reshape_001",
        (1, 1, 1, 256),
        (1, 1, 256, 128),
        (1, 1, 1, 128),
        (1, 1, 1, 128),
        (1, 1, 1, 64),
        ([16, 16], [64, 128], [64, 128]),
        marks=[],
        id="matmul_mul_with_reshape_001",
    ),
    pytest.param(
        "matmul_mul_with_reshape_002",
        (1, 1, 1, 256),
        (1, 1, 256, 128),
        (1, 1, 1, 128),
        (1, 1, 1, 128),
        (1, 1, 1, 64),
        ([16, 16], [64, 128], [64, 128]),
        marks=[pytest.mark.skip()],
        id="matmul_mul_with_reshape_002",
    ),
    pytest.param(
        "matmul_mul_with_reshape_003",
        (1, 16, 1, 256),
        (1, 16, 256, 128),
        (1, 16, 1, 128),
        (1, 16, 1, 128),
        (1, 1, 1, 64),
        ([16, 16], [64, 128], [64, 128]),
        marks=[pytest.mark.skip()],
        id="matmul_mul_with_reshape_003",
    ),
    pytest.param(
        "mul_matmul_with_reshape_001",
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        ([16, 16], [16, 16], [16, 16]),
        marks=[pytest.mark.skip()],
        id="mul_matmul_with_reshape_001",
    ),
    pytest.param(
        "mul_matmul_with_reshape_002",
        (1, 1, 1, 256),
        (1, 1, 1, 256),
        (1, 1, 256, 128),
        (1, 1, 1, 128),
        (1, 1, 1, 64),
        ([16, 16], [64, 128], [64, 128]),
        marks=[],
        id="mul_matmul_with_reshape_002",
    ),
    pytest.param(
        "mul_matmul_with_reshape_003",
        (1, 16, 1, 256),
        (1, 16, 1, 256),
        (1, 16, 256, 128),
        (1, 16, 1, 128),
        (1, 1, 1, 64),
        ([16, 16], [64, 128], [64, 128]),
        marks=[pytest.mark.skip()],
        id="mul_matmul_with_reshape_003",
    ),
]

TEST_CASES_ATTENTION = [
    # kernel_name: name of the kernel
    # q_shape: query tensor shape
    # k_shape: key tensor shape
    # v_shape: value tensor shape
    # attn_mask_shape: attention mask shape
    # output_shape: output tensor shape
    # vec_tile_shape: vector tile shape
    # cube_tile_shapes: cube tile shapes for matmul
    # marks: pytest marks
    pytest.param(
        "attention_mini",
        (1, 16, 16, 16),
        (1, 16, 16, 16),
        (1, 16, 16, 16),
        (1, 1, 16, 16),
        (1, 16, 16, 16),
        (1, 1, 16, 16),
        ([16, 16], [16, 16], [16, 16]),
        marks=[pytest.mark.skip()],
        id="attention_mini",
    ),
    pytest.param(
        "attention_prefill",
        (1, 16, 64, 128),
        (1, 16, 2048, 128),
        (1, 16, 2048, 128),
        (1, 1, 64, 2048),
        (1, 16, 64, 128),
        (1, 1, 64, 64),
        ([64, 64], [64, 64], [64, 64]),
        marks=[pytest.mark.skip()],
        id="attention_prefill",
    ),
    pytest.param(
        "attention_decoder",
        (1, 16, 1, 128),
        (1, 16, 2048, 128),
        (1, 16, 2048, 128),
        (1, 1, 1, 2048),
        (1, 16, 1, 128),
        (1, 1, 1, 64),
        ([16, 16], [64, 64], [64, 64]),
        marks=[pytest.mark.skip()],
        id="attention_decoder",
    ),
]


def _make_4input_kernel(soc_version, name, a_shape, b_shape, c_shape, out_shape, vec_tile_shape, cube_tile_shape):
    vec_tile = vec_tile_shape if vec_tile_shape is not None else (16, 16)
    cube_tile = cube_tile_shape if cube_tile_shape is not None else ([16, 16], [16, 16], [16, 16])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP16),
        b: pypto.Tensor(b_shape, pypto.DT_FP16),
        c: pypto.Tensor(c_shape, pypto.DT_FP16),
        out: pypto.Tensor(out_shape, pypto.DT_FP16),
    ):
        if "mul_matmul" in name:
            pypto.set_vec_tile_shapes(*vec_tile)
            pypto.set_cube_tile_shapes(*cube_tile)
            mul = pypto.mul(a, b)
            out[:] = pypto.matmul(mul, c, pypto.DT_FP16, a_trans=False, b_trans=False)
        else:
            pypto.set_cube_tile_shapes(*cube_tile)
            pypto.set_vec_tile_shapes(*vec_tile)
            mul = pypto.matmul(a, b, pypto.DT_FP16, a_trans=False, b_trans=False)
            out[:] = pypto.mul(mul, c)

    kernel.__name__ = name
    return kernel


def _make_attention_kernel(
    soc_version, name, q_shape, k_shape, v_shape, attn_mask_shape, output_shape, vec_tile_shape, cube_tile_shape
):
    vec_tile = vec_tile_shape if vec_tile_shape is not None else (1, 1, 16, 16)
    cube_tile = cube_tile_shape if cube_tile_shape is not None else ([16, 16], [16, 16], [16, 16])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        q: pypto.Tensor(q_shape, pypto.DT_FP16),
        k: pypto.Tensor(k_shape, pypto.DT_FP16),
        v: pypto.Tensor(v_shape, pypto.DT_FP16),
        attn_mask: pypto.Tensor(attn_mask_shape, pypto.DT_FP16),
        output: pypto.Tensor(output_shape, pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes(*cube_tile)
        q_k_t = pypto.matmul(q, k, pypto.DT_FP16, a_trans=False, b_trans=True)
        pypto.set_vec_tile_shapes(*vec_tile)
        q_k_t_mul = pypto.mul(q_k_t, 0.0883883461356163)
        q_k_t_mul_add = pypto.add(q_k_t_mul, attn_mask)
        softmax_q_k_t = pypto.softmax(q_k_t_mul_add, dim=-1)
        pypto.set_cube_tile_shapes(*cube_tile)
        output[:] = pypto.matmul(softmax_q_k_t, v, pypto.DT_FP16, a_trans=False, b_trans=False)

    kernel.__name__ = name
    return kernel


def create_attention_kernels(soc_version):
    kernels_4input = {
        p.values[0]: _make_4input_kernel(
            soc_version, p.values[0], p.values[1], p.values[2], p.values[3], p.values[4], p.values[5], p.values[6]
        )
        for p in TEST_CASES_4INPUT
    }

    kernels_attention = {
        p.values[0]: _make_attention_kernel(
            soc_version,
            p.values[0],
            p.values[1],
            p.values[2],
            p.values[3],
            p.values[4],
            p.values[5],
            p.values[6],
            p.values[7],
        )
        for p in TEST_CASES_ATTENTION
    }

    return {**kernels_4input, **kernels_attention}


def _compute_golden_4input(op_type, a, b, c):
    if "mul_matmul" in op_type:
        mul = torch.mul(a, b)
        return torch.matmul(mul, c)
    else:
        mul = torch.matmul(a, b)
        return torch.mul(mul, c)


def _compute_golden_attention(op_type, q, k, v, attn_mask):
    k_t = torch.transpose(k, 2, 3)
    q_k_t = torch.matmul(q, k_t)
    q_k_t_mul = torch.mul(q_k_t, 0.0883883461356163)
    q_k_t_mul_add = torch.add(q_k_t_mul, attn_mask)
    softmax = torch.softmax(q_k_t_mul_add, -1)
    return torch.matmul(softmax, v)


def run_4input_test(kernels, op_type, shapes):
    a = torch.rand(shapes[0], dtype=torch.float16, device="cpu")
    b = torch.rand(shapes[1], dtype=torch.float16, device="cpu")
    c = torch.rand(shapes[2], dtype=torch.float16, device="cpu")
    out = torch.rand(shapes[3], dtype=torch.float16, device="cpu")

    kernels[op_type](a, b, c, out)
    golden = _compute_golden_4input(op_type, a, b, c)

    check_nan(out, name=op_type)
    cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"cos_value {cos_value} < 0.9999")


def run_attention_test(kernels, op_type, shapes):
    q = torch.rand(shapes[0], dtype=torch.float16, device="cpu")
    k = torch.rand(shapes[1], dtype=torch.float16, device="cpu")
    v = torch.rand(shapes[2], dtype=torch.float16, device="cpu")
    attn_mask = torch.rand(shapes[3], dtype=torch.float16, device="cpu")
    output = torch.rand_like(q)

    kernels[op_type](q, k, v, attn_mask, output)
    golden = _compute_golden_attention(op_type, q, k, v, attn_mask)

    check_nan(output, name=op_type)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"cos_value {cos_value} < 0.9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
