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
Test qkv_rmsnorm_rope_scatternd codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def compute_rmsnorm_golden(input_tensor: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    hidden_dim = input_tensor.shape[-1]
    input_fp32 = input_tensor.float()
    rms = torch.sqrt(torch.sum(input_fp32 * input_fp32 * (1.0 / hidden_dim), dim=-1, keepdim=True) + epsilon)
    normalized = input_fp32 / rms
    return normalized.to(input_tensor.dtype)


def compute_rope_golden(query_tensor: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
    head_dim = query_tensor.shape[3]
    half_head_dim = head_dim // 2
    query_first_half, query_second_half = torch.split(query_tensor, half_head_dim, dim=3)
    query_second_half_negated = torch.mul(query_second_half, -1)
    rotated_qurey = torch.cat([query_first_half, query_second_half_negated], dim=3)
    query_scaled = torch.mul(query_tensor, cos_cache)
    rotated_scaled = torch.mul(rotated_qurey, sin_cache)
    output = torch.add(query_scaled, rotated_scaled)
    return output


def compute_qkv_fused_golden(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
    indices: torch.Tensor,
    epsilon: float = 1e-6,
):
    q_reshape = q_input.reshape(1, 64, 16, 128)
    q_rmsnorm = compute_rmsnorm_golden(q_reshape, epsilon)
    q_transposed = torch.transpose(q_rmsnorm, 1, 2)
    q_out = compute_rope_golden(q_transposed, cos_cache, sin_cache)

    k_reshape = k_input.reshape(1, 64, 8, 128)
    k_rmsnorm = compute_rmsnorm_golden(k_reshape, epsilon)
    k_transposed = torch.transpose(k_rmsnorm, 1, 2)
    k_rope = compute_rope_golden(k_transposed, cos_cache, sin_cache)
    k_final = k_rope.transpose(0, 2)
    indices_squeezed = indices.squeeze(1)
    k_out = past_key.clone()
    k_out.index_put_((indices_squeezed,), k_final, accumulate=False)

    v_reshape = v_input.reshape(1, 64, 8, 128)
    v_final = v_reshape.permute(1, 2, 0, 3)
    v_out = past_value.clone()
    v_out.index_put_((indices_squeezed,), v_final, accumulate=False)

    return q_out, k_out, v_out


def apply_rmsnorm(input_tensor: pypto.Tensor, epsilon: float = 1e-6) -> pypto.Tensor:
    normalized = pypto.rms_norm(input_tensor, epsilon=epsilon)
    return normalized


def apply_rope(input_tensor: pypto.Tensor, cos_cache: pypto.Tensor, sin_cache: pypto.Tensor) -> pypto.Tensor:
    batch_size = input_tensor.shape[0]
    num_heads = input_tensor.shape[1]
    seq_len = input_tensor.shape[2]
    head_dim = input_tensor.shape[3]
    half_head_dim = head_dim // 2

    first_half = pypto.view(input_tensor, [batch_size, num_heads, seq_len, half_head_dim], [0, 0, 0, 0])
    second_half = pypto.view(input_tensor, [batch_size, num_heads, seq_len, half_head_dim], [0, 0, 0, half_head_dim])
    pypto.set_vec_tile_shapes(1, 1, 32, 64)
    second_half_negated = pypto.mul(second_half, -1.0)
    rotated_tensor = pypto.concat([first_half, second_half_negated], dim=-1)
    pypto.set_vec_tile_shapes(1, 1, 32, 128)
    scaled_input = pypto.mul(input_tensor, cos_cache)
    scaled_rotated = pypto.mul(rotated_tensor, sin_cache)
    return pypto.add(scaled_input, scaled_rotated)


def create_fused_qkv_kernels(soc_version):
    """Factory function to create fused QKV kernels with specified soc_version."""

    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": pypto.RunMode.SIM},
        debug_options={"compile_debug_mode": 1},
    )
    def fused_qkv_kernel_fp16(
        q_input: pypto.Tensor([...], pypto.DT_FP16),
        k_input: pypto.Tensor([...], pypto.DT_FP16),
        v_input: pypto.Tensor([...], pypto.DT_FP16),
        cos_cache: pypto.Tensor([...], pypto.DT_FP16),
        sin_cache: pypto.Tensor([...], pypto.DT_FP16),
        indices: pypto.Tensor([...], pypto.DT_INT32),
        q_out: pypto.Tensor([...], pypto.DT_FP16),
        past_key_k_out: pypto.Tensor([...], pypto.DT_FP16),
        past_value_v_out: pypto.Tensor([...], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(2, 2048)
        q_reshape = pypto.reshape(q_input, [1, 64, 16, 128])
        pypto.set_vec_tile_shapes(1, 32, 1, 128)
        q_rmsnorm = apply_rmsnorm(q_reshape, epsilon=1e-6)
        q_transposed = pypto.transpose(q_rmsnorm, 1, 2)
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
        q_out[:] = apply_rope(q_transposed, cos_cache, sin_cache)

        pypto.set_vec_tile_shapes(4, 1024)
        k_reshape = pypto.reshape(k_input, [1, 64, 8, 128])
        pypto.set_vec_tile_shapes(1, 32, 1, 128)
        k_rmsnorm = apply_rmsnorm(k_reshape, epsilon=1e-6)
        k_transposed = pypto.transpose(k_rmsnorm, 1, 2)

        pypto.set_vec_tile_shapes(1, 1, 32, 128)
        k_rope = apply_rope(k_transposed, cos_cache, sin_cache)
        k_final = pypto.transpose(k_rope, 0, 2)
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
        indices_squeezed = pypto.reshape(indices, [64])
        pypto.set_vec_tile_shapes(16)
        pypto.index_put_(past_key_k_out, (indices_squeezed,), k_final, False)

        pypto.set_vec_tile_shapes(4, 1024)
        v_reshape = pypto.reshape(v_input, [1, 64, 8, 128])
        pypto.set_vec_tile_shapes(1, 32, 1, 128)
        v_transposed_02 = pypto.transpose(v_reshape, 0, 2)
        pypto.set_vec_tile_shapes(1, 32, 1, 128)
        v_transposed_12 = pypto.transpose(v_transposed_02, 1, 2)
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
        v_transposed_02_1 = pypto.transpose(v_transposed_12, 0, 2)
        pypto.set_vec_tile_shapes(32, 1, 1, 128)
        v_final = pypto.transpose(v_transposed_02_1, 1, 2)

        pypto.set_vec_tile_shapes(16)
        pypto.index_put_(past_value_v_out, (indices_squeezed,), v_final, False)

    return fused_qkv_kernel_fp16


TEST_CASES = [
    # torch_dtype: torch data type (float16)
    # q_input_shape: query input tensor shape
    # k_input_shape: key input tensor shape
    # v_input_shape: value input tensor shape
    # cos_shape: cos cache tensor shape
    # sin_shape: sin cache tensor shape
    # past_key_shape: past key tensor shape
    # past_value_shape: past value tensor shape
    # indices_shape: indices tensor shape
    # marks: pytest marks
    # - torch_dtype: torch data type (float16)
    # - q_input_shape: query input tensor shape
    # - k_input_shape: key input tensor shape
    # - v_input_shape: value input tensor shape
    # - cos_shape: cos cache tensor shape
    # - sin_shape: sin cache tensor shape
    # - past_key_shape: past key tensor shape
    # - past_value_shape: past value tensor shape
    # - indices_shape: indices tensor shape
    pytest.param(
        torch.float16,
        (64, 2048),
        (64, 1024),
        (64, 1024),
        (1, 1, 64, 128),
        (1, 1, 64, 128),
        (2048, 8, 1, 128),
        (2048, 8, 1, 128),
        (64, 1),
        marks=[pytest.mark.skip()],
        id="001",
    ),
]


def run_qkv_fused_test(
    kernels,
    dtype,
    q_input_shape,
    k_input_shape,
    v_input_shape,
    cos_shape,
    sin_shape,
    past_key_shape,
    past_value_shape,
    indices_shape,
):
    """Run a single qkv fused kernel test."""
    device = "cpu"

    torch.manual_seed(42)
    q_input = torch.randn(q_input_shape, dtype=dtype, device=device).contiguous()
    k_input = torch.randn(k_input_shape, dtype=dtype, device=device).contiguous()
    v_input = torch.randn(v_input_shape, dtype=dtype, device=device).contiguous()
    cos_cache = torch.randn(cos_shape, dtype=dtype, device=device).contiguous()
    sin_cache = torch.randn(sin_shape, dtype=dtype, device=device).contiguous()
    past_key = torch.randn(past_key_shape, dtype=dtype, device=device).contiguous()
    past_value = torch.randn(past_value_shape, dtype=dtype, device=device).contiguous()
    indices = torch.randperm(past_key_shape[0], dtype=torch.int32, device=device)[:indices_shape[0]].unsqueeze(1)

    q_golden, k_golden, v_golden = compute_qkv_fused_golden(
        q_input, k_input, v_input, cos_cache, sin_cache, past_key, past_value, indices
    )

    q_out = torch.empty_like(q_golden).contiguous()
    past_key_k_out = past_key.clone()
    past_value_v_out = past_value.clone()

    kernels["fused_qkv_kernel_fp16"](
        q_input, k_input, v_input, cos_cache, sin_cache, indices, q_out, past_key_k_out, past_value_v_out
    )

    check_nan(q_out, name="q_out")
    cos_value_q = abs(compare_cos(np.array(q_out.cpu()), np.array(q_golden.cpu())))
    if cos_value_q < 0.9999:
        raise AssertionError(f"cos_value_q {cos_value_q} < 0.9999")

    check_nan(past_key_k_out, name="past_key_k_out")
    cos_value_k = abs(compare_cos(np.array(past_key_k_out.cpu()), np.array(k_golden.cpu())))
    if cos_value_k < 0.9999:
        raise AssertionError(f"cos_value_k {cos_value_k} < 0.9999")

    check_nan(past_value_v_out, name="past_value_v_out")
    cos_value_v = abs(compare_cos(np.array(past_value_v_out.cpu()), np.array(v_golden.cpu())))
    if cos_value_v < 0.9999:
        raise AssertionError(f"cos_value_v {cos_value_v} < 0.9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
