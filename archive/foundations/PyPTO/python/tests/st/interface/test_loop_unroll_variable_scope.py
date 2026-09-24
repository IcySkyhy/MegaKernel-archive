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
"""
Test case for loop_unroll variable scope bugfix.

This test verifies that variables defined outside loop_unroll can be correctly
accessed inside nested loops without being prematurely deleted. This addresses
a bug where liveness analysis incorrectly marked variables for deletion after
inner loop exits, causing NameError when loop_unroll splits the loop into
multiple blocks.
"""

import os

from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_loop_unroll_variable_scope():
    """Test that variables defined outside loop_unroll are accessible in all unroll blocks.

    This test verifies the bugfix for liveness analysis incorrectly marking variables
    for deletion after inner loop exits. The variable should only be deleted after
    the outermost loop_unroll completes.
    """
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)

    bs = 32
    ne = 16

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def loop_unroll_kernel(
        input_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
        bias_input: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
        output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(16, 16)
        # Define variable outside loop_unroll
        # This variable should remain accessible throughout all unroll blocks
        bias_2d = pypto.reshape(bias_input, [1, input_tensor.shape[1]], inplace=True)

        # Use loop_unroll with multiple unroll factors
        # This will split the loop into multiple blocks (e.g., [2, 1] = 2 blocks)
        for bs_idx, tile_batch in pypto.loop_unroll(
            0,
            input_tensor.shape[0],
            1,
            name="LOOP_UNROLL_TEST",
            idx_name="bs_idx",
            unroll_list=[2, 1],  # Multiple blocks to test the bugfix
        ):
            # Use bias_2d in the outer loop (first use)
            tile_input = input_tensor[bs_idx:bs_idx + tile_batch, :]
            tile_bias = pypto.tensor([tile_batch, input_tensor.shape[1]], bias_2d.dtype, "tile_bias")

            # Nested loop that also uses bias_2d (second use)
            for tmp_idx in pypto.loop(tile_batch):
                # This should not cause NameError even after inner loop exits
                # because bias_2d should only be deleted after outer loop_unroll completes
                pypto.assemble(bias_2d, [tmp_idx, 0], tile_bias)

            # Use bias_2d.dtype after inner loop (this would fail if bug exists)
            # This verifies that bias_2d is still accessible
            tile_result = pypto.add(tile_input, tile_bias)
            output[bs_idx:bs_idx + tile_batch, :] = tile_result

    # Create test inputs
    input_tensor = torch.randn((bs, ne), dtype=torch.float32, device=device_id)
    bias_input = torch.randn((ne,), dtype=torch.float32, device=device_id)
    output = torch.zeros((bs, ne), dtype=torch.float32, device=device_id)

    # Expected output: input + bias broadcasted
    expected = input_tensor + bias_input.unsqueeze(0)

    # Run kernel - should not raise NameError
    loop_unroll_kernel(input_tensor, bias_input, output)

    # Verify output
    assert_allclose(output.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
