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
This test case verifies that the swimlane diagram generation for the costmodel works correctly
regardless of whether the CANN is installed in the environment or not.
"""

import numpy as np
from numpy.testing import assert_allclose
import torch

import pypto


@pypto.frontend.jit(
    runtime_options={"run_mode": 1}
)
def add_kernel(
    x: pypto.Tensor([...], pypto.DT_FP32),
    y: pypto.Tensor([...], pypto.DT_FP32),
    out: pypto.Tensor([...], pypto.DT_FP32),
):
    """
    Add implementation.

    Parameters
    ----------
    x : pypto.Tensor
        First input tensor
    y : pypto.Tensor
        Second input tensor
    out : pypto.Tensor
        Output tensor
    """
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out[:] = x + y


def test_add():
    """
    Test add implementation against PyTorch reference.
    """
    shape = (1, 4, 1, 64)

    # Prepare data
    input_data0 = torch.rand(shape, dtype=torch.float32)
    input_data1 = torch.rand(shape, dtype=torch.float32)
    output_data = torch.empty(shape, dtype=torch.float32)

    add_kernel(input_data0, input_data1, output_data)

    golden = torch.add(input_data0, input_data1)
    assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_add()
