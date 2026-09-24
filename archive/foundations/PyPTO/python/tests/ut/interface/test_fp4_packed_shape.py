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
Verify that a packed FP4x2 uint8 input is exposed as DT_FP4_E2M1 with logical K * 2 shape in kernel.
"""

from contextlib import contextmanager

import torch

import pypto

M = 64
PACKED_K = 32
LOGICAL_K = PACKED_K * 2


@contextmanager
def npuarch(arch: str):
    try:
        old_arch = pypto.platform.npuarch
        pypto.platform.npuarch = arch
        yield
    finally:
        pypto.platform.npuarch = old_arch


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM}, host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
)
def fp4_kernel(
    x: pypto.Tensor([], pypto.DT_FP4_E2M1),
    out: pypto.Tensor([], pypto.DT_INT32),
):
    assert x.dtype == pypto.DT_FP4_E2M1
    assert x.shape[-1] == PACKED_K * 2

    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    out.move(pypto.matmul(x, x, pypto.DT_INT32) + 1)


@npuarch("DAV_3510")
def test_fp4x2_to_fp4_logical_shape():
    packed_x = torch.zeros((M, PACKED_K), dtype=torch.uint8)
    out = torch.zeros((M, M), dtype=torch.int32)

    fp4_kernel(packed_x, out)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM}, host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
)
def fp4x2_kernel(
    x: pypto.Tensor([], pypto.DT_FP4_E2M1X2),
    out: pypto.Tensor([], pypto.DT_INT32),
):
    assert x.dtype == pypto.DT_FP4_E2M1X2
    assert x.shape[-1] == PACKED_K

    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    t = pypto.matmul(x, x, pypto.DT_INT32, b_trans=True)
    out.move(t + 1)


@npuarch("DAV_3510")
def test_fp4x2_to_fp4x2_logical_shape():
    packed_x = torch.zeros((M, PACKED_K), dtype=torch.uint8)
    out = torch.zeros((M, M), dtype=torch.int32)

    fp4x2_kernel(packed_x, out)


if __name__ == "__main__":
    test_fp4x2_to_fp4_logical_shape()
    test_fp4x2_to_fp4x2_logical_shape()
