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
""" """

import torch

import pypto


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM},
    host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH},
    pass_options={"vec_nbuffer_setting": {-1: 8}},
)
def loop_scope(
    a: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
    b: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
    result: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
):
    pypto.set_vec_tile_shapes(64, 64)

    for _ in pypto.loop(1, name="s0", idx_name="k"):
        pypto.set_vec_tile_shapes(32, 32)
        result.move(a + b)

    for _ in pypto.loop(1, name="s0", idx_name="k"):
        # loop之间的配置相互隔离
        assert [64, 64] == pypto.get_vec_tile_shapes()
        result.move(result + b)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM},
    host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH},
)
def loop_scope_1(
    a: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
    b: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
    result: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
):
    pypto.set_vec_tile_shapes(64, 64)

    # kernel之间的配置相互隔离
    assert {} == pypto.get_pass_options().get("vec_nbuffer_setting")
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        pypto.set_vec_tile_shapes(32, 32)
        result.move(a + b)


def test_loop_scope():
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)

    # prepare data
    a_data = torch.ones((n, m), dtype=torch.int32) * 2
    b_data = torch.ones((n, m), dtype=torch.int32)
    result = torch.zeros(shape, dtype=torch.int32)

    loop_scope(a_data, b_data, result)
    loop_scope_1(a_data, b_data, result)


if __name__ == "__main__":
    test_loop_scope()
