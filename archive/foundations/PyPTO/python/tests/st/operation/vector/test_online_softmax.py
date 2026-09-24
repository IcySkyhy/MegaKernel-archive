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
"""A5-only online softmax operation smoke tests."""

import pytest

import pypto

pytestmark = pytest.mark.soc("950")


@pypto.options(pass_options={"enable_slice": False})
def test_online_softmax_experimental_shapes():
    scores = pypto.tensor([128, 128], pypto.DT_FP32, "scores")

    with pypto.function("ONLINE_SOFTMAX_EXPERIMENTAL_SHAPES", scores):
        pypto.set_vec_tile_shapes(128, 64)
        exp_scores_bf16, column_max, column_sum = pypto.experimental.online_softmax(scores, 1.0)

        assert exp_scores_bf16.shape == [128, 128]
        assert exp_scores_bf16.dtype == pypto.DT_BF16
        assert column_max.shape == [1, 128]
        assert column_max.dtype == pypto.DT_FP32
        assert column_sum.shape == [1, 128]
        assert column_sum.dtype == pypto.DT_FP32


@pypto.options(pass_options={"enable_slice": False})
def test_online_softmax_update_experimental_shapes():
    previous_max = pypto.tensor([1, 128], pypto.DT_FP32, "previous_max")
    previous_sum = pypto.tensor([1, 128], pypto.DT_FP32, "previous_sum")
    previous_output = pypto.tensor([128, 128], pypto.DT_FP32, "previous_output")
    current_max = pypto.tensor([1, 128], pypto.DT_FP32, "current_max")
    current_sum = pypto.tensor([1, 128], pypto.DT_FP32, "current_sum")
    current_output = pypto.tensor([128, 128], pypto.DT_FP32, "current_output")

    with pypto.function(
        "ONLINE_SOFTMAX_UPDATE_EXPERIMENTAL_SHAPES",
        previous_max,
        previous_sum,
        previous_output,
        current_max,
        current_sum,
        current_output,
    ):
        pypto.set_vec_tile_shapes(128, 64)
        updated_max, updated_sum, updated_output = pypto.experimental.online_softmax_update(
            previous_max,
            previous_sum,
            previous_output,
            current_max,
            current_sum,
            current_output,
        )

        assert updated_max.shape == [1, 128]
        assert updated_max.dtype == pypto.DT_FP32
        assert updated_sum.shape == [1, 128]
        assert updated_sum.dtype == pypto.DT_FP32
        assert updated_output.shape == [128, 128]
        assert updated_output.dtype == pypto.DT_FP32
