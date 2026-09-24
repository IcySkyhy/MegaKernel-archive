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
"""Minimal moe_gating_topk verify case (compile + pass_verify + golden, no NPU)."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.moe_gating_topk import MoEGatingTopKConfig, moe_gating_topk, moe_gating_topk_cpu  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402


def test_moe_gating_topk_minimal():
    """batch=32, experts=16, k=2; pass_verify vs moe_gating_topk_cpu."""
    config = MoEGatingTopKConfig(
        k=2, k_group=1, group_count=1, group_select_mode=0,
        norm_type=1, routed_scaling_factor=1.0, eps=1e-20,
    )
    batch_size, num_experts = 32, 16
    np.random.seed(0)
    x = torch.tensor(np.random.uniform(0, 2, (batch_size, num_experts)).astype(np.float32)).contiguous()
    bias = torch.tensor(np.random.uniform(0, 2, (num_experts,)).astype(np.float32)).contiguous()
    y_g, idx_g, norm_g = moe_gating_topk_cpu(x, bias, config)
    y_g = y_g.contiguous()
    idx_g = idx_g.to(torch.int32).contiguous()
    norm_g = norm_g.contiguous()
    y_out = torch.empty((batch_size, config.k), dtype=torch.float32)
    idx_out = torch.empty((batch_size, config.k), dtype=torch.int32)
    norm_out = torch.empty((batch_size, num_experts), dtype=torch.float32)
    set_verify_goldens([None, None, y_g, idx_g, norm_g])
    moe_gating_topk(
        x_shape=tuple(x.shape), bias_shape=tuple(bias.shape), config=config, run_mode="sim"
    )(x, bias, y_out, idx_out, norm_out)
    assert_pass_verify_ok()
