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
Test amax block onboard
"""

import math
import os

from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_amax_block_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    input_shape = (8, 8)
    view_shape = (4, 8)
    tile_shape = (4, 8)

    pypto.runtime._device_init()

    input_tensor = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_SELF")
    dst_tensor = pypto.tensor((input_shape[0],), pypto.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(input_shape[0] / view_shape[0])

    with pypto.function("MAIN", input_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            # block view
            view_tensor = pypto.view(
                input_tensor,
                view_shape,
                [b_idx * view_shape[0], 0],
                valid_shape=[
                    pypto.min(input_shape[0] - b_idx * view_shape[0], pypto.symbolic_scalar(view_shape[0])),
                    pypto.symbolic_scalar(view_shape[1]),
                ],
            )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])

            block_result = pypto.amax(view_tensor, 1)
            pypto.assemble(block_result, [b_idx * view_shape[0]], dst_tensor)

    a_tensor = torch.randn(input_shape, dtype=torch.float32)
    b_tensor = torch.zeros(input_shape[0], dtype=torch.float32)

    pto_a = pypto.from_torch(a_tensor, "a_tensor")
    pto_b = pypto.from_torch(b_tensor, "b_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a, pto_b)

    golden = torch.amax(a_tensor, dim=1)

    assert_allclose(b_tensor, golden, rtol=1e-5, atol=1e-6)

    pypto.runtime._device_fini()
