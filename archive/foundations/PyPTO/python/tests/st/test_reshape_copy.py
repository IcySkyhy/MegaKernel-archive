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
Reshape Add Operator for PyPTO

Process: reshape 2D -> 1D -> add 0.5 -> reshape 1D -> 2D
Loop tile block: [64, 64], Vec tile shape: [64, 64]
"""

import logging
import os

from numpy.testing import assert_allclose
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_device_id() -> int:
    env_val = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    try:
        return int(env_val)
    except ValueError:
        logger.error("TILE_FWK_DEVICE_ID must be an integer, got: %s", env_val)
        raise


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def reshape_add_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    tile_m: int,
    tile_n: int,
    m_size: int,
    n_size: int,
):
    m_dyn = x.shape[0]
    n_dyn = x.shape[1]

    pypto.set_vec_tile_shapes(tile_m, tile_n)

    m_loop = (m_dyn + tile_m - 1) // tile_m
    n_loop = (n_dyn + tile_n - 1) // tile_n

    for m_idx in pypto.loop(m_loop, name="LOOP_M", idx_name="m_idx"):
        m_offset = m_idx * tile_m
        m_offset_end = pypto.min(m_offset + tile_m, m_dyn)
        valid_m = m_offset_end - m_offset

        for n_idx in pypto.loop(n_loop, name="LOOP_N", idx_name="n_idx"):
            n_offset = n_idx * tile_n
            n_offset_end = pypto.min(n_offset + tile_n, n_dyn)
            valid_n = n_offset_end - n_offset

            x_view = pypto.view(x, [tile_m, tile_n], [m_offset, n_offset], valid_shape=[valid_m, valid_n])
            x_1d = pypto.reshape(x_view, [tile_m * tile_n], valid_shape=[valid_m * valid_n])
            pypto.set_vec_tile_shapes(tile_m * tile_n)
            result_1d = pypto.add(x_1d, 0.5)
            result = pypto.reshape(result_1d, [tile_m, tile_n], valid_shape=[valid_m, valid_n])
            pypto.set_vec_tile_shapes(tile_m, tile_n)
            pypto.assemble(result, [m_offset, n_offset], y)


TEST_SHAPES = [
    (100, 100),
    (128, 100),
    (100, 128),
    (128, 128),
    (64, 64),
]


def main():
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    logger.info("Running on NPU device %d", device_id)

    tile_m, tile_n = 64, 64

    for m, n in TEST_SHAPES:
        x = torch.randn(m, n, dtype=torch.float32, device=f"npu:{device_id}")
        y = torch.zeros(m, n, dtype=torch.float32, device=f"npu:{device_id}")

        reshape_add_kernel(x, y, tile_m, tile_n, m, n)
        torch.npu.synchronize()

        golden = x + 0.5
        max_diff = (y - golden).abs().max().item()
        logger.info("shape [%d, %d] max_diff=%.6f", m, n, max_diff)

        assert_allclose(y.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)

    logger.info("All tests passed")


if __name__ == "__main__":
    main()
