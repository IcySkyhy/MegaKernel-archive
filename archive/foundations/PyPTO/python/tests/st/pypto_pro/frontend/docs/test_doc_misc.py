# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for doc examples — misc ops.

Verifies kernel examples from:
  docs/zh/pypto_pro/api/SIMD-API/operation/transpose_and_element_access/fill_index.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/setval.md
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

START = 0


# ============================================================================
# fill_index
# ============================================================================


@pl.jit(auto_mutex=True)
def fill_index_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        m_dim = out.shape[0]
        n_dim = out.shape[1]
        for i in pl.range(0, m_dim, 1):
            for j in pl.range(0, n_dim, 64):
                cur_out = tile_out.current()
                pl.fill_index(cur_out, START)
                pl.store(out, cur_out, [i, j])


# ============================================================================
# Test functions
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fill_index():
    torch.npu.set_device(ST_DEVICE)
    m_val, n_val = 1, 64
    out = torch.empty(m_val, n_val, device=ST_DEVICE, dtype=torch.int32)
    out_ref = torch.arange(START, START + n_val, dtype=torch.int32, device=ST_DEVICE).unsqueeze(0).contiguous()
    fill_index_kernel(out)
    torch.npu.synchronize()
    assert torch.equal(out, out_ref), f"fill_index mismatch:\n{out}\nvs\n{out_ref}"
    logging.info("test_fill_index passed!")


# ============================================================================
# make_tuple
# ============================================================================


@pl.jit()
def tuple_scalar_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TScalar", a=11, b=22)
    with pl.section_vector():
        t = pl.make_tuple(first=s.a, second=s.b)
        out[0] = t.first + t.second
        out[1] = t.second - t.first


@pl.jit()
def tuple_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("LoopT", v=0, cur=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            acc.cur = i
            t = pl.make_tuple(x=acc.cur, y=acc.cur * 10)
            acc.v = acc.v + t.x + t.y
        out[0] = acc.v


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_scalar():
    torch.npu.set_device(ST_DEVICE)
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_scalar_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([33, 11], device=ST_DEVICE, dtype=torch.int32)
    torch.testing.assert_close(out, expected)
    logging.info("test_tuple_scalar passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_in_loop():
    torch.npu.set_device(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    tuple_in_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([66], device=ST_DEVICE, dtype=torch.int32)
    torch.testing.assert_close(out, expected)
    logging.info("test_tuple_in_loop passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = [test_fill_index, test_tuple_scalar, test_tuple_in_loop]
    for case in cases:
        case()
    logging.info("\nAll misc tests passed!")
