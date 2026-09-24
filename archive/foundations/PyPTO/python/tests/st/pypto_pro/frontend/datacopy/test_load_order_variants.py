# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

"""One tensor read with two different ``order`` values in the same kernel.

``order`` picks which tensor axes an access walks, and that choice lands on the tensor's
GlobalTensor declaration -- the row/col strides are ctor arguments and ``Layout::DN`` is a
template argument, neither of which can change per access. A single declaration therefore
cannot serve two orders, so codegen emits one per layout and each access uses its own.

Before that, one declaration served every access and the first one to be collected decided
its strides. Two loads that differ only in *which axes* they walk produce the same
``Layout::ND``, so nothing rejected the mismatch -- the second load simply read the first
load's axes and returned the wrong elements. That silent case is what this test pins;
the generated declarations are checked in
tests/ut/pypto_pro/codegen/test_cce_tensor_order_variants.py.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO)

# Every axis is the tile size, so both axis pairs below are in bounds for the same tile.
DIM = 16
TILE = 64


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def _mixed_order_kernel(
    x: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
    inner: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
    middle: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    # Tile groups so auto_mutex inserts the MTE2->MTE3 sync between each load and its store.
    inner_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    middle_group = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[1])
    with pl.section_vector():
        inner_tile = inner_group.current()
        middle_tile = middle_group.current()
        # Same tensor, same offsets, different row axis: [2, 3] steps rows by axis 2 and
        # [1, 3] steps them by axis 1. Both are Layout::ND -- only the row stride differs,
        # which is exactly the mismatch nothing downstream can catch.
        pl.load(inner_tile, x, [0, 0, 0, 0], order=[2, 3])
        pl.load(middle_tile, x, [0, 0, 0, 0], order=[1, 3])
        pl.store(inner, inner_tile, [0, 0, 0, 0], order=[2, 3])
        pl.store(middle, middle_tile, [0, 0, 0, 0], order=[2, 3])


@pytest.mark.soc("950")
def test_mixed_order_loads_of_one_tensor():
    """Each load walks the axes its own ``order`` names, not the first load's."""
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)

    x = torch.rand([DIM, DIM, DIM, DIM], device=device, dtype=torch.float16)
    inner = torch.zeros([DIM, DIM, DIM, DIM], device=device, dtype=torch.float16)
    middle = torch.zeros([DIM, DIM, DIM, DIM], device=device, dtype=torch.float16)

    _mixed_order_kernel[None, 1](x, inner, middle)
    torch.npu.synchronize()

    # order=[2, 3] reads x[0, 0, i, j]; order=[1, 3] reads x[0, i, 0, j]. The second is the
    # one that used to be routed through the first's strides and come back as x[0, 0, i, j].
    torch.testing.assert_close(inner[0, 0], x[0, 0])
    torch.testing.assert_close(middle[0, 0], x[0, :, 0, :])
    assert not torch.equal(x[0, 0], x[0, :, 0, :]), "inputs must differ for the check to bite"
    logging.info("test_mixed_order_loads_of_one_tensor [%d^4] passed!", DIM)


@pl.jit(auto_mutex=True)
def _row_major_and_transposed_kernel(
    x: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    plain: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    product: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    """Read x row-major and transposed at once: out = x, and x @ x.T through L0.

    The transposed load has to land in a Mat tile -- the ISA requires the tile's layout to
    match the GlobalTensor's, so a DN read cannot target a row-major Vec tile.
    """
    vec = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    lhs_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[1])
    rhs_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[2])
    lhs_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[3])
    rhs_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[4])
    acc_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[5])
    with pl.section_vector():
        vec_tile = vec.current()
        pl.load(vec_tile, x, [0, 0], order=[0, 1])
        pl.store(plain, vec_tile, [0, 0])
    with pl.section_cube():
        lhs, rhs = lhs_l1.current(), rhs_l1.current()
        pl.load(lhs, x, [0, 0])
        pl.load(rhs, x, [0, 0], order=[1, 0])
        pl.move(lhs_l0a.current(), lhs)
        pl.move(rhs_l0b.current(), rhs)
        pl.matmul(acc_l0c.current(), lhs_l0a.current(), rhs_l0b.current())
        pl.store(product, acc_l0c.current(), [0, 0])


@pytest.mark.soc("950")
def test_row_major_and_transposed_loads_of_one_tensor():
    """order=[0, 1] and order=[1, 0] on one tensor, the pair that motivated the fix.

    A transposed load used to force Layout::DN on the tensor's only declaration, which the
    row-major load then had to read through -- so the two could not coexist at all.
    """
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)

    x = torch.rand([TILE, TILE], device=device, dtype=torch.float16)
    plain = torch.zeros([TILE, TILE], device=device, dtype=torch.float16)
    product = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)

    _row_major_and_transposed_kernel[None, 1](x, plain, product)
    torch.npu.synchronize()

    torch.testing.assert_close(plain, x)
    reference = x.float() @ x.float().t()
    torch.testing.assert_close(product, reference, atol=1e-1, rtol=1e-2)
    logging.info("test_row_major_and_transposed_loads_of_one_tensor [%d, %d] passed!", TILE, TILE)
