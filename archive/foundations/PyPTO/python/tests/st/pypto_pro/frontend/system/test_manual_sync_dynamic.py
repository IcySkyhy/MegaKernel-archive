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
"""A5 ST for manually inserted synchronization with dynamic tensor shapes.

Scenario ID prefixes used in test names and comments:
  - A: ``sync_src``/``sync_dst`` pipeline and buffer-dependency scenarios
  - B: event ID forms
  - C: control-flow placement
  - D: pipe barriers
  - E: ``sync_all``
  - F: ``mutex_lock``/``mutex_unlock``
  - H: manual synchronization combined with ``auto_mutex``
  - I: dynamic-shape-specific scenarios

The T prefix identifies the concrete test kernel that combines one or more
scenarios.  Every tensor has dynamic dimensions, the kernel reads
``tensor.shape``, and the result is checked against a torch reference.

Implemented scenario groups:
  - T01: A01/A04/A07/A08, B06/B09, F01/F02/F05/F06, I02/I03/I05
  - TI01: I01, sequential invocations of one JIT kernel with different shapes
  - T02: A06, B08, C01/C02, I05
  - T03a--T03f: D04; D05; D07--D11; D03; D02; D06
  - T04: E01
  - T05: H01/H02/H03/H04/H05
  - T06: A02/A05/A09/A10/A11
  - T07: A03
  - T08: B01/B02/B03/B04/B05/B07
  - T09: B10, C03/C04/C05/C06, I04
  - T10: F03/F04/F07/F08/F09

Together with ``test_manual_sync_all_dynamic.py`` (T11), these cases cover the
positive scenarios listed above.  Codegen-only checks are not counted as
end-to-end coverage.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_N = 64
EVENT_LOAD = (0, 1)
EVENT_STORE = (2, 3)
EVENT_REUSE = (4, 5)
EVENT_RELEASE = (6, 7)
MUTEX_INPUT = (0, 1)
MUTEX_OUTPUT = (2, 3)
STATIC_EVENT_ID = 2
STATIC_EVENT_IDS = (4, 5)
STATIC_EVENT_CONDITION = True
CONTROL_MUTEX_IDS = (8, 9)
LARGE_DYNAMIC_SHAPE = (2048, 2048)
REUSE_STRESS_SHAPE = (2048, 128)

DYNAMIC_SHAPES = [
    pytest.param((1, 64), id="one-tile"),
    pytest.param((2, 128), id="two-by-two-tiles"),
    pytest.param((5, 192), id="multi-tile"),
    pytest.param((3, 96), id="unaligned-tail"),
]

BARRIER_MTE2_SHAPES = [
    pytest.param((4, 64), id="compute-skip-skip-compute"),
    pytest.param((5, 96), id="compute-skip-skip-compute-tail"),
    pytest.param(LARGE_DYNAMIC_SHAPE, id="stress-2048x2048"),
]

BARRIER_MTE3_SHAPES = [
    pytest.param((3, 64), id="three-overlapping-stores"),
    pytest.param((3, 96), id="three-overlapping-stores-tail"),
]

CUBE_MTE1_BARRIER_SHAPES = [
    pytest.param((256, 256, 64), id="wide-skipped-moves"),
    pytest.param((320, 256, 64), id="dynamic-last-block"),
    pytest.param((2048, 256, 64), id="large-dynamic-last-block"),
]

CUBE_M_FIX_BARRIER_SHAPES = [
    pytest.param((128, 64, 64), id="two-blocks"),
    pytest.param((192, 64, 64), id="dynamic-last-block"),
    pytest.param((2048, 64, 64), id="large-dynamic-last-block"),
]

CUBE_DYNAMIC_SHAPES = [
    pytest.param((64, 64, 64), id="one-cube-tile"),
    pytest.param((128, 64, 64), id="two-cube-tiles"),
    pytest.param((192, 64, 64), id="multi-cube-tiles"),
    pytest.param((2048, 64, 64), id="stress-32-cube-tiles"),
]

SYNC_ALL_SHAPES = [
    pytest.param((3, 96), id="unaligned-tail"),
    pytest.param(LARGE_DYNAMIC_SHAPE, id="stress-2048x2048"),
]


def _require_a5() -> None:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        pytest.skip(f"Current device is {device_name}, not A5 (Ascend950)")


# === T01: vector pipeline + dynamic event/mutex + buffer reuse =====================
# A01/A04/A07/A08; B06/B09; F01/F02/F05/F06; I02/I03/I05


@pl.jit(auto_mutex=False)
def t01_vector_event_mutex_reuse(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        m = x.shape[0]
        n = x.shape[1]
        for row in pl.range(0, m):
            for col in pl.range(0, n, TILE_N):
                slot = (row * ((n + TILE_N - 1) // TILE_N) + col // TILE_N) % 2
                load_event = EVENT_LOAD[slot]
                store_event = EVENT_STORE[slot]
                reuse_event = EVENT_REUSE[slot]
                release_event = EVENT_RELEASE[slot]
                valid_n = n - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])

                pl.system.mutex_lock(
                    pipe=pl.PipeType.MTE2,
                    mutex_id=MUTEX_INPUT[slot],
                    mutex_ids=MUTEX_INPUT,
                )
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                pl.system.mutex_unlock(
                    pipe=pl.PipeType.MTE2,
                    mutex_id=MUTEX_INPUT[slot],
                    mutex_ids=MUTEX_INPUT,
                )

                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=load_event)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=load_event)
                pl.add(tile_out, tile_x, tile_y)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=store_event)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=store_event)

                pl.system.mutex_lock(
                    pipe=pl.PipeType.MTE3,
                    mutex_id=MUTEX_OUTPUT[slot],
                    mutex_ids=MUTEX_OUTPUT,
                )
                pl.store(out, tile_out, [row, col])
                pl.system.mutex_unlock(
                    pipe=pl.PipeType.MTE3,
                    mutex_id=MUTEX_OUTPUT[slot],
                    mutex_ids=MUTEX_OUTPUT,
                )
                # Reverse release chain: store completion releases V first,
                # then V releases MTE2 before the same physical UB addresses
                # are overwritten by the next loop iteration.
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=reuse_event)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=reuse_event)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=release_event)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=release_event)


# === T02: runtime bool/shape branches select complete synchronization pairs =========
# A06; B08; C01/C02; I05


@pl.jit(auto_mutex=False)
def t02_runtime_if_else_event(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    use_first_event: pl.DT_BOOL,
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        m = x.shape[0]
        n = x.shape[1]
        for row in pl.range(0, m):
            for col in pl.range(0, n, TILE_N):
                valid_n = n - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                if use_first_event:
                    load_event = 0 if n <= TILE_N else 1
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=load_event)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=load_event)
                else:
                    load_event = 2 if n <= TILE_N else 3
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=load_event)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=load_event)
                if n > TILE_N:
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=4)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=4)
                pl.add(tile_out, tile_x, tile_y)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.store(out, tile_out, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)


# === T03: pipeline-local hazards and barrier control flow ===========================
# T03a: D04; T03b: D05; T03c: D07/D08/D09/D10/D11
# T03d: D03; T03e: D02; T03f: D06


@pl.jit(auto_mutex=False)
def t03a_mte2_skipped_consumer(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0100, size=256)
    with pl.section_vector():
        n = x.shape[1]
        for row in pl.range(0, x.shape[0]):
            for col in pl.range(0, n, TILE_N):
                valid_n = n - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])

                # Every iteration writes the same UB address.  Rows 1/2 modulo 3
                # intentionally skip the Vector consumer, so MTE2 must complete
                # before the next iteration overwrites tile_x.
                pl.load(tile_x, x, [row, col])
                if row % 3 == 0:
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.add(tile_out, tile_x, tile_x)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(out, tile_out, [row, col])
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=3)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=3)
                else:
                    pl.system.bar_mte2()


@pl.jit(auto_mutex=False)
def t03b_mte3_overlapping_stores(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_first = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_second = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_last = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        n = x.shape[1]
        for col in pl.range(0, n, TILE_N):
            valid_n = n - col
            if valid_n >= TILE_N:
                pl.set_validshape(tile_first, [1, TILE_N])
                pl.set_validshape(tile_second, [1, TILE_N])
                pl.set_validshape(tile_last, [1, TILE_N])
            else:
                pl.set_validshape(tile_first, [1, valid_n])
                pl.set_validshape(tile_second, [1, valid_n])
                pl.set_validshape(tile_last, [1, valid_n])

            pl.load(tile_first, x, [0, col])
            pl.load(tile_second, x, [1, col])
            pl.load(tile_last, x, [2, col])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)

            # All stores overlap exactly.  The final GM value must come from
            # tile_last, so each MTE3 WAW transition is explicitly ordered.
            pl.store(out, tile_first, [0, col])
            pl.system.bar_mte3()
            pl.store(out, tile_second, [0, col])
            pl.system.bar_mte3()
            pl.store(out, tile_last, [0, col])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)


@pl.jit(auto_mutex=False)
def t03c_barrier_control_flow(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        n = x.shape[1]
        for row in pl.range(0, x.shape[0]):
            for col in pl.range(0, n, TILE_N):
                valid_n = n - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                if n > TILE_N:
                    # The first barrier is the real MTE2-to-V stage boundary;
                    # the adjacent second call checks consecutive preservation.
                    pl.system.bar_all()
                    pl.system.bar_all()
                else:
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.add(tile_out, tile_x, tile_y)

                # Keep a barrier lexically between the matching event pair.
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.bar_all()
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(out, tile_out, [row, col])
                # This is also the iteration boundary protecting all three
                # reused local buffers before the next dynamic-loop iteration.
                pl.system.bar_all()


@pl.jit(auto_mutex=False)
def t03d_mte1_skipped_consumer(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    wide_mat_type = pl.TileType(shape=[64, 256], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    wide_left_type = pl.TileType(shape=[64, 256], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    a_l1_wide_0 = pl.make_tile(wide_mat_type, addr=0x0000, size=32768)
    a_l1_wide_1 = pl.make_tile(wide_mat_type, addr=0x8000, size=32768)
    a_l1_last = pl.make_tile(mat_type, addr=0x10000, size=8192)
    b_l1 = pl.make_tile(mat_type, addr=0x12000, size=8192)
    wide_a_l0 = pl.make_tile(wide_left_type, addr=0x0000, size=32768)
    a_l0 = pl.make_tile(left_type, addr=0x0000, size=8192)
    b_l0 = pl.make_tile(right_type, addr=0x0000, size=8192)
    acc = pl.make_tile(acc_type, addr=0x0000, size=16384)
    with pl.section_cube():
        last_row = a.shape[0] - 64
        pl.load(a_l1_wide_0, a, [0, 0])
        pl.load(a_l1_wide_1, a, [64, 0])
        pl.load(a_l1_last, a, [last_row, 0])
        pl.load(b_l1, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        # Two wide MTE1 writes and the final narrow write reuse the same L0A
        # base address.  Each barrier drains the preceding MTE1 write before
        # the next differently shaped write reuses that storage.
        pl.move(wide_a_l0, a_l1_wide_0)
        pl.system.bar_mte1()
        pl.move(wide_a_l0, a_l1_wide_1)
        pl.system.bar_mte1()
        pl.move(a_l0, a_l1_last)
        pl.move(b_l0, b_l1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.matmul(acc, a_l0, b_l0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
        pl.store(out, acc, [0, 0])


@pl.jit(auto_mutex=False)
def t03e_m_overlapping_matmuls(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    a_l1_0 = pl.make_tile(mat_type, addr=0x0000, size=8192)
    a_l1_1 = pl.make_tile(mat_type, addr=0x2000, size=8192)
    b_l1 = pl.make_tile(mat_type, addr=0x4000, size=8192)
    a_l0_0 = pl.make_tile(left_type, addr=0x0000, size=8192)
    a_l0_1 = pl.make_tile(left_type, addr=0x2000, size=8192)
    b_l0 = pl.make_tile(right_type, addr=0x0000, size=8192)
    acc = pl.make_tile(acc_type, addr=0x0000, size=16384)
    with pl.section_cube():
        last_row = a.shape[0] - 64
        pl.load(a_l1_0, a, [0, 0])
        pl.load(a_l1_1, a, [last_row, 0])
        pl.load(b_l1, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(a_l0_0, a_l1_0)
        pl.move(a_l0_1, a_l1_1)
        pl.move(b_l0, b_l1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)

        # matmul_acc reads the Acc value produced by the first M operation.
        # bar_m orders this M-pipe RAW dependency; the normal M->FIX event is
        # still responsible for handing the final accumulated result to FIX.
        pl.matmul(acc, a_l0_0, b_l0)
        pl.system.bar_m()
        pl.matmul_acc(acc, acc, a_l0_1, b_l0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
        pl.store(out, acc, [0, 0])


@pl.jit(auto_mutex=False)
def t03f_fix_overlapping_stores(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    a_l1_0 = pl.make_tile(mat_type, addr=0x0000, size=8192)
    a_l1_1 = pl.make_tile(mat_type, addr=0x2000, size=8192)
    b_l1 = pl.make_tile(mat_type, addr=0x4000, size=8192)
    a_l0_0 = pl.make_tile(left_type, addr=0x0000, size=8192)
    a_l0_1 = pl.make_tile(left_type, addr=0x2000, size=8192)
    b_l0 = pl.make_tile(right_type, addr=0x0000, size=8192)
    acc_first = pl.make_tile(acc_type, addr=0x0000, size=16384)
    acc_last = pl.make_tile(acc_type, addr=0x4000, size=16384)
    with pl.section_cube():
        last_row = a.shape[0] - 64
        pl.load(a_l1_0, a, [0, 0])
        pl.load(a_l1_1, a, [last_row, 0])
        pl.load(b_l1, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(a_l0_0, a_l1_0)
        pl.move(a_l0_1, a_l1_1)
        pl.move(b_l0, b_l1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.matmul(acc_first, a_l0_0, b_l0)
        pl.matmul(acc_last, a_l0_1, b_l0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)

        # FIX stores distinct Acc results to the exact same GM range.  The
        # second result must win, so FIX-local WAW order affects precision.
        pl.store(out, acc_first, [0, 0])
        pl.system.bar_fix()
        pl.store(out, acc_last, [0, 0])


# === T04: HARD AIV_ONLY sync_all at uniform multi-core boundaries ==================
# E01


@pl.jit(auto_mutex=False)
def t04_hard_aiv_sync_all(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            for col in pl.range(0, x.shape[1], TILE_N):
                valid_n = x.shape[1] - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.add(tile_out, tile_x, tile_y)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(out, tile_out, [row, col])
                # The same three UB addresses are reused by the next loop
                # iteration.  Complete the reverse release chain instead of
                # relying on a loop-local sync_all as an in-core pipe barrier.
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=3)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)


# === T05: auto_mutex combined with all manual synchronization forms =================
# H01/H02/H03/H04/H05


@pl.jit(auto_mutex=True)
def t05_auto_mutex_manual_sync_barrier(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    x_group = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    y_group = pl.make_tile_group(type=tile_type, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])
    out_group = pl.make_tile_group(type=tile_type, addrs=[0x0400, 0x0500], mutex_ids=[4, 5])
    with pl.section_vector():
        # H05: cross-subcore synchronization coexists with all in-core manual
        # synchronization forms below; the test still judges the in-core
        # pipeline by its end-to-end precision result.
        pl.system.set_cross_core(
            pipe=pl.PipeType.V, event_id=14, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.V, event_id=14, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK
        )
        for row in pl.range(0, x.shape[0]):
            for col in pl.range(0, x.shape[1], TILE_N):
                tile_x = x_group.next()
                tile_y = y_group.next()
                tile_out = out_group.next()
                valid_n = x.shape[1] - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=12)
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=12)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=6)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=6)
                pl.add(tile_out, tile_x, tile_y)
                pl.system.bar_all()
                pl.store(out, tile_out, [row, col])
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)


# === T06: complete cube pipeline, reverse release, and buffer reuse =================
# A02/A05/A09/A10/A11


@pl.jit(auto_mutex=False)
def t06_cube_pipeline_reuse(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    a_l1 = pl.make_tile(mat_type, addr=0x0000, size=8192)
    b_l1 = pl.make_tile(mat_type, addr=0x2000, size=8192)
    a_l0 = pl.make_tile(left_type, addr=0x0000, size=8192)
    b_l0 = pl.make_tile(right_type, addr=0x0000, size=8192)
    acc = pl.make_tile(acc_type, addr=0x0000, size=16384)
    with pl.section_cube():
        for row in pl.range(0, a.shape[0], 64):
            pl.load(a_l1, a, [row, 0])
            pl.load(b_l1, b, [0, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.move(a_l0, a_l1)
            pl.move(b_l0, b_l1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
            # A11: non-adjacent pipe direct sync (MTE1→FIX).
            # Tests that the interface can generate set_flag/wait_flag between
            # non-adjacent pipes.  Data-correctness is already covered by the
            # MTE1→M→FIX chain (events 1/2); this sync is intentionally
            # redundant to exercise the non-adjacent path.
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=6)
            pl.matmul(acc, a_l0, b_l0)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=6)
            pl.store(out, acc, [row, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=3)
            pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=3)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=4)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=4)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=5)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=5)


# === T07: scalar pipeline ============================================================
# A03


@pl.jit(auto_mutex=False)
def t07_scalar_pipeline(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile = pl.make_tile(tile_type, addr=0x0000, size=256)
    with pl.section_vector():
        for row in pl.range(0, x.shape[0]):
            for col in pl.range(0, x.shape[1], TILE_N):
                valid_n = x.shape[1] - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile, [1, TILE_N])
                else:
                    pl.set_validshape(tile, [1, valid_n])
                pl.load(tile, x, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=6)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=6)
                value = tile[0, 0]
                tile[0, 0] = value + x.shape[0]
                pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=7)
                pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=7)
                pl.store(out, tile, [row, col])
                # MTE3 is the final reader of the shared tile.  Release it to
                # MTE2 before the next iteration overwrites the same address.
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)


# === T08: compile-time event ID forms on a real vector pipeline =====================
# B01/B02/B03/B04/B05/B07


@pl.jit(auto_mutex=False)
def t08_static_event_forms(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    local_event = 2
    folded_event = (1 + 2) % 8
    tile_type = pl.TileType(
        shape=[1, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        for row in pl.range(0, x.shape[0]):
            for col in pl.range(0, x.shape[1], TILE_N):
                valid_n = x.shape[1] - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=local_event)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=STATIC_EVENT_ID)
                pl.add(tile_out, tile_x, tile_y)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=folded_event)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=folded_event)
                if STATIC_EVENT_CONDITION:
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=STATIC_EVENT_IDS[0]
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=STATIC_EVENT_IDS[0]
                    )
                pl.store(out, tile_out, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=7)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=7)


# === T09: ternary ID and complete sync pairs in for/if/while control flow ============
# B10; C03/C04/C05/C06; I04


@pl.jit(auto_mutex=False)
def t09_dynamic_control_flow(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    scratch: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    event_id: pl.DT_INT64,
    condition: pl.DT_BOOL,
):
    tile_type = pl.TileType(
        shape=[1, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    scratch_tile_x = pl.make_tile(tile_type, addr=0x0300, size=256)
    scratch_tile_y = pl.make_tile(tile_type, addr=0x0400, size=256)
    scratch_tile_out = pl.make_tile(tile_type, addr=0x0500, size=256)
    with pl.section_vector():
        # C05: MTE3 stores the first result to scratch, sets event 5 before
        # the loop, and MTE2 waits on event 5 only after the loop before
        # loading that scratch value back into the final output.  A stale
        # scratch read is therefore visible in the precision assertion.
        first_valid_n = pl.min(x.shape[1], TILE_N)
        # Event 5 is intentionally not consumed until after the loop.  Use
        # dedicated local buffers so the loop cannot overwrite its producer
        # state before that delayed wait.
        pl.set_validshape(scratch_tile_x, [1, first_valid_n])
        pl.set_validshape(scratch_tile_y, [1, first_valid_n])
        pl.set_validshape(scratch_tile_out, [1, first_valid_n])
        pl.load(scratch_tile_x, x, [0, 0])
        pl.load(scratch_tile_y, y, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=6)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=6)
        pl.add(scratch_tile_out, scratch_tile_x, scratch_tile_y)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=7)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=7)
        pl.store(scratch, scratch_tile_out, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=5)
        row: pl.DT_INT64 = 0
        while row < x.shape[0]:
            for col in pl.range(0, x.shape[1], TILE_N):
                valid_n = x.shape[1] - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                selected_event = event_id if condition else (row + col // TILE_N) % 2
                pl.load(tile_x, x, [row, col])
                pl.load(tile_y, y, [row, col])
                if row % 2 == 0:
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=selected_event
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=selected_event
                    )
                else:
                    alt_event = (selected_event + 1) % 8
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=alt_event
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=alt_event
                    )
                pl.add(tile_out, tile_x, tile_y)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.store(out, tile_out, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
            row = row + 1
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=5)
        pl.set_validshape(tile_x, [1, first_valid_n])
        pl.load(tile_x, scratch, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=6)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=6)
        pl.store(out, tile_x, [0, 0])


# === T10: static boundary, local, branch, and consecutive mutex IDs =================
# F03/F04/F07/F08/F09


@pl.jit(auto_mutex=False)
def t10_mutex_id_forms(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    condition: pl.DT_BOOL,
):
    local_mutex = 5
    selected_mutex = CONTROL_MUTEX_IDS[0] if condition else CONTROL_MUTEX_IDS[1]
    tile_type = pl.TileType(
        shape=[1, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)
    with pl.section_vector():
        for row in pl.range(0, x.shape[0]):
            for col in pl.range(0, x.shape[1], TILE_N):
                valid_n = x.shape[1] - col
                if valid_n >= TILE_N:
                    pl.set_validshape(tile_x, [1, TILE_N])
                    pl.set_validshape(tile_y, [1, TILE_N])
                    pl.set_validshape(tile_out, [1, TILE_N])
                else:
                    pl.set_validshape(tile_x, [1, valid_n])
                    pl.set_validshape(tile_y, [1, valid_n])
                    pl.set_validshape(tile_out, [1, valid_n])
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=3)
                pl.load(tile_x, x, [row, col])
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=3)
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=31)
                pl.load(tile_y, y, [row, col])
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=31)
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=local_mutex)
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=local_mutex)
                pl.system.mutex_lock(
                    pipe=pl.PipeType.MTE2, mutex_id=selected_mutex, mutex_ids=CONTROL_MUTEX_IDS
                )
                pl.system.mutex_unlock(
                    pipe=pl.PipeType.MTE2, mutex_id=selected_mutex, mutex_ids=CONTROL_MUTEX_IDS
                )
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=10)
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=10)
                pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=10)
                pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=10)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.add(tile_out, tile_x, tile_y)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(out, tile_out, [row, col])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)


def _make_inputs(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device=ST_DEVICE, dtype=torch.float32)
    y = torch.randn(shape, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(shape, device=ST_DEVICE, dtype=torch.float32)
    return x, y, out


def _check_add(kernel, shape, *extra_args, used_cores=1):
    _require_a5()
    x, y, out = _make_inputs(shape)
    kernel[None, used_cores](x, y, out, *extra_args)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + y, rtol=1e-4, atol=1e-4)
    logging.info("%s shape=%s passed", getattr(kernel, "__name__", type(kernel).__name__), shape)


def _check_dynamic_control_flow(shape, event_id, condition):
    _require_a5()
    x, y, out = _make_inputs(shape)
    scratch = torch.full(shape, -12345.0, device=ST_DEVICE, dtype=torch.float32)
    t09_dynamic_control_flow[None, 1](x, y, out, scratch, event_id, condition)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + y, rtol=1e-4, atol=1e-4)
    first_valid_n = min(shape[1], TILE_N)
    torch.testing.assert_close(scratch[0, :first_valid_n], (x + y)[0, :first_valid_n], rtol=1e-4, atol=1e-4)


def _check_mte2_skipped_consumer(shape):
    _require_a5()
    torch.manual_seed(0)
    x = torch.randn(shape, device=ST_DEVICE, dtype=torch.float32)
    sentinel = -12345.0
    out = torch.full(shape, sentinel, device=ST_DEVICE, dtype=torch.float32)
    t03a_mte2_skipped_consumer[None, 1](x, out)
    torch.npu.synchronize()
    ref = torch.full_like(out, sentinel)
    ref[::3] = x[::3] * 2
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def _check_mte3_overlapping_stores(shape):
    _require_a5()
    torch.manual_seed(0)
    x = torch.randn(shape, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty((1, shape[1]), device=ST_DEVICE, dtype=torch.float32)
    t03b_mte3_overlapping_stores[None, 1](x, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x[2:3], rtol=0, atol=0)


def _make_cube_barrier_inputs(shape, out_rows=64):
    _require_a5()
    m, k, n = shape
    torch.manual_seed(0)
    # Small integers are represented exactly by FP16 and keep the synchronization
    # oracle strict: a stale/overwritten tile cannot be hidden by loose matmul tolerances.
    a = torch.randint(-2, 3, (m, k), dtype=torch.int32).to(device=ST_DEVICE, dtype=torch.float16)
    b = torch.randint(-2, 3, (k, n), dtype=torch.int32).to(device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty((out_rows, n), device=ST_DEVICE, dtype=torch.float32)
    return a, b, out


def _check_m_barrier(shape):
    a, b, out = _make_cube_barrier_inputs(shape)
    t03e_m_overlapping_matmuls[None, 1](a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a[:64].float(), b.float()) + torch.matmul(a[-64:].float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def _check_fix_barrier(shape):
    a, b, out = _make_cube_barrier_inputs(shape)
    t03f_fix_overlapping_stores[None, 1](a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a[-64:].float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def _check_mte1_barrier(kernel, shape):
    a, b, out = _make_cube_barrier_inputs(shape)
    kernel[None, 1](a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a[-64:, :64].float(), b[:64].float())
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def _check_cube(shape):
    _require_a5()
    m, k, n = shape
    torch.manual_seed(0)
    a = torch.randint(-2, 3, (m, k), dtype=torch.int32).to(device=ST_DEVICE, dtype=torch.float16)
    b = torch.randint(-2, 3, (k, n), dtype=torch.int32).to(device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty((m, n), device=ST_DEVICE, dtype=torch.float32)
    t06_cube_pipeline_reuse[None, 1](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-4, atol=1e-4)


def _check_scalar(shape):
    _require_a5()
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty_like(x)
    t07_scalar_pipeline[None, 1](x, out)
    torch.npu.synchronize()
    ref = x.clone()
    ref[:, ::TILE_N] += shape[0]
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
def test_t01_a01_a04_a07_a08_b06_b09_f01_f02_f05_f06_i02_i03_i05(shape):
    """T01: vector producer/consumer order, ping-pong IDs, mutexes and tails."""
    _check_add(t01_vector_event_mutex_reuse, shape)


@pytest.mark.soc("950")
def test_t01_large_shape_stresses_event_mutex_reuse():
    """T01 stress: 2048x2048 executes 65536 tile iterations with address reuse."""
    _check_add(t01_vector_event_mutex_reuse, LARGE_DYNAMIC_SHAPE)


@pytest.mark.soc("950")
def test_i01_same_jit_kernel_runs_sequential_dynamic_shapes():
    """I01: one JIT object executes several shapes sequentially in one test process."""
    for shape in ((1, 64), (3, 96), (2, 128), (5, 192)):
        _check_add(t01_vector_event_mutex_reuse, shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
@pytest.mark.parametrize("use_first_event", [True, False], ids=["if", "else"])
def test_t02_a06_b08_c01_c02_i05_runtime_if_else(shape, use_first_event):
    """T02: bool/shape branch pairs plus direct MTE3-to-MTE2 reuse synchronization."""
    _check_add(t02_runtime_if_else_event, shape, use_first_event)


@pytest.mark.soc("950")
@pytest.mark.parametrize("use_first_event", [True, False], ids=["if", "else"])
def test_t02_large_shape_stresses_transitive_reverse_release(use_first_event):
    """T02 stress: MTE3->MTE2 closes 4096 iterations on either runtime branch."""
    _check_add(t02_runtime_if_else_event, REUSE_STRESS_SHAPE, use_first_event)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", BARRIER_MTE2_SHAPES)
def test_t03a_d04_mte2_skipped_consumer(shape):
    """D04: bar_mte2 orders same-UB WAW after iterations that skip consumption."""
    _check_mte2_skipped_consumer(shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", BARRIER_MTE3_SHAPES)
def test_t03b_d05_mte3_overlapping_stores(shape):
    """D05: bar_mte3 orders consecutive stores that overwrite the same GM range."""
    _check_mte3_overlapping_stores(shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
def test_t03c_d07_d08_d09_d10_d11_barrier_control_flow(shape):
    """D07-D11: control-flow barriers preserve ordering."""
    _check_add(t03c_barrier_control_flow, shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", CUBE_MTE1_BARRIER_SHAPES)
def test_t03d_d03_mte1_skipped_consumer(shape):
    """D03: bar_mte1 drains wide MTE1 writes before the same L0A is reused."""
    _check_mte1_barrier(t03d_mte1_skipped_consumer, shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", CUBE_M_FIX_BARRIER_SHAPES)
def test_t03e_d02_m_overlapping_matmuls(shape):
    """D02: bar_m orders matmul -> matmul_acc through their shared Acc tile."""
    _check_m_barrier(shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", CUBE_M_FIX_BARRIER_SHAPES)
def test_t03f_d06_fix_overlapping_stores(shape):
    """D06: bar_fix orders distinct FIX stores that overwrite the same GM range."""
    _check_fix_barrier(shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", SYNC_ALL_SHAPES)
@pytest.mark.parametrize("used_cores", [1, 2], ids=["one-core", "two-cores"])
def test_t04_e01_hard_aiv_sync_all(shape, used_cores):
    """T04: every participating AIV reaches the two loop-external barriers."""
    _check_add(t04_hard_aiv_sync_all, shape, used_cores=used_cores)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
def test_t05_h01_h02_h03_h04_h05_auto_mutex_with_all_manual_sync_forms(shape):
    """T05: in-core manual sync remains correct alongside one cross-subcore pair."""
    _check_add(t05_auto_mutex_manual_sync_barrier, shape)


@pytest.mark.soc("950")
def test_t05_large_shape_stresses_manual_and_auto_mutex_coexistence():
    """T05 stress: mixed manual/automatic synchronization over 2048x2048."""
    _check_add(t05_auto_mutex_manual_sync_barrier, LARGE_DYNAMIC_SHAPE)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", CUBE_DYNAMIC_SHAPES)
def test_t06_a02_a05_a09_a10_a11_cube_pipeline(shape):
    """T06: complete forward/reverse cube dependencies and buffer reuse."""
    _check_cube(shape)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
def test_t07_a03_scalar_pipeline(shape):
    """T07: MTE2->S->MTE3 synchronization with runtime shape scalar use."""
    _check_scalar(shape)


@pytest.mark.soc("950")
def test_t07_large_shape_stresses_scalar_tile_reuse():
    """T07 stress: MTE2/S/MTE3 and reverse release repeat over 2048x2048."""
    _check_scalar(LARGE_DYNAMIC_SHAPE)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
def test_t08_b01_b02_b03_b04_b05_b07_static_event_forms(shape):
    """T08: all compile-time event ID spellings execute on the same vector flow."""
    _check_add(t08_static_event_forms, shape)


@pytest.mark.soc("950")
def test_t08_large_shape_stresses_transitive_reverse_release():
    """T08 stress: MTE3->MTE2 protects 4096 static-ID iterations."""
    _check_add(t08_static_event_forms, REUSE_STRESS_SHAPE)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
@pytest.mark.parametrize("condition", [True, False], ids=["ternary-first", "ternary-second"])
def test_t09_b10_c03_c04_c05_c06_i04_dynamic_control_flow(shape, condition):
    """T09: dynamic control flow plus a loop-spanning, precision-visible event pair."""
    _check_dynamic_control_flow(shape, 6, condition)


@pytest.mark.soc("950")
@pytest.mark.parametrize("condition", [True, False], ids=["dynamic-id", "derived-id"])
def test_t09_large_shape_stresses_loop_local_reverse_release(condition):
    """T09 stress: local tiles close 4096 iterations while event 5 remains pending."""
    _check_dynamic_control_flow(REUSE_STRESS_SHAPE, 6, condition)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
@pytest.mark.parametrize("condition", [True, False], ids=["mutex-8", "mutex-9"])
def test_t10_f03_f04_f07_f08_f09_mutex_id_forms(shape, condition):
    """T10: static, local, boundary, branch, and consecutive mutex IDs."""
    _check_add(t10_mutex_id_forms, shape, condition)


@pytest.mark.soc("950")
@pytest.mark.parametrize("condition", [True, False], ids=["mutex-8", "mutex-9"])
def test_t10_large_shape_stresses_mutex_pipeline_reuse(condition):
    """T10 stress: mutex forms coexist with 4096 closed pipeline iterations."""
    _check_add(t10_mutex_id_forms, REUSE_STRESS_SHAPE, condition)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for dynamic_shape in ((1, 64), (2, 128), (5, 192), (3, 96)):
        test_t01_a01_a04_a07_a08_b06_b09_f01_f02_f05_f06_i02_i03_i05(dynamic_shape)
        test_t02_a06_b08_c01_c02_i05_runtime_if_else(dynamic_shape, True)
        test_t02_a06_b08_c01_c02_i05_runtime_if_else(dynamic_shape, False)
        test_t03c_d07_d08_d09_d10_d11_barrier_control_flow(dynamic_shape)
        test_t04_e01_hard_aiv_sync_all(dynamic_shape, 1)
        test_t05_h01_h02_h03_h04_h05_auto_mutex_with_all_manual_sync_forms(dynamic_shape)
        test_t07_a03_scalar_pipeline(dynamic_shape)
        test_t08_b01_b02_b03_b04_b05_b07_static_event_forms(dynamic_shape)
        test_t09_b10_c03_c04_c05_c06_i04_dynamic_control_flow(dynamic_shape, True)
        test_t09_b10_c03_c04_c05_c06_i04_dynamic_control_flow(dynamic_shape, False)
        test_t10_f03_f04_f07_f08_f09_mutex_id_forms(dynamic_shape, True)
        test_t10_f03_f04_f07_f08_f09_mutex_id_forms(dynamic_shape, False)
    for barrier_shape in ((4, 64), (5, 96)):
        test_t03a_d04_mte2_skipped_consumer(barrier_shape)
    for barrier_shape in ((3, 64), (3, 96)):
        test_t03b_d05_mte3_overlapping_stores(barrier_shape)
    for cube_barrier_shape in ((256, 256, 64), (320, 256, 64), (2048, 256, 64)):
        test_t03d_d03_mte1_skipped_consumer(cube_barrier_shape)
    for cube_barrier_shape in ((128, 64, 64), (192, 64, 64), (2048, 64, 64)):
        test_t03e_d02_m_overlapping_matmuls(cube_barrier_shape)
        test_t03f_d06_fix_overlapping_stores(cube_barrier_shape)
    for cube_shape in ((64, 64, 64), (128, 64, 64), (192, 64, 64), (2048, 64, 64)):
        test_t06_a02_a05_a09_a10_a11_cube_pipeline(cube_shape)
