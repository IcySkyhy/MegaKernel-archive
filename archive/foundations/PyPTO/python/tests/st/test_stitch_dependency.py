#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Tests for CellMatch-based assemble dependency establishment.

Covers three dependency types:
  - WAW (Write After Write):   both ops are NORMAL_WRITE → same type + NORMAL_WRITE → mutex
  - RAW (Read After Write):    consumer=READ, producer=NORMAL_WRITE → different types → mutex
  - WAR (Write After Read):    consumer=NORMAL_WRITE, producer=READ → different types → mutex
"""

import os

from numpy.testing import assert_allclose
import pytest
import torch

import pypto

DTYPE = pypto.DT_FP32


@pytest.mark.skip(reason="temporarily skipped")
@pypto.options(pass_options={"enable_slice": True})
def test_stitch_all_dependencies():
    """
    Unified test covering WAW, RAW, and WAR in a single kernel.

    All phases converge on the SAME region out[8:16] (offset=8):

      out[0:8]   — WAW: 5 writes fully overlap
      out[8:16]  — COMMON: RAW producer write → RAW consumers read →
                    WAR init write → WAR readers read → WAR final write
      out[16:24] — RAW consumer 0 result
      out[24:32] — RAW consumer 1 result
      out[32:40] — WAR reader 0 result
      out[40:48] — WAR reader 1 result
      out[48:56] — WAR reader 2 result

    === WAW (out[0:8]) ===
    5 iterations, each views a different x slice (idx*8) as input.
    All assemble to out[0:8] — NORMAL_WRITE + NORMAL_WRITE → mutex (WAW).
    Last writer (iter 4, x[32:40]) determines golden.

    === RAW (out[8:16] written, read offset == WAR write offset) ===
    Producer: x[8:16] + 100 → out[8:16] (NORMAL_WRITE).
    2 consumers: view out[8:16] (READ) + x[16:24]/x[24:32] → out[16:24], out[24:32].
    READ vs NORMAL_WRITE → RAW mutex automatically.

    === WAR (out[8:16] read by 3 readers, WRITE offset == RAW READ offset) ===
    Init overwrite: x[32:40] + 400 → out[8:16].
    3 readers: view out[8:16] (READ) + x[0:8]/x[8:16]/x[16:24] → out[32:40]/[40:48]/[48:56].
    Final writer: x[32:40] + 800 → out[8:16] (NORMAL_WRITE).
    READ vs NORMAL_WRITE → WAR mutex — final write waits for all readers.
    """
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    x = pypto.tensor([40], DTYPE)
    out = pypto.tensor([56], DTYPE)

    with pypto.function("main_all_deps", x, out):
        pypto.set_vec_tile_shapes(8)

        # === WAW: 5 writes to out[0:8], each reads different x slice ===
        for idx in pypto.loop(5, name="waw", idx_name="idx"):
            tmp = pypto.view(x, [8], [idx * 8])
            t1 = pypto.add(tmp, 10.0)
            t2 = pypto.mul(t1, 2.0)
            t3 = pypto.add(t2, 1.0)
            pypto.assemble(t3, [0], out)

        # === RAW: 1 producer, 2 consumers — all on out[8:16] ===
        raw_src = pypto.view(x, [8], [8])
        raw_prod = pypto.add(raw_src, 100.0)
        pypto.assemble(raw_prod, [8], out)

        for idx in pypto.loop(2, name="raw_consumer", idx_name="idx"):
            raw_tmp = pypto.view(out, [8], [8])
            add_t = pypto.view(x, [8], [16 + idx * 8])
            raw_res = pypto.add(raw_tmp, add_t)
            pypto.assemble(raw_res, [16 + idx * 8], out)

        # === WAR: init, 3 readers, final write — all on out[8:16] ===
        war_init_src = pypto.view(x, [8], [32])
        war_init = pypto.add(war_init_src, 400.0)
        pypto.assemble(war_init, [8], out)

        for idx in pypto.loop(3, name="war_reader", idx_name="idx"):
            war_tmp = pypto.view(out, [8], [8])
            add_t = pypto.view(x, [8], [idx * 8])
            war_res = pypto.add(war_tmp, add_t)
            pypto.assemble(war_res, [32 + idx * 8], out)

        war_final_src = pypto.view(x, [8], [32])
        war_final = pypto.add(war_final_src, 800.0)
        pypto.assemble(war_final, [8], out)

    torch_x = torch.arange(40, dtype=torch.float32)
    golden = torch.zeros(56, dtype=torch.float32)

    golden[0:8] = torch_x[32:40] * 2.0 + 21.0
    golden[8:16] = torch_x[32:40] + 800.0
    golden[16:24] = torch_x[8:16] + torch_x[16:24] + 100.0
    golden[24:32] = torch_x[8:16] + torch_x[24:32] + 100.0
    golden[32:40] = torch_x[32:40] + torch_x[0:8] + 400.0
    golden[40:48] = torch_x[32:40] + torch_x[8:16] + 400.0
    golden[48:56] = torch_x[32:40] + torch_x[16:24] + 400.0

    pto_x = pypto.from_torch(torch_x, "x")
    torch_out = torch.zeros(56)
    pto_out = pypto.from_torch(torch_out, "out")
    pypto.runtime._device_run_once_data_from_host(pto_x, pto_out)

    assert_allclose(torch_out, golden, atol=1e-5, verbose=True)
    pypto.runtime._device_fini()
