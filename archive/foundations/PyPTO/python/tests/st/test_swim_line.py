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
profiling of aicpu pref  test for PyPTO
"""

import json
import multiprocessing as mp
import os
from typing import Dict, List

import pytest
import torch
import torch_npu

import pypto


@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
def matmul_add(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        out.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


def device_run_data_from_device_mix_nodep(queue):
    pypto.set_pass_options(enable_slice=False)
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    os.environ["DUMP_DEVICE_PERF"] = "true"
    if "Ascend950" in torch.npu.get_device_name():
        os.environ["ENABLE_AICORE_RESOLVE"] = "true"

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 3

    a_rawdata = torch.tensor([[1] * k] * n)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')

    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        matmul_add(a_data, b_data, c_data, d_data)

    torch_npu.npu.synchronize()
    pref_path = pypto.pypto_impl.LogTopFolder()
    queue.put(pref_path)


@pypto.options(pass_options={"enable_slice": True})
@pytest.mark.soc("910", "950")
def test_swim():
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    p = mp.Process(target=device_run_data_from_device_mix_nodep, args=(result_queue,))
    p.start()
    p.join()
    pref_path = ""
    if not result_queue.empty():
        pref_path = result_queue.get()
    else:
        assert False, "Could not Get pref path"
    aicpu_json_path = pref_path + "/machine_trace_perf_data_0.json"
    assert os.path.exists(aicpu_json_path), "Could not Get aicpu perf"

    with open(aicpu_json_path, 'r', encoding='utf-8') as f:
        core_list: List[Dict] = json.load(f)
        for core in core_list:
            tasks = core.get("tasks", [])
            if "Ascend950" in torch.npu.get_device_name() and core.get("coreType") == "AICPU-SCHED":
                continue
            block_idx = core.get("block_idx", -1)
            core_count = sum(1 for task in tasks if task["name"].startswith("BEGIN"))
            if core["coreType"].startswith("AICPU"):
                exit_core_count = sum(1 for task in tasks if task["name"].startswith("EXIT"))
                assert exit_core_count == 3, f"{block_idx} Multiple Turn with exit info is invalid"
            assert len(tasks) > 0, f"{block_idx} Could not Get aicpu perf"
            assert core_count == 3, f"{block_idx} Multiple Turn Get aicpu perf not success"
    swim_lane_json_path = pref_path + "/merged_swimlane.json"
    assert os.path.exists(swim_lane_json_path), "Could not Get swim lane"
    assert os.path.getsize(swim_lane_json_path) > 0, "Get swim lane is null"

    tilefwk_l1_prof_data_path = pref_path + "/tilefwk_L1_prof_data.json"
    assert os.path.exists(tilefwk_l1_prof_data_path), "Could not Get tilefwk_L1_prof_data"
    topo_path = pref_path + "/dyn_topo.txt"
    assert os.path.exists(topo_path), "Could not Get topo"
    # can not be empty list, need to have data
    with open(tilefwk_l1_prof_data_path, 'r', encoding='utf-8') as f:
        tilefwk_11_prof_data: List[Dict] = json.load(f)
        assert len(tilefwk_11_prof_data) > 0, "tilefwk_L1_prof_data is empty"
