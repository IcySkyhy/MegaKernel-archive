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

import os
from pathlib import Path
import shutil

import torch
import torch_npu

import pypto


@pypto.frontend.jit(debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=32,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c.move(a + b)


def check_output():
    out_path = os.environ.get('TILE_FWK_OUTPUT_DIR', "./output")
    latest_dir = ""
    if os.path.exists(out_path):
        subdirs = [os.path.join(out_path, d) for d in os.listdir(out_path) if os.path.isdir(os.path.join(out_path, d))]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getctime)
    assert os.path.exists(latest_dir)

    check_list = ["program.json", "dyn_topo.txt", "merged_swimlane.json", "tilefwk_L1_prof_data.json"]
    file_list = [os.path.join(latest_dir, d) for d in check_list]
    lost_file = None
    for file_path in file_list:
        if not os.path.exists(file_path):
            lost_file = file_path
            break
    shutil.rmtree(out_path)
    assert lost_file is None


def device_run(device_id):
    tiling = 32
    n, m = tiling * 1, tiling * 1
    _shape = (n, m)

    # prepare data
    a_data = torch.ones((n, m), dtype=torch.int32, device=f'npu:{device_id}') * 2
    b_data = torch.ones((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')

    add_kernel(a_data, b_data, c_data, tiling=tiling)
    torch_npu.npu.synchronize()

    check_output()


@pypto.options(pass_options={"enable_slice": True})
def test_debug_mode():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    output_name = "temp"
    os.environ["TILE_FWK_OUTPUT_DIR"] = f"{Path.cwd()}/{output_name}"
    torch.npu.set_device(int(device_id))
    device_run(device_id)
