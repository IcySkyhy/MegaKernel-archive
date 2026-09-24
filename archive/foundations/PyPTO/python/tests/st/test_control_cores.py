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
""" """

import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch_npu

import pypto
from st.test_swim_line import matmul_add

_OUTPUT_BASE = Path("./output")


def setup_function():
    from pypto.frontend.parser.entry import JitCallableWrapper

    getattr(JitCallableWrapper, '_kernel_module_cache').clear()
    if hasattr(torch_npu.npu.set_device_limit, 'called'):
        torch_npu.npu.set_device_limit.called = False


def _wait_for_prof_file(after_ts, timeout=10):
    my_pid = str(os.getpid())
    deadline = time.time() + timeout
    while time.time() < deadline:
        subdirs = sorted(
            [d for d in _OUTPUT_BASE.iterdir() if d.is_dir()],
            reverse=True,
        )
        for d in subdirs:
            if d.stat().st_mtime <= after_ts:
                continue
            if my_pid not in d.name:
                continue
            fpath = d / "tilefwk_L1_prof_data.json"
            if fpath.is_file():
                return str(fpath)
        time.sleep(0.1)
    raise RuntimeError(f"Prof file not found within {timeout}s after mark_ts={after_ts}")


def count_core_types(base_dir="./output"):
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        return 0, 0
    latest_dir = sorted(subdirs, reverse=True)

    file_path = str(latest_dir[0] / "tilefwk_L1_prof_data.json")
    for folder in latest_dir:
        target_path = folder / "tilefwk_L1_prof_data.json"
        if target_path.is_file():
            file_path = str(target_path)
            break

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    data = json.loads(content)
    aic_count = 0
    aiv_count = 0
    for item in data:
        core_type = item.get('coreType')
        if core_type == 'AIC':
            aic_count += 1
        elif core_type == 'AIV':
            aiv_count += 1

    return aic_count, aiv_count


def kernel_func(device_id):
    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    c_data_list = []
    d_data_list = []

    count = 2

    a_rawdata = torch.tensor([[1] * k] * n)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')

    mark_ts = time.time()
    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
        d_data_list.append(d_data)

        matmul_add(a_data, b_data, c_data, d_data)

    torch_npu.npu.synchronize()

    for idx in range(count):
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)

    prof_path = _wait_for_prof_file(mark_ts)
    with open(prof_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read().strip())
    aic_count = sum(1 for i in data if i.get('coreType') == 'AIC')
    aiv_count = sum(1 for i in data if i.get('coreType') == 'AIV')

    return aic_count, aiv_count


@pytest.mark.soc("910")
def test_not_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    kernel_func(device_id)


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": False})
def test_rts_stream_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    stream1 = torch.npu.current_stream()
    torch.npu.set_stream_limit(stream1, 8, 27)

    aic_count, aiv_count = kernel_func(device_id)
    assert aic_count <= 8, f"stream aic limit: expected <= 8, got {aic_count}"
    assert aiv_count <= 16, f"stream aiv limit: expected <= 16, got {aiv_count}"


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": False})
def test_rts_device_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.npu.set_device_limit(device_id, 8, 16)

    aic_count, aiv_count = kernel_func(device_id)
    assert aic_count <= 8, f"device aic limit: expected <= 8, got {aic_count}"
    assert aiv_count <= 16, f"device aiv limit: expected <= 16, got {aiv_count}"


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": False})
def test_rts_device_stream_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    torch.npu.set_device_limit(device_id, 15, 27)

    stream1 = torch.npu.current_stream()
    torch.npu.set_stream_limit(stream1, 8, 27)

    aic_count, aiv_count = kernel_func(device_id)
    assert aic_count <= 8, f"device+stream aic limit: expected <= 8, got {aic_count}"
    assert aiv_count <= 16, f"device+stream aiv limit: expected <= 16, got {aiv_count}"
