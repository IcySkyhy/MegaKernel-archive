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
PMU Trace codegen 看护：开启 enable_pmu_trace 后，kernel_aicore 生成的 CCE 代码应插入 mark_stamp。
"""

import glob
import os
from pathlib import Path
import shutil

import pytest
import torch
import torch_npu

import pypto


@pypto.frontend.jit(
    codegen_options={"enable_pmu_trace": True},
)
def add_kernel_pmu_trace(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=32,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c.move(a + b)


def _find_kernel_aicore_cce_files(log_dir: str):
    cce_dir = os.path.join(log_dir, "kernel_aicore")
    patterns = [
        os.path.join(cce_dir, "*.cpp"),
        os.path.join(cce_dir, "*.cce"),
    ]
    cce_files = []
    for pattern in patterns:
        cce_files.extend(glob.glob(pattern))
    return cce_files


def _assert_mark_stamp_in_cce(log_dir: str):
    cce_files = _find_kernel_aicore_cce_files(log_dir)
    assert cce_files, f"no CCE files under {log_dir}/kernel_aicore"

    files_with_stamp = []
    for cce_path in cce_files:
        with open(cce_path, encoding="utf-8") as f:
            content = f.read()
        if "mark_stamp" in content:
            files_with_stamp.append(cce_path)

    assert files_with_stamp, f"mark_stamp not found in any CCE file under {log_dir}/kernel_aicore, checked: {cce_files}"


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pmu_trace_codegen_inserts_mark_stamp():
    output_name = "temp_pmu_trace_codegen"
    os.environ["TILE_FWK_OUTPUT_DIR"] = f"{Path.cwd()}/{output_name}"
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling, tiling
    a_data = torch.ones((n, m), dtype=torch.int32, device=f"npu:{device_id}") * 2
    b_data = torch.ones((n, m), dtype=torch.int32, device=f"npu:{device_id}")
    c_data = torch.zeros((n, m), dtype=torch.int32, device=f"npu:{device_id}")

    try:
        add_kernel_pmu_trace(a_data, b_data, c_data, tiling=tiling)
        torch_npu.npu.synchronize()

        log_dir = pypto.pypto_impl.LogTopFolder()
        _assert_mark_stamp_in_cce(log_dir)
    finally:
        shutil.rmtree(output_name, ignore_errors=True)
