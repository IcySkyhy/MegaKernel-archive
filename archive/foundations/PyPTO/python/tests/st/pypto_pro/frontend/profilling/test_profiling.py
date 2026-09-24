#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""profiling 数据采集泛化测试 — 验证 CANN profiler 能正确采集 PyPTO kernel 信息。

环境要求:
  CANN 9.1: source /home/zhangjie/Ascend/cann-9.1.0/bin/setenv.bash

采集方式:
  1. torch_npu.profiler API（编程式，采集 kernel_details.csv）

参考:
  python/tests/st/test_profiling.py
"""

import csv
import glob
import logging
import os
import shutil

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _get_prof_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _build_experimental_config():
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    return experimental_config_cls(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


# =============================================================================
# 被采集的 kernel — 简单 add 操作
# =============================================================================
@pl.jit()
def prof_add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tt, addr=0x0000, size=16384)
    tb = pl.make_tile(tt, addr=0x4000, size=16384)
    tc = pl.make_tile(tt, addr=0x8000, size=16384)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


# =============================================================================
# Test 1: torch_npu.profiler API 采集 — 验证 kernel_details.csv
# =============================================================================
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_profiler_api_kernel_details():
    """测试 torch_npu.profiler API 采集模式，验证 profiler 能正确生成 kernel_details.csv 并包含 PyPTO kernel 记录。

    采集文件：profiling_output/api/ 下递归搜索 kernel_details.csv。
    验证内容：读取 CSV 文件，检查 Type 或 Name 列中是否包含 "prof_add_kernel"，
    确认 CANN profiler 能够识别并记录 PyPTO 自定义 kernel 的调用信息。

    输入：x,y=randn(64,64) FP32，z=zeros(64,64) FP32，循环调用 prof_add_kernel 10 次。
    预期：生成非空 kernel_details.csv，且至少有一行包含 "prof_add_kernel"。
    """
    _check_npu()
    logging.info("------------test_profiler_api_kernel_details--------------")

    root_dir = _get_prof_root_dir()
    output_dir = os.path.join(root_dir, "profiling_output", "api")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    x = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
    y = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
    z = torch.zeros(64, 64, device=ST_DEVICE, dtype=torch.float32)

    experimental_config = _build_experimental_config()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir, analyse_flag=True),
    ) as prof:
        for _ in range(10):
            prof_add_kernel(x, y, z)
            torch.npu.synchronize()
            prof.step()

    logging.info("profiler output dir: %s", output_dir)

    kernel_detail_files = glob.glob(os.path.join(output_dir, "**", "kernel_details.csv"), recursive=True)
    logging.info("Profiler output dir: %s", output_dir)
    logging.info("Found kernel_details files: %s", kernel_detail_files)
    assert len(kernel_detail_files) > 0, (
        f"未在 {output_dir} 下递归找到 kernel_details.csv\n"
        f"目录内容: {os.listdir(output_dir) if os.path.exists(output_dir) else '不存在'}"
    )

    matched = False
    for csv_file in kernel_detail_files:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dtype_val = row.get("Type", "")
                name_val = row.get("Name", "")
                if "prof_add_kernel" in dtype_val or "prof_add_kernel" in name_val:
                    logging.info("Found: Type=%s Name=%s", dtype_val, name_val)
                    matched = True
                    break
        if matched:
            break

    assert matched, (
        f"kernel_details.csv 中未找到 Name/Type 包含 'prof_add_kernel' 的记录。\n已检查文件: {kernel_detail_files}"
    )

    logging.info("profiler_api_kernel_details passed!")
    logging.info("profiling result: %s", output_dir)


# =============================================================================
# Test 2: torch_npu.profiler 采集 — 验证 trace_view.json 生成
# =============================================================================
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_profiler_api_trace_view():
    """测试 torch_npu.profiler API 采集模式，验证 profiler 能正确生成 trace_view.json（或 trace_result.json）且文件非空。

    采集文件：profiling_output/trace/ 下递归搜索 trace_view.json，若未找到则回退搜索 trace_result.json。
    验证内容：检查文件大小 > 0 字节，确认 trace 数据被成功采集写入。

    输入：x,y=randn(64,64) FP32，z=zeros(64,64) FP32，循环调用 prof_add_kernel 10 次。
    预期：生成的 trace 文件大小 > 0，可供可视化工具加载分析。
    """
    _check_npu()
    logging.info("------------test_profiler_api_trace_view--------------")

    root_dir = _get_prof_root_dir()
    output_dir = os.path.join(root_dir, "profiling_output", "trace")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    x = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
    y = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
    z = torch.zeros(64, 64, device=ST_DEVICE, dtype=torch.float32)

    experimental_config = _build_experimental_config()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir, analyse_flag=True),
    ) as prof:
        for _ in range(10):
            prof_add_kernel(x, y, z)
            torch.npu.synchronize()
            prof.step()

    trace_files = glob.glob(os.path.join(output_dir, "**", "trace_view.json"), recursive=True)
    if not trace_files:
        trace_files = glob.glob(os.path.join(output_dir, "**", "trace_result.json"), recursive=True)

    assert len(trace_files) > 0, f"未在 {output_dir} 下找到 trace_view.json 或 trace_result.json"

    for tf in trace_files:
        sz = os.path.getsize(tf)
        logging.info("Trace file: %s, size=%d bytes", tf, sz)
        assert sz > 0, f"trace 文件为空: {tf}"

    logging.info("profiler_api_trace_view passed!")
    logging.info("profiling result: %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_profiler_api_kernel_details,
        test_profiler_api_trace_view,
    ]
    for t in tests:
        t()
        logging.info(f"{t.__name__} passed!")
    logging.info("\nAll profiling NPU tests passed!")
    logging.info("\nProfiling result files are kept at:")
    logging.info("  %s/build/profiling_test/", _get_prof_root_dir())
