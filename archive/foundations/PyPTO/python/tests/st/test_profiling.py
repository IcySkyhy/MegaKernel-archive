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

import csv
import glob
import os
import shutil
import subprocess

import pytest
import torch
import torch_npu

import pypto


def _get_root_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def _get_prof_base_dir(root_dir: str) -> str:
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        return os.path.join(ascend_work_path, "profiling_data")
    return root_dir


def _clean_prof_dirs(prof_base_dir: str) -> None:
    for old_dir in glob.glob(os.path.join(prof_base_dir, "PROF*")):
        shutil.rmtree(old_dir, ignore_errors=True)


def _run_msprof(root_dir: str, script_path: str) -> subprocess.CompletedProcess:
    cmd = ["msprof", "python", script_path]
    try:
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        raise pytest.fail("msprof 命令执行超时") from exc
    except FileNotFoundError as exc:
        raise pytest.fail("msprof 命令未找到，请确保 CANN 环境已正确配置") from exc


def _collect_op_summary_files(prof_dirs):
    op_summary_files_found = []
    for prof_dir in prof_dirs:
        op_summary_pattern = os.path.join(prof_dir, "mindstudio_profiler_output", "op_summary_*.csv")
        op_summary_files_found.extend(glob.glob(op_summary_pattern))
    return op_summary_files_found


def _csv_contains_pypto(csv_file: str) -> bool:
    try:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return any(
                "PYPTO_add_kernel" in row.get("Op Name", "") and "PyPTO" in row.get("OP Type", "") for row in reader
            )
    except Exception:
        return False


def _find_pypto_in_csv(op_summary_files):
    return any(_csv_contains_pypto(csv_file) for csv_file in op_summary_files)


def _kernel_details_contains_pypto(csv_file: str) -> bool:
    try:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return any(
                "PyPTO" in row.get("Type", "") and "PYPTO_add_direct_kernel" in row.get("Name", "") for row in reader
            )
    except Exception:
        return False


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_direct_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    z: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    z.move(x + y)


def _build_experimental_config():
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return experimental_config


def _run_add_direct_profiler(device_id: int, shape: tuple, profiler_output_dir: str) -> None:
    input_data0 = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    input_data1 = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    output_data = torch.zeros(shape, dtype=torch.float, device=f"npu:{device_id}")
    experimental_config = _build_experimental_config()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiler_output_dir, analyse_flag=True),
    ) as prof:
        for _ in range(10):
            add_direct_kernel(input_data0, input_data1, output_data)
            torch_npu.npu.synchronize()
            prof.step()


def _collect_kernel_detail_files(profiler_output_dir: str):
    return glob.glob(
        os.path.join(profiler_output_dir, "**", "kernel_details.csv"),
        recursive=True,
    )


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": True})
def test_msprof_profiling_pypto_op_summary():
    """
    看护用例：验证 msprof 性能采集功能
    1. 执行 msprof python examples/00_hello_world/hello_world.py
    2. 验证 PROF*/mindstudio_profiler_output/op_summary_*.csv 文件生成
    3. 验证 CSV 文件中 Op Name 包含 PYPTO_add_kernel 字样，且 OP Type 包含 PyPTO 字样
    """
    root_dir = _get_root_dir()
    prof_base_dir = _get_prof_base_dir(root_dir)
    _clean_prof_dirs(prof_base_dir)
    add_direct_script = os.path.join(root_dir, "examples", "00_hello_world", "hello_world.py")
    assert os.path.exists(add_direct_script), f"脚本不存在: {add_direct_script}"

    _run_msprof(root_dir, add_direct_script)
    prof_dirs = glob.glob(os.path.join(prof_base_dir, "PROF*"))
    assert len(prof_dirs) > 0, f"未在 {prof_base_dir} 下找到 PROF* 文件夹"

    op_summary_files_found = _collect_op_summary_files(prof_dirs)
    pypto_found = _find_pypto_in_csv(op_summary_files_found)
    assert len(op_summary_files_found) > 0, (
        f"未在 PROF* 文件夹中找到 mindstudio_profiler_output/op_summary_*.csv 文件。\n已检查的 PROF 目录: {prof_dirs}"
    )

    assert pypto_found, (
        f"在 op_summary CSV 文件中未找到同时满足 Op Name 包含 PYPTO_add_kernel 且 OP Type 包含 PyPTO 的记录。\n"
        f"已检查的 CSV 文件: {op_summary_files_found}"
    )

    for prof_dir in prof_dirs:
        shutil.rmtree(prof_dir, ignore_errors=True)


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": True})
def test_torch_npu_profiler_collect_pypto_kernel_details():
    """
    看护用例：验证 torch_npu.profiler 能正确采集到 PyPTO 内核信息
    1. 在测试中直接定义并执行 add_direct_kernel
    2. 在 ./add_direct_profiler 下递归查找 kernel_details.csv
    3. 校验 Type 包含 PyPTO 且 Name 包含 PyPYPTO_add_kernel
    4. 清理 add_direct_profiler 文件夹
    """
    root_dir = _get_root_dir()
    profiler_output_dir = os.path.join(root_dir, "add_direct_profiler")
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    shape = (1, 4, 1, 64)

    shutil.rmtree(profiler_output_dir, ignore_errors=True)
    try:
        _run_add_direct_profiler(device_id, shape, profiler_output_dir)
        kernel_detail_files = _collect_kernel_detail_files(profiler_output_dir)
        assert len(kernel_detail_files) > 0, f"未在 {profiler_output_dir} 下递归找到 kernel_details.csv"

        matched = any(_kernel_details_contains_pypto(csv_file) for csv_file in kernel_detail_files)
        assert matched, (
            "在 kernel_details.csv 中未找到 Type 包含 PyPTO 且 "
            "Name 包含 PYPTO_add_direct_kernel 的记录。\n"
            f"已检查文件: {kernel_detail_files}"
        )
    finally:
        shutil.rmtree(profiler_output_dir, ignore_errors=True)
