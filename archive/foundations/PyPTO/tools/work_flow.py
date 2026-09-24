#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
This is a subscript of build_ci.py
"""

import os
from pathlib import Path
import shutil
import subprocess


def ini(path, prof, pe):
    """
    L0----------------------------------------------------------------------暂无使用
    export PROFILER_SAMPLECONFIG='{"stars_acsq_task":"off","app":"test_dynshape","prof_level":"l0","taskTime":"l0",
    "result_dir":"/home/chenjz/tilefwk_code/build","app_dir":"/home/chenjz/tilefwk_code/build/.",
    "ai_core_profiling":"off","aicpuTrace":"on"}'
    L1-----------------------------------------------------------------------泳道图数据打点采集
    export PROFILER_SAMPLECONFIG='{"stars_acsq_task":"off","app":"test_dynshape","prof_level":"l1","taskTime":"l1",
    "result_dir":"/home/chenjz/tilefwk_code/build","app_dir":"/home/chenjz/tilefwk_code/build/.",
    "ai_core_profiling":"off","aicpuTrace":"on"}'
    L2-----------------------------------------------------------------------打点采集泳道图、PMU数据
    export PROFILER_SAMPLECONFIG='{"stars_acsq_task":"off","app":"test_dynshape","prof_level":"l2","taskTime":"l2",
    "result_dir":"/home/chenjz/tilefwk_code/build","app_dir":"/home/chenjz/tilefwk_code/build/.",
    "ai_core_profiling":"off","aicpuTrace":"on"}'
    """
    # 定义字典
    level = None
    real_pe = None
    if prof == 1:
        level = "l1"
    elif prof == 2:
        level = "l2"
        real_pe = pe
    else:
        assert False, f'Current {prof} is invalid, only support[1, 2]'
    level = "l2"
    env = {
        "PROFILER_SAMPLECONFIG": "{"
        + "\"stars_acsq_task\":\"off\","
        + "\"app\":\"test_dynshape\","
        + f"\"prof_level\":\"{level}\","
        + f"\"taskTime\":\"{level}\","
        + f"\"result_dir\":\"{str(path)}\","
        + f"\"app_dir\":\"{str(path)}\","
        + "\"ai_core_profiling\":\"off\","
        + "\"aicpuTrace\":\"on\""
        + "}"
    }
    if real_pe:
        env["PYPTO_PROF_PMU_EVENT_TYPE"] = str(real_pe)
    return env


def find_file_and_get_parent_dirs(target_file, search_dir='.'):
    """
    在指定目录及其子目录中查找目标文件，并返回其父目录的绝对路径列表
    :param target_file: 要查找的目标文件名称
    :param search_dir: 开始查找的目录，默认为当前目录
    :return: 包含匹配文件父目录绝对路径的列表
    """
    search_path = Path(search_dir).resolve()
    parent_dirs = []
    for file_path in search_path.rglob(target_file):
        parent_dirs.append(str(file_path.parent.resolve()))
    return parent_dirs


def remove_prof_dirs(base_path):
    base = Path(base_path).resolve()
    for prof_dir in base.glob("PROF*"):
        if prof_dir.is_dir():
            shutil.rmtree(prof_dir)


def work_flow_plot(path, level, pe):
    base_path = Path(path).resolve()
    path_pro = find_file_and_get_parent_dirs("aicpu.data.*", base_path)
    if len(path_pro) < 1:
        remove_prof_dirs(base_path)
        assert False, f'{str(base_path)} No Profiling, Do not to plot'
    save_path = base_path / "Profiling/work_flow"
    if not save_path.exists():
        os.makedirs(save_path)
    print(save_path)
    cmd = [
        "python3",
        "./tools/profiling/tilefwk_prof_data_parser.py",
        "-p",
        str(path_pro[0]),
        f"--output={str(save_path)}",
        "-t",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, shell=False, capture_output=False, check=True, text=True, encoding='utf-8')
    if level == 2:
        cmd = [
            "python3",
            "./tools/profiling/tilefwk_pmu_to_csv.py",
            "-p",
            str(path_pro[0]),
            f"--output={str(save_path)}",
            f"-pe={pe}",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, shell=False, capture_output=False, check=True, text=True, encoding='utf-8')
    # 删除Prof落盘日志，避免有干扰
    print(f"cleanup PROF* under {str(base_path)}")
    remove_prof_dirs(base_path)
