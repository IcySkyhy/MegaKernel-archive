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
"""Shared utility for locating the system-level cmake binary."""

import os
from pathlib import Path
import shutil
from typing import Optional


def which_cmake() -> Optional[Path]:
    """查找系统级 CMake 可执行文件路径

    排除 cmake pip 包的干扰, 通过遍历 PATH 环境变量查找 ELF 格式的 CMake 可执行文件.

    :return: 系统 CMake 可执行文件路径, 找不到则返回 None
    :rtype: Optional[Path]
    """
    # 拆分 PATH 环境变量为单个目录列表(排除空目录)
    path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]
    # 遍历每个 PATH 目录, 逐个调用 shutil.which 检查, 限定 shutil.which 只在当前单个目录下查找 cmake
    valid_path_lst = []
    for path_dir in path_dir_lst:
        # 避免 PATH 环境变量中有重复的单元
        if path_dir in valid_path_lst:
            continue
        valid_path_lst.append(path_dir)
        # 检查当前目录
        cmake_str = shutil.which("cmake", path=path_dir)
        if not cmake_str:
            continue
        cmake_file = Path(cmake_str).resolve()
        if not cmake_file.exists() or not cmake_file.is_file():
            continue
        if cmake_file.stat().st_size <= 4:  # 下文读取前 4 字节判断文件是否是 ELF 文件
            continue
        with open(cmake_file, 'rb') as fh:
            header = fh.read(4)  # 前 4 字节是 ELF 文件标识
        if header != b'\x7fELF':
            continue
        return cmake_file
    return None
