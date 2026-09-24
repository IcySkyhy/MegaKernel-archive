#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from shutil import which
import subprocess
from typing import List, Optional


def setup_logging(level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_path(path: str, must_exist: bool = True) -> Path:
    """验证并返回路径对象"""
    path_obj = Path(path).expanduser().resolve()
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    return path_obj


def find_tool(tool_name: str, alternatives: Optional[List[str]] = None) -> Optional[str]:
    """查找可用的工具"""
    # 首先尝试直接查找
    tool_path = which(tool_name)
    if tool_path:
        return tool_path

    # 尝试替代工具
    if alternatives:
        for alt in alternatives:
            alt_path = which(alt)
            if alt_path:
                return alt_path

    return None


def check_required_tools() -> dict:
    """检查必需的工具"""
    tools = {}

    # 检查 addr2line 工具
    tools['addr2line'] = find_tool('llvm-addr2line', ['addr2line'])

    # 检查 objdump 工具
    tools['objdump'] = find_tool('llvm-objdump', ['objdump'])

    return tools


def run_command(cmd: List[str], capture: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    """运行命令"""
    logger = logging.getLogger(__name__)
    logger.debug("运行命令: %s", ' '.join(cmd))

    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        else:
            result = subprocess.run(cmd, timeout=timeout)
        return result
    except subprocess.TimeoutExpired:
        logger.error("命令执行超时: %s", ' '.join(cmd))
        raise
    except Exception as e:
        logger.error("命令执行失败: %s, 错误: %s", ' '.join(cmd), e)
        raise


def is_valid_binary(path: Path) -> bool:
    """检查是否是有效的二进制文件"""
    if not path.exists():
        return False

    # 检查文件类型
    try:
        file_tool = which('file')
        if not file_tool:
            return False
        result = subprocess.run([file_tool, str(path)], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.lower()
            # 检查是否是 ELF 或 Mach-O 文件
            return 'elf' in output or 'mach-o' in output
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("检查二进制文件类型失败: %s, 错误: %s", path, e, exc_info=True)

    return False
