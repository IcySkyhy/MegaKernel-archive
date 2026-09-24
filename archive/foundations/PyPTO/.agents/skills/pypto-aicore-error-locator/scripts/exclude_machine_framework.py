#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
步骤 2: 排除 machine 框架调度问题

将 SKILL.md 中原来的 2.1-2.4 四个子步骤合并为单个脚本调用:
  2.1 - 定位已安装 pypto 中的 aicore_entry.h（通过 pip show pypto）
  2.2 - 备份并注释 CallSubFuncTask（直接修改安装路径，无需重编）
  2.3 - 运行测试验证 aicore error 是否消失
  2.4 - 从备份恢复原始 aicore_entry.h

用法:
    python3 exclude_machine_framework.py --test-cmd "<test_cmd>" --run-path <run_path>

返回值:
    Exit 0: 注释后 aicore error 消失 → 问题在 kernel 代码中，继续执行后续步骤
    Exit 1: 注释后仍有 aicore error → 问题在 machine 框架调度，停止执行后续步骤
    Exit 2: 定位 aicore_entry.h 或 CallSubFuncTask 失败

设计参考: tools/scripts/locate_aicore_error.py 中的 exclude_machine_framework_issues() 函数
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    _PIP_CMD,
    DEFAULT_TIMEOUT,
    comment_lines_by_range,
    read_file,
    setup_logging,
    write_file,
)

setup_logging()
logger = logging.getLogger(__name__)


def _find_installed_aicore_entry():
    logger.info("")
    logger.info("---步骤 2.1: 定位已安装 pypto 中的 aicore_entry.h")
    logger.info("")

    result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.info("✗ pip show pypto 执行失败")
        return None

    location = None
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            location = line.split(':', 1)[1].strip()
            break

    if not location:
        logger.info("✗ 无法解析 pypto 安装位置")
        return None

    for root, _, files in os.walk(location):
        if 'aicore_entry.h' in files:
            installed_path = os.path.join(root, 'aicore_entry.h')
            logger.info("---步骤 2.1 结果: ✓ %s", installed_path)
            return installed_path

    logger.info("✗ 安装路径下未找到 aicore_entry.h")
    return None


def _find_callsubfunctask_range(lines):
    callsub_idx = -1
    for i, line in enumerate(lines):
        if 'CallSubFuncTask' in line:
            callsub_idx = i
            break

    if callsub_idx == -1:
        return None, None

    end_idx = callsub_idx
    for i in range(callsub_idx, len(lines)):
        if ');' in lines[i]:
            end_idx = i
            break

    start_idx = callsub_idx
    for i in range(callsub_idx, -1, -1):
        if '#if ENABLE_AICORE_PRINT' in lines[i]:
            start_idx = i
            break

    return start_idx, end_idx


def _comment_callsubfunctask(file_path):
    lines = read_file(file_path)
    start, end = _find_callsubfunctask_range(lines)
    if start is None or end is None:
        return False, None, None
    comment_lines_by_range(lines, start, end)
    write_file(file_path, lines)
    logger.info("成功注释 CallSubFuncTask 部分（行 %d-%d）", start + 1, end + 1)
    return True, start, end


def _run_test(test_cmd, run_dir, use_pypto_test_framework=False):
    logger.info("")
    logger.info("[测试用例执行] %s", test_cmd[:100])
    logger.info("")

    if isinstance(test_cmd, str):
        cmd_list = shlex.split(test_cmd)
    else:
        cmd_list = test_cmd

    result = subprocess.run(
        cmd_list,
        cwd=run_dir,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=DEFAULT_TIMEOUT,
        check=False,
    )
    stderr_output = result.stdout + result.stderr
    if use_pypto_test_framework:
        has_aicore = "aicore error" in stderr_output.lower()
    else:
        has_aicore = result.returncode != 0 and "aicore error" in stderr_output.lower()

    logger.info("")
    logger.info("aicore error %s", "存在" if has_aicore else "已消失")
    return has_aicore


def main():
    parser = argparse.ArgumentParser(description='步骤 2: 排除 machine 框架调度问题（合并 2.1-2.4）')
    parser.add_argument('--test-cmd', required=True, help='触发 aicore error 的测试命令')
    parser.add_argument('--run-path', default=os.getcwd(), help='运行测试的目录路径（默认: 当前目录）')
    parser.add_argument('--use-pypto-test-framework', action='store_true', help='使用 Pypto_Test 框架模式')
    args = parser.parse_args()

    run_path = os.path.abspath(args.run_path)
    if not os.path.exists(run_path):
        logger.info("错误：运行目录不存在: %s", run_path)
        sys.exit(2)

    installed_path = _find_installed_aicore_entry()
    if not installed_path:
        logger.info("")
        logger.info("=" * 80)
        logger.info("步骤 2 结果: ✗ 步骤 2.1 定位 aicore_entry.h 失败")
        logger.info("=" * 80)
        sys.exit(2)

    lines = read_file(installed_path)
    start, end = _find_callsubfunctask_range(lines)
    if start is None or end is None:
        logger.info("")
        logger.info("✗ 未找到 CallSubFuncTask")
        logger.info("")
        logger.info("=" * 80)
        logger.info("步骤 2 结果: ✗ 步骤 2.2 定位 CallSubFuncTask 失败")
        logger.info("=" * 80)
        sys.exit(2)

    already_commented = all(not lines[i].strip() or lines[i].strip().startswith('//') for i in range(start, end + 1))
    did_backup = False

    try:
        logger.info("")
        logger.info("---步骤 2.2: 注释 CallSubFuncTask — 修改安装路径下的 aicore_entry.h")
        logger.info("")

        if already_commented:
            logger.info("CallSubFuncTask 已为注释状态，跳过修改")
        else:
            backup_path = installed_path + ".step2_bak"
            shutil.copy(installed_path, backup_path)
            did_backup = True
            if not _comment_callsubfunctask(installed_path)[0]:
                logger.info("✗ 注释 CallSubFuncTask 失败")
                sys.exit(2)
            logger.info("")
            logger.info("---步骤 2.2 结果: ✓ 已注释 CallSubFuncTask")

        logger.info("")
        logger.info("---步骤 2.3: 运行测试验证 — 判断 aicore error 是否消失")
        has_aicore = _run_test(args.test_cmd, run_path, use_pypto_test_framework=args.use_pypto_test_framework)

    finally:
        if did_backup:
            logger.info("")
            logger.info("---步骤 2.4: 恢复 aicore_entry.h — 从备份恢复安装路径下的原始文件")
            backup_path = installed_path + ".step2_bak"
            if os.path.exists(backup_path):
                shutil.copy(backup_path, installed_path)
                os.remove(backup_path)
                logger.info("---步骤 2.4 结果: ✓ aicore_entry.h 已恢复")

    logger.info("")
    logger.info("=" * 80)
    if not has_aicore:
        logger.info("步骤 2 结果: ✓ 注释后无 aicore error → 问题在 kernel 代码中，继续排查")
        logger.info("=" * 80)
        sys.exit(0)
    else:
        logger.info("步骤 2 结果: ✗ 注释后仍有 aicore error → 问题在 machine 框架调度")
        logger.info("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
