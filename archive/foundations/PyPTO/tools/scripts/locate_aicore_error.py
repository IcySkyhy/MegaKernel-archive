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
AICore Error 一键定位脚本

此脚本用于定位 PyPTO 测试中出现的 aicore error，通过系统化的排查流程，
找出导致错误的 CCE 文件和具体代码行，并映射到前端源代码。

用法:
    python3 locate_aicore_error.py --pypto-path <pypto_path> --run-path <run_path> --test-cmd <test_cmd>

示例:
    python3 locate_aicore_error.py --pypto-path /path/to/pypto --run-path /path/to/run --test-cmd "python test_my_op.py"

工作流程（5 个步骤）:
    步骤 1: 初始测试 — 问题复现，检查是否为该脚本可适用的 aicore error 场景
    步骤 2: 排除 machine 框架调度问题 — 判断问题在 kernel 代码还是 machine 调度框架
    步骤 3: 定位问题 CCE 文件
        3.1 启用追踪日志
        3.2 重新编译和安装
        3.3 清理日志并运行测试
        3.4 检查 kernel 二进制文件大小
        3.5 分析追踪日志并定位 CCE 文件
        3.6 测试验证 CCE 文件
    步骤 4: 二分 CCE 文件找到问题代码行
        4.1 确定错误范围（是否在 T 操作中）
        4.2 获取可注释行范围
        4.3 二分查找迭代
    步骤 5: 问题代码行映射到前端代码
        5.1 CCE 行映射到前端源文件
        5.2 输出最终结果
"""

import argparse
from collections.abc import Callable
import dataclasses
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time

# ============================================================================
# Constants
# ============================================================================

RESTART = "RESTART"
DEFAULT_TIMEOUT = 1800
_PIP_CMD = shutil.which("pip3") or shutil.which("pip") or "/usr/bin/pip3"
SKIP_KEYWORDS = ['set_flag', 'wait_flag', 'pipe_barrier']

# Known file locations for locate_file()
_KNOWN_LOCATIONS = {
    'device_switch.h': "framework/src/machine/utils/device_switch.h",
}

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Context for Output-related Helpers
# ============================================================================


@dataclasses.dataclass
class _OutputCtx:
    """Groups output-related configuration shared across I/O functions."""

    show_output: bool
    indent: str
    source_tag: str = ""


@dataclasses.dataclass
class _CceOpRef:
    """Groups the CCE operation identification fields for source mapping."""

    cce_op_lines: list
    op_index: int
    op_name: str
    cce_line: str


# ============================================================================
# Logging Helpers — Major/Minor Step Formatting
# ============================================================================


def _log_major(step_id, description):
    logger.info("=" * 80)
    logger.info("步骤 %s: %s", step_id, description)
    logger.info("=" * 80)


def _log_major_result(step_id, fmt, *args):
    msg = fmt % args if args else fmt
    logger.info("")
    logger.info("步骤 %s 结果: %s", step_id, msg)
    logger.info("")


def _log_minor(step_id, description):
    logger.info("")
    logger.info("---步骤 %s: %s", step_id, description)
    logger.info("")


def _log_minor_result(step_id, fmt, *args):
    msg = fmt % args if args else fmt
    logger.info("")
    logger.info("---步骤 %s 结果: %s", step_id, msg)
    logger.info("")


# ============================================================================
# Logging & Command Execution
# ============================================================================


def _log_if_relevant(line, line_num, ctx):
    if not ctx.show_output:
        return
    stripped = line.strip()

    if line_num <= 20:
        if stripped:
            tag_prefix = f"{ctx.source_tag} " if ctx.source_tag else ""
            logger.info("%s%s  %s", ctx.indent, tag_prefix, stripped[:200])
        return

    if stripped and 'error' in stripped.lower():
        tag_prefix = f"{ctx.source_tag} " if ctx.source_tag else ""
        logger.info("%s%s  %s", ctx.indent, tag_prefix, stripped[:200])


def _log_stderr(line, ctx):
    if not ctx.show_output:
        return
    stripped = line.strip()
    if stripped:
        logger.info("%s[STDERR] %s", ctx.indent, stripped[:200])


def _wait_process(process, timeout, start_time, indent):
    """Wait for a process to complete, terminating it if it exceeds timeout."""
    while process.poll() is None:
        if time.time() - start_time > timeout:
            logger.info("%s[超时] 命令执行超过 %d 秒，终止进程", indent, timeout)
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(0.1)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _capture_process_output(process, timeout, ctx):
    """Launch I/O reader threads, wait for completion, collect output."""
    stderr_lines = []
    start_time = time.time()

    def _read_stdout():
        if process.stdout:
            for i, line in enumerate(process.stdout, 1):
                _log_if_relevant(line, i, ctx)

    def _read_stderr():
        if process.stderr:
            for line in process.stderr:
                stderr_lines.append(line)
                _log_stderr(line, ctx)

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    _wait_process(process, timeout, start_time, ctx.indent)

    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    return process.returncode, ''.join(stderr_lines), start_time


def run_command_with_live_output(
    cmd, *, cwd=None, env=None, timeout=DEFAULT_TIMEOUT, show_output=True, indent="", source_tag=""
):
    """Execute a command with live output display.

    Returns (returncode, stderr_output). stdout and stderr are read in separate
    threads to avoid deadlocks.
    """
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd

    if show_output:
        display = ' '.join(cmd_list[:3])
        display += '...' if len(cmd_list) > 3 else ''
        logger.info("")
        logger.info("%s[测试用例执行] %s", indent, display)
        logger.info("")

    try:
        process = subprocess.Popen(
            cmd_list,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='ignore',
            bufsize=1,
        )
        returncode, stderr_output, start_time = _capture_process_output(
            process, timeout, _OutputCtx(show_output, indent, source_tag)
        )

        if show_output:
            elapsed = time.time() - start_time
            status = "成功" if returncode == 0 else "失败"
            logger.info("")
            logger.info("%s[%s] 耗时 %.1f 秒, 返回码: %d", indent, status, elapsed, returncode)

        return returncode, stderr_output

    except Exception as e:
        logger.info("%s[异常] 执行命令失败: %s", indent, str(e))
        return -1, str(e)


# ============================================================================
# File & Path Utilities
# ============================================================================


def validate_path(path, path_type="路径"):
    if not os.path.exists(path):
        return False, f"错误：{path_type}不存在: {path}"
    return True, None


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def locate_file(base_dir, known_rel_path, filename):
    """Find a file: try the known relative path first, then walk base_dir.

    Returns the full path, or None if not found.
    """
    primary = os.path.join(base_dir, known_rel_path)
    if os.path.exists(primary):
        return primary
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    logger.info("错误：在 %s 下未找到 %s", base_dir, filename)
    return None


def _try_restore(file_path, backup_suffix, label):
    """Restore a file from its backup. Returns True if restored."""
    if not file_path:
        return False
    backup_path = file_path + backup_suffix
    if not os.path.exists(backup_path):
        return False
    logger.info("[恢复 %s]", label)
    shutil.copy(backup_path, file_path)
    os.remove(backup_path)
    logger.info("  ✓ 已恢复 %s", label)
    return True


@dataclasses.dataclass(frozen=True)
class _ModifyConfig:
    """Configuration for checked file modification operations."""

    file_path: str
    backup_suffix: str
    check_already_done: Callable
    do_modify: Callable
    label: str


def _checked_modify_text_file(config):
    """Check if text file is already in target state. If not, backup and modify.

    Returns:
        True if the file was modified, False if already in target state.
    """
    if not os.path.exists(config.file_path):
        logger.info("警告：未找到 %s", config.file_path)
        return False

    lines = read_file(config.file_path)
    if config.check_already_done(lines):
        logger.info("%s: 已为目标状态，无需修改", config.label)
        return False

    backup_path = config.file_path + config.backup_suffix
    shutil.copy(config.file_path, backup_path)
    config.do_modify(lines)
    write_file(config.file_path, lines)
    return True


def _checked_modify_json_file(config):
    """Check if JSON config file is already in target state. If not, backup and modify.

    Returns:
        True if the file was modified, False if already in target state.
    """
    if not os.path.exists(config.file_path):
        logger.info("警告：未找到 %s", config.file_path)
        return False

    with open(config.file_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)

    if config.check_already_done(json_config):
        logger.info("%s: 已为目标状态，无需修改", config.label)
        return False

    backup_path = config.file_path + config.backup_suffix
    shutil.copy(config.file_path, backup_path)
    config.do_modify(json_config)
    with open(config.file_path, 'w', encoding='utf-8') as f:
        json.dump(json_config, f, indent=4)
    return True


# ============================================================================
# Build & Install
# ============================================================================


def build_and_install_pypto(pypto_path):
    """Build pypto from source and pip install the resulting wheel.

    Returns True on success.
    """
    logger.info("编译 pypto 包...")
    result = subprocess.run(
        ["python3", "build_ci.py", "-f", "python3", "--disable_auto_execute"],
        shell=False,
        capture_output=True,
        text=True,
        cwd=pypto_path,
        timeout=DEFAULT_TIMEOUT,
    )
    if result.returncode != 0:
        logger.info("编译失败: %s", result.stderr[-500:] if result.stderr else "无输出")
        return False

    whl_files = list(Path(pypto_path).glob("build_out/pypto*.whl"))
    if not whl_files:
        logger.info("错误：未找到 pypto whl 文件")
        return False

    logger.info("安装 pypto 包...")
    target = _get_pypto_install_location()
    if target:
        _remove_pypto_from_target(target)
    install_cmd = [_PIP_CMD, "install", str(whl_files[0]), "--force", "--no-deps"]
    if target:
        install_cmd += ["--target", target]
    result = subprocess.run(install_cmd, shell=False, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.info("安装失败: %s", result.stderr[-500:] if result.stderr else "无输出")
        return False
    return True


# ============================================================================
# Text Manipulation (CCE code commenting/uncommenting)
# ============================================================================


def comment_special_lines(lines):
    """Comment out set_flag/wait_flag/pipe_barrier lines in-place."""
    for i, line in enumerate(lines):
        if any(kw in line for kw in SKIP_KEYWORDS):
            if not line.strip().startswith('//'):
                lines[i] = '// ' + line
    return lines


def _is_uncommentable(stripped_line):
    """Check if a line should be skipped during CCE code commenting."""
    if not stripped_line:
        return True
    if stripped_line.startswith('//'):
        return True
    if '#' in stripped_line:
        return True
    if any(kw in stripped_line for kw in SKIP_KEYWORDS):
        return True
    if '{' in stripped_line and '[aicore]' in stripped_line:
        return True
    return False


def _find_tensor_func_close(lines):
    """Find the last } in the file ([aicore] main function's closing brace)."""
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if '}' in stripped and not stripped.startswith('//'):
            return i + 1
    return None


def get_commentable_lines(lines, error_in_t=False):
    """Get line numbers (1-indexed) that are safe to comment out."""
    result = []
    fast_result = []
    tensor_close = _find_tensor_func_close(lines)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if i == tensor_close:
            continue
        if _is_uncommentable(stripped):
            continue
        result.append(i)
        if stripped.startswith('T') and '<' in stripped and '>' in stripped:
            fast_result.append(i)
    return fast_result if error_in_t else result


def _build_brace_pairs(lines):
    """Build {open_idx: close_idx} mapping for all brace pairs in the file."""
    pairs = {}
    stack = []
    for i, line in enumerate(lines):
        for ch in line:
            if ch == '{':
                stack.append(i)
            elif ch == '}' and stack:
                pairs[stack.pop()] = i
    return pairs


# Module-level state set by comment_lines_by_indices, read by uncomment_lines_by_indices
_BRACE_PAIRS = {}


def comment_lines_by_indices(lines, line_indices):
    global _BRACE_PAIRS
    _BRACE_PAIRS = _build_brace_pairs(lines)

    # Auto-comment matching } for any for-loop { being commented
    extra = set()
    for ln in set(line_indices):
        close = _BRACE_PAIRS.get(ln - 1)
        if close is not None:
            extra.add(close + 1)

    all_lns = sorted(set(line_indices) | extra, reverse=True)
    for ln in all_lns:
        lines[ln - 1] = '// ' + lines[ln - 1]
    return lines


def uncomment_lines_by_indices(lines, line_indices):
    global _BRACE_PAIRS

    # Auto-uncomment matching } for any for-loop { being uncommented
    extra = set()
    for ln in set(line_indices):
        close = _BRACE_PAIRS.get(ln - 1)
        if close is not None:
            extra.add(close + 1)

    all_lns = sorted(set(line_indices) | extra)
    for ln in all_lns:
        idx = ln - 1
        if lines[idx].strip().startswith('//'):
            lines[idx] = lines[idx][3:]
    return lines


def comment_lines_by_range(lines, start_idx, end_idx):
    for i in range(start_idx, end_idx + 1):
        if not lines[i].strip().startswith('//'):
            lines[i] = '// ' + lines[i]


# ============================================================================
# Error Detection
# ============================================================================
def has_error(returncode, stderr_output, use_pypto_test_framework=False):
    """Check if stderr_output contains an aicore error.

    When use_pypto_test_framework is True, the returncode is always 0
    (Pypto_Test framework), so we only check the output content.
    """
    if not use_pypto_test_framework and returncode == 0:
        return False
    return "aicore error" in stderr_output.lower()


# ============================================================================
# Test Runners
# ============================================================================


def run_test(test_cmd, run_dir, show_output=True, source_tag=""):
    return run_command_with_live_output(
        test_cmd,
        cwd=run_dir,
        timeout=DEFAULT_TIMEOUT,
        show_output=show_output,
        source_tag=source_tag,
    )


def backup_and_test(cce_file, test_ctx, modify_func, source_tag=""):
    """Backup, modify, test, and restore a CCE file. Returns (error_exists, stderr_output, original_lines)."""
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)

    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()
    cce_lines = comment_special_lines(cce_lines)
    modified_lines = modify_func(cce_lines)

    try:
        write_file(cce_file, modified_lines)
        returncode, stderr_output = run_test(
            test_ctx.test_cmd, test_ctx.run_dir, show_output=True, source_tag=source_tag
        )
        error_exists = has_error(returncode, stderr_output, test_ctx.use_pypto_test_framework)
    finally:
        write_file(cce_file, original_lines)
        if os.path.exists(backup_file):
            os.remove(backup_file)

    return error_exists, stderr_output, original_lines


# ============================================================================
# Parallel Compile Error Handling
# ============================================================================


def fix_parallel_compile_config():
    installed_config = _find_installed_tile_fwk_config()
    if not installed_config:
        logger.info("警告：未找到已安装的 tile_fwk_config.json")
        return False

    with open(installed_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if config.get('parallel_compile') == 1:
        logger.info("parallel_compile 已为 1，无需修改")
        return True

    backup_path = installed_config + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy(installed_config, backup_path)

    config['parallel_compile'] = 1
    with open(installed_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    logger.info("已修改已安装的 tile_fwk_config.json: parallel_compile = 1")
    return True


def handle_parallel_compile_error(output):
    if 'ld.lld: error: undefined' not in output:
        return False

    logger.info("")
    logger.info("--- 检测到并行编译错误: ld.lld: error: undefined ---")

    installed_config = _find_installed_tile_fwk_config()
    if installed_config:
        with open(installed_config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if config.get('parallel_compile') == 1:
            logger.info("parallel_compile 已为 1，但错误仍然存在，无法自动修复")
            logger.info("停止执行，请手动排查其他原因")
            return False

    logger.info("自动修复：直接修改已安装的 tile_fwk_config.json，parallel_compile = 1")

    if not fix_parallel_compile_config():
        logger.info("无法修复，停止执行")
        return False

    logger.info("✓ 配置已修复（已安装目录直接修改，无需重新编译）")
    logger.info("需要重新执行定位流程")
    return True


# ============================================================================
# Step 2 Helpers (Exclude Machine Framework Issues)
# ============================================================================


def _find_callsubfunctask_range(lines):
    """Find the range of lines containing CallSubFuncTask."""
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
        return False
    comment_lines_by_range(lines, start, end)
    write_file(file_path, lines)
    logger.info("成功注释 CallSubFuncTask 部分（行 %d-%d）", start + 1, end + 1)
    return True


# ============================================================================
# Step 2: Exclude Machine Framework Scheduling Issues
# ============================================================================


def _find_installed_aicore_entry():
    """Locate aicore_entry.h in the installed pypto package. Returns path or None."""
    _log_minor("2.1", "定位已安装 pypto 中的 aicore_entry.h")
    result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
    if result.returncode != 0:
        _log_minor_result("2.1", "✗ pip show pypto 执行失败")
        _log_major_result("2", "✗ 步骤 2.1 定位 aicore_entry.h 失败：pip show pypto 执行失败")
        return None
    location = None
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            location = line.split(':', 1)[1].strip()
            break
    if not location:
        _log_minor_result("2.1", "✗ 无法解析 pypto 安装位置")
        _log_major_result("2", "✗ 步骤 2.1 定位 aicore_entry.h 失败：无法解析 pypto 安装位置")
        return None
    for root, _, files in os.walk(location):
        if 'aicore_entry.h' in files:
            installed_path = os.path.join(root, 'aicore_entry.h')
            _log_minor_result("2.1", "✓ %s", installed_path)
            return installed_path
    _log_minor_result("2.1", "✗ 安装路径下未找到 aicore_entry.h")
    _log_major_result("2", "✗ 步骤 2.1 定位 aicore_entry.h 失败：安装路径下未找到 aicore_entry.h")
    return None


def _get_pypto_install_location():
    """Get the currently installed pypto Location via pip show. Returns path or None."""
    result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            return line.split(':', 1)[1].strip()
    return None


def _remove_pypto_from_target(target_dir):
    """Remove pypto directory and pypto-*.dist-info from target_dir."""
    for name in os.listdir(target_dir):
        if name == 'pypto' or (name.startswith('pypto-') and name.endswith('.dist-info')):
            path = os.path.join(target_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)


def _find_installed_tile_fwk_config():
    result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    location = None
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            location = line.split(':', 1)[1].strip()
            break
    if not location:
        return None
    for root, _, files in os.walk(location):
        if 'tile_fwk_config.json' in files:
            return os.path.join(root, 'tile_fwk_config.json')
    return None


def exclude_machine_framework_issues(test_ctx):
    """Returns True if the issue is in kernel code (continue), False if in framework (stop)."""
    _log_major("2", "排除 machine 框架调度问题 — 判断 aicore error 源于 kernel 还是框架")

    installed_path = _find_installed_aicore_entry()
    if not installed_path:
        return False

    _log_minor("2.2", "注释 CallSubFuncTask — 修改安装路径下的 aicore_entry.h")
    lines = read_file(installed_path)
    start, end = _find_callsubfunctask_range(lines)
    if start is None or end is None:
        _log_minor_result("2.2", "✗ 未找到 CallSubFuncTask")
        _log_major_result("2", "✗ 步骤 2.2 注释 CallSubFuncTask 失败：未找到 CallSubFuncTask")
        return False
    already_commented = all(not lines[i].strip() or lines[i].strip().startswith('//') for i in range(start, end + 1))
    if already_commented:
        logger.info("CallSubFuncTask 已为注释状态，跳过修改")
        did_backup = False
    else:
        shutil.copy(installed_path, installed_path + ".step2_bak")
        did_backup = True
        _comment_callsubfunctask(installed_path)
    if did_backup:
        _log_minor_result("2.2", "✓ 已注释 CallSubFuncTask")
    else:
        _log_minor_result("2.2", "CallSubFuncTask 已为注释状态，跳过修改")

    _log_minor("2.3", "运行测试验证 — 判断 aicore error 是否消失")
    returncode, stderr_output = run_test(
        test_ctx.test_cmd, test_ctx.run_dir, show_output=True, source_tag="[步骤2-测试用例输出]"
    )
    has_aicore_error = has_error(returncode, stderr_output, test_ctx.use_pypto_test_framework)
    _log_minor_result("2.3", "aicore error %s", "存在" if has_aicore_error else "已消失")

    if did_backup:
        _log_minor("2.4", "恢复 aicore_entry.h — 从备份恢复安装路径下的原始文件")
        shutil.copy(installed_path + ".step2_bak", installed_path)
        os.remove(installed_path + ".step2_bak")
        _log_minor_result("2.4", "✓ aicore_entry.h 已恢复")

    if not has_aicore_error:
        _log_major_result("2", "✓ 注释后无 aicore error → 问题在 kernel 代码中，继续排查")
    else:
        _log_major_result("2", "✗ 注释后仍有 aicore error → 问题在 machine 框架调度")
    return not has_aicore_error


# ============================================================================
# Step 3.1: Enable Trace Logs
# ============================================================================


def _modify_device_switch_config(pypto_path):
    """Modify device_switch.h to enable compile verbose log. Returns True if modified."""
    switch_path = locate_file(pypto_path, _KNOWN_LOCATIONS['device_switch.h'], 'device_switch.h')

    def _switch_has_define(lines):
        return any('#define ENABLE_COMPILE_VERBOSE_LOG' in line for line in lines)

    def _switch_already_1(lines):
        return any('#define ENABLE_COMPILE_VERBOSE_LOG 1' in line for line in lines)

    def _switch_set_1(lines):
        for i, line in enumerate(lines):
            if '#define ENABLE_COMPILE_VERBOSE_LOG' in line:
                lines[i] = '#define ENABLE_COMPILE_VERBOSE_LOG 1\n'
                return

    if not switch_path:
        logger.info("警告：未找到 device_switch.h")
        return False
    if not _switch_has_define(read_file(switch_path)):
        logger.info("警告：device_switch.h 中未找到 ENABLE_COMPILE_VERBOSE_LOG 定义，跳过此文件")
        return False
    return _checked_modify_text_file(
        _ModifyConfig(switch_path, ".backup", _switch_already_1, _switch_set_1, "device_switch.h")
    )


def _trace_already_set(config):
    if 'global' in config and 'codegen' in config['global']:
        return (
            config['global']['codegen'].get('fixed_output_path') is True
            and config['global']['codegen'].get('force_overwrite') is False
        )
    return config.get('fixed_output_path') is True and config.get('force_overwrite') is False


def _trace_set(config):
    if 'global' in config and 'codegen' in config['global']:
        config['global']['codegen']['fixed_output_path'] = True
        config['global']['codegen']['force_overwrite'] = False
    else:
        config['fixed_output_path'] = True
        config['force_overwrite'] = False


def enable_trace_logs(pypto_path):
    """Modify device_switch.h (needs rebuild). tile_fwk_config is handled later.

    Returns (any_modified, needs_rebuild).
    """
    _log_minor("3.1", "启用追踪日志 — 修改 device_switch.h，使 aicore 内核启动/完成事件被记录到设备日志")

    if _modify_device_switch_config(pypto_path):
        _log_minor_result("3.1", "✓ 已修改追踪日志配置，需要重新编译")
        return True, True
    else:
        _log_minor_result("3.1", "追踪日志配置已为目标状态，无需修改")
        return False, False


def _apply_tile_fwk_config():
    """Modify installed tile_fwk_config.json after rebuild (no recompilation)."""
    _log_minor("3.1", "启用追踪日志 — 修改已安装的 tile_fwk_config.json（直接修改，无需重编）")
    installed = _find_installed_tile_fwk_config()
    if installed and _checked_modify_json_file(
        _ModifyConfig(installed, ".backup", _trace_already_set, _trace_set, "tile_fwk_config.json (已安装)")
    ):
        _log_minor_result("3.1", "✓ tile_fwk_config.json 已修改")
    elif not installed:
        _log_minor_result("3.1", "✗ 未找到已安装的 tile_fwk_config.json")


# ============================================================================
# Step 3.2: Rebuild and Install
# ============================================================================


def rebuild_and_install(pypto_path):
    _log_minor("3.2", "重新编译和安装 — 使步骤 3.1 的追踪日志配置生效")
    if not build_and_install_pypto(pypto_path):
        _log_minor_result("3.2", "✗ 编译安装失败")
        return False
    _log_minor_result("3.2", "✓ 编译安装成功")
    return True


# ============================================================================
# Step 3.3: Clean Logs and Run Test
# ============================================================================


def _collect_output_subdirs(output_path):
    """Recursively find all output_* subdirectories under output/."""
    subdirs = []
    for root, dirs, _ in os.walk(output_path):
        subdirs.extend(os.path.join(root, d) for d in dirs if d.startswith('output_'))
    return subdirs


def get_latest_program_json(output_path):
    """Get the latest program.json from the output directory.

    Searches recursively for output_* directories — works for both
    standard output (output/output_xxx/) and Pypto_Test framework
    (output/xx/xx/xx/pass_output/output_xxx/).
    """
    if not os.path.exists(output_path):
        logger.info("警告：output 目录不存在: %s", output_path)
        return None

    subdirs = _collect_output_subdirs(output_path)
    if not subdirs:
        logger.info("警告：未找到以 'output_' 开头的子文件夹")
        return None

    subdirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = subdirs[0]
    logger.info("最新 output 子文件夹: %s", latest_dir)

    json_path = os.path.join(latest_dir, 'program.json')
    if not os.path.exists(json_path):
        logger.info("警告：program.json 不存在: %s", json_path)
        return None
    return json_path


def _clean_plog_output(run_path):
    """Remove all plog_output directories under output/ to avoid stale trace logs."""
    output_dir = os.path.join(run_path, "output")
    if os.path.exists(output_dir):
        for root, _, _ in os.walk(output_dir):
            if os.path.basename(root) == 'plog_output':
                shutil.rmtree(root)


def clean_and_run_test(test_ctx):
    """Step 3.3: Clean logs and run test. Returns program_json_path or None."""
    _log_minor("3.3", "清理日志并运行测试 — 重新运行测试以生成带有 LActStart/LActFinish 事件的设备日志")

    device_log_path = test_ctx.device_log_path
    if test_ctx.use_pypto_test_framework:
        _clean_plog_output(test_ctx.run_dir)
    else:
        if os.path.exists(device_log_path):
            shutil.rmtree(device_log_path)
        os.makedirs(device_log_path, exist_ok=True)

    # Clean kernel_aicore (regenerated by test)
    for pattern_path in Path(test_ctx.run_dir).glob("kernel_aic*"):
        if pattern_path.is_dir():
            shutil.rmtree(pattern_path)
        else:
            pattern_path.unlink()

    # Run test with trace env vars
    logger.info("ASCEND_PROCESS_LOG_PATH=%s", device_log_path)
    logger.info("ASCEND_GLOBAL_LOG_LEVEL=0")
    logger.info("ASCEND_HOST_LOG_FILE_NUM=1000")

    env = os.environ.copy()
    env['ASCEND_PROCESS_LOG_PATH'] = device_log_path
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
    env['ASCEND_HOST_LOG_FILE_NUM'] = '1000'

    returncode, stderr_output = run_command_with_live_output(
        test_ctx.test_cmd,
        cwd=test_ctx.run_dir,
        env=env,
        timeout=DEFAULT_TIMEOUT,
        show_output=True,
        source_tag="[步骤3-测试用例输出]",
    )

    if not has_error(returncode, stderr_output, test_ctx.use_pypto_test_framework):
        _log_minor_result("3.3", "✗ 未检测到 aicore error，停止执行")
        return None

    _log_minor_result("3.3", "✓ 检测到 aicore error，追踪日志已生成")
    return get_latest_program_json(os.path.join(test_ctx.run_dir, "output"))


# ============================================================================
# Step 3.5: Analyze Trace Logs and Locate CCE File
# ============================================================================


def parse_luid(luid_str):
    match = re.search(r'LUid\{(\d+),(\d+),(\d+),(\d+),(\d+)\}', luid_str)
    if match:
        return {
            'deviceTaskId': int(match.group(1)),
            'funcId': int(match.group(2)),
            'rootIndex': int(match.group(3)),
            'opIdx': int(match.group(4)),
            'leafIndex': int(match.group(5)),
        }
    return None


def _parse_event_line(line, event_name):
    """Parse a single trace log line for an event. Returns dict or None."""
    if 'trace' not in line or event_name not in line:
        return None
    luid_match = re.search(r'LUid\{[^}]+\}', line)
    event_match = re.search(rf'{event_name}\{{[^}}]+\}}', line)
    if not (luid_match and event_match):
        return None
    luid = parse_luid(luid_match.group(0))
    core_idx_match = re.search(rf'{event_name}\{{(\d+)\}}', event_match.group(0))
    if not (luid and core_idx_match):
        return None
    return {'luid': luid, 'coreIdx': int(core_idx_match.group(1))}


def find_trace_log_file(test_ctx):
    """Find the device log file containing #trace markers.

    Without Pypto_Test framework: wk/debug/device-xx/device-xx.log
    With Pypto_Test framework: output/xx/xx/xx/plog_output/debug/device-xx/device-xx.log
    """
    if test_ctx.use_pypto_test_framework:
        base_path = os.path.join(test_ctx.run_dir, "output")
        if not os.path.exists(base_path):
            logger.info("错误：output 目录不存在: %s", base_path)
            return None

        plog_dirs = []
        for root, _, _ in os.walk(base_path):
            if os.path.basename(root) == 'plog_output':
                plog_dirs.append(root)

        if not plog_dirs:
            logger.info("错误：在 %s 下未找到 plog_output 目录", base_path)
            return None

        log_files = []
        for plog_dir in plog_dirs:
            debug_dir = os.path.join(plog_dir, "debug")
            if os.path.isdir(debug_dir):
                log_files.extend(Path(debug_dir).rglob("device*.log"))
    else:
        search_path = os.path.join(test_ctx.run_dir, "wk", "debug")
        if not os.path.isdir(search_path):
            logger.info("错误：日志目录不存在: %s", search_path)
            return None
        log_files = list(Path(search_path).rglob("device*.log"))

    logger.info("搜索包含 #trace 的日志文件...")

    for log_file in log_files:
        try:
            content = log_file.read_text(encoding='utf-8')
            if '#trace' not in content:
                continue
            logger.info("找到包含 #trace 的日志文件: %s", log_file)

            if 'LActStart' not in content:
                logger.info("日志文件不包含 LActStart 事件，不适用于该方法定位 aicore error")
                logger.info("  文件: %s", log_file)
                return None

            logger.info("日志文件包含 LActStart 事件，适用于该方法定位")
            return str(log_file)
        except OSError as e:
            logger.warning("处理文件异常: %s, 原因: %s", log_file, e)

    logger.info("错误：未找到包含 #trace 的日志文件")
    return None


def analyze_trace(log_file):
    """Parse trace log to find missing leaf indices (crashed kernels)."""
    lactstart_events = []
    lactfinish_events = []

    logger.info("--- 分析追踪日志 ---")
    logger.info("日志文件: %s", log_file)

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            event = _parse_event_line(line, 'LActStart')
            if event:
                lactstart_events.append(event)
                continue
            event = _parse_event_line(line, 'LActFinish')
            if event:
                lactfinish_events.append(event)

    start_idxs = sorted(set(e['coreIdx'] for e in lactstart_events))
    finish_idxs = sorted(set(e['coreIdx'] for e in lactfinish_events))

    logger.info("LActStart 事件数量: %d", len(lactstart_events))
    logger.info("LActStart coreIdxs: %s", start_idxs)
    logger.info("LActFinish 事件数量: %d", len(lactfinish_events))
    logger.info("LActFinish coreIdxs: %s", finish_idxs)

    missing_core_idxs = [i for i in start_idxs if i not in finish_idxs]
    logger.info("\n缺失的 coreIdxs: %s", missing_core_idxs)

    missing_leaf_indices = []
    for event in lactstart_events:
        if event['coreIdx'] in missing_core_idxs:
            leaf = event['luid']['leafIndex']
            if leaf not in missing_leaf_indices:
                missing_leaf_indices.append(leaf)

    logger.info("对应的 leafIndices: %s", missing_leaf_indices)

    if not missing_leaf_indices:
        logger.info("\n警告: 没有发现缺失的 leaf index")
        logger.info("无法确定问题 kernel，停止执行后续步骤")
        return []

    return missing_leaf_indices


def find_cce_file(kernel_aicore_dir, leaf_index):
    """Locate the CCE source file for a given leaf index.

    Deterministic approach:
    1. Search sub_func_*_call_*.h for 'case {leaf_index}:' → extract function name.
    2. Grep function name across kernel_aicore_dir → 2 files found:
       sub_func_xx_call_0.h + the actual CCE .cpp file.
    3. Return the .cpp file.
    """
    logger.info("定位问题 CCE 文件 (leafIndex: %d)", leaf_index)

    record_files = list(Path(kernel_aicore_dir).glob("**/sub_func_*_call_*.h"))
    logger.info("找到 %d 个 Record 文件", len(record_files))

    for record_file in record_files:
        content = record_file.read_text(encoding='utf-8')
        if f'case {leaf_index}:' not in content:
            continue

        logger.info("\n在文件中找到 leafIndex %d: %s", leaf_index, record_file)

        func_match = re.search(rf'case {leaf_index}:\s*\{{\s*(\w+)\(', content)
        func_name = func_match.group(1) if func_match else None
        if not func_name:
            logger.info("警告：无法从 Record 文件中提取函数名")
            continue
        logger.info("Function name: %s", func_name)

        # Grep func_name in file contents across kernel_aicore_dir
        result = subprocess.run(
            ["/usr/bin/grep", "-rl", func_name, kernel_aicore_dir], capture_output=True, text=True, timeout=30
        )
        matched_files = [Path(p) for p in result.stdout.strip().split('\n') if p]
        logger.info("grep 函数名找到 %d 个匹配文件", len(matched_files))
        for mf in matched_files:
            logger.info("  %s", mf)

        cce_files = [f for f in matched_files if f.suffix == '.cpp']
        if cce_files:
            logger.info("\n找到 CCE 文件: %s", cce_files[0])
            return str(cce_files[0])

        logger.info("警告：未在匹配文件中找到 CCE 源文件 (.cpp)")
        break

    logger.info("\n错误：无法在任何 Record 文件中找到 leafIndex %d", leaf_index)
    return None


def find_all_cce_files(kernel_aicore_dir, missing_leaf_indices):
    return [f for idx in missing_leaf_indices if (f := find_cce_file(kernel_aicore_dir, idx))]


@dataclasses.dataclass(frozen=True)
class _TestContext:
    """Immutable test environment configuration shared across diagnostic steps."""

    test_cmd: str
    run_dir: str
    device_log_path: str = './wk'
    use_pypto_test_framework: bool = False


@dataclasses.dataclass
class _TestParams:
    cce_file: str
    test_ctx: _TestContext
    error_in_t: bool = False


def _test_with_comment_strategy(params, source_tag=""):
    """Common pattern: comment lines strategy, run test, check parallel compile.

    Returns (error_exists, restart_indicator).
    """

    def _modify(cce_lines):
        commentable = get_commentable_lines(cce_lines, error_in_t=params.error_in_t)
        if not commentable:
            return None
        return comment_lines_by_indices(cce_lines.copy(), commentable)

    error_exists, stderr_output, original_lines = backup_and_test(
        params.cce_file, params.test_ctx, _modify, source_tag=source_tag
    )

    if original_lines is None:
        return False, None

    if handle_parallel_compile_error(stderr_output):
        return False, RESTART

    return error_exists, None


def test_cce_file(cce_file, test_ctx, source_tag=""):
    """Test if a CCE file is the problem file. Returns True, False, or RESTART."""
    logger.info("\n测试 CCE 文件: %s", cce_file)

    error_exists, restart = _test_with_comment_strategy(
        _TestParams(cce_file, test_ctx, error_in_t=False), source_tag=source_tag
    )
    if restart == RESTART:
        return RESTART

    if error_exists:
        logger.info("结果: 注释所有行后仍有 error，此文件可能不是问题文件")
        return False
    else:
        logger.info("结果: 注释所有行后运行成功，此文件是问题文件")
        return True


def _find_kernel_aicore_dir(run_dir):
    """Find the kernel_aicore directory under run_dir. Returns path or None."""
    for root, dirs, _ in os.walk(run_dir):
        if 'kernel_aicore' in dirs:
            return os.path.join(root, 'kernel_aicore')
    return None


def _confirm_cce_files(problem_cce_files, test_ctx, source_tag):
    """Test each candidate CCE file to confirm the root cause. Returns cce_file, None, or RESTART."""
    _log_minor("3.6", "测试验证 CCE 文件 — 通过注释代码并重测，逐个验证 CCE 文件确认根因")

    for cce_file in problem_cce_files:
        result = test_cce_file(cce_file, test_ctx, source_tag=source_tag)
        if result == RESTART:
            return RESTART
        if result is True:
            _log_minor_result("3.6", "✓ 已确认问题 CCE 文件: %s", cce_file)
            return cce_file

    logger.info("\n所有 CCE 文件测试结果均为 False（注释后仍报错），说明均非问题文件")
    _log_minor_result("3.6", "✗ 未定位到问题 CCE 文件，场景不适用，停止执行")
    return None


def analyze_and_locate_cce(test_ctx, source_tag=""):
    """Step 6: Analyze trace logs and locate the CCE file.

    Returns cce_file, None, or RESTART.
    """
    _log_minor("3.5", "分析追踪日志并定位 CCE 文件 — 解析 LActStart/LActFinish 事件找出崩溃 kernel 对应的 CCE 源文件")

    log_file = find_trace_log_file(test_ctx)
    if not log_file:
        _log_minor_result("3.5", "✗ 未找到包含 #trace 的日志文件")
        return None

    missing_leaf_indices = analyze_trace(log_file)
    if not missing_leaf_indices:
        logger.info("\n结果：没有发现缺失的 leaf index")
        _log_minor_result("3.5", "✗ 没有发现缺失的 leaf index")
        return None

    kernel_aicore_path = _find_kernel_aicore_dir(test_ctx.run_dir)
    if not kernel_aicore_path or not os.path.exists(kernel_aicore_path):
        logger.info("错误：未找到 kernel_aicore 目录")
        _log_minor_result("3.5", "✗ 未找到 kernel_aicore 目录")
        return None

    logger.info("\nkernel_aicore 目录: %s", kernel_aicore_path)

    problem_cce_files = find_all_cce_files(kernel_aicore_path, missing_leaf_indices)
    if not problem_cce_files:
        logger.info("\n未找到问题 CCE 文件")
        _log_minor_result("3.5", "✗ 未找到问题 CCE 文件")
        return None

    logger.info("\n找到 %d 个问题 CCE 文件:", len(problem_cce_files))
    for i, f in enumerate(problem_cce_files, 1):
        logger.info("  %d. %s", i, f)
    _log_minor_result("3.5", "✓ 找到 %d 个候选 CCE 文件", len(problem_cce_files))

    return _confirm_cce_files(problem_cce_files, test_ctx, source_tag)


# ============================================================================
# Step 4: Binary Search to Locate Problem Line
# ============================================================================


def determine_error_scope(cce_file, test_ctx, source_tag=""):
    """Check if the error is in T operations. Returns True, False, or RESTART."""
    _log_minor("4.1", "确定错误范围 — 判断 aicore error 是否位于 T 操作中")

    error_exists, restart = _test_with_comment_strategy(
        _TestParams(cce_file, test_ctx, error_in_t=True), source_tag=source_tag
    )
    if restart == RESTART:
        return RESTART

    if error_exists:
        _log_minor_result("4.1", "问题不在 T 操作行")
        return False
    else:
        _log_minor_result("4.1", "✓ 问题在 T 操作行")
        return True


def get_commentable_range(cce_file, error_in_t):
    """Get the initial range for binary search.

    Returns (left, right, commentable_lines) or (None, None, None).
    """
    _log_minor("4.2", "获取可注释行范围 — 确定 CCE 文件中可安全注释的代码行索引，作为二分搜索空间")
    logger.info("获取可注释行范围...")
    logger.info("error_in_t: %s", error_in_t)

    cce_lines = comment_special_lines(read_file(cce_file))
    commentable = get_commentable_lines(cce_lines, error_in_t)

    n = len(commentable)
    logger.info("可注释的行数: %d", n)
    if n <= 0:
        logger.info("错误：没有可注释的行")
        _log_minor_result("4.2", "✗ 没有可注释的行")
        return None, None, None

    logger.info("初始范围: left=%d, right=%d", 0, n - 1)
    _log_minor_result("4.2", "✓ 搜索范围 [0, %d]，共 %d 个可注释行", n - 1, n)
    return 0, n - 1, commentable


@dataclasses.dataclass
class _SearchConfig:
    """State for binary search iteration."""

    cce_file: str
    test_ctx: _TestContext
    left: int
    right: int
    commentable_lines: list


def binary_search_iteration(config, source_tag=""):
    """Execute one iteration of binary search.

    Returns (new_left, new_right, problem_line or None).
    """
    logger.info("\n二分查找迭代: left=%d, right=%d", config.left, config.right)

    backup_file = config.cce_file + ".bak"
    shutil.copy(config.cce_file, backup_file)
    cce_lines = read_file(config.cce_file)
    original_lines = cce_lines.copy()

    mid = (config.left + config.right) // 2
    logger.info("mid = (left + right) // 2 = (%d + %d) // 2 = %d", config.left, config.right, mid)

    # Build test state: comment everything, then uncomment [0..mid]
    current = comment_lines_by_indices(cce_lines.copy(), config.commentable_lines)
    current = uncomment_lines_by_indices(current, config.commentable_lines[:mid + 1])

    write_file(config.cce_file, current)
    returncode, stderr_output = run_test(
        config.test_ctx.test_cmd, config.test_ctx.run_dir, show_output=True, source_tag=source_tag
    )
    error_exists = has_error(returncode, stderr_output, config.test_ctx.use_pypto_test_framework)

    write_file(config.cce_file, original_lines)
    os.remove(backup_file)

    if error_exists:
        logger.info("结果: 运行失败（有 error），问题在 [%d, %d] 中", config.left, mid)
        new_left, new_right = config.left, mid
    else:
        logger.info("结果: 运行成功（无 error），问题在 [%d, %d] 中", mid + 1, config.right)
        new_left, new_right = mid + 1, config.right

    logger.info("下一轮: left=%d, right=%d", new_left, new_right)

    if new_left == new_right:
        problem_line = config.commentable_lines[new_left]
        logger.info("找到问题代码行: %d", problem_line)
        return new_left, new_right, problem_line

    return new_left, new_right, None


def binary_search_problem_line(cce_file, test_ctx, source_tag=""):
    """Binary search to locate the problem code line.

    Returns problem_line number, None, or RESTART.
    """
    _log_major("4", "二分查找定位问题代码行 — 通过迭代注释缩小范围，精确定位到导致 aicore error 的具体代码行")

    error_in_t = determine_error_scope(cce_file, test_ctx, source_tag=source_tag)
    if error_in_t == RESTART:
        return RESTART

    left, right, commentable = get_commentable_range(cce_file, error_in_t)
    if left is None or right is None or commentable is None:
        _log_major_result("4", "✗ 无法获取可注释行范围，停止执行")
        return None

    _log_minor("4.3", "二分查找迭代 — 每次注释一半代码行并测试，快速收敛到问题行")

    iteration = 0
    while left <= right:
        iteration += 1
        logger.info("\n--- 迭代 %d ---", iteration)

        new_left, new_right, problem_line = binary_search_iteration(
            _SearchConfig(cce_file, test_ctx, left, right, commentable), source_tag=source_tag
        )

        if problem_line is not None:
            _log_major_result("4", "✓ 二分查找完成，共 %d 次迭代，问题行: %d", iteration, problem_line)
            return problem_line

        left, right = new_left, new_right

    _log_major_result("4", "✗ 二分查找未收敛，无法定位问题行")

    return None


# ============================================================================
# Step 5.1: Map to Frontend Source Code
# ============================================================================


def _read_cce_for_mapping(cce_path):
    """Read CCE file and extract funcHash + line-to-code mapping."""
    lines = read_file(cce_path)

    func_hash = None
    for line in lines:
        match = re.search(r'//\s*funcHash:\s*(\d+)', line)
        if match:
            func_hash = match.group(1)
            break

    if not func_hash:
        return None, None

    return func_hash, {i: line.strip() for i, line in enumerate(lines, start=1)}


def _get_cce_operations(line_code_map):
    """Extract T operations and framework operations from CCE line map."""
    target_ops = {'set_flag', 'wait_flag', 'pipe_barrier', 'SUBKERNEL'}
    cce_op = {}
    for idx, line in line_code_map.items():
        is_t = line.startswith('T') and (('<' in line and '>' in line) or ('(' in line and ')' in line))
        if is_t or any(op in line for op in target_ops):
            cce_op[idx] = line
    return cce_op


def _extract_op_names(op_values):
    """Extract operation type names from CCE operation strings."""
    names = []
    for val in op_values:
        if val.startswith('T') and '<' in val and '>' in val:
            names.append(val.split("<")[0])
        elif val.startswith('T') and '(' in val and ')' in val:
            names.append(val.split("(")[0])
        elif (val.startswith('w') or val.startswith('s')) and '(' in val:
            names.append(val.split("(")[0])
        else:
            names.append(val)
    return names


def _resolve_from_program_json(json_path, func_hash, op_ref):
    """Match a CCE operation to its source location in program.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        program_data = json.load(f)

    func_data = next((f for f in program_data.get('functions', []) if f.get('hash') == func_hash), None)
    if not func_data:
        logger.info("错误：未找到 hash 为 %s 的函数", func_hash)
        return None

    program_ops = func_data.get('operations', [])

    logger.info("")
    logger.info("[统计信息]")
    logger.info("  CCE 文件中操作数: %d 个", len(op_ref.cce_op_lines))
    logger.info("  program.json 中操作数: %d 个", len(program_ops))

    if len(op_ref.cce_op_lines) != len(program_ops):
        return {
            'matched': False,
            'reason': 'CCE 文件与 program.json 操作数不一致',
            'cce_line_code': op_ref.cce_line,
        }

    matched_op = program_ops[op_ref.op_index]
    return {
        'matched': True,
        'cce_line_code': op_ref.cce_line,
        'operation_type': op_ref.op_name,
        'operation_index': op_ref.op_index + 1,
        'opcode': matched_op.get('opcode'),
        'source_file': matched_op.get('file'),
        'source_line': matched_op.get('line'),
    }


def find_source_location(cce_path, json_path, cce_line_number):
    """Map a CCE line number to the frontend source file and line."""
    func_hash, line_code_map = _read_cce_for_mapping(cce_path)
    if not func_hash or not line_code_map:
        logger.info("错误：无法找到 funcHash 或无法解析 CCE 文件")
        return None

    if cce_line_number not in line_code_map:
        logger.info("错误：CCE 文件中没有第 %d 行", cce_line_number)
        return None

    cce_line = line_code_map[cce_line_number]
    cce_op = _get_cce_operations(line_code_map)
    cce_op_lines = list(cce_op.keys())
    cce_op_names = _extract_op_names(list(cce_op.values()))

    if cce_line_number not in cce_op_lines:
        return {
            'matched': False,
            'reason': '该代码为框架自动生成代码，非客户前端编写的代码',
            'cce_line_code': cce_line,
        }

    op_index = cce_op_lines.index(cce_line_number)
    op_name = cce_op_names[op_index]

    return _resolve_from_program_json(json_path, func_hash, _CceOpRef(cce_op_lines, op_index, op_name, cce_line))


def print_source_code_line(file_path, line_number):
    if not file_path or not line_number:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if line_number < 1 or line_number > len(lines):
            return
        start = max(0, line_number - 3)
        end = min(len(lines), line_number + 2)
        logger.info("")
        logger.info("[源代码] %s:%d", file_path, line_number)
        logger.info("-" * 80)
        for i in range(start, end):
            marker = ">>>" if i == line_number - 1 else "   "
            logger.info("%s %4d: %s", marker, i + 1, lines[i].rstrip())
        logger.info("-" * 80)
    except OSError as e:
        logger.error("[ERROR] 无法读取源代码文件: %s", e)


def map_to_source_code(cce_file, program_json_path, problem_line):
    """Step 5.1: Map CCE line to frontend source code."""
    _log_minor("5.1", "CCE 行映射到前端源文件 — 通过 funcHash 和 program.json 查找对应的源码位置")
    logger.info("CCE 文件: %s", cce_file)
    logger.info("问题行号: %d", problem_line)
    logger.info("program.json: %s", program_json_path)

    result = find_source_location(cce_file, program_json_path, problem_line)

    if result is None:
        _log_minor_result("5.1", "✗ 无法映射到源代码")
        return None

    logger.info("")
    logger.info("[CCE 问题代码]")
    logger.info("  %s", result['cce_line_code'])

    if result['matched']:
        logger.info("")
        logger.info("✓ 操作: %s (第 %d 个)", result['operation_type'], result['operation_index'])
        logger.info("✓ 匹配: %s", result['opcode'])
        if result['source_file'] and result['source_line']:
            logger.info("✓ 源代码: %s:%d", result['source_file'], result['source_line'])
            print_source_code_line(result['source_file'], result['source_line'])
        else:
            logger.info("✗ 该代码为框架自动生成代码，无源码映射")
    else:
        logger.info("")
        logger.info("✗ 无法匹配")
        logger.info("  原因: %s", result['reason'])

    _log_minor_result("5.1", "✓ 映射完成，操作: %s", result.get('operation_type', '未知'))
    return result


# ============================================================================
# Step 5.2: Print Final Result
# ============================================================================


def print_final_result(cce_file, problem_line, source_result):
    _log_minor("5.2", "输出最终结果")

    logger.info("[问题定位]")
    logger.info("  CCE 文件: %s", cce_file)
    logger.info("  CCE 问题行号: %d", problem_line)

    cce_lines = read_file(cce_file)
    if problem_line <= len(cce_lines):
        logger.info("  CCE 问题代码: %s", cce_lines[problem_line - 1].strip())

    if source_result and source_result.get('matched'):
        src_file = source_result.get('source_file')
        src_line = source_result.get('source_line')
        logger.info("\n[源代码映射]")
        logger.info("  源文件: %s", src_file)
        if src_line is not None:
            logger.info("  源行号: %d", src_line)
        if src_file and src_line is not None:
            src_lines = read_file(src_file)
            if src_line <= len(src_lines):
                logger.info("  源代码: %s", src_lines[src_line - 1].strip())

    _log_minor_result("5.2", "✓ 最终结果已输出")


# ============================================================================
# Restore Original State
# ============================================================================


def restore_original_state(pypto_path):
    """Restore all modified files and rebuild pypto to initial state."""
    _log_major("恢复", "恢复初始状态")

    # Restore installed tile_fwk_config.json (if backup exists)
    installed_config = _find_installed_tile_fwk_config()
    if installed_config:
        _try_restore(installed_config, ".backup", "tile_fwk_config.json (已安装)")

    # Restore source device_switch.h from backup
    switch_path = locate_file(pypto_path, _KNOWN_LOCATIONS['device_switch.h'], 'device_switch.h') or ""
    source_modified = _try_restore(switch_path, ".backup", "device_switch.h")

    if not source_modified:
        logger.info("源代码未修改，跳过重新编译")
        _log_major_result("恢复", "✓ 所有文件已恢复")
        return True

    # Rebuild to restore compiled state
    logger.info("[重新编译和安装]")
    if not build_and_install_pypto(pypto_path):
        logger.info("  警告：未能恢复到初始编译状态")
        _log_major_result("恢复", "✗ 未能恢复到初始编译状态")
        return False

    logger.info("  ✓ 已恢复到初始编译状态")
    logger.info("")
    logger.info("✓ 所有文件已恢复，pypto 已重新编译安装")
    _log_major_result("恢复", "✓ 所有文件已恢复，pypto 已重新编译安装")
    return True


# ============================================================================
# Main Entry Point
# ============================================================================


def _setup_trace_and_rebuild(args, pypto_path):
    """Steps 3.1-3.2: Enable trace logs and rebuild if needed.

    Returns trace_modified (bool), or None if compilation failed.
    """
    _log_major("3", "定位问题 CCE 文件 — 通过追踪日志精确定位出问题的 CCE 源文件")
    trace_modified, needs_rebuild = enable_trace_logs(pypto_path)

    if needs_rebuild and not args.skip_rebuild:
        if not rebuild_and_install(pypto_path):
            _log_major_result("3", "✗ 步骤 3.2 编译安装失败")
            return None
    else:
        if not needs_rebuild:
            logger.info("\n步骤 3.1 未修改任何文件，跳过步骤 3.2: 重新编译和安装")
        else:
            logger.info("\n跳过步骤 3.2: 重新编译和安装")

    # Apply tile_fwk_config AFTER rebuild (modify installed copy, no recompilation)
    _apply_tile_fwk_config()

    return trace_modified


def _map_and_report(result):
    """Step 5: Map CCE problem line to frontend source code and print result."""
    cce_file, problem_line, program_json_path = result
    _log_major("5", "问题代码行映射到前端代码 — 将 CCE 问题行映射回前端源文件，便于开发者定位修复")
    source_result = map_to_source_code(cce_file, program_json_path, problem_line)
    print_final_result(cce_file, problem_line, source_result)
    _log_major_result("5", "✓ 定位完成")


def _run_initial_test(test_ctx):
    """Step 1: Run initial test to check for aicore error. Returns True if error found."""
    _log_major("1", "初始测试 — 复现问题，检查是否为该脚本可适用的 aicore error 场景")

    env = os.environ.copy()
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '0'

    returncode, stderr_output = run_command_with_live_output(
        test_ctx.test_cmd,
        cwd=test_ctx.run_dir,
        env=env,
        timeout=DEFAULT_TIMEOUT,
        show_output=True,
        source_tag="[步骤1-测试用例输出]",
    )

    if not has_error(returncode, stderr_output, test_ctx.use_pypto_test_framework):
        _log_major_result("1", "✗ 未检测到 aicore error，场景不适用，停止执行")
        return False

    _log_major_result("1", "✓ 有 aicore error，该场景可检测，继续执行")
    return True


def _parse_args():
    parser = argparse.ArgumentParser(
        description='AICore Error 一键定位脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pypto-path', default='./', help='pypto 项目根目录路径（默认: ./）')
    parser.add_argument('--run-path', default='./', help='运行测试的目录路径（默认: ./）')
    parser.add_argument('--test-cmd', default=None, help='触发 aicore error 的测试命令')
    parser.add_argument('--device-log-path', default='./wk', help='device log 落盘路径（默认: ./wk）')
    parser.add_argument('--skip-machine-check', action='store_true', help='跳过 machine 框架调度问题检查')
    parser.add_argument('--skip-rebuild', action='store_true', help='跳过重新编译（使用当前安装的 pypto）')
    parser.add_argument(
        '--use-pypto-test-framework',
        action='store_true',
        help='使用 Pypto_Test 框架（不检查 returncode，适配不同的输出目录结构）',
    )
    parser.add_argument('--cce-file', default=None, help='直接指定问题 CCE 文件，跳过步骤 1-3，从二分查找开始')
    parser.add_argument(
        '--cce-line',
        type=int,
        default=None,
        help='直接指定 CCE 问题代码行号，与 --cce-file 配合跳过步骤 1-4，直达源码映射',
    )
    return parser.parse_args()


def _init_environment(args):
    """Parse args, set up logging, validate paths. Returns config dict."""
    setup_logging()

    pypto_path = os.path.abspath(args.pypto_path)
    run_path = os.path.abspath(args.run_path)
    device_log_path = os.path.abspath(args.device_log_path)

    for path, label in [(pypto_path, "pypto 路径"), (run_path, "运行目录")]:
        valid, msg = validate_path(path, label)
        if not valid:
            logger.info(msg)
            return None

    logger.info("=" * 80)
    logger.info("AICore Error 一键定位")
    logger.info("=" * 80)
    logger.info("")
    logger.info("pypto_path: %s", pypto_path)
    logger.info("run_path: %s", run_path)
    logger.info("device_log_path: %s", device_log_path)
    logger.info("test_cmd: %s", args.test_cmd or "(未指定)")
    logger.info("")

    return {
        'pypto_path': pypto_path,
        'run_path': run_path,
        'device_log_path': device_log_path,
    }


def _run_diagnostic_loop(test_ctx):
    """Steps 3.3-5: Retry loop for CCE file location and binary search.

    Returns (cce_file, problem_line, program_json_path) or None.
    """
    while True:
        # Step 3.3: Clean logs and run test
        program_json_path = clean_and_run_test(test_ctx)
        if program_json_path is None:
            _log_major_result("3", "✗ 步骤 3.3 未检测到 aicore error 或未找到 program.json")
            return None

        # Step 3.4: Check kernel binary file sizes
        _log_minor("3.4", "检查 kernel 二进制文件大小")
        kernel_aicore_dir = os.path.join(test_ctx.run_dir, "kernel_aicore")
        kernel_files = sorted(Path(kernel_aicore_dir).glob("dy_kernel_*_0.o"))
        if not kernel_files:
            _log_minor_result("3.4", "未找到 dy_kernel_*_0.o 文件")
        else:
            for kf in kernel_files:
                size_mb = kf.stat().st_size / (1024 * 1024)
                logger.info("%s  —  %.2f MB", kf.name, size_mb)
            _log_minor_result("3.4", "✓ 已检查 %d 个 kernel 文件", len(kernel_files))

        # Steps 3.5-3.6: Analyze trace and locate CCE file
        cce_file = analyze_and_locate_cce(test_ctx, source_tag="[步骤3-测试用例输出]")
        if cce_file == RESTART:
            logger.info("\n并行编译配置已修复，重新执行定位流程")
            continue
        if cce_file is None:
            _log_major_result("3", "✗ 步骤 3.5/3.6 未定位到问题 CCE 文件")
            return None

        # Step 4: Binary search to locate problem line
        _log_major_result("3", "✓ 已定位问题 CCE 文件: %s", cce_file)
        problem_line = binary_search_problem_line(cce_file, test_ctx, source_tag="[步骤4-测试用例输出]")
        if problem_line == RESTART:
            logger.info("\n并行编译配置已修复，重新执行定位流程")
            _log_major("3", "定位问题 CCE 文件")
            continue
        if problem_line is None:
            return None

        return cce_file, problem_line, program_json_path


def _check_tile_fwk_config():
    """Ensure tile_fwk_config.json has trace enabled for the installed pypto.

    Modifies the installed config directly (no rebuild needed) if not already set.
    """
    installed_config = _find_installed_tile_fwk_config()
    if not installed_config:
        logger.info("警告：未找到已安装的 tile_fwk_config.json，无法启用 trace")
        return

    if _checked_modify_json_file(
        _ModifyConfig(installed_config, ".backup", _trace_already_set, _trace_set, "tile_fwk_config.json (已安装)")
    ):
        logger.info("tile_fwk_config.json: trace 已启用 ✓（已安装目录直接修改，无需重新编译）")
    else:
        logger.info("tile_fwk_config.json: trace 已为目标状态 ✓")


def _run_quick_mode_a(args, test_ctx):
    """Quick mode A: user provides CCE file, script does binary search + mapping."""
    if not args.test_cmd:
        logger.info("错误：--cce-file 模式需要 --test-cmd（二分查找需要运行测试）")
        return

    _check_tile_fwk_config()

    try:
        # Step 4.0: Verify CCE file is the problem file
        _log_major("4.0", "验证 CCE 文件 — 确认该文件是否为问题文件")
        result = test_cce_file(args.cce_file, test_ctx, source_tag="[步骤4.0-测试用例输出]")
        if result == RESTART:
            logger.info("\n并行编译配置已修复，请重新运行")
            return
        if result is not True:
            _log_major_result("4.0", "✗ 该 CCE 文件不是问题文件，停止执行")
            return
        _log_major_result("4.0", "✓ 已确认 CCE 文件为问题文件")

        # Step 4: Binary search
        _log_major("4", "二分查找定位问题代码行 — 通过迭代注释缩小范围，精确定位到导致 aicore error 的具体代码行")
        problem_line = binary_search_problem_line(args.cce_file, test_ctx, source_tag="[步骤4-测试用例输出]")
        if problem_line == RESTART:
            logger.info("\n并行编译配置已修复，请重新运行")
            return
        if problem_line is None:
            _log_major_result("4", "✗ 未定位到问题代码行")
            return
        _log_major_result("4", "✓ 二分查找完成，问题行: %d", problem_line)

        # Step 5: Map to source code
        program_json_path = get_latest_program_json(os.path.join(test_ctx.run_dir, "output"))
        if not program_json_path:
            logger.info("错误：未找到 program.json，无法进行源码映射")
            return
        _map_and_report((args.cce_file, problem_line, program_json_path))
    finally:
        installed_config = _find_installed_tile_fwk_config()
        if installed_config:
            _try_restore(installed_config, ".backup", "tile_fwk_config.json (已安装)")


def _run_quick_mode_b(args, run_path):
    """Quick mode B: user provides CCE file + line number, script does mapping only."""
    _check_tile_fwk_config()

    try:
        _log_major("5", "问题代码行映射到前端代码 — 将 CCE 问题行映射回前端源文件，便于开发者定位修复")
        program_json_path = get_latest_program_json(os.path.join(run_path, "output"))
        if not program_json_path:
            logger.info("错误：未找到 program.json，无法进行源码映射")
            return
        _map_and_report((args.cce_file, args.cce_line, program_json_path))
    finally:
        installed_config = _find_installed_tile_fwk_config()
        if installed_config:
            _try_restore(installed_config, ".backup", "tile_fwk_config.json (已安装)")


def main():
    args = _parse_args()
    env = _init_environment(args)
    if env is None:
        return
    pypto_path = env['pypto_path']
    run_path = env['run_path']
    device_log_path = env['device_log_path']
    test_cmd = args.test_cmd
    use_pypto_test_framework = args.use_pypto_test_framework

    test_ctx = _TestContext(
        test_cmd=test_cmd or '',
        run_dir=run_path,
        device_log_path=device_log_path,
        use_pypto_test_framework=use_pypto_test_framework,
    )

    # Quick mode B: user provides both CCE file and line → mapping only
    if args.cce_file and args.cce_line is not None:
        _run_quick_mode_b(args, run_path)
        return

    # Quick mode A: user provides CCE file only → binary search + mapping
    if args.cce_file:
        _run_quick_mode_a(args, test_ctx)
        return

    # Full flow
    if not test_cmd:
        logger.info("错误：完整流程需要 --test-cmd")
        return

    # Step 1: Initial test
    if not _run_initial_test(test_ctx):
        return

    state_modified = False
    try:
        # Step 2: Exclude machine framework scheduling issues
        if not args.skip_machine_check:
            if not exclude_machine_framework_issues(test_ctx):
                return
        else:
            logger.info("跳过步骤 2: 排除 machine 框架调度问题")

        # Steps 3.1-3.2: Enable trace and rebuild
        trace_modified = _setup_trace_and_rebuild(args, pypto_path)
        if trace_modified is None:
            return
        if trace_modified:
            state_modified = True

        # Steps 3.3-5: Diagnostic retry loop
        result = _run_diagnostic_loop(test_ctx)
        if result is None:
            return

        # Step 5: Map to source code
        _map_and_report(result)

    except KeyboardInterrupt:
        logger.info("\n\n用户中断执行")
        raise
    except Exception as e:
        logger.info("\n\n脚本执行异常: %s", str(e))
        raise
    finally:
        if state_modified:
            logger.info("\n开始恢复初始状态...")
            restore_original_state(pypto_path)


if __name__ == '__main__':
    main()
