#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
步骤 3: 定位问题 CCE 文件

将 SKILL.md 中原来的 3.1-3.7 七个子步骤合并为单个脚本调用:
  3.1 - 启用追踪日志（修改配置文件）
  3.2 - 重新编译和安装（条件执行）
  3.3 - 清理日志并运行测试
  3.4 - 检查 kernel 二进制文件大小
  3.5 - 获取 program.json 路径
  3.6 - 分析追踪日志并定位 CCE 文件
  3.7 - 测试验证 CCE 文件

用法:
    python3 locate_problem_cce.py --pypto-path <pypto_path> --test-cmd "<test_cmd>"
        --run-path <run_path> --device-log-path <device_log_path>

返回值:
    Exit 0: 成功定位问题 CCE 文件（输出 CCE 文件路径和 program.json 路径）
    Exit 1: 未找到问题 CCE 文件 / 场景不适用 / 构建失败
    Exit 2: 并行编译错误无法自动修复（parallel_compile 已为 1 但仍报错），停止执行

设计参考: tools/scripts/locate_aicore_error.py 中的 _run_diagnostic_loop
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    _PIP_CMD,
    DEFAULT_TIMEOUT,
    KNOWN_LOCATIONS,
    comment_lines_by_indices,
    comment_special_lines,
    find_installed_tile_fwk_config,
    get_commentable_lines,
    locate_file,
    read_file,
    setup_logging,
    validate_path,
    write_file,
)

setup_logging()
logger = logging.getLogger(__name__)

RESTART = "RESTART"


def _find_kernel_aicore_dir(run_dir):
    for root, dirs, _ in os.walk(run_dir):
        if 'kernel_aicore' in dirs:
            return os.path.join(root, 'kernel_aicore')
    return None


def check_test_result(returncode, output, use_pypto_test_framework=False):
    if not use_pypto_test_framework and returncode == 0:
        return False, False
    output_lower = output.lower()
    has_aicore = "aicore error" in output_lower
    has_parallel_error = "ld.lld: error: undefined" in output_lower
    return has_aicore, has_parallel_error


def _print_error_lines(stderr_output):
    error_lines = [line for line in stderr_output.split('\n') if 'error' in line.lower()]
    if error_lines:
        for line in error_lines[:10]:
            logger.info("  %s", line)


def run_test_cmd(test_cmd, run_dir):
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
    return result.returncode, result.stdout + result.stderr


def backup_and_test_cce(cce_file, test_cmd, run_dir, modify_func, use_pypto_test_framework=False):
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)
    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()
    cce_lines = comment_special_lines(cce_lines)
    modified_lines = modify_func(cce_lines)
    if modified_lines is None:
        shutil.copy(backup_file, cce_file)
        os.remove(backup_file)
        return False, "", original_lines
    try:
        write_file(cce_file, modified_lines)
        returncode, output = run_test_cmd(test_cmd, run_dir)
        has_aicore, has_parallel = check_test_result(returncode, output, use_pypto_test_framework)
        if has_parallel:
            return False, output, original_lines, RESTART
        return has_aicore, output, original_lines
    finally:
        write_file(cce_file, original_lines)
        if os.path.exists(backup_file):
            os.remove(backup_file)


def _modify_tile_fwk_config():
    config_path = find_installed_tile_fwk_config()
    if not config_path:
        logger.info("警告：未找到已安装的 tile_fwk_config.json")
        return False, config_path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if 'global' in config and 'codegen' in config['global']:
        already_set = (
            config['global']['codegen'].get('fixed_output_path') is True
            and config['global']['codegen'].get('force_overwrite') is False
        )
    else:
        already_set = config.get('fixed_output_path') is True and config.get('force_overwrite') is False
    if already_set:
        logger.info("tile_fwk_config.json: 已为目标状态，无需修改")
        return False, config_path
    backup_path = config_path + ".backup"
    shutil.copy(config_path, backup_path)
    if 'global' in config and 'codegen' in config['global']:
        config['global']['codegen']['fixed_output_path'] = True
        config['global']['codegen']['force_overwrite'] = False
    else:
        config['fixed_output_path'] = True
        config['force_overwrite'] = False
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    logger.info("tile_fwk_config.json (已安装): 已备份并修改")
    return True, config_path


def _modify_device_switch(pypto_path):
    switch_path = locate_file(pypto_path, KNOWN_LOCATIONS['device_switch.h'], 'device_switch.h')
    if not switch_path:
        logger.info("警告：未找到 device_switch.h")
        return False
    lines = read_file(switch_path)
    if not any('#define ENABLE_COMPILE_VERBOSE_LOG' in line for line in lines):
        logger.info("警告：device_switch.h 中未找到 ENABLE_COMPILE_VERBOSE_LOG 定义，跳过")
        return False
    if any('#define ENABLE_COMPILE_VERBOSE_LOG 1' in line for line in lines):
        logger.info("device_switch.h: 已为目标状态，无需修改")
        return False
    backup_path = switch_path + ".backup"
    shutil.copy(switch_path, backup_path)
    for i, line in enumerate(lines):
        if '#define ENABLE_COMPILE_VERBOSE_LOG' in line:
            lines[i] = '#define ENABLE_COMPILE_VERBOSE_LOG 1\n'
            break
    write_file(switch_path, lines)
    logger.info("device_switch.h: 已备份并修改")
    return True


def enable_trace_logs(pypto_path):
    logger.info("")
    logger.info("---步骤 3.1: 启用追踪日志")
    logger.info("")

    modified_switch = _modify_device_switch(pypto_path)

    if modified_switch:
        logger.info("")
        logger.info("---步骤 3.1 结果: 已修改追踪日志配置，需要重新编译")
    else:
        logger.info("")
        logger.info("---步骤 3.1 结果: 追踪日志配置已为目标状态，无需修改")
    return modified_switch, modified_switch


def _modify_tile_fwk_config():
    """Modify installed tile_fwk_config.json to enable trace."""
    logger.info("")
    logger.info("---步骤 3.1b: 修改已安装的 tile_fwk_config.json（直接修改，无需重编）")
    logger.info("")
    config_path = find_installed_tile_fwk_config()
    if not config_path:
        logger.info("警告：未找到已安装的 tile_fwk_config.json")
        return False
    backup_path = config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy(config_path, backup_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if 'global' in config and 'codegen' in config['global']:
        codegen = config['global']['codegen']
        already_set = codegen.get('fixed_output_path') is True and codegen.get('force_overwrite') is False
        if already_set:
            logger.info("tile_fwk_config.json 已为目标状态，无需修改")
            return False
        codegen['fixed_output_path'] = True
        codegen['force_overwrite'] = False
    else:
        config['fixed_output_path'] = True
        config['force_overwrite'] = False
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    logger.info("已修改 tile_fwk_config.json (已安装)")
    return True


def clean_and_run_test(device_log_path, run_path, test_cmd, use_pypto_test_framework=False):
    logger.info("")
    logger.info("---步骤 3.3: 清理日志并运行测试")
    logger.info("")

    if use_pypto_test_framework:
        output_dir = os.path.join(run_path, "output")
        if os.path.exists(output_dir):
            for root, _, _ in os.walk(output_dir):
                if os.path.basename(root) == 'plog_output':
                    shutil.rmtree(root)
    else:
        if os.path.exists(device_log_path):
            shutil.rmtree(device_log_path)
        os.makedirs(device_log_path, exist_ok=True)

    for pattern_path in Path(run_path).glob("kernel_aic*"):
        if pattern_path.is_dir():
            shutil.rmtree(pattern_path)
        else:
            pattern_path.unlink()

    logger.info("ASCEND_PROCESS_LOG_PATH=%s", device_log_path)
    logger.info("ASCEND_GLOBAL_LOG_LEVEL=0")
    logger.info("ASCEND_HOST_LOG_FILE_NUM=1000")

    env = os.environ.copy()
    env['ASCEND_PROCESS_LOG_PATH'] = device_log_path
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
    env['ASCEND_HOST_LOG_FILE_NUM'] = '1000'

    logger.info("")
    logger.info("[测试用例执行] %s", test_cmd[:100])

    if isinstance(test_cmd, str):
        cmd_list = shlex.split(test_cmd)
    else:
        cmd_list = test_cmd

    result = subprocess.run(
        cmd_list,
        cwd=run_path,
        env=env,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=DEFAULT_TIMEOUT,
        check=False,
    )
    stderr_output = result.stdout + result.stderr
    has_aicore, has_parallel = check_test_result(result.returncode, stderr_output, use_pypto_test_framework)

    if not has_aicore:
        logger.info("")
        logger.info("---步骤 3.3 结果: 未检测到 aicore error，停止执行")
        return None, None

    logger.info("---步骤 3.3 结果: 检测到 aicore error，追踪日志已生成")
    return stderr_output, has_parallel


def check_kernel_files(run_path):
    logger.info("")
    logger.info("---步骤 3.4: 检查 kernel 二进制文件大小")
    kernel_dir = os.path.join(run_path, "kernel_aicore")
    if not os.path.isdir(kernel_dir):
        logger.info("---步骤 3.4 结果: 未找到 kernel_aicore 目录")
        return
    kernel_files = sorted(Path(kernel_dir).glob("dy_kernel_*_0.o"))
    if not kernel_files:
        logger.info("---步骤 3.4 结果: 未找到 dy_kernel_*_0.o 文件")
        return
    for kf in kernel_files:
        size_mb = kf.stat().st_size / (1024 * 1024)
        logger.info("  %s  —  %.2f MB", kf.name, size_mb)
    logger.info("---步骤 3.4 结果: 已检查 %d 个 kernel 文件", len(kernel_files))


def get_latest_program_json(output_path, use_pypto_test_framework=False):
    if not os.path.exists(output_path):
        logger.info("警告：output 目录不存在: %s", output_path)
        return None
    if use_pypto_test_framework:
        subdirs = []
        for root, dirs, _ in os.walk(output_path):
            if os.path.basename(root) == 'pass_output':
                for d in dirs:
                    if d.startswith('output_'):
                        subdirs.append(os.path.join(root, d))
    else:
        subdirs = [
            os.path.join(output_path, d)
            for d in os.listdir(output_path)
            if os.path.isdir(os.path.join(output_path, d)) and d.startswith('output_')
        ]
    if not subdirs:
        logger.info("警告：未找到以 'output_' 开头的子文件夹")
        return None
    subdirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = subdirs[0]
    json_path = os.path.join(latest_dir, 'program.json')
    if not os.path.exists(json_path):
        logger.info("警告：program.json 不存在: %s", json_path)
        return None
    return json_path


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


def find_trace_log_file(run_path, use_pypto_test_framework=False):
    if use_pypto_test_framework:
        base_path = os.path.join(run_path, "output")
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
        search_path = os.path.join(run_path, "wk", "debug")
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
    lactstart_events = []
    lactfinish_events = []
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
    logger.info("")
    logger.info("缺失的 coreIdxs: %s", missing_core_idxs)
    missing_leaf_indices = []
    for event in lactstart_events:
        if event['coreIdx'] in missing_core_idxs:
            leaf = event['luid']['leafIndex']
            if leaf not in missing_leaf_indices:
                missing_leaf_indices.append(leaf)
    logger.info("对应的 leafIndices: %s", missing_leaf_indices)
    if not missing_leaf_indices:
        logger.info("")
        logger.info("警告: 没有发现缺失的 leaf index")
        logger.info("无法确定问题 kernel，停止执行后续步骤")
        return []
    return missing_leaf_indices


def find_cce_file(kernel_aicore_dir, leaf_index):
    logger.info("定位问题 CCE 文件 (leafIndex: %d)", leaf_index)
    record_files = list(Path(kernel_aicore_dir).glob("**/sub_func_*_call_*.h"))
    logger.info("找到 %d 个 Record 文件", len(record_files))
    for record_file in record_files:
        content = record_file.read_text(encoding='utf-8')
        if f'case {leaf_index}:' not in content:
            continue
        logger.info("")
        logger.info("在文件中找到 leafIndex %d: %s", leaf_index, record_file)
        func_match = re.search(rf'case {leaf_index}:\s*\{{\s*(\w+)\(', content)
        func_name = func_match.group(1) if func_match else None
        if not func_name:
            logger.info("警告：无法从 Record 文件中提取函数名")
            continue
        logger.info("Function name: %s", func_name)
        result = subprocess.run(
            ["/usr/bin/grep", "-rl", func_name, kernel_aicore_dir], capture_output=True, text=True, timeout=30
        )
        matched_files = [Path(p) for p in result.stdout.strip().split('\n') if p]
        logger.info("grep 函数名找到 %d 个匹配文件", len(matched_files))
        for mf in matched_files:
            logger.info("  %s", mf)
        cce_files = [f for f in matched_files if f.suffix == '.cpp']
        if cce_files:
            logger.info("找到 CCE 文件: %s", cce_files[0])
            return str(cce_files[0])
        logger.info("警告：未在匹配文件中找到 CCE 源文件 (.cpp)")
        break
    logger.info("")
    logger.info("错误：无法在任何 Record 文件中找到 leafIndex %d", leaf_index)
    return None


def find_all_cce_files(kernel_aicore_dir, missing_leaf_indices):
    return [f for idx in missing_leaf_indices if (f := find_cce_file(kernel_aicore_dir, idx))]


def test_single_cce(cce_file, test_cmd, run_dir):
    logger.info("")
    logger.info("测试 CCE 文件: %s", cce_file)

    def _modify_func(lines):
        commentable = get_commentable_lines(lines, error_in_t=False)
        logger.info("可注释的行数: %d", len(commentable))
        if not commentable:
            return None
        return comment_lines_by_indices(lines.copy(), commentable)

    result = backup_and_test_cce(cce_file, test_cmd, run_dir, _modify_func)
    if len(result) == 4:
        return RESTART
    has_aicore, output, original_lines = result
    if original_lines is None:
        return False
    if has_aicore:
        _print_error_lines(output)
        logger.info("结果: 注释所有行后仍有 error，此文件可能不是问题文件")
        return False
    else:
        logger.info("结果: 注释所有行后运行成功（无 error），此文件可能是问题文件")
        return True


def confirm_cce_files(problem_cce_files, test_cmd, run_dir):
    logger.info("")
    logger.info("---步骤 3.7: 测试验证 CCE 文件")
    for cce_file in problem_cce_files:
        result = test_single_cce(cce_file, test_cmd, run_dir)
        if result == RESTART:
            return RESTART
        if result is True:
            logger.info("")
            logger.info("---步骤 3.7 结果: 已确认问题 CCE 文件: %s", cce_file)
            return cce_file
    logger.info("")
    logger.info("所有 CCE 文件测试结果均为 False（注释后仍报错），说明均非问题文件")
    logger.info("---步骤 3.7 结果: 未定位到问题 CCE 文件，场景不适用")
    return None


def fix_parallel_compile():
    config_path = find_installed_tile_fwk_config()
    if not config_path:
        logger.info("警告：未找到已安装的 tile_fwk_config.json")
        return False
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if config.get('parallel_compile') == 1:
        logger.info("parallel_compile 已为 1，但错误仍然存在，无法自动修复")
        logger.info("停止执行，请手动排查其他原因")
        return False

    backup_path = config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy(config_path, backup_path)

    config['parallel_compile'] = 1
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    logger.info("已修改已安装的 tile_fwk_config.json: parallel_compile = 1（无需重新编译）")
    logger.info("需要重新执行定位流程")
    return True


def _build_and_install_with_lld_handling(pypto_path):
    """Build and install PyPTO, auto-handling parallel compile (lld) errors.

    Returns:
        True  — build + install succeeded
        False — build failed for non-lld reasons
        None  — lld error persists even after fixing parallel_compile → caller exits 2
    """
    for _ in range(2):
        result = subprocess.run(
            ["python3", "build_ci.py", "-f", "python3", "--disable_auto_execute"],
            shell=False,
            capture_output=True,
            text=True,
            cwd=pypto_path,
            timeout=DEFAULT_TIMEOUT,
        )
        build_output = result.stdout + result.stderr
        has_lld = "ld.lld: error: undefined" in build_output

        if result.returncode == 0:
            whl_files = list(Path(pypto_path).glob("build_out/pypto*.whl"))
            if not whl_files:
                logger.info("错误：未找到 pypto whl 文件")
                return False
            target = None
            show_result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
            if show_result.returncode == 0:
                for line in show_result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        target = line.split(':', 1)[1].strip()
                        break
            if target:
                for name in os.listdir(target):
                    if name == 'pypto' or (name.startswith('pypto-') and name.endswith('.dist-info')):
                        path = os.path.join(target, name)
                        if os.path.isdir(path):
                            shutil.rmtree(path)
            install_cmd = [_PIP_CMD, "install", str(whl_files[0]), "--force", "--no-deps"]
            if target:
                install_cmd += ["--target", target]
            install_result = subprocess.run(install_cmd, shell=False, capture_output=True, text=True, timeout=300)
            if install_result.returncode != 0:
                logger.info("安装失败: %s", install_result.stderr[-500:])
                return False
            return True

        if has_lld:
            logger.info("检测到并行编译错误 (ld.lld: error: undefined)")
            if fix_parallel_compile():
                logger.info("已修复并行编译配置，重新编译...")
                continue
            return None

        logger.info("编译失败: %s", result.stderr[-500:])
        return False

    return False


def main():
    parser = argparse.ArgumentParser(description='步骤 3: 定位问题 CCE 文件（合并 3.1-3.7）')
    parser.add_argument('--pypto-path', default=os.getcwd(), help='pypto 项目根目录路径（默认: 当前目录）')
    parser.add_argument('--test-cmd', required=True, help='触发 aicore error 的测试命令')
    parser.add_argument('--run-path', default=os.getcwd(), help='运行测试的目录路径（默认: 当前目录）')
    parser.add_argument('--device-log-path', default='./device_log', help='device log 落盘路径（默认: ./device_log）')
    parser.add_argument('--use-pypto-test-framework', action='store_true', help='使用 Pypto_Test 框架模式')
    args = parser.parse_args()

    pypto_path = os.path.abspath(args.pypto_path)
    run_path = os.path.abspath(args.run_path)
    device_log_path = os.path.abspath(args.device_log_path)

    for path, label in [(pypto_path, "pypto 路径"), (run_path, "运行目录")]:
        valid, msg = validate_path(path, label)
        if not valid:
            logger.info(msg)
            sys.exit(1)

    logger.info("=" * 80)
    logger.info("步骤 3: 定位问题 CCE 文件")
    logger.info("=" * 80)
    logger.info("pypto_path: %s", pypto_path)
    logger.info("run_path: %s", run_path)
    logger.info("device_log_path: %s", device_log_path)
    logger.info("test_cmd: %s", args.test_cmd)
    use_pypto = args.use_pypto_test_framework

    trace_modified, needs_rebuild = enable_trace_logs(pypto_path)

    if needs_rebuild:
        build_result = _build_and_install_with_lld_handling(pypto_path)
        if build_result is None:
            sys.exit(2)
        if not build_result:
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 编译安装失败")
            logger.info("=" * 80)
            sys.exit(1)
    else:
        logger.info("")
        logger.info("步骤 3.1 未修改任何文件，跳过步骤 3.2: 重新编译和安装")

    # Apply tile_fwk_config AFTER rebuild (modify installed copy)
    _modify_tile_fwk_config()

    while True:
        stderr_output, has_parallel = clean_and_run_test(
            device_log_path, run_path, args.test_cmd, use_pypto_test_framework=use_pypto
        )
        if stderr_output is None:
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 未检测到 aicore error，停止执行")
            logger.info("=" * 80)
            sys.exit(1)

        if has_parallel:
            if fix_parallel_compile():
                continue
            sys.exit(2)

        check_kernel_files(run_path)

        logger.info("")
        logger.info("---步骤 3.5: 获取 program.json 路径")
        output_dir = os.path.join(run_path, "output")
        program_json_path = get_latest_program_json(output_dir, use_pypto_test_framework=use_pypto)
        if program_json_path:
            logger.info("program.json: %s", program_json_path)
            logger.info("---步骤 3.5 结果: %s", program_json_path)
        else:
            logger.info("---步骤 3.5 结果: 未找到 program.json")
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 未找到 program.json，停止执行")
            logger.info("=" * 80)
            sys.exit(1)

        logger.info("")
        logger.info("---步骤 3.6: 分析追踪日志并定位 CCE 文件")
        log_file = find_trace_log_file(run_path, use_pypto_test_framework=use_pypto)
        if not log_file:
            logger.info("---步骤 3.6 结果: 未找到包含 #trace 的日志文件")
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 未找到可用的追踪日志，停止执行")
            logger.info("=" * 80)
            sys.exit(1)

        missing_leaf_indices = analyze_trace(log_file)
        if not missing_leaf_indices:
            logger.info("---步骤 3.6 结果: 没有发现缺失的 leaf index")
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 无法定位问题 CCE 文件")
            logger.info("=" * 80)
            sys.exit(1)

        kernel_aicore_dir = _find_kernel_aicore_dir(run_path)
        if not kernel_aicore_dir:
            kernel_aicore_dir = os.path.join(run_path, "kernel_aicore")
        if not os.path.exists(kernel_aicore_dir):
            logger.info("错误：未找到 kernel_aicore 目录")
            logger.info("---步骤 3.6 结果: 未找到 kernel_aicore 目录")
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 未找到 kernel_aicore 目录，停止执行")
            logger.info("=" * 80)
            sys.exit(1)

        logger.info("")
        logger.info("kernel_aicore 目录: %s", kernel_aicore_dir)
        problem_cce_files = find_all_cce_files(kernel_aicore_dir, missing_leaf_indices)

        if not problem_cce_files:
            logger.info("---步骤 3.6 结果: 未找到问题 CCE 文件")
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 未找到问题 CCE 文件")
            logger.info("=" * 80)
            sys.exit(1)

        logger.info("")
        logger.info("找到 %d 个候选 CCE 文件:", len(problem_cce_files))
        for i, f in enumerate(problem_cce_files, 1):
            logger.info("  %d. %s", i, f)
        logger.info("---步骤 3.6 结果: 找到 %d 个候选 CCE 文件", len(problem_cce_files))

        confirmed = confirm_cce_files(problem_cce_files, args.test_cmd, run_path)
        if confirmed == RESTART:
            if fix_parallel_compile():
                continue
            sys.exit(2)
        if confirmed is None:
            logger.info("=" * 80)
            logger.info("步骤 3 结果: 所有候选 CCE 文件均非问题文件，停止执行")
            logger.info("=" * 80)
            sys.exit(1)

        logger.info("")
        logger.info("=" * 80)
        logger.info("步骤 3 结果: 已定位问题 CCE 文件")
        logger.info("=" * 80)
        logger.info("CCE_FILE=%s", confirmed)
        logger.info("PROGRAM_JSON=%s", program_json_path)
        sys.exit(0)


if __name__ == '__main__':
    main()
