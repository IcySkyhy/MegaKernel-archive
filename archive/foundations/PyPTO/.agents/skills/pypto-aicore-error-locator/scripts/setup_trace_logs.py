#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
启用追踪日志 — 修改已安装的 tile_fwk_config.json

设置 fixed_output_path=True, force_overwrite=False，使测试运行时生成 program.json
到固定路径 (run_path/output/)，供后续源码映射使用。

用法:
    python3 setup_trace_logs.py

退出码:
    0: 成功（已修改或已为目标状态）
    1: 未找到已安装的 tile_fwk_config.json
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys

_PIP_CMD = shutil.which("pip3") or shutil.which("pip") or "/usr/bin/pip3"

logger = logging.getLogger(__name__)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


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


def main():
    parser = argparse.ArgumentParser(description="启用追踪日志 — 修改已安装的 tile_fwk_config.json")
    parser.parse_args()

    setup_logging()

    installed_config = _find_installed_tile_fwk_config()
    if not installed_config:
        logger.info("错误：未找到已安装的 tile_fwk_config.json")
        sys.exit(1)

    logger.info("tile_fwk_config.json: %s", installed_config)

    with open(installed_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if _trace_already_set(config):
        logger.info("✓ trace 已为目标状态，无需修改")
        sys.exit(0)

    backup_path = installed_config + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy(installed_config, backup_path)

    _trace_set(config)
    with open(installed_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    logger.info("✓ trace 已启用 (fixed_output_path=True, force_overwrite=False)")
    logger.info("  备份: %s", backup_path)


if __name__ == "__main__":
    main()
