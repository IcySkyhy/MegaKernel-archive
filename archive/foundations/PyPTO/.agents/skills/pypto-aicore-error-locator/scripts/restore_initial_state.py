#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
恢复初始状态

定位完成后，恢复步骤 3 修改的配置文件（tile_fwk_config.json、device_switch.h）
并重新编译安装 pypto，回到原始编译状态。

用法:
    python3 restore_initial_state.py --pypto-path <pypto_path>

设计参考: tools/scripts/locate_aicore_error.py 中的 restore_original_state 函数
"""

import argparse
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    KNOWN_LOCATIONS,
    build_and_install,
    find_installed_tile_fwk_config,
    locate_file,
    setup_logging,
    validate_path,
)

setup_logging()
logger = logging.getLogger(__name__)


def restore_from_backup(base_dir, known_rel_path, filename, label):
    file_path = locate_file(base_dir, known_rel_path, filename)
    if not file_path:
        logger.info("未找到 %s，跳过恢复", filename)
        return False
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        logger.info("%s 的 .backup 备份不存在，跳过恢复", filename)
        return False
    shutil.copy(backup_path, file_path)
    os.remove(backup_path)
    logger.info("已恢复 %s", label)
    return True


def main():
    parser = argparse.ArgumentParser(description='恢复定位过程中修改的配置文件并重新编译安装 pypto')
    parser.add_argument('--pypto-path', default=os.getcwd(), help='pypto 项目根目录路径（默认: 当前目录）')
    args = parser.parse_args()

    pypto_path = os.path.abspath(args.pypto_path)
    valid, msg = validate_path(pypto_path, "pypto 路径")
    if not valid:
        logger.info(msg)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("恢复初始状态")
    logger.info("=" * 80)
    logger.info("pypto_path: %s", pypto_path)
    logger.info("")

    # Restore installed tile_fwk_config.json (if backup exists)
    installed_config = find_installed_tile_fwk_config()
    tile_restored = False
    if installed_config:
        backup_path = installed_config + ".backup"
        if os.path.exists(backup_path):
            shutil.copy(backup_path, installed_config)
            os.remove(backup_path)
            logger.info("已恢复 tile_fwk_config.json (已安装)")
            tile_restored = True

    # Restore source device_switch.h from backup
    switch_restored = restore_from_backup(
        pypto_path, KNOWN_LOCATIONS['device_switch.h'], 'device_switch.h', 'device_switch.h'
    )

    any_restored = tile_restored or switch_restored

    if not any_restored:
        logger.info("没有需要恢复的备份文件")
        logger.info("=" * 80)
        logger.info("无需操作")
        logger.info("=" * 80)
        sys.exit(0)

    if not switch_restored:
        logger.info("源代码未修改，跳过重新编译")
        logger.info("=" * 80)
        logger.info("已恢复到初始状态")
        logger.info("=" * 80)
        sys.exit(0)

    if not build_and_install(pypto_path):
        logger.info("")
        logger.info("=" * 80)
        logger.info("恢复失败：编译安装出错")
        logger.info("=" * 80)
        sys.exit(1)

    logger.info("")
    logger.info("=" * 80)
    logger.info("已恢复到初始编译状态")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
