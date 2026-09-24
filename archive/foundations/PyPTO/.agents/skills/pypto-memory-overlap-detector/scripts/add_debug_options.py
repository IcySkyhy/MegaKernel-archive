#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import logging
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_file, setup_logging, write_file

setup_logging()

logger = logging.getLogger(__name__)


def add_debug_options_to_file(file_path):
    content = read_file(file_path)
    original_content = content

    if 'runtime_debug_mode' in content:
        logger.info("文件已包含 runtime_debug_mode，无需修改: %s", file_path)
        return False, None

    patterns = [
        (r'@pypto\.frontend\.jit\s*\n', '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1}\n)\n'),
        (r'@pypto\.frontend\.jit\(\)\s*\n', '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1}\n)\n'),
    ]

    for pattern, replacement in patterns:
        content, count = re.subn(pattern, replacement, content)
        if count > 0:
            break

    if count == 0:
        pattern = r'@pypto\.frontend\.jit\(([^)]+)\)\s*\n'
        match = re.search(pattern, content)
        if match:
            params = match.group(1).strip()
            if '\n' in match.group(1):
                replacement = (
                    f'@pypto.frontend.jit(\n    debug_options={{"runtime_debug_mode": 1}},\n{match.group(1)})\n'
                )
            else:
                replacement = f'@pypto.frontend.jit(\n    debug_options={{"runtime_debug_mode": 1}},\n    {params}\n)\n'
            content = re.sub(pattern, replacement, content)
            count = 1

    if content != original_content:
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        logger.info("已创建备份: %s", backup_path)

        write_file(file_path, content)
        logger.info("已添加 debug_options 到文件: %s", file_path)
        return True, backup_path

    logger.warning("未找到需要修改的 @pypto.frontend.jit 装饰器: %s", file_path)
    return False, None


def restore_from_backup(test_file_path):
    backup_path = f"{test_file_path}.backup"

    if not os.path.exists(backup_path):
        logger.warning("备份文件不存在: %s", backup_path)
        return False

    shutil.copy2(backup_path, test_file_path)
    logger.info("已从备份恢复: %s", test_file_path)

    os.remove(backup_path)
    logger.info("已删除备份: %s", backup_path)

    return True


def print_usage():
    logger.info("用法:")
    logger.info("  添加 debug_options:")
    logger.info("    python3 add_debug_options.py add <test_file_path>")
    logger.info("")
    logger.info("  从备份恢复并删除备份:")
    logger.info("    python3 add_debug_options.py restore <test_file_path>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  test_file_path: 测试用例文件路径")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == 'add':
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)

        test_file_path = os.path.abspath(sys.argv[2])

        if not os.path.exists(test_file_path):
            logger.error("测试文件不存在: %s", test_file_path)
            sys.exit(1)

        logger.info("测试文件路径: %s", test_file_path)

        modified, backup_path = add_debug_options_to_file(test_file_path)

        if modified:
            logger.info("")
            logger.info("=" * 60)
            logger.info("修改完成！")
            logger.info("备份文件: %s", backup_path)
            logger.info("如需恢复，请运行:")
            logger.info("  python3 add_debug_options.py restore %s", test_file_path)
            logger.info("=" * 60)
            sys.exit(0)
        else:
            sys.exit(1)

    elif command == 'restore':
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)

        test_file_path = os.path.abspath(sys.argv[2])

        if not os.path.exists(test_file_path):
            logger.error("测试文件不存在: %s", test_file_path)
            sys.exit(1)

        if restore_from_backup(test_file_path):
            logger.info("")
            logger.info("=" * 60)
            logger.info("恢复成功！已删除备份文件")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.error("恢复失败！")
            sys.exit(1)

    else:
        logger.error("未知命令: %s", command)
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
