#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
验证 CCE 文件是否为问题文件

注释 CCE 文件中所有可注释的代码行，运行测试，判断 aicore error 是否消失：
- 消失 → 此文件是问题文件（退出码 0）
- 仍存在 → 此文件不是问题文件（退出码 1）

用法:
    python3 verify_cce_file.py <cce_file> <test_cmd> <run_path> [--use-pypto-test-framework]

退出码:
    0: 此文件是问题文件
    1: 此文件不是问题文件，或验证失败
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    backup_and_test,
    comment_lines_by_indices,
    get_commentable_lines,
    print_error_info,
    setup_logging,
    validate_path,
)

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="验证 CCE 文件是否为问题文件")
    parser.add_argument('cce_file', help="CCE 文件路径")
    parser.add_argument('test_cmd', help="触发 aicore error 的测试命令")
    parser.add_argument('run_path', help="运行测试命令的目录路径")
    parser.add_argument('--use-pypto-test-framework', action='store_true', help="使用 Pypto_Test 框架")
    args = parser.parse_args()

    cce_file = os.path.abspath(args.cce_file)
    test_cmd = args.test_cmd
    run_dir = os.path.abspath(args.run_path)

    for path, label in [(cce_file, "CCE 文件"), (run_dir, "运行目录")]:
        valid, msg = validate_path(path, label)
        if not valid:
            logger.info(msg)
            sys.exit(1)

    logger.info("=" * 80)
    logger.info("验证 CCE 文件")
    logger.info("=" * 80)
    logger.info("CCE 文件: %s", cce_file)
    logger.info("")

    def _modify(lines):
        commentable = get_commentable_lines(lines, error_in_t=False)
        logger.info("可注释行数: %d", len(commentable))
        if not commentable:
            logger.info("没有可注释的行")
            return None
        return comment_lines_by_indices(lines.copy(), commentable)

    error_exists, output, original_lines = backup_and_test(
        cce_file, test_cmd, run_dir, _modify, use_pypto_test_framework=args.use_pypto_test_framework
    )

    if original_lines is None:
        logger.info("错误：无法完成验证")
        sys.exit(1)

    if error_exists:
        print_error_info(output, logger)
        logger.info("")
        logger.info("结果: 注释所有行后仍有 error，此文件不是问题文件")
        sys.exit(1)
    else:
        logger.info("")
        logger.info("结果: 注释所有行后运行成功，此文件是问题文件 ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
