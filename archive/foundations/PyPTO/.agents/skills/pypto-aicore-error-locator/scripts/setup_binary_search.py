#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
步骤 4.1 + 4.2: 确定错误范围并获取二分查找初始范围

合并 SKILL.md 中的 4.1 和 4.2 为单个脚本调用:
  4.1 - 注释所有 T 操作行并测试，判断 aicore error 是否在 T 操作中
  4.2 - 根据判断结果获取二分查找的初始范围 (LEFT, RIGHT)

用法:
    python3 setup_binary_search.py <cce_file> <test_cmd> <run_path>

输出:
    ERROR_IN_T=True|False
    LEFT=<left>
    RIGHT=<right>
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    backup_and_test,
    comment_lines_by_indices,
    comment_special_lines,
    get_commentable_lines,
    print_error_info,
    read_file,
    setup_logging,
    validate_path,
)

setup_logging()
logger = logging.getLogger(__name__)


def determine_error_in_t(cce_file, test_cmd, run_dir, use_pypto_test_framework=False):
    logger.info("=" * 80)
    logger.info("步骤 4.1: 确定错误范围")
    logger.info("=" * 80)
    logger.info("CCE 文件: %s", cce_file)
    logger.info("测试命令: %s", test_cmd)
    logger.info("运行目录: %s", run_dir)
    logger.info("")

    def _modify(lines):
        commentable = get_commentable_lines(lines, True)
        logger.info("T 操作可注释行数: %d", len(commentable))
        if not commentable:
            logger.info("没有可注释的 T 操作行，问题不在 T 操作中")
            return None
        return comment_lines_by_indices(lines.copy(), commentable)

    error_exists, output, original_lines = backup_and_test(
        cce_file, test_cmd, run_dir, _modify, use_pypto_test_framework=use_pypto_test_framework
    )

    if original_lines is None:
        return False

    if error_exists:
        print_error_info(output, logger)
        logger.info("")
        logger.info("结果: 注释 T 操作行后仍有 error，问题不在 T 操作行")
        return False
    else:
        logger.info("")
        logger.info("结果: 注释 T 操作行后运行成功，问题在 T 操作行")
        return True


def get_initial_range(cce_file, error_in_t):
    logger.info("")
    logger.info("=" * 80)
    logger.info("步骤 4.2: 获取可注释行范围")
    logger.info("=" * 80)
    logger.info("CCE 文件: %s", cce_file)
    logger.info("error_in_t: %s", error_in_t)
    logger.info("")

    cce_lines = read_file(cce_file)
    cce_lines = comment_special_lines(cce_lines)
    commentable = get_commentable_lines(cce_lines, error_in_t)

    n = len(commentable)
    logger.info("可注释的行数: %d", n)

    if n <= 0:
        logger.info("错误：没有可注释的行")
        return None, None

    left, right = 0, n - 1
    logger.info("初始范围: left=%d, right=%d", left, right)
    return left, right


def main():
    parser = argparse.ArgumentParser(description="确定错误范围并获取二分查找初始范围")
    parser.add_argument('cce_file', help="CCE 文件路径")
    parser.add_argument('test_cmd', help="触发 aicore error 的测试命令")
    parser.add_argument('run_path', help="运行测试命令的目录路径")
    parser.add_argument('--use-pypto-test-framework', action='store_true', help="使用 Pypto_Test 框架")
    args = parser.parse_args()

    cce_file = os.path.abspath(args.cce_file)
    test_cmd = args.test_cmd
    run_dir = os.path.abspath(args.run_path)
    use_pypto = args.use_pypto_test_framework

    for path, label in [(cce_file, "CCE 文件"), (run_dir, "运行目录")]:
        valid, msg = validate_path(path, label)
        if not valid:
            logger.info(msg)
            sys.exit(1)

    error_in_t = determine_error_in_t(cce_file, test_cmd, run_dir, use_pypto_test_framework=use_pypto)
    if error_in_t is None:
        sys.exit(2)

    left, right = get_initial_range(cce_file, error_in_t)
    if left is None:
        sys.exit(1)

    logger.info("")
    logger.info("ERROR_IN_T=%s", str(error_in_t))
    logger.info("LEFT=%d", left)
    logger.info("RIGHT=%d", right)


if __name__ == "__main__":
    main()
