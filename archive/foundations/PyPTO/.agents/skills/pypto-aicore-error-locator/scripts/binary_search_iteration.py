#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import argparse
from dataclasses import dataclass
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    comment_lines_by_indices,
    get_commentable_lines,
    has_error,
    print_error_info,
    read_file,
    run_test,
    setup_logging,
    uncomment_lines_by_indices,
    validate_path,
    write_file,
)

setup_logging()

logger = logging.getLogger(__name__)


@dataclass
class BinarySearchParams:
    cce_file: str
    test_cmd: str
    run_dir: str
    left: int
    right: int
    error_in_t: bool
    use_pypto_test_framework: bool = False


def binary_search_iteration(params):
    logger.info("二分查找迭代: left=%d, right=%d", params.left, params.right)
    logger.info("CCE 文件: %s", params.cce_file)
    logger.info("error_in_t: %s", params.error_in_t)
    logger.info("")

    backup_file = params.cce_file + ".bak"
    shutil.copy(params.cce_file, backup_file)
    cce_lines = read_file(params.cce_file)
    original_lines = cce_lines.copy()

    commentable_lines = get_commentable_lines(cce_lines, params.error_in_t)

    if not commentable_lines:
        logger.info("错误：没有可注释的行")
        write_file(params.cce_file, original_lines)
        os.remove(backup_file)
        return None, None, None

    mid = (params.left + params.right) // 2
    logger.info("mid = (left + right) // 2 = (%d + %d) // 2 = %d", params.left, params.right, mid)
    logger.info("取消注释范围 [0, %d] 的行", mid)
    logger.info(commentable_lines[0:mid + 1])

    current_lines = cce_lines.copy()
    current_lines = comment_lines_by_indices(current_lines, commentable_lines)

    lines_to_uncomment = commentable_lines[0:mid + 1]
    current_lines = uncomment_lines_by_indices(current_lines, lines_to_uncomment)

    write_file(params.cce_file, current_lines)
    logger.info("运行测试...")
    returncode, output = run_test(params.test_cmd, params.run_dir)
    error_exists = has_error(returncode, output, params.use_pypto_test_framework)

    write_file(params.cce_file, original_lines)
    os.remove(backup_file)

    if error_exists:
        print_error_info(output, logger)
        logger.info("结果: 运行失败（有 error），问题在 [%d, %d] 中", params.left, mid)
        new_left = params.left
        new_right = mid
    else:
        logger.info("结果: 运行成功（无 error），问题在 [%d, %d] 中", mid + 1, params.right)
        new_left = mid + 1
        new_right = params.right

    logger.info("下一轮: left=%d, right=%d", new_left, new_right)
    logger.info("")

    if new_left == new_right:
        problem_line = commentable_lines[new_left]
        logger.info("找到问题代码行: %d", problem_line)
        return new_left, new_right, problem_line

    return new_left, new_right, None


def main():
    parser = argparse.ArgumentParser(description="二分查找迭代定位问题代码行")
    parser.add_argument('cce_file', help="CCE 文件路径")
    parser.add_argument('test_cmd', help="触发 aicore error 的测试命令")
    parser.add_argument('run_dir', help="运行测试命令的目录路径")
    parser.add_argument('left', type=int, help="二分查找左边界")
    parser.add_argument('right', type=int, help="二分查找右边界")
    parser.add_argument('error_in_t', type=lambda x: x.lower() == 'true', help="错误是否在T操作中 (True/False)")
    parser.add_argument('--use-pypto-test-framework', action='store_true', help="使用 Pypto_Test 框架")
    args = parser.parse_args()

    cce_file = os.path.abspath(args.cce_file)
    test_cmd = args.test_cmd
    run_dir = os.path.abspath(args.run_dir)
    left = args.left
    right = args.right
    error_in_t = args.error_in_t
    use_pypto = args.use_pypto_test_framework

    valid, error_msg = validate_path(cce_file, "CCE 文件")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    valid, error_msg = validate_path(run_dir, "运行目录")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    params = BinarySearchParams(
        cce_file=cce_file,
        test_cmd=test_cmd,
        run_dir=run_dir,
        left=left,
        right=right,
        error_in_t=error_in_t,
        use_pypto_test_framework=use_pypto,
    )
    new_left, new_right, problem_line = binary_search_iteration(params)

    if new_left is not None:
        logger.info(f"NEXT_LEFT {new_left}")
        logger.info(f"NEXT_RIGHT {new_right}")
        if problem_line is not None:
            logger.info(f"FOUND {problem_line}")


if __name__ == "__main__":
    main()
