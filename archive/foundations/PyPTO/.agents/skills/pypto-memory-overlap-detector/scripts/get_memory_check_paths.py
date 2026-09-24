#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_trace_log_dir, get_latest_output_subdir, setup_logging, validate_path

setup_logging()

logger = logging.getLogger(__name__)


def print_usage():
    logger.info("用法: python3 get_memory_check_paths.py <log_base_path> <output_path>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  log_base_path: 落盘日志根目录")
    logger.info("  output_path:   运行目录中的 output 目录路径")


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    log_base_path = os.path.abspath(sys.argv[1])
    output_path = os.path.abspath(sys.argv[2])

    valid, error_msg = validate_path(log_base_path, "落盘日志根目录")
    if not valid:
        logger.error(error_msg)
        sys.exit(1)

    valid, error_msg = validate_path(output_path, "output 目录")
    if not valid:
        logger.error(error_msg)
        sys.exit(1)

    logger.info("落盘日志根目录: %s", log_base_path)
    logger.info("Output 路径: %s", output_path)

    trace_log_dir = find_trace_log_dir(log_base_path)
    if not trace_log_dir:
        logger.error("未找到包含 #trace 的 device*.log 文件")
        sys.exit(1)

    logger.info("找到 trace 日志目录: %s", trace_log_dir)

    dyn_topo_path, error_msg = get_latest_output_subdir(output_path, 'dyn_topo.txt')
    if not dyn_topo_path:
        logger.error(error_msg)
        sys.exit(1)

    logger.info("找到 dyn_topo.txt: %s", dyn_topo_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("内存重叠检测命令:")
    logger.info("python3 tools/schema/schema_memory_check.py -d %s -t %s", trace_log_dir, dyn_topo_path)
    logger.info("=" * 60)

    sys.exit(0)


if __name__ == '__main__':
    main()
