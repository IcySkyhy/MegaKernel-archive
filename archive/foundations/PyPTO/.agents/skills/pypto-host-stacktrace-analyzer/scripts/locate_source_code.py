#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_command, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SourceCodeLocator:
    """源码定位器"""

    def __init__(self, binary_path=None):
        if binary_path is None:
            # 查找默认的 libtile_fwk_interface.so
            from auto_find_binary import BinaryFinder

            finder = BinaryFinder()
            binary_path = finder.find_binary("libtile_fwk_interface.so")

        self.binary_path = Path(binary_path) if binary_path else None

    def locate_address(self, address):
        """定位地址到源码行"""
        if not self.binary_path:
            return None

        try:
            cmd = ['addr2line', '-e', str(self.binary_path), '-f', '-C', address]
            result = run_command(cmd, capture=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    return {'function': lines[0], 'location': lines[1]}
        except Exception as e:
            logger.warning("地址定位失败: %s, 错误: %s", address, e)

        return None

    def locate_symbol(self, symbol):
        """定位符号到源码行"""
        if not self.binary_path:
            return None

        try:
            # 首先获取符号地址
            cmd = ['nm', '-D', str(self.binary_path)]
            result = run_command(cmd, capture=True, timeout=10)

            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if symbol in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            address = parts[0]
                            return self.locate_address(address)
        except Exception as e:
            logger.warning("符号定位失败: %s, 错误: %s", symbol, e)

        return None

    def check_debug_symbols(self):
        """检查是否有调试符号"""
        if not self.binary_path:
            return False

        try:
            cmd = ['nm', '-C', str(self.binary_path)]
            result = run_command(cmd, capture=True, timeout=10)

            if result.returncode == 0:
                # 检查是否有带源码路径的符号
                for line in result.stdout.split('\n'):
                    if '.cpp:' in line or '.cc:' in line:
                        return True
        except Exception as e:
            logger.warning("调试符号检查失败: %s", e)

        return False


def main():
    parser = argparse.ArgumentParser(description='源码定位')
    parser.add_argument('-b', '--binary', help='二进制文件路径')
    parser.add_argument('-a', '--address', help='要定位的地址')
    parser.add_argument('-s', '--symbol', help='要定位的符号')
    parser.add_argument('--check-debug', action='store_true', help='检查是否有调试符号')

    args = parser.parse_args()

    # 创建定位器
    locator = SourceCodeLocator(args.binary)

    if not locator.binary_path:
        logger.error("未找到二进制文件")
        sys.exit(1)

    logger.info("二进制文件: %s", locator.binary_path)

    # 检查调试符号
    if args.check_debug:
        has_debug = locator.check_debug_symbols()
        if has_debug:
            logger.info("✓ 二进制文件包含调试符号")
        else:
            logger.info("✗ 二进制文件不包含调试符号")
            logger.info("建议：使用 Debug 版本编译")
        sys.exit(0)

    # 定位地址
    if args.address:
        result = locator.locate_address(args.address)
        if result:
            logger.info("=" * 80)
            logger.info("地址定位结果")
            logger.info("=" * 80)
            logger.info("地址: %s", args.address)
            logger.info("函数: %s", result['function'])
            logger.info("位置: %s", result['location'])
        else:
            logger.error("无法定位地址: %s", args.address)

    # 定位符号
    if args.symbol:
        result = locator.locate_symbol(args.symbol)
        if result:
            logger.info("=" * 80)
            logger.info("符号定位结果")
            logger.info("=" * 80)
            logger.info("符号: %s", args.symbol)
            logger.info("函数: %s", result['function'])
            logger.info("位置: %s", result['location'])
        else:
            logger.error("无法定位符号: %s", args.symbol)


if __name__ == '__main__':
    main()
