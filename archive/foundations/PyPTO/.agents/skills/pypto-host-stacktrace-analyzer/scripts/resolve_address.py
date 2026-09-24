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
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check_required_tools, run_command, setup_logging, validate_path

setup_logging()
logger = logging.getLogger(__name__)


class AddressResolver:
    """地址解析器"""

    def __init__(self):
        self.tools = check_required_tools()
        self._check_tools()

    @staticmethod
    def format_results(results: List[Dict[str, Optional[str]]]) -> str:
        """格式化输出结果"""
        output = []

        output.append("=" * 80)
        output.append("地址解析结果")
        output.append("=" * 80)

        for result in results:
            output.append(f"地址: {result['address']}")

            if result['success']:
                if result['function']:
                    output.append(f"  函数: {result['function']}")
                if result['file']:
                    line_str = f":{result['line']}" if result['line'] else ""
                    output.append(f"  源码: {result['file']}{line_str}")
            else:
                output.append("  状态: 解析失败")

            output.append("")

        return '\n'.join(output)

    def resolve_address(self, binary: str, address: str) -> Dict[str, Optional[str]]:
        """解析单个地址"""
        result = {'address': address, 'function': None, 'file': None, 'line': None, 'success': False}

        try:
            # 使用 addr2line 解析地址
            cmd = [self.tools['addr2line'], '-e', binary, '-f', '-C', address]
            output = run_command(cmd, capture=True)

            if output.returncode == 0:
                lines = output.stdout.strip().split('\n')
                if len(lines) >= 2:
                    result['function'] = lines[0]
                    file_line = lines[1]

                    # 解析文件和行号
                    if ':' in file_line:
                        parts = file_line.rsplit(':', 1)
                        result['file'] = parts[0]
                        try:
                            result['line'] = int(parts[1])
                        except ValueError:
                            pass

                    result['success'] = True
            else:
                logger.warning("地址解析失败: %s", address)
                logger.debug("错误输出: %s", output.stderr)

        except Exception as e:
            logger.error("地址解析异常: %s, 错误: %s", address, e)

        return result

    def resolve_addresses(self, binary: str, addresses: List[str]) -> List[Dict[str, Optional[str]]]:
        """解析多个地址"""
        results = []

        logger.info("=" * 80)
        logger.info("地址解析")
        logger.info("=" * 80)
        logger.info("二进制文件: %s", binary)
        logger.info("地址数量: %d", len(addresses))

        for idx, address in enumerate(addresses):
            logger.info("解析地址 %d/%d: %s", idx + 1, len(addresses), address)
            result = self.resolve_address(binary, address)
            results.append(result)

        return results

    def _check_tools(self):
        """检查必需的工具"""
        if not self.tools.get('addr2line'):
            logger.error("未找到 addr2line 工具")
            logger.error("请安装 binutils 或 llvm-tools")
            raise RuntimeError("未找到 addr2line 工具")

        logger.info("使用 addr2line 工具: %s", self.tools['addr2line'])


def main():
    parser = argparse.ArgumentParser(description='地址到源码行映射')
    parser.add_argument('binary', help='二进制文件路径')
    parser.add_argument('addresses', nargs='+', help='地址列表')
    parser.add_argument('-o', '--output', help='输出文件路径')

    args = parser.parse_args()

    # 验证二进制文件
    try:
        binary_path = validate_path(args.binary)
    except Exception as e:
        logger.error("二进制文件验证失败: %s", e)
        sys.exit(1)

    # 解析地址
    resolver = AddressResolver()
    results = resolver.resolve_addresses(str(binary_path), args.addresses)

    # 格式化输出
    output = resolver.format_results(results)
    logger.info("%s", output)

    # 保存到文件
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info("结果已保存到: %s", args.output)
        except Exception as e:
            logger.error("保存文件失败: %s", e)


if __name__ == '__main__':
    main()
