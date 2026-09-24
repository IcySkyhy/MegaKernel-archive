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
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check_required_tools, run_command, setup_logging, validate_path

setup_logging()
logger = logging.getLogger(__name__)


class SymbolResolver:
    """符号解析器"""

    def __init__(self):
        self.tools = check_required_tools()
        self._check_tools()

    @staticmethod
    def format_results(results: List[Dict[str, Optional[str]]]) -> str:
        """格式化输出结果"""
        output = []

        output.append("=" * 80)
        output.append("符号解析结果")
        output.append("=" * 80)

        for result in results:
            output.append(f"地址: {result['address']}")

            if result['success']:
                if result['symbol']:
                    output.append(f"  符号: {result['symbol']}")
                if result['section']:
                    output.append(f"  段: {result['section']}")
                if result['offset']:
                    output.append(f"  偏移: {result['offset']}")
            else:
                output.append("  状态: 解析失败")

            output.append("")

        return '\n'.join(output)

    @staticmethod
    def _parse_symbol_table(table: str, address: str) -> Optional[Dict[str, Optional[str]]]:
        """解析符号表"""
        # 移除 '0x' 前缀并转换为整数
        try:
            target_addr = int(address, 16)
        except ValueError:
            return None

        best_match = None
        best_distance = float('inf')

        # 解析每一行
        for line in table.split('\n'):
            # 格式: 0000000000123456 g     F .text  0000000000000010 function_name
            match = re.match(r'^([0-9a-f]+)\s+\w+\s+\w+\s+([0-9a-f]+)\s+([^\s]+)', line)
            if match:
                addr_str, size_str, symbol = match.groups()
                addr = int(addr_str, 16)
                size = int(size_str, 16)

                # 检查地址是否在符号范围内
                if addr <= target_addr < addr + size:
                    return {
                        'symbol': symbol,
                        'section': '.text',  # 简化处理
                        'offset': hex(target_addr - addr),
                    }

                # 记录最近的符号
                distance = abs(addr - target_addr)
                if distance < best_distance:
                    best_distance = distance
                    best_match = {'symbol': symbol, 'section': '.text', 'offset': hex(target_addr - addr)}

        return best_match

    def resolve_symbol(self, binary: str, address: str) -> Dict[str, Optional[str]]:
        """解析单个地址的符号信息"""
        result = {'address': address, 'symbol': None, 'section': None, 'offset': None, 'success': False}

        try:
            # 使用 objdump 获取符号表
            cmd = [self.tools['objdump'], '-t', binary]
            output = run_command(cmd, capture=True)

            if output.returncode == 0:
                # 解析符号表
                symbol_info = self._parse_symbol_table(output.stdout, address)
                if symbol_info:
                    result.update(symbol_info)
                    result['success'] = True
            else:
                logger.warning("符号解析失败: %s", address)
                logger.debug("错误输出: %s", output.stderr)

        except Exception as e:
            logger.error("符号解析异常: %s, 错误: %s", address, e)

        return result

    def resolve_symbols(self, binary: str, addresses: List[str]) -> List[Dict[str, Optional[str]]]:
        """解析多个地址的符号信息"""
        results = []

        logger.info("=" * 80)
        logger.info("符号解析")
        logger.info("=" * 80)
        logger.info("二进制文件: %s", binary)
        logger.info("地址数量: %d", len(addresses))

        for idx, address in enumerate(addresses):
            logger.info("解析符号 %d/%d: %s", idx + 1, len(addresses), address)
            result = self.resolve_symbol(binary, address)
            results.append(result)

        return results

    def _check_tools(self):
        """检查必需的工具"""
        if not self.tools.get('objdump'):
            logger.error("未找到 objdump 工具")
            logger.error("请安装 binutils 或 llvm-tools")
            raise RuntimeError("未找到 objdump 工具")

        logger.info("使用 objdump 工具: %s", self.tools['objdump'])


def main():
    parser = argparse.ArgumentParser(description='符号解析')
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

    # 解析符号
    resolver = SymbolResolver()
    results = resolver.resolve_symbols(str(binary_path), args.addresses)

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
