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
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_command, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SymbolDemangler:
    """符号反混淆器"""

    def __init__(self):
        self.cxxfilt_available = self._check_cxxfilt()

    @staticmethod
    def _check_cxxfilt() -> bool:
        """检查 c++filt 工具是否可用"""
        result = run_command(['which', 'c++filt'], capture=True)
        return result.returncode == 0

    def demangle_symbol(self, mangled_symbol: str) -> str:
        """反混淆单个符号"""
        if not self.cxxfilt_available:
            return mangled_symbol

        try:
            cmd = ['c++filt', mangled_symbol]
            result = run_command(cmd, capture=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning("符号反混淆失败: %s, 错误: %s", mangled_symbol, e)

        return mangled_symbol

    def demangle_symbols(self, symbols: List[str]) -> List[dict]:
        """反混淆多个符号"""
        results = []

        for symbol in symbols:
            demangled = self.demangle_symbol(symbol)
            results.append({'mangled': symbol, 'demangled': demangled, 'changed': symbol != demangled})

        return results


def main():
    parser = argparse.ArgumentParser(description='符号反混淆')
    parser.add_argument('symbols', nargs='+', help='符号列表')
    parser.add_argument('-j', '--json', action='store_true', help='输出 JSON 格式')

    args = parser.parse_args()

    # 反混淆符号
    demangler = SymbolDemangler()
    results = demangler.demangle_symbols(args.symbols)

    # 输出结果
    if args.json:
        import json

        logger.info("%s", json.dumps(results, indent=2))
    else:
        logger.info("=" * 80)
        logger.info("符号反混淆结果")
        logger.info("=" * 80)

        for result in results:
            logger.info("混淆符号: %s", result['mangled'])
            if result['changed']:
                logger.info("反混淆: %s", result['demangled'])
            else:
                logger.info("状态: 无需反混淆")
            logger.info("")


if __name__ == '__main__':
    main()
