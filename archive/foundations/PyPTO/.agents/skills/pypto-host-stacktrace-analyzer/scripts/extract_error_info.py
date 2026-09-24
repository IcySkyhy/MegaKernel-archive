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
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ErrorInfoExtractor:
    """错误信息提取器"""

    def __init__(self):
        self.error_info = {}

    @staticmethod
    def extract_errcode(text: str) -> Optional[str]:
        """提取错误码"""
        pattern = r'Errcode:\s*([A-F0-9]+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_error_location(text: str) -> Optional[Dict[str, str]]:
        """提取错误位置"""
        # 格式: file, line, func
        pattern = r'file\s+(\S+),\s+line\s+(\d+),\s+func\s+(\S+)'
        match = re.search(pattern, text)
        if match:
            return {'file': match.group(1), 'line': match.group(2), 'function': match.group(3)}
        return None

    @staticmethod
    def extract_error_message(text: str) -> Optional[str]:
        """提取错误消息"""
        # 尝试多种模式
        patterns = [
            r'Error:\s*(.+?)(?:\n|$)',  # Error: message
            r'ERROR:\s*(.+?)(?:\n|$)',  # ERROR: message
            r'RuntimeError:\s*(.+?)(?:\n|$)',  # RuntimeError: message
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def extract_pypto_error_info(text: str) -> Optional[Dict[str, str]]:
        """提取 PyPTO 特定错误信息"""
        info = {}

        # 提取错误码
        errcode = ErrorInfoExtractor.extract_errcode(text)
        if errcode:
            info['errcode'] = errcode

        # 提取错误位置
        location = ErrorInfoExtractor.extract_error_location(text)
        if location:
            info['location'] = location

        # 提取错误消息
        message = ErrorInfoExtractor.extract_error_message(text)
        if message:
            info['message'] = message

        return info if info else None

    @staticmethod
    def format_error_info(error_info: Dict[str, str]) -> str:
        """格式化错误信息"""
        output = []

        output.append("=" * 80)
        output.append("错误信息")
        output.append("=" * 80)

        if 'errcode' in error_info:
            output.append(f"错误码: {error_info['errcode']}")

        if 'location' in error_info:
            loc = error_info['location']
            output.append(f"位置: {loc['file']}:{loc['line']}")
            output.append(f"函数: {loc['function']}")

        if 'message' in error_info:
            output.append(f"消息: {error_info['message']}")

        return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='提取错误信息')
    parser.add_argument('input', help='错误文本或文件路径')
    parser.add_argument('-f', '--file', action='store_true', help='输入是文件路径')
    parser.add_argument('-j', '--json', action='store_true', help='输出 JSON 格式')

    args = parser.parse_args()

    # 读取输入
    if args.file:
        try:
            from common import validate_path

            path = validate_path(args.input)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error("读取文件失败: %s", e)
            sys.exit(1)
    else:
        text = args.input

    # 提取错误信息
    extractor = ErrorInfoExtractor()
    error_info = extractor.extract_pypto_error_info(text)

    if not error_info:
        logger.warning("未能提取错误信息")
        sys.exit(1)

    # 输出结果
    if args.json:
        import json

        logger.info("%s", json.dumps(error_info, indent=2))
    else:
        logger.info("%s", extractor.format_error_info(error_info))


if __name__ == '__main__':
    main()
