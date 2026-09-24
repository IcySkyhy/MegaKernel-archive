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
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_valid_binary, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class BinaryFinder:
    """二进制文件查找器"""

    def __init__(self):
        self.search_paths = self._get_default_search_paths()

    @staticmethod
    def _get_default_search_paths() -> List[Path]:
        """获取默认搜索路径"""
        paths = []

        # 当前目录
        paths.append(Path.cwd())

        # PATH 环境变量
        if 'PATH' in os.environ:
            for path in os.environ['PATH'].split(':'):
                paths.append(Path(path))

        # Conda 环境路径（如果当前在 conda 环境中）
        # 说明：不依赖 sys.prefix，优先使用 CONDA_PREFIX 来判断“当前激活的 conda 环境”
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_prefix_path = Path(conda_prefix).expanduser().resolve()
            if conda_prefix_path.exists():
                paths.extend(
                    [
                        conda_prefix_path / 'bin',
                        conda_prefix_path / 'lib',
                        conda_prefix_path / 'lib64',
                    ]
                )

                # Conda site-packages（用 glob 展开 python 版本）
                import glob

                conda_site_patterns = [
                    conda_prefix_path / 'lib' / 'python3.*' / 'site-packages',
                    conda_prefix_path / 'lib' / 'python3.*' / 'site-packages' / 'pypto',
                    conda_prefix_path / 'lib64' / 'python3.*' / 'site-packages',
                    conda_prefix_path / 'lib64' / 'python3.*' / 'site-packages' / 'pypto',
                ]
                for pattern in conda_site_patterns:
                    for p in glob.glob(str(pattern)):
                        paths.append(Path(p))

        # PyPTO 安装路径
        pypto_paths = [
            Path.home() / '.local' / 'lib' / 'python3.*' / 'site-packages' / 'pypto',
            Path('/usr/local/lib/python3.*') / 'site-packages' / 'pypto',
        ]

        # 扩展通配符
        import glob

        for pattern in pypto_paths:
            for path in glob.glob(str(pattern)):
                paths.append(Path(path))

        # 常见库路径
        paths.extend(
            [
                Path('/usr/lib'),
                Path('/usr/local/lib'),
                Path('/lib'),
                Path('/lib64'),
            ]
        )

        # 去重并过滤不存在的路径
        unique_paths = []
        seen = set()
        for path in paths:
            path = path.resolve()
            if path not in seen and path.exists():
                unique_paths.append(path)
                seen.add(path)

        return unique_paths

    def find_binary(self, binary_name: str, custom_paths: Optional[List[str]] = None) -> Optional[Path]:
        """查找二进制文件"""
        logger.info("查找二进制文件: %s", binary_name)

        # 如果是绝对路径，直接验证
        if Path(binary_name).is_absolute():
            path = Path(binary_name)
            if is_valid_binary(path):
                logger.info("找到二进制文件: %s", path)
                return path
            else:
                logger.warning("文件存在但不是有效的二进制文件: %s", path)
                return None

        # 自定义搜索路径
        search_paths = self.search_paths.copy()
        if custom_paths:
            for custom_path in custom_paths:
                path = Path(custom_path).expanduser().resolve()
                if path.exists() and path not in search_paths:
                    search_paths.append(path)

        # 在所有搜索路径中查找
        for search_path in search_paths:
            # 直接匹配
            candidate = search_path / binary_name
            if is_valid_binary(candidate):
                logger.info("找到二进制文件: %s", candidate)
                return candidate

            # 递归搜索（仅限前几层）
            for depth in range(3):
                pattern = str(search_path / ('*/' * depth) / binary_name)
                import glob

                matches = glob.glob(pattern)
                for match in matches:
                    path = Path(match)
                    if is_valid_binary(path):
                        logger.info("找到二进制文件: %s", path)
                        return path

        logger.warning("未找到二进制文件: %s", binary_name)
        return None

    def list_potential_binaries(self, binary_name: str) -> List[Path]:
        """列出所有潜在的二进制文件"""
        candidates = []

        for search_path in self.search_paths:
            # 直接匹配
            candidate = search_path / binary_name
            if candidate.exists():
                candidates.append(candidate)

            # 递归搜索
            for depth in range(3):
                pattern = str(search_path / ('*/' * depth) / binary_name)
                import glob

                matches = glob.glob(pattern)
                for match in matches:
                    path = Path(match)
                    if path not in candidates:
                        candidates.append(path)

        return candidates


def main():
    parser = argparse.ArgumentParser(description='自动搜索二进制文件')
    parser.add_argument('binary_name', help='二进制文件名')
    parser.add_argument('-p', '--path', action='append', help='额外的搜索路径')

    args = parser.parse_args()

    # 查找二进制文件
    finder = BinaryFinder()
    binary_path = finder.find_binary(args.binary_name, args.path)

    if binary_path:
        logger.info("=" * 80)
        logger.info("找到二进制文件")
        logger.info("=" * 80)
        logger.info("路径: %s", binary_path)
        logger.info("绝对路径: %s", binary_path.resolve())

        # 验证有效性
        if is_valid_binary(binary_path):
            logger.info("状态: 有效")
        else:
            logger.warning("状态: 无效")

        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("未找到二进制文件")
        logger.error("=" * 80)
        logger.error("文件名: %s", args.binary_name)

        # 列出潜在文件
        candidates = finder.list_potential_binaries(args.binary_name)
        if candidates:
            logger.info("潜在文件:")
            for candidate in candidates:
                logger.info("  - %s", candidate)

        sys.exit(1)


if __name__ == '__main__':
    main()
