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
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_command, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class DebugPyptoBuilder:
    """Debug版本PyPTO编译器"""

    def __init__(self, pypto_root=None):
        if pypto_root is None:
            pypto_root = os.getcwd()
        self.pypto_root = Path(pypto_root).resolve()
        self.build_script = self.pypto_root / "build_ci.py"
        self.build_output = self.pypto_root / "build_out"

    def check_prerequisites(self):
        """检查编译前提条件"""
        logger.info("检查编译前提条件...")

        # 检查build_ci.py是否存在
        if not self.build_script.exists():
            logger.error("未找到 build_ci.py: %s", self.build_script)
            return False

        # 检查Python版本
        result = run_command(['python3', '--version'], capture=True)
        if result.returncode == 0:
            logger.info("Python版本: %s", result.stdout.strip())
        else:
            logger.error("无法获取Python版本")
            return False

        # 检查cmake
        result = run_command(['which', 'cmake'], capture=True)
        if result.returncode == 0:
            logger.info("CMake路径: %s", result.stdout.strip())
        else:
            logger.error("未找到cmake，请先安装cmake")
            return False

        logger.info("前提条件检查通过")
        return True

    def build_debug(self, timeout=1200):
        """编译Debug版本"""
        logger.info("=" * 80)
        logger.info("开始编译DebugDebug版本PyPTO")
        logger.info("=" * 80)

        cmd = ['python3.11', str(self.build_script), '-f', 'python3', '--build_type', 'Debug']

        logger.info("编译命令: %s", ' '.join(cmd))
        logger.info("超时时间: %d 秒", timeout)

        try:
            result = run_command(cmd, capture=False, timeout=timeout)

            if result.returncode == 0:
                logger.info("=" * 80)
                logger.info("✓ Debug版本编译成功")
                logger.info("=" * 80)
                return True
            else:
                logger.error("=" * 80)
                logger.error("✗ Debug版本编译失败")
                logger.error("=" * 80)
                return False

        except subprocess.TimeoutExpired:
            logger.error("=" * 80)
            logger.error("✗ 编译超时（%d秒）", timeout)
            logger.error("=" * 80)
            logger.error("建议：")
            logger.error("  1. 增加超时时间")
            logger.error("  2. 检查系统资源")
            logger.error("  3. 查看编译日志")
            return False
        except Exception as e:
            logger.error("=" * 80)
            logger.error("✗ 编译异常: %s", e)
            logger.error("=" * 80)
            return False

    def find_wheel(self):
        """查找编译生成的wheel文件"""
        logger.info("查找wheel文件...")

        if not self.build_output.exists():
            logger.error("build_out目录不存在: %s", self.build_output)
            return None

        wheels = list(self.build_output.glob("pypto*.whl"))

        if not wheels:
            logger.error("未找到wheel文件")
            return None

        # 返回最新的wheel文件
        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        logger.info("找到wheel文件: %s", latest_wheel)
        return latest_wheel

    def get_build_info(self):
        """获取编译信息"""
        info = {
            'pypto_root': str(self.pypto_root),
            'build_script': str(self.build_script),
            'build_output': str(self.build_output),
            'wheel_file': None,
        }

        wheel = self.find_wheel()
        if wheel:
            info['wheel_file'] = str(wheel)

        return info


def main():
    parser = argparse.ArgumentParser(description='编译Debug版本PyPTO')
    parser.add_argument('-p', '--pypto-root', help='PyPTO项目根目录（默认：当前目录）')
    parser.add_argument('-t', '--timeout', type=int, default=1200, help='编译超时时间（秒，默认：1200）')
    parser.add_argument('--skip-check', action='store_true', help='跳过前提条件检查')

    args = parser.parse_args()

    # 创建编译器
    builder = DebugPyptoBuilder(args.pypto_root)

    # 检查前提条件
    if not args.skip_check:
        if not builder.check_prerequisites():
            logger.error("前提条件检查失败，退出")
            sys.exit(1)

    # 编译Debug版本
    success = builder.build_debug(args.timeout)

    if success:
        # 输出编译信息
        info = builder.get_build_info()
        logger.info("=" * 80)
        logger.info("编译信息")
        logger.info("=" * 80)
        logger.info("PyPTO根目录: %s", info['pypto_root'])
        logger.info("编译脚本: %s", info['build_script'])
        logger.info("输出目录: %s", info['build_output'])
        if info['wheel_file']:
            logger.info("Wheel文件: %s", info['wheel_file'])
        logger.info("=" * 80)

        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
