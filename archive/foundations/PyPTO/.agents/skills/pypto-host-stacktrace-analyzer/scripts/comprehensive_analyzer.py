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
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging, validate_path

setup_logging()
logger = logging.getLogger(__name__)


class ComprehensiveAnalyzer:
    """综合堆栈分析器"""

    def __init__(self):
        self.error_info = None
        self.python_frames = []
        self.cpp_frames = []
        self.binary_files = {}

    def analyze(self, text: str):
        """综合分析堆栈信息"""
        # 1. 提取错误信息
        self._extract_error_info(text)

        # 2. 解析 Python traceback
        self._parse_python_traceback(text)

        # 3. 解析 C++ stack trace
        self._parse_cpp_trace(text)

        # 4. 查找二进制文件
        self._find_binaries()

        # 5. 分析符号
        self._analyze_symbols()

    def generate_report(self) -> str:
        """生成分析报告"""
        report = []

        # 标题
        report.append("=" * 80)
        report.append("综合堆栈分析报告")
        report.append("=" * 80)
        report.append("")

        # 错误信息
        if self.error_info:
            report.append("## 错误信息")
            report.append("")
            if 'errcode' in self.error_info:
                report.append(f"错误码: {self.error_info['errcode']}")
            if 'location' in self.error_info:
                loc = self.error_info['location']
                report.append(f"位置: {loc['file']}:{loc['line']}")
                report.append(f"函数: {loc['function']}")
            if 'message' in self.error_info:
                report.append(f"消息: {self.error_info['message']}")
            report.append("")

        # Python traceback
        if self.python_frames:
            report.append("## Python Traceback")
            report.append("")
            report.append(f"总帧数: {len(self.python_frames)}")
            report.append("")

            for idx, frame in enumerate(self.python_frames):
                marker = " ⚠️ 错误触发点" if idx == len(self.python_frames) - 1 else ""
                report.append(f"### 帧 #{idx}{marker}")
                report.append(f"- 文件: {frame['file']}")
                report.append(f"- 行号: {frame['line']}")
                report.append(f"- 函数: {frame['function']}")
                report.append("")

        # C++ stack trace
        if self.cpp_frames:
            report.append("## C++ Stack Trace")
            report.append("")
            report.append(f"总帧数: {len(self.cpp_frames)}")
            report.append("")

            for idx, frame in enumerate(self.cpp_frames):
                marker = " ⚠️ 错误发生点" if idx == 0 else ""
                report.append(f"### 帧 #{idx}{marker}")
                report.append(f"- 二进制: {frame['binary']}")

                if 'symbol' in frame:
                    report.append(f"- 符号: {frame['symbol']}")
                    if 'demangled_symbol' in frame:
                        report.append(f"- 反混淆: {frame['demangled_symbol']}")

                if 'offset' in frame:
                    report.append(f"- 偏移: {frame['offset']}")
                if 'address' in frame:
                    report.append(f"- 地址: {frame['address']}")
                if 'source_location' in frame:
                    report.append(f"- 源码: {frame['source_location']}")  # ⭐ 新增
                report.append("")

        # 二进制文件
        if self.binary_files:
            report.append("## 二进制文件")
            report.append("")
            for name, path in self.binary_files.items():
                report.append(f"- {name}: {path}")
            report.append("")

        # 源码映射失败提示
        if self.cpp_frames:
            all_unknown = all(
                frame.get('source_location', '').strip() in ('??:0', '?:0', '')
                for frame in self.cpp_frames
                if 'source_location' in frame
            )
            if all_unknown:
                report.append("## ⚠️ 源码映射提示")
                report.append("")
                report.append(
                    "所有 C++ 帧的源码位置均为 `??:0`，这通常表示当前安装的 PyPTO 为 **Release 构建**，不包含调试符号。"
                )
                report.append("")
                report.append("**解决方案**：")
                report.append("1. 按步骤 2 编译安装 Debug 版本 PyPTO（包含调试符号），然后重新运行分析")
                report.append("2. 或使用 `addr2line` 手动定位（需要二进制文件路径）：")
                report.append("   ```bash")
                report.append("   addr2line -e <binary_path> -f -C <address>")
                report.append("   ```")
                report.append("3. 或使用 `objdump` 反汇编查看上下文：")
                report.append("   ```bash")
                report.append("   objdump -d -C <binary_path> | grep -A 20 '<symbol>'")
                report.append("   ```")
                report.append("")

        return '\n'.join(report)

    def _extract_error_info(self, text: str):
        """提取错误信息"""
        import re

        self.error_info = {}

        # 提取错误码
        errcode_match = re.search(r'Errcode:\s*([A-F0-9]+)', text)
        if errcode_match:
            self.error_info['errcode'] = errcode_match.group(1)

        # 提取错误位置
        location_match = re.search(r'file\s+(\S+),\s+line\s+(\d+),\s+func\s+(\S+)', text)
        if not location_match:
            location_match = re.search(r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\S+))?', text)
        if location_match:
            self.error_info['location'] = {
                'file': location_match.group(1),
                'line': location_match.group(2),
                'function': location_match.group(3),
            }

        # 提取错误消息
        message_match = re.search(r'Error:\s*(.+?)(?:\n|$)', text)
        if message_match:
            self.error_info['message'] = message_match.group(1).strip()

    def _parse_python_traceback(self, text: str):
        """解析 Python traceback"""
        import re

        pattern = r'File "([^"]+)", line (\d+), in ([^\n]+)'
        for match in re.finditer(pattern, text):
            frame = {
                'index': len(self.python_frames),
                'file': match.group(1),
                'line': int(match.group(2)),
                'function': match.group(3),
            }
            self.python_frames.append(frame)

    def _parse_cpp_trace(self, text: str):
        """解析 C++ stack trace"""
        import re

        # 格式: libtile_fwk_interface.so(function+offset) [address]
        # 例如: libtile_fwk_interface.so(npu::tile_fwk::HostMachine::
        # CompileFunction(npu::tile_fwk::Function*) const+0x744) [0xfffef3d2c3bc]
        pattern = r'([^\s\(\)]+)\(([^+]+)\+([0-9a-fx]+)\)\s+\[0x([0-9a-f]+)\]'
        for match in re.finditer(pattern, text):
            frame = {
                'index': len(self.cpp_frames),
                'binary': match.group(1),
                'symbol': match.group(2),
                'offset': match.group(3),
                'address': '0x' + match.group(4),
            }
            self.cpp_frames.append(frame)

    def _find_binaries(self):
        """查找二进制文件"""
        from auto_find_binary import BinaryFinder

        finder = BinaryFinder()

        # 从 C++ 帧中提取二进制文件名
        binary_names = set()
        for frame in self.cpp_frames:
            if 'binary' in frame:
                binary_names.add(frame['binary'])

        # 查找每个二进制文件
        for binary_name in binary_names:
            binary_path = finder.find_binary(binary_name)
            if binary_path:
                self.binary_files[binary_name] = str(binary_path)
                logger.info("找到二进制文件: %s -> %s", binary_name, binary_path)

    def _analyze_symbols(self):
        """分析符号"""
        for frame in self.cpp_frames:
            if 'symbol' in frame and 'binary' in frame:
                binary_name = frame['binary']
                if binary_name in self.binary_files:
                    binary_path = self.binary_files[binary_name]

                    # 尝试反混淆符号
                    try:
                        cmd = ['c++filt', frame['symbol']]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            frame['demangled_symbol'] = result.stdout.strip()
                    except Exception as e:
                        logger.error(
                            "符号反混淆失败: symbol=%s, binary=%s, 错误: %s",
                            frame.get('symbol'),
                            binary_path,
                            e,
                        )
                        continue

                    # 尝试定位源码行号
                    if 'address' in frame:
                        try:
                            cmd = ['addr2line', '-e', binary_path, '-f', '-C', frame['address']]
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                            if result.returncode == 0:
                                lines = result.stdout.strip().split('\n')
                                if len(lines) >= 2:
                                    frame['source_location'] = lines[1]
                        except Exception as e:
                            logger.warning("源码行定位失败: %s, 错误: %s", frame['address'], e)


def main():
    parser = argparse.ArgumentParser(description='综合堆栈分析')
    parser.add_argument('input', help='堆栈文本或文件路径')
    parser.add_argument('-f', '--file', action='store_true', help='输入是文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')

    args = parser.parse_args()

    # 读取输入
    if args.file:
        try:
            path = validate_path(args.input)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error("读取文件失败: %s", e)
            sys.exit(1)
    else:
        text = args.input

    # 分析堆栈
    analyzer = ComprehensiveAnalyzer()
    analyzer.analyze(text)

    # 生成报告
    report = analyzer.generate_report()
    logger.info("%s", report)

    # 保存到文件
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info("报告已保存到: %s", args.output)
        except Exception as e:
            logger.error("保存文件失败: %s", e)


if __name__ == '__main__':
    main()
