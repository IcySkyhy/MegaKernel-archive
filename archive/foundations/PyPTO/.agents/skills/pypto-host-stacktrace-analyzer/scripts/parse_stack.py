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
from dataclasses import dataclass
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging, validate_path

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class StackFrame:
    """堆栈帧数据结构"""

    index: int
    address: Optional[str] = None
    symbol: Optional[str] = None
    binary: Optional[str] = None
    offset: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None


class StackParser:
    """堆栈解析器"""

    def __init__(self):
        self.frames: List[StackFrame] = []

    @staticmethod
    def parse_python_traceback(text: str) -> List[StackFrame]:
        """解析 Python traceback 格式"""
        frames = []
        pattern = r'File "([^"]+)", line (\d+), in ([^\n]+)'

        for match in re.finditer(pattern, text):
            frame = StackFrame(
                index=len(frames), source_file=match.group(1), source_line=int(match.group(2)), symbol=match.group(3)
            )
            frames.append(frame)

        return frames

    @staticmethod
    def parse_cpp_stacktrace(text: str) -> List[StackFrame]:
        """解析 C++/C stack trace 格式（gdb、libunwind）"""
        frames = []

        # 格式 1: #0 0x7f1234567890 in function_name (args) at file.c:123
        pattern1 = r'#(\d+)\s+(0x[0-9a-f]+)\s+in\s+([^\s(]+)\s*(?:\([^)]*\))?\s*(?:at\s+([^:]+):(\d+))?'
        for match in re.finditer(pattern1, text):
            frame = StackFrame(
                index=int(match.group(1)),
                address=match.group(2),
                symbol=match.group(3),
                source_file=match.group(4),
                source_line=int(match.group(5)) if match.group(5) else None,
            )
            frames.append(frame)

        # 格式 2: 0x7f1234567890 function_name+0x1234 (/path/to/binary)
        if not frames:
            pattern2 = r'(0x[0-9a-f]+)\s+([^\s+]+)(?:\+([0-9a-fx]+))?\s*(?:\(([^)]+)\))?'
            for idx, match in enumerate(re.finditer(pattern2, text)):
                frame = StackFrame(
                    index=idx,
                    address=match.group(1),
                    symbol=match.group(2),
                    offset=match.group(3),
                    binary=match.group(4),
                )
                frames.append(frame)

        # 格式 3: libtile_fwk_interface.so(function+offset) [address]
        # 例如: libtile_fwk_interface.so(npu::tile_fwk::HostMachine::
        # CompileFunction(npu::tile_fwk::Function*) const+0x744) [0xfffef3d2c3bc]
        if not frames:
            pattern3 = r'([^\s(]+)\(([^+]+)\+([0-9a-fx]+)\)\s+\[([0-9a-f]+)\]'
            for idx, match in enumerate(re.finditer(pattern3, text)):
                frame = StackFrame(
                    index=idx,
                    binary=match.group(1),
                    symbol=match.group(2),
                    offset=match.group(3),
                    address=match.group(4),
                )
                frames.append(frame)

        return frames

    @staticmethod
    def parse_pypto_format(text: str) -> List[StackFrame]:
        """解析 PyPTO 特定格式"""
        frames = []

        # PyPTO 格式: [0] 0x7f1234567890 in function_name at file.py:123
        pattern = r'\[(\d+)\]\s+(0x[0-9a-f]+)\s+in\s+([^\s]+)\s+at\s+([^:]+):(\d+)'

        for match in re.finditer(pattern, text):
            frame = StackFrame(
                index=int(match.group(1)),
                address=match.group(2),
                symbol=match.group(3),
                source_file=match.group(4),
                source_line=int(match.group(5)),
            )
            frames.append(frame)

        return frames

    @staticmethod
    def parse_generic_format(text: str) -> List[StackFrame]:
        """解析通用格式（基于正则表达式）"""
        frames = []

        # 尝试提取所有可能的地址和符号
        # 格式: address symbol+offset (binary)
        pattern = r'(0x[0-9a-f]+)\s+([^\s+]+)(?:\+([0-9a-fx]+))?\s*(?:\(([^)]+)\))?'

        for idx, match in enumerate(re.finditer(pattern, text)):
            frame = StackFrame(
                index=idx, address=match.group(1), symbol=match.group(2), offset=match.group(3), binary=match.group(4)
            )
            frames.append(frame)

        return frames

    @staticmethod
    def parse(text: str) -> Tuple[List[StackFrame], str]:
        """自动检测并解析堆栈信息，返回帧列表和检测到的格式"""
        # 尝试 Python traceback
        python_frames = StackParser.parse_python_traceback(text)
        if python_frames:
            logger.info("检测到 Python traceback 格式")
            return python_frames, "python"

        # 尝试 PyPTO 格式
        pypto_frames = StackParser.parse_pypto_format(text)
        if pypto_frames:
            logger.info("检测到 PyPTO 格式")
            return pypto_frames, "pypto"

        # 尝试 C++/C stack trace
        cpp_frames = StackParser.parse_cpp_stacktrace(text)
        if cpp_frames:
            logger.info("检测到 C++/C stack trace 格式")
            return cpp_frames, "cpp"

        # 使用通用格式
        generic_frames = StackParser.parse_generic_format(text)
        if generic_frames:
            logger.info("使用通用格式解析")
            return generic_frames, "generic"

        logger.warning("未能识别堆栈格式")
        return [], "unknown"

    def get_binary_names(self) -> List[str]:
        """获取所有二进制文件名"""
        binaries = set()
        for frame in self.frames:
            if frame.binary:
                binaries.add(frame.binary)
        return list(binaries)

    def get_addresses(self) -> List[str]:
        """获取所有地址"""
        addresses = []
        for frame in self.frames:
            if frame.address:
                addresses.append(frame.address)
        return addresses

    def get_offsets(self) -> List[str]:
        """获取所有偏移量"""
        offsets = []
        for frame in self.frames:
            if frame.offset:
                offsets.append(frame.offset)
        return offsets


def main():
    parser = argparse.ArgumentParser(description='解析堆栈信息')
    parser.add_argument('input', help='堆栈文本或文件路径')
    parser.add_argument('-f', '--file', action='store_true', help='输入是文件路径')
    parser.add_argument('-j', '--json', action='store_true', help='输出 JSON 格式')

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

    # 解析堆栈
    stack_parser = StackParser()
    frames, format_type = stack_parser.parse(text)

    if not frames:
        logger.error("未能解析出堆栈信息")
        sys.exit(1)

    # 输出结果
    if args.json:
        import json

        result = {'format': format_type, 'frames': []}
        for frame in frames:
            frame_dict = {'index': frame.index}
            if frame.address:
                frame_dict['address'] = frame.address
            if frame.symbol:
                frame_dict['symbol'] = frame.symbol
            if frame.binary:
                frame_dict['binary'] = frame.binary
            if frame.offset:
                frame_dict['offset'] = frame.offset
            if frame.source_file:
                frame_dict['source_file'] = frame.source_file
                if frame.source_line:
                    frame_dict['source_line'] = frame.source_line
            result['frames'].append(frame_dict)

        logger.info("%s", json.dumps(result, indent=2))
    else:
        logger.info("=" * 80)
        logger.info("解析结果")
        logger.info("=" * 80)

        for frame in frames:
            logger.info("帧 #%d:", frame.index)
            if frame.address:
                logger.info("  地址: %s", frame.address)
            if frame.symbol:
                logger.info("  符号: %s", frame.symbol)
            if frame.binary:
                logger.info("  二进制: %s", frame.binary)
            if frame.offset:
                logger.info("  偏移: %s", frame.offset)
            if frame.source_file:
                logger.info("  源码: %s:%s", frame.source_file, frame.source_line)
            logger.info("")

        # 输出摘要
        binaries = stack_parser.get_binary_names()
        addresses = stack_parser.get_addresses()
        offsets = stack_parser.get_offsets()

        logger.info("=" * 80)
        logger.info("摘要")
        logger.info("=" * 80)
        logger.info("格式: %s", format_type)
        logger.info("总帧数: %d", len(frames))
        if binaries:
            logger.info("二进制文件: %s", ', '.join(binaries))
        if addresses:
            logger.info("地址数量: %d", len(addresses))
        if offsets:
            logger.info("偏移量数量: %d", len(offsets))


if __name__ == '__main__':
    main()
