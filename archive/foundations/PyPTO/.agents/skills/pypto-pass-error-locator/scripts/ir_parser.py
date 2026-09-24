#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO IR 文本格式解析器

本模块提供 IR 文本文件的解析能力，支持解析 .tifwkgr 格式的 IR 文件。
"""

from dataclasses import dataclass, field
from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple

ERROR_CODES = {
    "SUCCESS": 0,
    "FILE_NOT_FOUND": 1,
    "PARSE_ERROR": 2,
    "INVALID_HEADER": 3,
    "DUPLICATE_RAWTENSOR": 4,
    "DUPLICATE_OPERATION": 5,
    "SHAPE_MISMATCH": 6,
    "DATAFLOW_ERROR": 7,
}


@dataclass
class RawTensor:
    """RAWTENSOR 信息"""

    index: int
    shape: List[int]
    data_type: str
    raw_id: int
    name: str
    line: int = 0


@dataclass
class Cast:
    """INCAST/OUTCAST 信息"""

    cast_type: str
    index: int
    shape: List[int]
    valid_shape: List[int]
    logic_tensor_id: int
    raw_tensor_id: int
    subgraph_id: int
    slot: int
    line: int = 0


@dataclass
class Operation:
    """操作节点信息"""

    op_id: int
    opcode: str
    output_shape: List[int]
    output_valid_shape: List[int]
    output_logic_id: int
    output_raw_id: int
    subgraph_id: int
    read_mem_type: str
    write_mem_type: str
    graph_id: int
    scope_id: int
    input_tensors: List[Dict[str, Any]]
    attributes: Dict[str, Any]
    offset: List[int] = field(default_factory=list)
    dynoffset: List[int] = field(default_factory=list)
    dynvalidshape: List[int] = field(default_factory=list)
    offset_from: bool = False
    offset_direction: str = ""  # "from" or "to"
    dynoffset_direction: str = ""  # "from" or "to"
    dynvalidshape_direction: str = ""  # "from" or "to"
    line: int = 0


@dataclass
class IRHeader:
    """IR 文件头信息"""

    function_name: str
    function_magic: int
    hash_value: int
    function_type: str
    graph_type: str


@dataclass
class IRFile:
    """IR 文件完整信息"""

    header: Optional[IRHeader]
    rawtensors: Dict[int, RawTensor]
    incasts: Dict[int, Cast]
    outcasts: Dict[int, Cast]
    operations: Dict[int, Operation]
    file_path: str


class IRParser:
    """IR 文本格式解析器"""

    def __init__(self):
        self.header_pattern = re.compile(r'^Function\s+(\S+)\[(\d+)\]\s+(-?\d+)\s+(\w+)\s+(\w+)\s+\{$')
        self.rawtensor_pattern = re.compile(r'RAWTENSOR\[\s*(\d+)\]\s+<([^>]+)>\s+@(\d+)"?([^"]*)"?')
        self.cast_pattern = re.compile(
            r'(IN|OUT)CAST\[\s*(\d+)\]\s+<([^>]+)>\s+%(\d+)@(\d+)#(\([^)]+\))\s+(from|to)Slot\[(\d+)\]'
        )
        self.operation_pattern = re.compile(
            r'<([^>]+)>\s+%(\d+)@(\d+)#(\([^)]+\))(\S+)::(\S+)\s+=\s+!(\d+)\s+(\S+)\s*'
            r'\(g:(-?\d+),\s*s:(-?\d+)\)(.*)'
        )

    @staticmethod
    def _parse_shape_and_dtype(shape_str: str) -> Tuple[List[int], str]:
        """解析 shape 和数据类型"""
        parts = shape_str.strip().split()
        shape = []
        data_type = ""

        for part in parts:
            if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                shape.append(int(part))
            elif part.startswith('DT_'):
                data_type = part

        return shape, data_type

    @staticmethod
    def _parse_array(array_str: str) -> List[int]:
        """解析数组字符串 [x, y, z, ...] 或 offset:[x, y, z, ...]"""
        start_idx = array_str.find('[')
        end_idx = array_str.rfind(']')

        if start_idx == -1 or end_idx == -1:
            return []

        array_content = array_str[start_idx + 1:end_idx].strip()
        if not array_content:
            return []

        values = []
        for item in array_content.split(','):
            item = item.strip()
            if item:
                try:
                    values.append(int(item))
                except ValueError:
                    pass

        return values

    @staticmethod
    def _skip_array_tokens(tokens: List[str], start_idx: int) -> int:
        """跳过数组tokens，返回新的索引"""
        if start_idx >= len(tokens):
            return start_idx

        i = start_idx
        while i < len(tokens):
            if tokens[i].endswith(']'):
                return i + 1
            i += 1

        return i

    def parse_file(self, file_path: str) -> Tuple[Optional[IRFile], Optional[str]]:
        """解析 IR 文件

        Args:
            file_path: IR 文件路径

        Returns:
            (IRFile, error_message): 成功时返回 (IRFile, None)，失败时返回 (None, error_message)
        """
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return None, f"Failed to read file: {str(e)}"

        ir_file = IRFile(header=None, rawtensors={}, incasts={}, outcasts={}, operations={}, file_path=file_path)

        in_function = False

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line or line.startswith('/*') or line.startswith('*/'):
                continue

            if line == '-------------':
                continue

            if line.startswith('Function'):
                header, error = self._parse_header(line)
                if error:
                    return None, error
                ir_file.header = header
                in_function = True
                continue

            if line == '}':
                in_function = False
                continue

            if not in_function:
                continue

            if line.startswith('RAWTENSOR'):
                rawtensor, error = self._parse_rawtensor(line, line_num)
                if error:
                    return None, error
                if rawtensor:
                    ir_file.rawtensors[rawtensor.index] = rawtensor

            elif line.startswith('INCAST'):
                cast, error = self._parse_cast(line, line_num, 'IN')
                if error:
                    return None, error
                if cast:
                    ir_file.incasts[cast.index] = cast

            elif line.startswith('OUTCAST'):
                cast, error = self._parse_cast(line, line_num, 'OUT')
                if error:
                    return None, error
                if cast:
                    ir_file.outcasts[cast.index] = cast

            elif '=' in line and '!' in line:
                operation, error = self._parse_operation(line, line_num)
                if error:
                    return None, error
                if operation:
                    ir_file.operations[operation.op_id] = operation

        return ir_file, None

    def _parse_header(self, line: str) -> Tuple[Optional[IRHeader], Optional[str]]:
        """解析文件头"""
        match = self.header_pattern.match(line)
        if not match:
            return None, f"Invalid header format: {line}"

        return IRHeader(
            function_name=match.group(1),
            function_magic=int(match.group(2)),
            hash_value=int(match.group(3)),
            function_type=match.group(4),
            graph_type=match.group(5),
        ), None

    def _parse_rawtensor(self, line: str, line_num: int) -> Tuple[Optional[RawTensor], Optional[str]]:
        """解析 RAWTENSOR"""
        match = self.rawtensor_pattern.match(line)
        if not match:
            return None, f"Invalid RAWTENSOR format: {line}"

        shape_str = match.group(2)
        shape, data_type = self._parse_shape_and_dtype(shape_str)

        return RawTensor(
            index=int(match.group(1)),
            shape=shape,
            data_type=data_type,
            raw_id=int(match.group(3)),
            name=match.group(4).strip(),
            line=line_num,
        ), None

    def _parse_cast(self, line: str, line_num: int, cast_type: str) -> Tuple[Optional[Cast], Optional[str]]:
        """解析 INCAST/OUTCAST"""
        match = self.cast_pattern.match(line)
        if not match:
            return None, f"Invalid CAST format: {line}"

        shape_str = match.group(3)
        shape, _ = self._parse_shape_and_dtype(shape_str)

        valid_shape_str = match.group(3).split('/')[1].strip() if '/' in match.group(3) else shape_str
        valid_shape, _ = self._parse_shape_and_dtype(valid_shape_str)

        return Cast(
            cast_type=cast_type,
            index=int(match.group(2)),
            shape=shape,
            valid_shape=valid_shape,
            logic_tensor_id=int(match.group(4)),
            raw_tensor_id=int(match.group(5)),
            subgraph_id=int(match.group(6).strip('()')),
            slot=int(match.group(8)),
            line=line_num,
        ), None

    def _parse_operation(self, line: str, line_num: int) -> Tuple[Optional[Operation], Optional[str]]:
        """解析操作节点"""
        match = self.operation_pattern.match(line)
        if not match:
            return None, f"Invalid operation format: {line}"

        output_shape_str = match.group(1)
        output_shape, _ = self._parse_shape_and_dtype(output_shape_str.split('/')[0].strip())
        valid_shape_part = output_shape_str.split('/')[1].strip() if '/' in output_shape_str else output_shape_str
        output_valid_shape, _ = self._parse_shape_and_dtype(valid_shape_part)

        input_tensors, attributes, offset_info = self._parse_operation_params(match.group(11))

        return Operation(
            op_id=int(match.group(7)),
            opcode=match.group(8),
            output_shape=output_shape,
            output_valid_shape=output_valid_shape,
            output_logic_id=int(match.group(2)),
            output_raw_id=int(match.group(3)),
            subgraph_id=int(match.group(4).strip('()')),
            read_mem_type=match.group(5),
            write_mem_type=match.group(6),
            graph_id=int(match.group(9)),
            scope_id=int(match.group(10)),
            input_tensors=input_tensors,
            attributes=attributes,
            offset=offset_info.get('offset', []),
            dynoffset=offset_info.get('dynoffset', []),
            dynvalidshape=offset_info.get('dynvalidshape', []),
            offset_from=offset_info.get('offset_from', False),
            offset_direction=offset_info.get('offset_direction', ''),
            dynoffset_direction=offset_info.get('dynoffset_direction', ''),
            dynvalidshape_direction=offset_info.get('dynvalidshape_direction', ''),
            line=line_num,
        ), None

    def _parse_array_from_tokens(self, tokens: List[str], start_idx: int) -> List[int]:
        """从tokens列表中解析数组，处理 offset:[...] 格式"""
        if start_idx >= len(tokens):
            return []

        array_content = []
        i = start_idx

        if '[' not in tokens[i]:
            return []

        while i < len(tokens):
            token = tokens[i]
            array_content.append(token)
            i += 1

        array_str = ' '.join(array_content)
        return self._parse_array(array_str)

    def _parse_operation_params(self, params_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """解析操作参数和属性"""

        input_tensors = []
        attributes = {}
        offset_info = {
            'offset': [],
            'dynoffset': [],
            'dynvalidshape': [],
            'offset_from': False,
            'offset_direction': '',
            'dynoffset_direction': '',
            'dynvalidshape_direction': '',
        }

        tokens = params_str.strip().split()
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.startswith('%'):
                tensor_info = {
                    'logic_id': int(token[1:].split('@')[0]),
                    'raw_id': int(token.split('@')[1].split('#')[0]),
                    'subgraph_id': int(token.split('#')[1].strip('()').split(')')[0]),
                    'mem_type': token.split(')')[1] if ')' in token else '',
                }
                input_tensors.append(tensor_info)
                i += 1

            elif token.startswith('#'):
                attr_name = token[1:].split('{')[0]
                attr_value = token.split('{')[1].rstrip('}')
                attributes[attr_name] = attr_value
                i += 1

            elif token == 'from':
                offset_info['offset_from'] = True
                i += 1

            elif token == 'to':
                offset_info['offset_from'] = False
                i += 1

            elif token.startswith('offset:['):
                offset_info['offset'] = self._parse_array_from_tokens(tokens, i)
                offset_info['offset_direction'] = 'from' if offset_info['offset_from'] else 'to'
                i = self._skip_array_tokens(tokens, i)

            elif token.startswith('dynoffset:['):
                offset_info['dynoffset'] = self._parse_array_from_tokens(tokens, i)
                offset_info['dynoffset_direction'] = 'from' if offset_info['offset_from'] else 'to'
                i = self._skip_array_tokens(tokens, i)

            elif token.startswith('dynvalidshape:['):
                offset_info['dynvalidshape'] = self._parse_array_from_tokens(tokens, i)
                offset_info['dynvalidshape_direction'] = 'from' if offset_info['offset_from'] else 'to'
                i = self._skip_array_tokens(tokens, i)

            else:
                i += 1

        return input_tensors, attributes, offset_info


def create_response(
    status: str, error_code: int, error_message: str, data: Dict[str, Any], file_path: str = ""
) -> Dict[str, Any]:
    """创建标准化响应"""
    response = {
        "status": status,
        "error_code": error_code,
        "error_message": error_message,
        "data": data,
        "metadata": {"file": file_path, "timestamp": datetime.utcnow().isoformat() + "Z", "version": "1.0.0"},
    }
    return response
