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
PyPTO OP 信息查询工具

本工具用于从 IR 文件中快速查询指定 OP 的详细信息，包括：
- OP magic 和 opcode
- 输入输出 tensor 的 shape、validshape
- 内存类型
- 属性信息
- 上下文信息（graph_id、scope_id）

使用场景：
- 精度工具报错时，快速查看 OP 信息
- 与精度工具报错信息进行比对
- 定位 Pass 修改的 OP 信息
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_parser import IRParser  # noqa: E402


class OpInfoExtractor:
    """OP 信息提取器"""

    def __init__(self):
        self.parser = IRParser()

    @staticmethod
    def format_as_text(op_info: Dict[str, Any]) -> str:
        """格式化为文本输出

        Args:
            op_info: OP 信息字典

        Returns:
            格式化后的文本字符串
        """
        lines = []

        lines.append("=== Operation Information ===")
        lines.append(f"OP Magic: {op_info['op_magic']}")
        lines.append(f"Opcode: {op_info['opcode']}")
        lines.append(f"Line: {op_info['line']}")
        lines.append("")

        lines.append("=== Output Tensor ===")
        output = op_info['output']
        lines.append(f"Logic Tensor ID: {output['logic_id']}")
        lines.append(f"Raw Tensor ID: {output['raw_id']}")
        lines.append(f"Shape: {output['shape']}")
        lines.append(f"Valid Shape: {output['valid_shape']}")
        lines.append(f"Data Type: {output['data_type']}")
        lines.append(f"Memory Type: {output['memory_type']['read']}::{output['memory_type']['write']}")
        lines.append(f"Subgraph ID: {output['subgraph_id']}")
        lines.append("")

        lines.append("=== Input Tensors ===")
        for input_tensor in op_info['inputs']:
            logic_id = input_tensor['logic_id']
            raw_id = input_tensor['raw_id']
            lines.append(f"[{input_tensor['index']}] Logic ID: {logic_id}, Raw ID: {raw_id}")
            if 'shape' in input_tensor:
                lines.append(f"    Shape: {input_tensor['shape']}")
                lines.append(f"    Valid Shape: {input_tensor['valid_shape']}")
                lines.append(f"    Data Type: {input_tensor['data_type']}")
            lines.append(f"    Memory Type: {input_tensor['memory_type']}")
            lines.append(f"    Subgraph ID: {input_tensor['subgraph_id']}")
        lines.append("")

        lines.append("=== Attributes ===")
        if op_info['attributes']:
            for attr_name, attr_value in op_info['attributes'].items():
                lines.append(f"{attr_name}: {attr_value}")
        else:
            lines.append("(none)")
        lines.append("")

        lines.append("=== Offset Information ===")
        offset_info = op_info['offset_info']
        if offset_info['offset'] or offset_info['dynoffset'] or offset_info['dynvalidshape']:
            if offset_info['offset'] and offset_info['offset_direction']:
                lines.append(f"{offset_info['offset_direction']} offset: {offset_info['offset']}")
            if offset_info['dynoffset'] and offset_info['dynoffset_direction']:
                lines.append(f"{offset_info['dynoffset_direction']} dynoffset: {offset_info['dynoffset']}")
            if offset_info['dynvalidshape'] and offset_info['dynvalidshape_direction']:
                lines.append(f"{offset_info['dynvalidshape_direction']} dynvalidshape: {offset_info['dynvalidshape']}")
        else:
            lines.append("(none)")
        lines.append("")

        lines.append("=== Context ===")
        context = op_info['context']
        lines.append(f"Graph ID: {context['graph_id']}")
        lines.append(f"Scope ID: {context['scope_id']}")

        return "\n".join(lines)

    @staticmethod
    def format_as_json(op_info: Dict[str, Any]) -> str:
        """格式化为 JSON 输出

        Args:
            op_info: OP 信息字典

        Returns:
            JSON 字符串
        """
        return json.dumps(op_info, indent=2)

    @staticmethod
    def format_ops_list_as_text(ops_list: List[Dict[str, Any]]) -> str:
        """格式化 OP 列表为文本输出

        Args:
            ops_list: OP 列表

        Returns:
            格式化后的文本字符串
        """
        lines = []
        lines.append(f"Total Operations: {len(ops_list)}")
        lines.append("")

        for op in ops_list:
            lines.append(f"OP Magic: {op['op_magic']}, Opcode: {op['opcode']}, Line: {op['line']}")
            lines.append(f"  Input Count: {op['input_count']}, Output Shape: {op['output_shape']}")

        return "\n".join(lines)

    @staticmethod
    def format_ops_list_as_json(ops_list: List[Dict[str, Any]]) -> str:
        """格式化 OP 列表为 JSON 输出

        Args:
            ops_list: OP 列表

        Returns:
            JSON 字符串
        """
        return json.dumps(ops_list, indent=2)

    @staticmethod
    def _extract_dtype_from_shape(shape: List[int]) -> str:
        """从 shape 中提取数据类型"""
        if not shape:
            return ""

        for item in shape:
            if isinstance(item, str) and item.startswith('DT_'):
                return item

        return ""

    def get_op_info(self, ir_file_path: str, op_magic: int) -> Optional[Dict[str, Any]]:
        """获取指定 OP 的信息

        Args:
            ir_file_path: IR 文件路径
            op_magic: OP 的 magic ID

        Returns:
            OP 信息字典，如果 OP 不存在则返回 None
        """
        ir_file, error = self.parser.parse_file(ir_file_path)
        if error:
            return None

        if op_magic not in ir_file.operations:
            return None

        op = ir_file.operations[op_magic]

        op_info = {
            "op_magic": op.op_id,
            "opcode": op.opcode,
            "line": op.line,
            "output": {
                "logic_id": op.output_logic_id,
                "raw_id": op.output_raw_id,
                "shape": op.output_shape,
                "valid_shape": op.output_valid_shape,
                "data_type": self._extract_dtype_from_shape(op.output_shape),
                "memory_type": {"read": op.read_mem_type, "write": op.write_mem_type},
                "subgraph_id": op.subgraph_id,
            },
            "inputs": [],
            "attributes": op.attributes,
            "offset_info": {
                "offset": op.offset,
                "dynoffset": op.dynoffset,
                "dynvalidshape": op.dynvalidshape,
                "offset_direction": op.offset_direction,
                "dynoffset_direction": op.dynoffset_direction,
                "dynvalidshape_direction": op.dynvalidshape_direction,
            },
            "context": {"graph_id": op.graph_id, "scope_id": op.scope_id},
        }

        for idx, input_tensor in enumerate(op.input_tensors):
            input_info = {
                "index": idx,
                "logic_id": input_tensor['logic_id'],
                "raw_id": input_tensor['raw_id'],
                "subgraph_id": input_tensor['subgraph_id'],
                "memory_type": input_tensor.get('mem_type', ''),
            }

            tensor_ref = self._find_tensor_info(ir_file, input_tensor['logic_id'])
            if tensor_ref:
                input_info["shape"] = tensor_ref.get("shape", [])
                input_info["valid_shape"] = tensor_ref.get("valid_shape", [])
                input_info["data_type"] = tensor_ref.get("data_type", "")

            op_info["inputs"].append(input_info)

        return op_info

    def list_all_ops(self, ir_file_path: str) -> List[Dict[str, Any]]:
        """列出所有 OP 的基本信息

        Args:
            ir_file_path: IR 文件路径

        Returns:
            OP 信息列表
        """
        ir_file, error = self.parser.parse_file(ir_file_path)
        if error:
            return []

        ops_list = []
        for _, op in sorted(ir_file.operations.items()):
            op_summary = {
                "op_magic": op.op_id,
                "opcode": op.opcode,
                "line": op.line,
                "input_count": len(op.input_tensors),
                "output_shape": op.output_shape,
            }
            ops_list.append(op_summary)

        return ops_list

    def _find_tensor_info(self, ir_file, logic_id: int) -> Optional[Dict[str, Any]]:
        """查找 tensor 信息"""
        for cast in ir_file.incasts.values():
            if cast.logic_tensor_id == logic_id:
                return {"shape": cast.shape, "valid_shape": cast.valid_shape, "data_type": ""}

        for cast in ir_file.outcasts.values():
            if cast.logic_tensor_id == logic_id:
                return {"shape": cast.shape, "valid_shape": cast.valid_shape, "data_type": ""}

        for op in ir_file.operations.values():
            if op.output_logic_id == logic_id:
                return {
                    "shape": op.output_shape,
                    "valid_shape": op.output_valid_shape,
                    "data_type": self._extract_dtype_from_shape(op.output_shape),
                }

        return None


def main():
    parser = argparse.ArgumentParser(
        description='Query operation information from IR file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query specific operation
  python3 get_op_info.py --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr --op-magic 10003

  # List all operations
  python3 get_op_info.py --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr --list-ops

  # Query operation with JSON output
  python3 get_op_info.py --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic 10003 --format json
        """,
    )

    parser.add_argument('--ir-file', required=True, help='Path to IR file')
    parser.add_argument('--op-magic', type=int, help='Operation magic ID to query')
    parser.add_argument('--list-ops', action='store_true', help='List all operations')
    parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format (default: text)')

    args = parser.parse_args()

    if not args.op_magic and not args.list_ops:
        parser.error("Either --op-magic or --list-ops must be specified")

    extractor = OpInfoExtractor()

    if args.list_ops:
        ops_list = extractor.list_all_ops(args.ir_file)
        if not ops_list:
            logger.error(f"Failed to parse IR file or no operations found: {args.ir_file}")
            sys.exit(1)

        if args.format == 'json':
            logger.info(extractor.format_ops_list_as_json(ops_list))
        else:
            logger.info(extractor.format_ops_list_as_text(ops_list))

    else:
        op_info = extractor.get_op_info(args.ir_file, args.op_magic)
        if op_info is None:
            logger.error(f"Operation with magic {args.op_magic} not found in IR file: {args.ir_file}")
            sys.exit(1)

        if args.format == 'json':
            logger.info(extractor.format_as_json(op_info))
        else:
            logger.info(extractor.format_as_text(op_info))


if __name__ == '__main__':
    main()
