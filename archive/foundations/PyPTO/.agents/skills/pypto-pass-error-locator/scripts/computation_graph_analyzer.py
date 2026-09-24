#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO 计算图 JSON 分析工具

本工具提供完整的计算图分析能力，包括：
- 加载和解析计算图JSON文件
- 获取各类数据（Tensor、Operation、RawTensor、Incast/Outcast等）
- 追踪数据流转
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")


class GraphType(Enum):
    """计算图类型"""

    TENSOR_GRAPH = 1
    TILE_GRAPH = 2
    BLOCK_GRAPH = 3
    EXECUTE_GRAPH = 4

    @classmethod
    def from_value(cls, value: int) -> Optional['GraphType']:
        return cls(value) if value in cls._value2member_map_ else None

    @classmethod
    def to_string(cls, value: int) -> str:
        mapping = {1: 'TensorGraph', 2: 'TileGraph', 3: 'BlockGraph', 4: 'ExecuteGraph'}
        return mapping.get(value, 'Unknown')


class MemoryType(Enum):
    """内存层级类型"""

    MEM_UB = 0
    MEM_L1 = 1
    MEM_L0A = 2
    MEM_L0B = 3
    MEM_L0C = 4
    MEM_DEVICE_DDR = 15

    @classmethod
    def to_string(cls, value: int) -> str:
        mapping = {
            0: 'MEM_UB (Unified Buffer)',
            1: 'MEM_L1 (L1 Cache)',
            2: 'MEM_L0A (L0A Cache)',
            3: 'MEM_L0B (L0B Cache)',
            4: 'MEM_L0C (L0C Cache)',
            15: 'MEM_DEVICE_DDR (Global Memory)',
        }
        return mapping.get(value, f'Unknown ({value})')


@dataclass
class TensorInfo:
    """Tensor信息"""

    magic: int
    shape: List[int]
    validshape: List[int]
    dynvalidshape: List[List[int]]
    offset: List[int]
    rawtensor: int
    nodetype: int
    mem_id: int
    mem_range: List[int]
    mem_type_asis: int
    mem_type_tobe: int
    subgraphid: int
    subgraph_boundary: bool
    life_range: List[int]
    kind: int
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def get_memory_type_str(self) -> Tuple[str, str]:
        """获取内存类型字符串"""
        return (MemoryType.to_string(self.mem_type_asis), MemoryType.to_string(self.mem_type_tobe))

    def is_input(self) -> bool:
        """是否为输入Tensor"""
        return self.nodetype == 1

    def is_output(self) -> bool:
        """是否为输出Tensor"""
        return self.nodetype == 2


@dataclass
class OperationInfo:
    """Operation信息"""

    opmagic: int
    opcode: str
    ioperands: List[int]
    ooperands: List[int]
    attr: List[Any]
    subgraphid: int
    latency: int
    kind: int
    tile: Optional[Dict[str, Any]]
    op_attr: Dict[str, Any]
    file: Optional[str]
    line: Optional[int]
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def get_input_tensor_magics(self) -> List[int]:
        """获取输入Tensor的magic ID列表"""
        return self.ioperands

    def get_output_tensor_magics(self) -> List[int]:
        """获取输出Tensor的magic ID列表"""
        return self.ooperands


@dataclass
class RawTensorInfo:
    """RawTensor信息"""

    rawmagic: int
    rawshape: List[int]
    ori_rawshape: List[int]
    datatype: int
    format: int
    kind: int
    symbol: Optional[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInfo:
    """Function信息"""

    func_magicname: str
    funcmagic: int
    graphtype: int
    incasts: List[List[Any]]
    outcasts: List[List[Any]]
    operations: List[OperationInfo]
    tensors: List[TensorInfo]
    rawtensors: List[RawTensorInfo]
    total_subgraph_count: int
    file: Optional[str]
    line: Optional[int]
    global_tensors: List[int]
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def get_graph_type(self) -> Optional[GraphType]:
        """获取图类型"""
        return GraphType.from_value(self.graphtype)

    def get_graph_type_str(self) -> str:
        """获取图类型字符串"""
        return GraphType.to_string(self.graphtype)

    def get_input_tensor_magics(self) -> List[int]:
        """获取输入Tensor的magic ID列表"""
        return [incast[0] for incast in self.incasts]

    def get_output_tensor_magics(self) -> List[int]:
        """获取输出Tensor的magic ID列表"""
        return [outcast[0] for outcast in self.outcasts]


@dataclass
class GraphInfo:
    """计算图信息"""

    entryhash: str
    version: str
    functions: List[FunctionInfo]
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def get_main_function(self) -> Optional[FunctionInfo]:
        """获取主函数（通常只有一个）"""
        return self.functions[0] if self.functions else None


@dataclass
class EdgeInfo:
    """Operation级边信息"""

    src_opmagic: int
    dst_opmagic: int
    tensor_magic: int
    src_opcode: str
    dst_opcode: str


@dataclass
class SubgraphEdgeInfo:
    """Subgraph级边信息"""

    src_subgraph: int
    dst_subgraph: int
    tensor_magic: int
    producer_ops: List[int]
    consumer_ops: List[int]


@dataclass
class CycleDetectionResult:
    """成环检测结果"""

    has_cycle: bool
    cycle_paths: List[Dict[str, Any]]
    node_count: int
    edge_count: int
    detection_basis: str
    limitations: List[str]


class ComputationGraphAnalyzer:
    """计算图分析器"""

    def __init__(self):
        self.graph: Optional[GraphInfo] = None

    @staticmethod
    def build_tensor_producer_map(func: FunctionInfo) -> Dict[int, List[OperationInfo]]:
        """构建Tensor到生产者Op的映射"""
        producer_map: Dict[int, List[OperationInfo]] = defaultdict(list)
        for op in func.operations:
            for tensor_magic in op.ooperands:
                producer_map[tensor_magic].append(op)
        return dict(producer_map)

    @staticmethod
    def build_tensor_consumer_map(func: FunctionInfo) -> Dict[int, List[OperationInfo]]:
        """构建Tensor到消费者Op的映射"""
        consumer_map: Dict[int, List[OperationInfo]] = defaultdict(list)
        for op in func.operations:
            for tensor_magic in op.ioperands:
                consumer_map[tensor_magic].append(op)
        return dict(consumer_map)

    @staticmethod
    def _parse_operation(op_data: Dict[str, Any]) -> OperationInfo:
        """解析Operation数据"""
        return OperationInfo(
            opmagic=op_data.get('opmagic', -1),
            opcode=op_data.get('opcode', ''),
            ioperands=op_data.get('ioperands', []),
            ooperands=op_data.get('ooperands', []),
            attr=op_data.get('attr', []),
            subgraphid=op_data.get('subgraphid', -1),
            latency=op_data.get('latency', 0),
            kind=op_data.get('kind', -1),
            tile=op_data.get('tile'),
            op_attr=op_data.get('op_attr', {}),
            file=op_data.get('file'),
            line=op_data.get('line'),
            raw_data=op_data,
        )

    @staticmethod
    def _parse_tensor(tensor_data: Dict[str, Any]) -> TensorInfo:
        """解析Tensor数据"""
        mem_type = tensor_data.get('mem_type', {})
        return TensorInfo(
            magic=tensor_data.get('magic', -1),
            shape=tensor_data.get('shape', []),
            validshape=tensor_data.get('validshape', []),
            dynvalidshape=tensor_data.get('dynvalidshape', []),
            offset=tensor_data.get('offset', []),
            rawtensor=tensor_data.get('rawtensor', -1),
            nodetype=tensor_data.get('nodetype', 0),
            mem_id=tensor_data.get('mem_id', -1),
            mem_range=tensor_data.get('mem_range', []),
            mem_type_asis=mem_type.get('asis', 0),
            mem_type_tobe=mem_type.get('tobe', 0),
            subgraphid=tensor_data.get('subgraphid', -1),
            subgraph_boundary=tensor_data.get('subgraph_boundary', False),
            life_range=tensor_data.get('life_range', []),
            kind=tensor_data.get('kind', -1),
            raw_data=tensor_data,
        )

    @staticmethod
    def _parse_rawtensor(rt_data: Dict[str, Any]) -> RawTensorInfo:
        """解析RawTensor数据"""
        return RawTensorInfo(
            rawmagic=rt_data.get('rawmagic', -1),
            rawshape=rt_data.get('rawshape', []),
            ori_rawshape=rt_data.get('ori_rawshape', []),
            datatype=rt_data.get('datatype', -1),
            format=rt_data.get('format', -1),
            kind=rt_data.get('kind', -1),
            symbol=rt_data.get('symbol'),
            raw_data=rt_data,
        )

    @staticmethod
    def _default_limitations() -> List[str]:
        return [
            "dependOperand is not dumped in graph json",
            "operation group ordering is not analyzed",
            "cycle detection is based on explicit tensor dataflow only",
        ]

    @staticmethod
    def _cycle_slice_from_stack(node_path: List[int], repeated_node: int) -> List[int]:
        start_idx = node_path.index(repeated_node)
        return node_path[start_idx:] + [repeated_node]

    @staticmethod
    def _op_detail_map(func: FunctionInfo) -> Dict[int, OperationInfo]:
        return {op.opmagic: op for op in func.operations}

    @staticmethod
    def _build_op_cycle_path(
        cycle_nodes: List[int],
        path_edges: List[EdgeInfo],
        op_map: Dict[int, OperationInfo],
    ) -> Dict[str, Any]:
        edge_records = [
            {
                "src_opmagic": edge.src_opmagic,
                "dst_opmagic": edge.dst_opmagic,
                "tensor_magic": edge.tensor_magic,
                "src_opcode": edge.src_opcode,
                "dst_opcode": edge.dst_opcode,
            }
            for edge in path_edges
        ]
        node_details = []
        seen_nodes: Set[int] = set()
        for node in cycle_nodes[:-1]:
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            op = op_map[node]
            node_details.append(
                {
                    "opmagic": op.opmagic,
                    "opcode": op.opcode,
                    "file": op.file,
                    "line": op.line,
                    "subgraphid": op.subgraphid,
                }
            )
        return {
            "node_path": cycle_nodes,
            "edges": edge_records,
            "node_details": node_details,
        }

    @staticmethod
    def _build_subgraph_cycle_path(cycle_nodes: List[int], path_edges: List[SubgraphEdgeInfo]) -> Dict[str, Any]:
        edge_records = [
            {
                "src_subgraph": edge.src_subgraph,
                "dst_subgraph": edge.dst_subgraph,
                "tensor_magic": edge.tensor_magic,
                "producer_ops": edge.producer_ops,
                "consumer_ops": edge.consumer_ops,
            }
            for edge in path_edges
        ]
        return {
            "node_path": cycle_nodes,
            "edges": edge_records,
        }

    @classmethod
    def compare_cycle_state(cls, before_json: str, after_json: str, max_cycle_paths: int = 10) -> Dict[str, Any]:
        """对比前后两个图的成环状态"""
        before_analyzer = cls()
        before_analyzer.load_graph(before_json)
        after_analyzer = cls()
        after_analyzer.load_graph(after_json)

        before_op = before_analyzer.detect_op_cycles(max_cycle_paths=max_cycle_paths)
        after_op = after_analyzer.detect_op_cycles(max_cycle_paths=max_cycle_paths)
        before_subgraph = before_analyzer.detect_subgraph_cycles(max_cycle_paths=max_cycle_paths)
        after_subgraph = after_analyzer.detect_subgraph_cycles(max_cycle_paths=max_cycle_paths)

        first_cycle_introduced_in_after = (not before_op.has_cycle and after_op.has_cycle) or (
            not before_subgraph.has_cycle and after_subgraph.has_cycle
        )
        if first_cycle_introduced_in_after:
            root_cause_hint = "cycle first appears in after graph"
        elif before_op.has_cycle or before_subgraph.has_cycle:
            root_cause_hint = "cycle already exists in before graph"
        else:
            root_cause_hint = "no explicit cycle detected in either graph"

        return {
            "analysis_type": "compare_cycle",
            "before_json": before_json,
            "after_json": after_json,
            "before_has_op_cycle": before_op.has_cycle,
            "after_has_op_cycle": after_op.has_cycle,
            "before_has_subgraph_cycle": before_subgraph.has_cycle,
            "after_has_subgraph_cycle": after_subgraph.has_cycle,
            "first_cycle_introduced_in_after": first_cycle_introduced_in_after,
            "root_cause_hint": root_cause_hint,
            "limitations": cls._default_limitations(),
        }

    @classmethod
    def _detect_cycles_iterative(
        cls,
        graph: Dict[int, List[Any]],
        get_dst: Callable[[Any], int],
        build_cycle_path: Callable[[List[int], List[Any]], Dict[str, Any]],
        max_cycle_paths: int,
    ) -> List[Dict[str, Any]]:
        states: Dict[int, str] = {node: "TODO" for node in graph}
        stack_nodes: List[int] = []
        stack_edges: List[Any] = []
        cycle_paths: List[Dict[str, Any]] = []
        seen_cycles: Set[Tuple[int, ...]] = set()

        for start_node in graph:
            if states[start_node] != "TODO":
                continue

            states[start_node] = "IN_STACK"
            stack_nodes.append(start_node)
            frame_stack: List[Tuple[int, int, Optional[Any]]] = [(start_node, 0, None)]

            while frame_stack and len(cycle_paths) < max_cycle_paths:
                node, edge_idx, incoming_edge = frame_stack[-1]
                edges = graph.get(node, [])
                if edge_idx >= len(edges):
                    states[node] = "DONE"
                    frame_stack.pop()
                    stack_nodes.pop()
                    if incoming_edge is not None:
                        stack_edges.pop()
                    continue

                edge = edges[edge_idx]
                frame_stack[-1] = (node, edge_idx + 1, incoming_edge)
                dst = get_dst(edge)
                state = states.get(dst, "TODO")
                if state == "TODO":
                    states[dst] = "IN_STACK"
                    stack_nodes.append(dst)
                    stack_edges.append(edge)
                    frame_stack.append((dst, 0, edge))
                    continue

                if state == "IN_STACK":
                    start_idx = stack_nodes.index(dst)
                    cycle_nodes = stack_nodes[start_idx:] + [dst]
                    cycle_edge_path = stack_edges[start_idx:] + [edge]
                    cycle_key = tuple(cycle_nodes)
                    if cycle_key not in seen_cycles:
                        seen_cycles.add(cycle_key)
                        cycle_paths.append(build_cycle_path(cycle_nodes, cycle_edge_path))

            if len(cycle_paths) >= max_cycle_paths:
                break

        return cycle_paths

    def load_graph(self, json_path: str) -> GraphInfo:
        """加载计算图JSON文件"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"计算图文件不存在: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.graph = self._parse_graph(data)
        return self.graph

    def find_tensor_by_magic(self, magic: int) -> Optional[TensorInfo]:
        """根据magic ID查找Tensor"""
        if not self.graph:
            return None

        func = self.graph.get_main_function()
        if not func:
            return None

        for tensor in func.tensors:
            if tensor.magic == magic:
                return tensor
        return None

    def find_operation_by_magic(self, opmagic: int) -> Optional[OperationInfo]:
        """根据opmagic ID查找Operation"""
        if not self.graph:
            return None

        func = self.graph.get_main_function()
        if not func:
            return None

        for op in func.operations:
            if op.opmagic == opmagic:
                return op
        return None

    def find_operations_by_input_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找使用指定Tensor作为输入的所有Operation"""
        if not self.graph:
            return []

        func = self.graph.get_main_function()
        if not func:
            return []

        return [op for op in func.operations if tensor_magic in op.ioperands]

    def find_operations_by_output_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找产生指定Tensor作为输出的所有Operation"""
        if not self.graph:
            return []

        func = self.graph.get_main_function()
        if not func:
            return []

        return [op for op in func.operations if tensor_magic in op.ooperands]

    def find_producer_of_tensor(self, tensor_magic: int) -> Optional[OperationInfo]:
        """查找产生指定Tensor的Operation（生产者）"""
        ops = self.find_operations_by_output_tensor(tensor_magic)
        return ops[0] if ops else None

    def find_consumers_of_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找使用指定Tensor的所有Operation（消费者）"""
        return self.find_operations_by_input_tensor(tensor_magic)

    def get_main_function(self) -> Optional[FunctionInfo]:
        """获取主Function"""
        if not self.graph:
            return None
        return self.graph.get_main_function()

    def build_op_graph(self, func: FunctionInfo) -> Dict[int, List[EdgeInfo]]:
        """基于显式Tensor数据流构建Op依赖图"""
        graph: Dict[int, List[EdgeInfo]] = {op.opmagic: [] for op in func.operations}
        consumer_map = self.build_tensor_consumer_map(func)
        for op in func.operations:
            for tensor_magic in op.ooperands:
                for consumer in consumer_map.get(tensor_magic, []):
                    graph[op.opmagic].append(
                        EdgeInfo(
                            src_opmagic=op.opmagic,
                            dst_opmagic=consumer.opmagic,
                            tensor_magic=tensor_magic,
                            src_opcode=op.opcode,
                            dst_opcode=consumer.opcode,
                        )
                    )
        return graph

    def build_subgraph_graph(self, func: FunctionInfo) -> Dict[int, List[SubgraphEdgeInfo]]:
        """基于跨subgraph Tensor流构建subgraph依赖图"""
        graph: Dict[int, List[SubgraphEdgeInfo]] = defaultdict(list)
        producer_map = self.build_tensor_producer_map(func)
        consumer_map = self.build_tensor_consumer_map(func)
        known_subgraphs = {op.subgraphid for op in func.operations if op.subgraphid >= 0}
        for subgraph_id in known_subgraphs:
            graph[subgraph_id] = []

        for tensor_magic, producers in producer_map.items():
            consumers = consumer_map.get(tensor_magic, [])
            if not consumers:
                continue
            producer_subgraphs = {op.subgraphid for op in producers if op.subgraphid >= 0}
            consumer_subgraphs = {op.subgraphid for op in consumers if op.subgraphid >= 0}
            for src_subgraph in producer_subgraphs:
                for dst_subgraph in consumer_subgraphs:
                    if src_subgraph == dst_subgraph:
                        continue
                    graph[src_subgraph].append(
                        SubgraphEdgeInfo(
                            src_subgraph=src_subgraph,
                            dst_subgraph=dst_subgraph,
                            tensor_magic=tensor_magic,
                            producer_ops=sorted([op.opmagic for op in producers if op.subgraphid == src_subgraph]),
                            consumer_ops=sorted([op.opmagic for op in consumers if op.subgraphid == dst_subgraph]),
                        )
                    )
        return dict(graph)

    def detect_op_cycles(self, max_cycle_paths: int = 10) -> CycleDetectionResult:
        """检测Op级显式Tensor数据流成环"""
        func = self.get_main_function()
        if func is None:
            return CycleDetectionResult(False, [], 0, 0, "explicit_tensor_dataflow", self._default_limitations())

        op_graph = self.build_op_graph(func)
        op_map = self._op_detail_map(func)
        cycle_paths = self._detect_cycles_iterative(
            graph=op_graph,
            get_dst=lambda edge: edge.dst_opmagic,
            build_cycle_path=lambda cycle_nodes, cy_edges: self._build_op_cycle_path(cycle_nodes, cy_edges, op_map),
            max_cycle_paths=max_cycle_paths,
        )

        edge_count = sum(len(edges) for edges in op_graph.values())
        return CycleDetectionResult(
            has_cycle=bool(cycle_paths),
            cycle_paths=cycle_paths,
            node_count=len(op_graph),
            edge_count=edge_count,
            detection_basis="explicit_tensor_dataflow",
            limitations=self._default_limitations(),
        )

    def detect_subgraph_cycles(self, max_cycle_paths: int = 10) -> CycleDetectionResult:
        """检测Subgraph级显式Tensor数据流成环"""
        func = self.get_main_function()
        if func is None:
            return CycleDetectionResult(False, [], 0, 0, "explicit_tensor_dataflow", self._default_limitations())

        subgraph_graph = self.build_subgraph_graph(func)
        cycle_paths = self._detect_cycles_iterative(
            graph=subgraph_graph,
            get_dst=lambda edge: edge.dst_subgraph,
            build_cycle_path=self._build_subgraph_cycle_path,
            max_cycle_paths=max_cycle_paths,
        )

        edge_count = sum(len(edges) for edges in subgraph_graph.values())
        return CycleDetectionResult(
            has_cycle=bool(cycle_paths),
            cycle_paths=cycle_paths,
            node_count=len(subgraph_graph),
            edge_count=edge_count,
            detection_basis="explicit_tensor_dataflow",
            limitations=self._default_limitations(),
        )

    def _parse_graph(self, data: Dict[str, Any]) -> GraphInfo:
        """解析计算图数据"""
        functions = []
        for func_data in data.get('functions', []):
            functions.append(self._parse_function(func_data))

        return GraphInfo(
            entryhash=data.get('entryhash', ''), version=data.get('version', ''), functions=functions, raw_data=data
        )

    def _parse_function(self, func_data: Dict[str, Any]) -> FunctionInfo:
        """解析Function数据"""
        operations = []
        for op_data in func_data.get('operations', []):
            operations.append(self._parse_operation(op_data))

        tensors = []
        for tensor_data in func_data.get('tensors', []):
            tensors.append(self._parse_tensor(tensor_data))

        rawtensors = []
        for rt_data in func_data.get('rawtensors', []):
            rawtensors.append(self._parse_rawtensor(rt_data))

        return FunctionInfo(
            func_magicname=func_data.get('func_magicname', ''),
            funcmagic=func_data.get('funcmagic', -1),
            graphtype=func_data.get('graphtype', -1),
            incasts=func_data.get('incasts', []),
            outcasts=func_data.get('outcasts', []),
            operations=operations,
            tensors=tensors,
            rawtensors=rawtensors,
            total_subgraph_count=func_data.get('_total_subgraph_count', 0),
            file=func_data.get('file'),
            line=func_data.get('line'),
            global_tensors=func_data.get('global_tensors', []),
            raw_data=func_data,
        )


def _build_summary(analyzer: ComputationGraphAnalyzer, args: argparse.Namespace) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "json_path": args.json_path,
        "op_magic": args.op_magic,
        "tensor_magic": args.tensor_magic,
    }

    if args.op_magic is not None:
        op = analyzer.find_operation_by_magic(args.op_magic)
        summary["operation"] = (
            None
            if op is None
            else {
                "opmagic": op.opmagic,
                "opcode": op.opcode,
                "ioperands": op.ioperands,
                "ooperands": op.ooperands,
                "file": op.file,
                "line": op.line,
            }
        )

    if args.tensor_magic is not None:
        tensor = analyzer.find_tensor_by_magic(args.tensor_magic)
        summary["tensor"] = (
            None
            if tensor is None
            else {
                "magic": tensor.magic,
                "shape": tensor.shape,
                "validshape": tensor.validshape,
                "mem_type": tensor.get_memory_type_str(),
                "life_range": tensor.life_range,
            }
        )
        if tensor is not None:
            producer = analyzer.find_producer_of_tensor(args.tensor_magic)
            consumers = analyzer.find_consumers_of_tensor(args.tensor_magic)
            summary["producer"] = (
                None
                if producer is None
                else {
                    "opmagic": producer.opmagic,
                    "opcode": producer.opcode,
                    "file": producer.file,
                    "line": producer.line,
                }
            )
            summary["consumers"] = [
                {
                    "opmagic": consumer.opmagic,
                    "opcode": consumer.opcode,
                    "file": consumer.file,
                    "line": consumer.line,
                }
                for consumer in consumers
            ]

    return summary


def _result_to_dict(result: CycleDetectionResult, analysis_type: str, json_path: str) -> Dict[str, Any]:
    return {
        "json_path": json_path,
        "analysis_type": analysis_type,
        "has_cycle": result.has_cycle,
        "cycle_paths": result.cycle_paths,
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "detection_basis": result.detection_basis,
        "limitations": result.limitations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze PyPTO computation graph JSON")
    parser.add_argument("--json-path", help="Path to graph JSON file")
    parser.add_argument("--op-magic", type=int, default=None, help="Operation magic id")
    parser.add_argument("--tensor-magic", type=int, default=None, help="Tensor magic id")
    parser.add_argument("--detect-op-cycle", action="store_true", help="Detect op-level cycles")
    parser.add_argument("--detect-subgraph-cycle", action="store_true", help="Detect subgraph-level cycles")
    parser.add_argument("--before-json", help="Path to before graph JSON")
    parser.add_argument("--after-json", help="Path to after graph JSON")
    parser.add_argument("--compare-cycle", action="store_true", help="Compare cycle state between two graph JSON files")
    parser.add_argument("--max-cycle-paths", type=int, default=10, help="Maximum cycle paths to report")
    args = parser.parse_args()

    if args.compare_cycle:
        if not args.before_json or not args.after_json:
            parser.error("--compare-cycle requires both --before-json and --after-json")
        result = ComputationGraphAnalyzer.compare_cycle_state(
            args.before_json,
            args.after_json,
            max_cycle_paths=args.max_cycle_paths,
        )
        logging.info(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not args.json_path:
        parser.error("--json-path is required unless --compare-cycle is used")

    analyzer = ComputationGraphAnalyzer()
    analyzer.load_graph(args.json_path)

    if args.detect_op_cycle:
        result = analyzer.detect_op_cycles(max_cycle_paths=args.max_cycle_paths)
        logging.info(
            json.dumps(
                _result_to_dict(result, "detect_op_cycle", args.json_path),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.detect_subgraph_cycle:
        result = analyzer.detect_subgraph_cycles(max_cycle_paths=args.max_cycle_paths)
        logging.info(
            json.dumps(
                _result_to_dict(result, "detect_subgraph_cycle", args.json_path),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    summary = _build_summary(analyzer, args)
    logging.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
