#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List, Optional

TASK_ID_OP_BITS = 16
TASK_ID_OP_MASK = (1 << TASK_ID_OP_BITS) - 1


def encode_task_id(func_idx: int, op_idx: int) -> int:
    return (func_idx << TASK_ID_OP_BITS) | op_idx


class Category:
    WRITE_RACE = "ConcurrentWriteOverlap"
    MISSING_DEPENDENCY = "MissingDependency"
    ILLEGAL_LINKAGE = "IllegalReadWriteLinkage"


CATEGORY_TITLES: Dict[str, str] = {
    Category.WRITE_RACE: "Concurrent write overlap (producers may overwrite each other)",
    Category.MISSING_DEPENDENCY: "Missing producer/consumer dependency (data flow broken)",
    Category.ILLEGAL_LINKAGE: "Illegal read/write linkage (kernel I/O does not match)",
}

CATEGORY_ORDER: List[str] = [
    Category.WRITE_RACE,
    Category.MISSING_DEPENDENCY,
    Category.ILLEGAL_LINKAGE,
]


@dataclass
class StaticOp:
    func_key: int
    root_hash: str
    raw_name: str
    op_idx: int
    incast_slots: List[int]
    outcast_slots: List[int]
    static_successors_op_idx: List[int]


@dataclass
class StaticFunction:
    func_key: int
    root_hash: str
    raw_name: str
    ops: Dict[int, StaticOp] = field(default_factory=dict)


@dataclass
class DynTask:
    seq_no: int
    task_id: int
    root_index: int
    root_hash: str
    opmagic: int
    leaf_index: int
    leaf_hash: str
    core_type: int
    psg_id: int
    wrap_id: int
    static_succ_count: int
    successors: List[int]

    @property
    def func_idx(self) -> int:
        return self.task_id >> TASK_ID_OP_BITS

    @property
    def op_idx(self) -> int:
        return self.task_id & TASK_ID_OP_MASK

    @property
    def static_successors(self) -> List[int]:
        return self.successors[:self.static_succ_count]


@dataclass
class StitchEdge:
    stitch_kind: str
    slot_idx: int
    producer_func_key: int
    producer_func_idx: int
    producer_op_idx: int
    producer_task_id: int
    consumer_func_key: int
    consumer_func_idx: int
    consumer_op_idx: int
    consumer_task_id: int
    inferred_seq_no: Optional[int] = None


@dataclass
class CellTableDesc:
    slot_idx: int
    stitch_policy: str
    root_hash: str
    func_key: int
    cell_shape: List[int]
    cell_count: int
    outcast_count: int


@dataclass
class SlotAccessEvent:
    seq_no: int
    slot_idx: int
    func_idx: int
    op_idx: int
    task_id: int
    access_type: str
    cell_idx_list: List[int]
    all_concrete: bool

    @property
    def is_writer(self) -> bool:
        return self.access_type == "W"

    @property
    def is_reader(self) -> bool:
        return self.access_type == "R"


@dataclass
class Violation:
    rule_id: str
    message: str
    slot_idx: Optional[int] = None
    cell_idx: Optional[int] = None
    category: str = ""
