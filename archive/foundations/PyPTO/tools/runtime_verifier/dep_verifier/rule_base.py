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
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    CellTableDesc,
    DynTask,
    SlotAccessEvent,
    StaticFunction,
    StitchEdge,
    Violation,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleContext:
    static_functions: Dict[int, StaticFunction]
    dyn_tasks: List[DynTask]
    stitch_edges: List[StitchEdge]
    slot_cell_tables: Dict[int, List[CellTableDesc]] = field(default_factory=dict)
    slot_accesses: List[SlotAccessEvent] = field(default_factory=list)
    slot_roles: Dict[int, str] = field(default_factory=dict)
    slot_tensor_names: Dict[int, str] = field(default_factory=dict)
    slot_func_names: Dict[int, str] = field(default_factory=dict)

    _task_by_id: Optional[Dict[Tuple[int, int], DynTask]] = None
    _writers_of_cell: Optional[Dict[Tuple[int, int, int], List[SlotAccessEvent]]] = None
    _readers_of_cell: Optional[Dict[Tuple[int, int, int], List[SlotAccessEvent]]] = None
    _descendants_cache: Optional[Dict[Tuple[int, int], Set[int]]] = None
    _slots_with_xroot_static_reader: Optional[Set[int]] = None

    @property
    def task_by_id(self) -> Dict[Tuple[int, int], DynTask]:
        if self._task_by_id is None:
            self._task_by_id = {(t.seq_no, t.task_id): t for t in self.dyn_tasks}
        return self._task_by_id

    @property
    def writers_of_cell(self) -> Dict[Tuple[int, int, int], List[SlotAccessEvent]]:
        if self._writers_of_cell is None:
            d: Dict[Tuple[int, int, int], List[SlotAccessEvent]] = defaultdict(list)
            for e in self.slot_accesses:
                if not e.is_writer:
                    continue
                for c in e.cell_idx_list:
                    d[(e.seq_no, e.slot_idx, c)].append(e)
            self._writers_of_cell = dict(d)
        return self._writers_of_cell

    @property
    def readers_of_cell(self) -> Dict[Tuple[int, int, int], List[SlotAccessEvent]]:
        if self._readers_of_cell is None:
            d: Dict[Tuple[int, int, int], List[SlotAccessEvent]] = defaultdict(list)
            for e in self.slot_accesses:
                if not e.is_reader:
                    continue
                for c in e.cell_idx_list:
                    d[(e.seq_no, e.slot_idx, c)].append(e)
            self._readers_of_cell = dict(d)
        return self._readers_of_cell

    @staticmethod
    def _collect_slot_func_keys(
        fn: StaticFunction, writers_by_slot: Dict[int, Set[int]], readers_by_slot: Dict[int, Set[int]]
    ) -> None:
        func_key = fn.func_key
        for op in fn.ops.values():
            for s in op.outcast_slots:
                writers_by_slot.setdefault(s, set()).add(func_key)
            for s in op.incast_slots:
                readers_by_slot.setdefault(s, set()).add(func_key)

    def descendants(self, seq_no: int, task_id: int) -> Set[int]:
        if self._descendants_cache is None:
            self._descendants_cache = {}
        key = (seq_no, task_id)
        cached = self._descendants_cache.get(key)
        if cached is not None:
            return cached
        task_by_id = self.task_by_id
        visited: Set[int] = set()
        stack: List[int] = [task_id]
        while stack:
            cur = stack.pop()
            t = task_by_id.get((seq_no, cur))
            if t is None:
                continue
            for s in t.successors:
                if s in visited:
                    continue
                visited.add(s)
                stack.append(s)
        self._descendants_cache[key] = visited
        return visited

    def reaches(self, seq_no: int, u: int, v: int) -> bool:
        if u == v:
            return True
        return v in self.descendants(seq_no, u)

    def get_static_op(self, func_key: int, op_idx: int):
        fn = self.static_functions.get(func_key)
        if fn is None:
            return None
        return fn.ops.get(op_idx)

    def is_partial_slot(self, slot_idx: int) -> bool:
        rows = self.slot_cell_tables.get(slot_idx, [])
        for row in rows:
            if (row.stitch_policy or "").lower() == "partial":
                return True
        return False

    def has_xroot_static_reader(self, slot_idx: int) -> bool:
        if self._slots_with_xroot_static_reader is None:
            self._slots_with_xroot_static_reader = self._build_xroot_static_reader_set()
        return slot_idx in self._slots_with_xroot_static_reader

    def _build_xroot_static_reader_set(self) -> Set[int]:
        writers_by_slot: Dict[int, Set[int]] = {}
        readers_by_slot: Dict[int, Set[int]] = {}
        for fn in self.static_functions.values():
            self._collect_slot_func_keys(fn, writers_by_slot, readers_by_slot)
        xroot: Set[int] = set()
        for s, writer_keys in writers_by_slot.items():
            reader_keys = readers_by_slot.get(s)
            if not reader_keys:
                continue
            if len(writer_keys | reader_keys) > 1:
                xroot.add(s)
        return xroot


class Rule(ABC):
    RULE_ID: str = ""
    DESCRIPTION: str = ""
    CATEGORY: str = ""

    def __init__(self):
        if not self.RULE_ID:
            raise ValueError(f"{type(self).__name__}.RULE_ID is not set")
        if not self.CATEGORY:
            raise ValueError(f"{type(self).__name__}.CATEGORY is not set")

    @abstractmethod
    def check(self, ctx: RuleContext) -> List[Violation]:
        raise NotImplementedError

    def run(self, ctx: RuleContext) -> List[Violation]:
        try:
            vs = self.check(ctx) or []
        except Exception as exc:
            logger.exception("rule %s execution failed", self.RULE_ID)
            return [
                Violation(
                    rule_id=self.RULE_ID,
                    category=self.CATEGORY,
                    message=f"rule execution failed: {exc}",
                )
            ]
        for v in vs:
            if not v.category:
                v.category = self.CATEGORY
        return vs
