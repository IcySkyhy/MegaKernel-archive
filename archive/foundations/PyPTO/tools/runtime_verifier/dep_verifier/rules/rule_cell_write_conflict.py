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
from typing import Dict, List, Set, Tuple

from ..models import Category, SlotAccessEvent, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellWriteConflictRule(Rule):
    RULE_ID = "rule_cell_write_conflict"
    DESCRIPTION = "Multiple producers writing to the same region must have a determined order"
    CATEGORY = Category.WRITE_RACE

    HARD_LIMIT = 4096

    @staticmethod
    def _all_pairs_ordered(ctx: RuleContext, seq_no: int, writers: List[SlotAccessEvent]) -> bool:
        desc_list: List[Set[int]] = [ctx.descendants(seq_no, w.task_id) for w in writers]
        n = len(writers)
        for i in range(n):
            for j in range(i + 1, n):
                if writers[j].task_id in desc_list[i]:
                    continue
                if writers[i].task_id in desc_list[j]:
                    continue
                return False
        return True

    @staticmethod
    def _is_legal_parallel_outcast(
        ctx: RuleContext, seq_no: int, slot: int, cell: int, writer_list: List[SlotAccessEvent]
    ) -> bool:
        if ctx.has_xroot_static_reader(slot):
            return False
        readers = ctx.readers_of_cell.get((seq_no, slot, cell), [])
        if readers:
            reader_tasks = {r.task_id for r in readers}
            if any(w.task_id in reader_tasks for w in writer_list):
                return False
        return True

    @staticmethod
    def _is_total_order_chain(ctx: RuleContext, seq_no: int, writers: List[SlotAccessEvent]) -> bool:
        if len(writers) <= 1:
            return True
        ordered = sorted(writers, key=lambda w: (w.func_idx, w.op_idx, w.task_id))
        for a, b in zip(ordered, ordered[1:]):
            if not ctx.reaches(seq_no, a.task_id, b.task_id):
                return CellWriteConflictRule._all_pairs_ordered(ctx, seq_no, ordered)
        return True

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []

        reuse_slots: Set[Tuple[int, int]] = set()
        for e in ctx.stitch_edges:
            if "reuse" in (e.stitch_kind or "").lower() and e.inferred_seq_no is not None:
                reuse_slots.add((e.inferred_seq_no, e.slot_idx))

        violations: List[Violation] = []
        for (seq_no, slot, cell), writers in ctx.writers_of_cell.items():
            if not ctx.is_partial_slot(slot):
                continue
            if len(writers) <= 1:
                continue
            if (seq_no, slot) in reuse_slots:
                continue

            uniq: Dict[int, SlotAccessEvent] = {}
            for w in writers:
                uniq.setdefault(w.task_id, w)
            writer_list = list(uniq.values())
            if len(writer_list) <= 1:
                continue
            if any(not w.all_concrete for w in writer_list):
                continue
            if len(writer_list) > self.HARD_LIMIT:
                continue
            if self._is_legal_parallel_outcast(ctx, seq_no, slot, cell, writer_list):
                continue
            if self._is_total_order_chain(ctx, seq_no, writer_list):
                continue

            violations.append(
                Violation(
                    rule_id=self.RULE_ID,
                    slot_idx=slot,
                    cell_idx=cell,
                    message=(
                        f"{len(writer_list)} producer instances write to the same "
                        f"region without a determined order, later writes may "
                        f"overwrite earlier ones"
                    ),
                )
            )
        return violations
