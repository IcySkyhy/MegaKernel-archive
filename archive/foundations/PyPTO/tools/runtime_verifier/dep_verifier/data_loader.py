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
from collections import defaultdict
import csv
import logging
from typing import Dict, List, Set, Tuple

from .models import (
    CellTableDesc,
    DynTask,
    SlotAccessEvent,
    StaticFunction,
    StaticOp,
    StitchEdge,
)

logger = logging.getLogger(__name__)


def _parse_int_list(cells: List[str]) -> List[int]:
    out: List[int] = []
    for c in cells:
        c = (c or "").strip()
        if not c:
            continue
        try:
            out.append(int(c))
        except ValueError:
            pass
    return out


def _parse_bracket_int_list(cell: str) -> List[int]:
    if not cell:
        return []
    s = cell.strip().strip('"')
    if not s or s == "[]":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    out: List[int] = []
    for p in s.replace(",", ";").split(";"):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            pass
    return out


def _header_idx(path: str, reader: "csv._reader") -> Dict[str, int]:
    header = next(reader, None)
    if not header:
        raise ValueError(f"empty file or missing header: {path}")
    return {name: i for i, name in enumerate(header)}


def _make_pick(col: Dict[str, int]):
    def pick(row, key, default=""):
        idx = col.get(key)
        if idx is None or idx >= len(row):
            return default
        return row[idx]

    return pick


def load_static_topo(path: str) -> Dict[int, StaticFunction]:
    logger.debug("load static_topo: %s", path)
    functions: Dict[int, StaticFunction] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        col = _header_idx(path, reader)
        tail = col["staticSuccessors"]
        for row in reader:
            if not row or not row[0].strip():
                continue
            func_key = int(row[col["funcKey"]])
            op = StaticOp(
                func_key=func_key,
                root_hash=row[col["rootHash"]],
                raw_name=row[col["rawName"]],
                op_idx=int(row[col["opIdx"]]),
                incast_slots=_parse_bracket_int_list(row[col["incastSlots"]]),
                outcast_slots=_parse_bracket_int_list(row[col["outcastSlots"]]),
                static_successors_op_idx=_parse_int_list(row[tail:]),
            )
            fn = functions.get(func_key)
            if fn is None:
                fn = StaticFunction(func_key=func_key, root_hash=op.root_hash, raw_name=op.raw_name)
                functions[func_key] = fn
            fn.ops[op.op_idx] = op
    logger.debug("  static function=%d, op=%d", len(functions), sum(len(fn.ops) for fn in functions.values()))
    return functions


def load_dyn_topo(path: str) -> List[DynTask]:
    logger.debug("load dyn_topo: %s", path)
    tasks: List[DynTask] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        col = _header_idx(path, reader)
        succ_start = col["successors"]
        for row in reader:
            if not row or not row[0].strip():
                continue
            tasks.append(
                DynTask(
                    seq_no=int(row[col["seqNo"]]),
                    task_id=int(row[col["taskId"]]),
                    root_index=int(row[col["rootIndex"]]),
                    root_hash=row[col["rootHash"]],
                    opmagic=int(row[col["opmagic"]]),
                    leaf_index=int(row[col["leafIndex"]]),
                    leaf_hash=row[col["leafHash"]],
                    core_type=int(row[col["coreType"]]),
                    psg_id=int(row[col["psgId"]]),
                    wrap_id=int(row[col["wrapId"]]),
                    static_succ_count=int(row[col["staticSuccCount"]]),
                    successors=_parse_int_list(row[succ_start:]),
                )
            )
    logger.debug("  dynamic task instances=%d", len(tasks))
    return tasks


def load_dyn_stitch_edges(path: str) -> List[StitchEdge]:
    logger.debug("load dyn_stitch_edges: %s", path)
    edges: List[StitchEdge] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        col = _header_idx(path, reader)
        pick = _make_pick(col)
        for row in reader:
            if not row or not row[0].strip():
                continue
            edges.append(
                StitchEdge(
                    stitch_kind=pick(row, "stitchKind"),
                    slot_idx=int(pick(row, "slotIdx")),
                    producer_func_key=int(pick(row, "producerFuncKey")),
                    producer_func_idx=int(pick(row, "producerFuncIdx")),
                    producer_op_idx=int(pick(row, "producerOpIdx")),
                    producer_task_id=int(pick(row, "producerTaskId")),
                    consumer_func_key=int(pick(row, "consumerFuncKey")),
                    consumer_func_idx=int(pick(row, "consumerFuncIdx")),
                    consumer_op_idx=int(pick(row, "consumerOpIdx")),
                    consumer_task_id=int(pick(row, "consumerTaskId")),
                )
            )
    logger.debug("  stitch edges=%d", len(edges))
    return edges


def _merge_unique(existing: str, incoming: str) -> str:
    if not existing:
        return incoming
    merged = existing.split(";")
    seen = set(merged)
    for token in incoming.split(";"):
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            merged.append(token)
    return ";".join(merged)


def load_slot_mapping(
    path: str,
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, str], Dict[int, str]]:
    logger.debug("load slot_mapping: %s", path)
    f2r: Dict[int, int] = {}
    r2role: Dict[int, str] = {}
    r2tensor: Dict[int, str] = {}
    r2func: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        col = _header_idx(path, reader)
        pick = _make_pick(col)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                front = int(pick(row, "frontendSlotIdx"))
                rt = int(pick(row, "runtimeSlotIdx"))
            except ValueError:
                continue
            role = (pick(row, "slotRole", "INTERNAL") or "INTERNAL").strip().upper()
            if role not in ("INPUT", "OUTPUT", "INOUT", "INTERNAL"):
                role = "INTERNAL"
            tensor_name = (pick(row, "tensorName", "") or "").strip()
            func_name = (pick(row, "funcRawName", "") or "").strip()
            f2r[front] = rt
            r2role[rt] = role
            if tensor_name:
                r2tensor[rt] = _merge_unique(r2tensor.get(rt, ""), tensor_name)
            if func_name:
                r2func[rt] = _merge_unique(r2func.get(rt, ""), func_name)
    logger.debug("  slot mapping=%d", len(f2r))
    return f2r, r2role, r2tensor, r2func


def load_slot_cell_table(path: str) -> Dict[int, List[CellTableDesc]]:
    logger.debug("load slot_cell_table: %s", path)
    out: Dict[int, List[CellTableDesc]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        col = _header_idx(path, reader)
        pick = _make_pick(col)
        for row in reader:
            if not row or not row[0].strip():
                continue
            slot_idx = int(pick(row, "slotIdx"))
            out[slot_idx].append(
                CellTableDesc(
                    slot_idx=slot_idx,
                    stitch_policy=pick(row, "stitchPolicy", "partial"),
                    root_hash=pick(row, "rootHash", "0"),
                    func_key=int(pick(row, "funcKey", "-1")),
                    cell_shape=_parse_bracket_int_list(pick(row, "cellShape")),
                    cell_count=int(pick(row, "cellCount", "1")),
                    outcast_count=int(pick(row, "outcastCount", "1")),
                )
            )
    logger.debug("  slots with cell table=%d", len(out))
    return dict(out)


def load_slot_access(path: str) -> List[SlotAccessEvent]:
    logger.debug("load dyn_slot_access: %s", path)
    events: List[SlotAccessEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        col = _header_idx(path, reader)
        pick = _make_pick(col)
        for row in reader:
            if not row or not row[0].strip():
                continue
            events.append(
                SlotAccessEvent(
                    seq_no=int(pick(row, "seqNo")),
                    slot_idx=int(pick(row, "slotIdx")),
                    func_idx=int(pick(row, "funcIdx", "0")),
                    op_idx=int(pick(row, "opIdx", "0")),
                    task_id=int(pick(row, "taskId", "0")),
                    access_type=pick(row, "accessType", "W"),
                    cell_idx_list=_parse_bracket_int_list(pick(row, "cellIdxList")),
                    all_concrete=bool(int(pick(row, "allConcrete", "1"))),
                )
            )
    logger.debug("  slot access events=%d", len(events))
    return events


def infer_edge_seq_no(edges: List[StitchEdge], tasks: List[DynTask]) -> None:
    tasks_by_seq: Dict[int, Set[int]] = defaultdict(set)
    for t in tasks:
        tasks_by_seq[t.seq_no].add(t.task_id)
    for e in edges:
        cands = [seq for seq, ids in tasks_by_seq.items() if e.producer_task_id in ids and e.consumer_task_id in ids]
        e.inferred_seq_no = cands[0] if len(cands) == 1 else None
