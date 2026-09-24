#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the LICENSE.
# -----------------------------------------------------------------------------------------------------------
"""基于 controlflow 内存 dump 和 dyn_topo 的离线内存重叠检查。

只需要传入一次运行的 output 目录。脚本不依赖 profiling/swimlane 时间：
同一 DeviceTask 内，dyn_topo 上互相不可达的两个 Op 视为可能并发，然后检查
Read-Write 和 Write-Write Operand 的物理地址及各维访问范围。
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

DYN_TOPO_NAME = "dyn_topo.txt"
ACCESS_CSV_NAME = "mem_rawtensor_access.csv"
TASK_OP_BITS = 16
SEPARATOR = "=" * 100


@dataclass(frozen=True, order=True)
class TaskKey:
    seq_no: int
    task_id: int


@dataclass
class TopoTask:
    key: TaskKey
    root_index: int
    root_hash: str
    opmagic: int
    leaf_index: int
    leaf_hash: str
    core_type: int
    psg_id: int
    wrap_id: int
    successors: List[int]


@dataclass
class Access:
    row_number: int
    seq_no: int
    func_idx: int
    op_idx: int
    task_id: int
    root_index: int
    access_type: str
    operand_index: int
    base: int
    end: int
    offset: List[int]
    shape: List[int]
    raw_shape: Optional[List[int]]
    location: str
    all_concrete: bool
    raw_magic: Optional[int]

    @property
    def key(self) -> TaskKey:
        return TaskKey(self.seq_no, self.task_id)

    @property
    def is_write(self) -> bool:
        return self.access_type == "W"

    @property
    def operand_name(self) -> str:
        kind = "outputOperand" if self.is_write else "inputOperand"
        return f"{kind}[{self.operand_index}]"


@dataclass
class Conflict:
    race_kind: str
    overlap_kind: str
    src: Access
    dst: Access
    src_topo: TopoTask
    dst_topo: TopoTask


def _parse_int_list(value: str) -> List[int]:
    text = (value or "").strip().strip('"')
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    result = []
    for token in text.replace(",", ";").split(";"):
        token = token.strip()
        if token:
            result.append(int(token))
    return result


def _find_input(output_dir: str, name: str, preferred_relative_path: str) -> str:
    preferred = os.path.join(output_dir, preferred_relative_path)
    matches = []
    for root, _, files in os.walk(output_dir):
        if name in files:
            matches.append(os.path.abspath(os.path.join(root, name)))
    if not matches:
        raise FileNotFoundError(f"{name} not found in output directory: {output_dir}")
    if len(matches) > 1:
        joined = "\n  ".join(sorted(matches))
        raise ValueError(
            f"Multiple {name} found, cannot determine which run they belong to. "
            f"Please pass a single-run output directory:\n  {joined}"
        )
    preferred = os.path.abspath(preferred)
    return preferred if preferred in matches else matches[0]


def resolve_inputs(output_dir: str) -> Tuple[str, str]:
    dyn_topo = _find_input(output_dir, DYN_TOPO_NAME, DYN_TOPO_NAME)
    access_csv = _find_input(
        output_dir,
        ACCESS_CSV_NAME,
        os.path.join("dep_verify_dump", ACCESS_CSV_NAME),
    )
    return dyn_topo, access_csv


def _header_index(path: str, reader) -> Dict[str, int]:
    header = next(reader, None)
    if not header:
        raise ValueError(f"File is empty or missing header: {path}")
    return {name.strip(): index for index, name in enumerate(header)}


def _require_columns(path: str, columns: Dict[str, int], required: Set[str]):
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def load_dyn_topo(path: str) -> Dict[TaskKey, TopoTask]:
    required = {
        "seqNo", "taskId", "rootIndex", "rootHash", "opmagic", "leafIndex", "leafHash",
        "coreType", "psgId", "wrapId", "successors",
    }
    tasks: Dict[TaskKey, TopoTask] = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        columns = _header_index(path, reader)
        _require_columns(path, columns, required)
        successor_start = columns["successors"]
        for row_number, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            try:
                key = TaskKey(int(row[columns["seqNo"]]), int(row[columns["taskId"]]))
                task = TopoTask(
                    key=key,
                    root_index=int(row[columns["rootIndex"]]),
                    root_hash=row[columns["rootHash"]],
                    opmagic=int(row[columns["opmagic"]]),
                    leaf_index=int(row[columns["leafIndex"]]),
                    leaf_hash=row[columns["leafHash"]],
                    core_type=int(row[columns["coreType"]]),
                    psg_id=int(row[columns["psgId"]]),
                    wrap_id=int(row[columns["wrapId"]]),
                    successors=[
                        int(value) for value in row[successor_start:] if value.strip()
                    ],
                )
            except (IndexError, ValueError) as error:
                raise ValueError(f"{path}:{row_number} dyn_topo row parse failed: {error}") from error
            if key in tasks:
                raise ValueError(
                    f"{path}:{row_number} duplicate (seqNo, taskId)=({key.seq_no}, {key.task_id})"
                )
            tasks[key] = task
    return tasks


def load_accesses(path: str) -> Tuple[Dict[TaskKey, List[Access]], Counter]:
    required = {
        "seqNo", "funcIdx", "opIdx", "taskId", "rootIndex", "accessType",
        "base", "end", "offset", "shape", "rawShape", "location", "allConcrete",
    }
    accesses: Dict[TaskKey, List[Access]] = defaultdict(list)
    counters: Counter = Counter()
    operand_counters: Counter = Counter()
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        columns = _header_index(path, reader)
        _require_columns(path, columns, required)
        raw_magic_column = columns.get("rawMagic")
        for row_number, row in enumerate(reader, start=2):
            counters["totalRows"] += 1
            if not row or not row[0].strip():
                counters["skippedEmpty"] += 1
                continue
            try:
                seq_no = int(row[columns["seqNo"]])
                func_idx = int(row[columns["funcIdx"]])
                op_idx = int(row[columns["opIdx"]])
                task_id = int(row[columns["taskId"]])
                root_index = int(row[columns["rootIndex"]])
                access_type = row[columns["accessType"]].strip().upper()
                base = int(row[columns["base"]])
                end = int(row[columns["end"]])
                offset = _parse_int_list(row[columns["offset"]])
                shape = _parse_int_list(row[columns["shape"]])
                raw_shape_values = _parse_int_list(row[columns["rawShape"]])
                location = row[columns["location"]].strip()
                all_concrete = row[columns["allConcrete"]].strip().lower() in {"1", "true"}
                raw_magic = None
                if raw_magic_column is not None and raw_magic_column < len(row):
                    raw_magic_text = row[raw_magic_column].strip()
                    raw_magic = int(raw_magic_text) if raw_magic_text else None
            except (IndexError, ValueError) as error:
                raise ValueError(f"{path}:{row_number} access row parse failed: {error}") from error

            if access_type not in {"R", "W"}:
                raise ValueError(f"{path}:{row_number} accessType is not R/W: {access_type!r}")
            expected_task_id = (func_idx << TASK_OP_BITS) | (op_idx & 0xffff)
            if task_id != expected_task_id:
                raise ValueError(
                    f"{path}:{row_number} taskId={task_id} vs "
                    f"(funcIdx << 16 | opIdx)={expected_task_id} mismatch"
                )
            operand_counter_key = (seq_no, task_id, access_type)
            operand_index = operand_counters[operand_counter_key]
            operand_counters[operand_counter_key] += 1
            if not all_concrete:
                counters["skippedNonConcrete"] += 1
                continue
            if end <= base:
                counters["skippedInvalidRange"] += 1
                continue
            if len(offset) != len(shape) or not shape or any(value <= 0 for value in shape):
                counters["skippedInvalidShape"] += 1
                continue

            access = Access(
                row_number=row_number,
                seq_no=seq_no,
                func_idx=func_idx,
                op_idx=op_idx,
                task_id=task_id,
                root_index=root_index,
                access_type=access_type,
                operand_index=operand_index,
                base=base,
                end=end,
                offset=offset,
                shape=shape,
                raw_shape=raw_shape_values or None,
                location=location,
                all_concrete=all_concrete,
                raw_magic=raw_magic,
            )
            accesses[access.key].append(access)
            counters["keptRows"] += 1
            counters[f"kept{access_type}"] += 1
    return dict(accesses), counters


def validate_task_sets(topo_tasks: Dict[TaskKey, TopoTask], accesses: Dict[TaskKey, List[Access]]):
    missing = sorted(set(accesses) - set(topo_tasks))
    if not missing:
        return
    examples = ", ".join(f"({key.seq_no},{key.task_id})" for key in missing[:10])
    suffix = " ..." if len(missing) > 10 else ""
    raise ValueError(
        f"{len(missing)} access task(s) have no corresponding dyn_topo node: {examples}{suffix}."
        "Please confirm both files are from the same run."
    )


def validate_topology(topo_tasks: Dict[TaskKey, TopoTask]):
    dangling = []
    for key, task in topo_tasks.items():
        for successor in task.successors:
            successor_key = TaskKey(key.seq_no, successor)
            if successor_key not in topo_tasks:
                dangling.append((key, successor))
    if not dangling:
        return
    examples = ", ".join(
        f"({key.seq_no},{key.task_id})->{successor}" for key, successor in dangling[:10]
    )
    suffix = " ..." if len(dangling) > 10 else ""
    raise ValueError(
        f"dyn_topo has {len(dangling)} dangling successor edge(s): {examples}{suffix}."
        "Cannot reliably determine task ordering when topology is incomplete."
    )


def build_graph(topo_tasks: Dict[TaskKey, TopoTask]) -> Dict[int, Dict[int, Set[int]]]:
    graph: Dict[int, Dict[int, Set[int]]] = defaultdict(dict)
    for key, task in topo_tasks.items():
        graph[key.seq_no][key.task_id] = set(task.successors)
    return dict(graph)


class ReachabilityIndex:
    """Lazily compute and cache topology reachability sets only for tasks with spatial conflict candidates."""

    def __init__(self, graph: Dict[int, Dict[int, Set[int]]]):
        self._graph = graph
        self._closure_cache: Dict[TaskKey, Set[int]] = {}
        self._ordered_pair_cache: Dict[Tuple[TaskKey, TaskKey], bool] = {}

    @property
    def queried_pair_count(self) -> int:
        return len(self._ordered_pair_cache)

    @property
    def closure_source_count(self) -> int:
        return len(self._closure_cache)

    def _reachable_from(self, key: TaskKey) -> Set[int]:
        cached = self._closure_cache.get(key)
        if cached is not None:
            return cached
        seq_graph = self._graph.get(key.seq_no, {})
        visited: Set[int] = set()
        stack = list(seq_graph.get(key.task_id, set()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(seq_graph.get(current, set()) - visited)
        self._closure_cache[key] = visited
        return visited

    def is_ordered(self, lhs: TaskKey, rhs: TaskKey) -> bool:
        if lhs.seq_no != rhs.seq_no:
            raise ValueError("Can only query task dependencies within the same seqNo")
        pair = (lhs, rhs) if lhs < rhs else (rhs, lhs)
        cached = self._ordered_pair_cache.get(pair)
        if cached is not None:
            return cached
        ordered = (
            rhs.task_id in self._reachable_from(lhs)
            or lhs.task_id in self._reachable_from(rhs)
        )
        self._ordered_pair_cache[pair] = ordered
        return ordered


def _access_overlap_kind(src: Access, dst: Access) -> Optional[str]:
    if src.end <= dst.base or dst.end <= src.base:
        return None
    if src.base != dst.base or src.end != dst.end:
        # 与 schema_memory_check 的 "memory reuse must happen for full match" 对应。
        return "INVALID_PARTIAL_ADDRESS_OVERLAP"
    if src.raw_shape and dst.raw_shape and src.raw_shape != dst.raw_shape:
        # 相同物理分配的逻辑布局不同，offset/shape 已不处于可直接比较的同一坐标系。
        return "INVALID_RAW_SHAPE_MISMATCH"
    if len(src.offset) != len(dst.offset) or len(src.shape) != len(dst.shape):
        # 与 schema_memory_check 的 "memory reuse must happen for same dimension" 对应。
        return "INVALID_DIMENSION_MISMATCH"
    for dim in range(len(src.offset)):
        if src.offset[dim] + src.shape[dim] <= dst.offset[dim]:
            return None
        if dst.offset[dim] + dst.shape[dim] <= src.offset[dim]:
            return None
    return "ALL_DIMENSIONS_OVERLAP"


def check_overlaps(
    topo_tasks: Dict[TaskKey, TopoTask],
    accesses: Dict[TaskKey, List[Access]],
) -> Tuple[List[Conflict], Counter]:
    graph = build_graph(topo_tasks)
    reachability = ReachabilityIndex(graph)
    by_seq: Dict[int, List[Access]] = defaultdict(list)
    for task_accesses in accesses.values():
        for access in task_accesses:
            by_seq[access.seq_no].append(access)

    conflicts: List[Conflict] = []
    counters: Counter = Counter()
    for seq_no, seq_accesses in sorted(by_seq.items()):
        seq_accesses.sort(key=lambda access: (access.base, access.end, access.row_number))
        counters["deviceTasks"] += 1
        counters["sweepAccesses"] += len(seq_accesses)
        active: List[Access] = []
        for current in seq_accesses:
            # 半开区间：candidate.end == current.base 时二者不相交，可从 active 移除。
            active = [candidate for candidate in active if candidate.end > current.base]
            for candidate in active:
                counters["addressCandidatePairs"] += 1
                if candidate.key == current.key:
                    counters["skippedSameTaskPairs"] += 1
                    continue
                src, dst = (
                    (candidate, current)
                    if candidate.key < current.key
                    else (current, candidate)
                )
                if not src.is_write and not dst.is_write:
                    counters["skippedReadReadPairs"] += 1
                    continue
                counters["spatialRuleChecks"] += 1
                overlap_kind = _access_overlap_kind(src, dst)
                if overlap_kind is None:
                    counters["dimensionDisjointPairs"] += 1
                    continue
                counters["spatialConflictCandidates"] += 1
                if reachability.is_ordered(src.key, dst.key):
                    counters["orderedSpatialCandidates"] += 1
                    continue
                counters["possiblyConcurrentCandidates"] += 1
                race_kind = (
                    "RACE_WRITE_WRITE"
                    if src.is_write and dst.is_write
                    else "RACE_READ_WRITE"
                )
                conflicts.append(
                    Conflict(
                        race_kind=race_kind,
                        overlap_kind=overlap_kind,
                        src=src,
                        dst=dst,
                        src_topo=topo_tasks[src.key],
                        dst_topo=topo_tasks[dst.key],
                    )
                )
                counters[race_kind] += 1
                counters[overlap_kind] += 1
            active.append(current)
    counters["topologyTaskPairQueries"] = reachability.queried_pair_count
    counters["reachabilityClosureSources"] = reachability.closure_source_count
    counters["conflicts"] = len(conflicts)
    return conflicts, counters


def _format_interval(begin: int, end: int) -> str:
    return f"[{begin}, {end})"


def _format_hex_interval(begin: int, end: int) -> str:
    return f"[0x{begin:x}, 0x{end:x})"


def _print_task(role: str, access: Access, topo: TopoTask):
    print(f"{role}:")
    print(
        f"  DeviceTask: seqNo={access.seq_no}; "
        f"RootFunction: funcIdx={access.func_idx}, rootIndex={access.root_index}, "
        f"topoRootIndex={topo.root_index}, rootHash={topo.root_hash}"
    )
    print(
        f"  Op: taskId={access.task_id}, opIdx={access.op_idx}, opmagic={topo.opmagic}, "
        f"leafIndex={topo.leaf_index}, leafHash={topo.leaf_hash}, "
        f"coreType={topo.core_type}, psgId={topo.psg_id}, wrapId={topo.wrap_id}"
    )


def _print_operand(role: str, access: Access):
    access_mode = "WRITE" if access.is_write else "READ"
    print(
        f"  {role}: {access.operand_name}, access={access_mode}, csvRow={access.row_number}, "
        f"location={access.location}, rawMagic={access.raw_magic}"
    )
    print(
        f"  RawTensor: address={_format_hex_interval(access.base, access.end)}, "
        f"size={access.end - access.base} bytes"
    )
    print(f"  Shape: offset={access.offset}, shape={access.shape}, rawShape={access.raw_shape}")


def _print_overlap_detail(conflict: Conflict):
    src, dst = conflict.src, conflict.dst
    address_begin = max(src.base, dst.base)
    address_end = min(src.end, dst.end)
    print("Overlap:")
    print(
        f"  Physical intersection: {_format_hex_interval(address_begin, address_end)}, "
        f"size={address_end - address_begin} bytes"
    )
    if conflict.overlap_kind != "ALL_DIMENSIONS_OVERLAP":
        print("  Dimension intersection: unavailable because allocation ranges/dimensions are inconsistent")
        return
    for dim in range(len(src.offset)):
        src_begin = src.offset[dim]
        src_end = src_begin + src.shape[dim]
        dst_begin = dst.offset[dim]
        dst_end = dst_begin + dst.shape[dim]
        overlap_begin = max(src_begin, dst_begin)
        overlap_end = min(src_end, dst_end)
        print(
            f"  dim[{dim}]: src={_format_interval(src_begin, src_end)}, "
            f"dst={_format_interval(dst_begin, dst_end)}, "
            f"intersection={_format_interval(overlap_begin, overlap_end)}"
        )


def _print_conflict_kind_guide():
    print()
    print("Overlap Kind descriptions:")
    print(
        "  ALL_DIMENSIONS_OVERLAP: Two possibly concurrent tasks access the same full physical allocation,"
        "and all dimension access ranges overlap."
    )
    print(
        "  INVALID_PARTIAL_ADDRESS_OVERLAP: Two RawTensor physical ranges partially overlap,"
        "but not the same full allocation; check address, size and allocation logic."
    )
    print(
        "  INVALID_RAW_SHAPE_MISMATCH: Two RawTensors use the same full physical range,"
        "but rawShape differs, cannot reliably compare in the same coordinate system."
    )
    print(
        "  INVALID_DIMENSION_MISMATCH: Two RawTensors use the same full physical range,"
        "but offset/shape dimension count differs, cannot reliably compare per-dimension intersection."
    )
    print("Race Kind descriptions:")
    print("  RACE_READ_WRITE: One task reads, another writes to conflicting memory.")
    print("  RACE_WRITE_WRITE: Both tasks write to conflicting memory.")


def print_report(
    conflicts: List[Conflict],
    unchecked_access_count: int,
):
    print(SEPARATOR)
    print("PyPTO Memory Overlap Check (dyn_topo based)")
    print(SEPARATOR)
    if conflicts:
        print(f"RESULT: FAIL - Found {len(conflicts)} memory overlap(s).")
        if unchecked_access_count:
            print(f"WARNING: {unchecked_access_count} access record(s) are invalid and were not checked.")
    elif unchecked_access_count:
        print(f"RESULT: FAIL - {unchecked_access_count} access record(s) are invalid, cannot complete full check.")
    else:
        print("RESULT: PASS - No Read-Write/Write-Write memory overlap found between topology-independent tasks.")

    if not conflicts:
        return

    _print_conflict_kind_guide()
    indexed_conflicts = list(enumerate(conflicts, start=1))
    compact_conflicts = [
        (index, conflict)
        for index, conflict in indexed_conflicts
        if conflict.overlap_kind != "ALL_DIMENSIONS_OVERLAP"
    ]
    if compact_conflicts:
        print()
        print(SEPARATOR)
        print("Non-full-dimension overlap anomalies (compact list)")
        print(SEPARATOR)
        for conflict_index, conflict in compact_conflicts:
            print(
                f"[Conflict #{conflict_index}] overlapKind={conflict.overlap_kind}, "
                f"raceKind={conflict.race_kind}, "
                f"src=(funcIdx={conflict.src.func_idx}, opIdx={conflict.src.op_idx}), "
                f"dst=(funcIdx={conflict.dst.func_idx}, opIdx={conflict.dst.op_idx})"
            )

    all_dimension_conflicts: Dict[
        Tuple[TaskKey, TaskKey], List[Tuple[int, Conflict]]
    ] = defaultdict(list)
    for conflict_index, conflict in indexed_conflicts:
        if conflict.overlap_kind == "ALL_DIMENSIONS_OVERLAP":
            all_dimension_conflicts[(conflict.src.key, conflict.dst.key)].append(
                (conflict_index, conflict)
            )

    if all_dimension_conflicts:
        print()
        print(SEPARATOR)
        print("ALL_DIMENSIONS_OVERLAP details")
        print(SEPARATOR)
        for group_index, (task_pair, indexed_group) in enumerate(
            sorted(all_dimension_conflicts.items(), key=lambda item: item[0]),
            start=1,
        ):
            group = [conflict for _, conflict in indexed_group]
            representative = group[0]
            print()
            print(SEPARATOR)
            print(
                f"[Task Pair #{group_index}] seqNo={task_pair[0].seq_no}, "
                f"srcTaskId={task_pair[0].task_id}, dstTaskId={task_pair[1].task_id}, "
                f"operandConflicts={len(group)}"
            )
            print(
                "Dependency: Two Ops are in the same DeviceTask and mutually unreachable "
                "in dyn_topo transitive closure, thus may execute concurrently."
            )
            _print_task("Source Task", representative.src, representative.src_topo)
            _print_task("Destination Task", representative.dst, representative.dst_topo)
            for conflict_index, conflict in indexed_group:
                print()
                print(
                    f"  [Operand Conflict #{conflict_index}] raceKind={conflict.race_kind}, "
                    f"overlapKind={conflict.overlap_kind}"
                )
                _print_operand("Source Operand", conflict.src)
                _print_operand("Destination Operand", conflict.dst)
                _print_overlap_detail(conflict)


def run(output_dir: str) -> int:
    dyn_topo_path, access_csv_path = resolve_inputs(output_dir)
    logging.info("dyn_topo: %s", dyn_topo_path)
    logging.info("memory access dump: %s", access_csv_path)

    topo_tasks = load_dyn_topo(dyn_topo_path)
    accesses, access_stats = load_accesses(access_csv_path)
    validate_topology(topo_tasks)
    validate_task_sets(topo_tasks, accesses)
    logging.info("loaded topo tasks=%d, access tasks=%d", len(topo_tasks), len(accesses))

    conflicts, _ = check_overlaps(topo_tasks, accesses)
    unchecked_access_count = sum(
        access_stats[name]
        for name in ("skippedNonConcrete", "skippedInvalidRange", "skippedInvalidShape")
    )
    print_report(conflicts, unchecked_access_count)
    return 1 if conflicts or unchecked_access_count else 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check memory overlap between topology-independent Ops based on dyn_topo.txt and mem_rawtensor_access.csv "
            "in the output directory."
        )
    )
    parser.add_argument("output_dir", help="Output directory of a single run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.isdir(args.output_dir):
        print(f"RESULT: FAIL - output directory does not exist: {args.output_dir}")
        return 1
    try:
        return run(os.path.abspath(args.output_dir))
    except (OSError, ValueError) as error:
        print(f"RESULT: FAIL - memory overlap check failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
