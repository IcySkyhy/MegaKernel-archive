#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import unicodedata


def parse_log_file(log_file_path):
    all_blocks_data = []
    current_block_aicpu = []
    current_block_aicore = []
    in_perf_trace_block = False
    block_start_line = 0

    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if "Begin dump machine perf trace:" in line:
                if in_perf_trace_block:
                    print(
                        f"Error: Nested perf trace block at line {line_num}. "
                        f"Block started at line {block_start_line} is not properly closed."
                    )
                    return {"error": f"Nested perf trace block at line {line_num}"}

                in_perf_trace_block = True
                block_start_line = line_num
                current_block_aicpu = []
                current_block_aicore = []
                print(f"Found perf trace block start at line {line_num}")
                continue

            if "Finish dump machine perf trace." in line:
                if not in_perf_trace_block:
                    print(f"Error: Finish without matching begin at line {line_num}")
                    return {"error": f"Finish without matching begin at line {line_num}"}

                in_perf_trace_block = False
                if current_block_aicpu and current_block_aicore:
                    block_json_str = ''.join(current_block_aicpu) + ',' + ''.join(current_block_aicore)
                    block_json_str = block_json_str.replace(",]", "]")
                    all_blocks_data.append(block_json_str)
                    print(f"Successfully parsed performance trace block from line {block_start_line} to {line_num}")
                else:
                    print(f"Warning: Empty performance trace block from line {block_start_line} to {line_num}")

                current_block_aicpu = []
                current_block_aicore = []
                continue

            if in_perf_trace_block:
                if "tile_fwk aicpu prof:" in line:
                    match = re.search(r'tile_fwk aicpu prof:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        if content.endswith('"'):
                            content = content[:-1]
                        current_block_aicpu.append(content)
                elif "tile_fwk aicore prof:" in line:
                    match = re.search(r'tile_fwk aicore prof:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        if content.endswith('"'):
                            content = content[:-1]
                        current_block_aicore.append(content)
            else:
                if "tile_fwk aicpu prof:" in line or "tile_fwk aicore prof:" in line:
                    print(f"Warning: Ignoring prof data outside of perf trace block at line {line_num}")

    if in_perf_trace_block:
        print(f"Error: Unclosed perf trace block started at line {block_start_line}. Discarding incomplete block data.")
        return {"error": f"Unclosed perf trace block started at line {block_start_line}"}

    if all_blocks_data:
        full_json_str = '[' + ','.join(all_blocks_data) + ']'
        try:
            parsed_data = json.loads(full_json_str)
            print(f"Successfully parsed {len(all_blocks_data)} performance trace blocks")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw JSON string: {full_json_str[:500]}...")  # Print first 500 chars for debugging
            return {"error": str(e), "raw_blocks": all_blocks_data}
    else:
        print("No valid performance trace blocks found in log file")
        return []


def save_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        if isinstance(data, dict) and "raw_aicpu" in data:
            file.write('[' + data["raw_aicpu"] + ',' + data["raw_aicore"] + ']')
        else:
            json.dump(data, file, indent=2, ensure_ascii=False)


def convert_to_perfetto_format(input_json: List[Dict]) -> List[Dict]:
    trace_events = []
    thread_id = 0
    trace_events_head = {
        "args": {"name": "AICPU View", "type": "aicpu"},
        "cat": "__metadata",
        "name": "process_name",
        "ph": "M",
        "pid": 0,
    }
    trace_events.append(trace_events_head)
    for block in input_json:
        block_idx = block.get("blockIdx", 0)
        core_type = block.get("coreType", "UNKNOWN")
        freq = block.get("freq", 0)
        thread_name = f"{core_type}-{block_idx}"
        trace_events.append(
            {"name": "thread_name", "ph": "M", "pid": 0, "tid": thread_id, "args": {"name": thread_name}}
        )

        tasks = block.get("tasks", [])
        sorted_tasks = sorted(tasks, key=lambda x: x.get("end", 0))

        # 处理相同end_time的情况
        adjusted_tasks = []
        prev_end = None
        for task in sorted_tasks:
            task_copy = task.copy()
            current_end = task_copy.get("end", 0)

            # 如果当前end与上一个相同，则递增1
            if prev_end is not None and current_end == prev_end:
                task_copy["end"] = prev_end + 1
                current_end = prev_end + 1

            adjusted_tasks.append(task_copy)
            prev_end = current_end

        prev_end = None
        for task in adjusted_tasks:
            task_name = task.get("name", "UNKNOWN")
            end_time = task.get("end", 0)
            if task_name.startswith("BEGIN"):
                start_time = end_time - 1
            else:
                start_time = prev_end
            ts = start_time / freq
            dur = (end_time - start_time) / freq
            if dur <= 0:
                dur = 1 / freq
            perfetto_event = {
                "name": f"{task_name}",
                "cat": core_type,
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": 0,
                "tid": thread_id,
                "freq": freq,
            }
            trace_events.append(perfetto_event)
            prev_end = end_time
        thread_id += 1
    trace_events_pypto = {"traceEvents": trace_events}
    return trace_events_pypto


def parse_log_command(input_file, output_file):
    parsed_data = parse_log_file(input_file)
    save_json(parsed_data, output_file)
    print(f"Parsing completed, result saved to: {output_file}")


def merge_aicpu_aicore_swim_lane(aicpu_perfetto_data, input_kernel_file):
    if input_kernel_file is not None and os.path.exists(input_kernel_file):
        with open(input_kernel_file, 'r', encoding='utf-8') as f:
            aicore_perfetto_data = json.load(f)
            # kernel swim lane + aicpu swim lane
            merged_trace_events = aicpu_perfetto_data["traceEvents"] + aicore_perfetto_data["traceEvents"]
            merged_data = {'traceEvents': merged_trace_events}
        with open(input_kernel_file, 'w', encoding='utf-8') as fw:
            json.dump(merged_data, fw, indent=2)


def gen_perfetto_command(input_file, output_file, input_kernel_file=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        perfetto_data = convert_to_perfetto_format(input_data)
        merge_aicpu_aicore_swim_lane(perfetto_data, input_kernel_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(perfetto_data, f, ensure_ascii=False, indent=2)
        print(f"Success to generate perfetto file: {output_file}")

        complete_events = [e for e in perfetto_data["traceEvents"] if e.get('ph') == 'X']
        metadata_events = [e for e in perfetto_data["traceEvents"] if e.get('ph') == 'M']
        print(f"Process {len(complete_events)} task events, {len(metadata_events)} meta data events")

        print("\ninfo: upload this json file to https://ui.perfetto.dev/")
    except FileNotFoundError:
        print(f"error: cannot find input file {input_file}")
    except json.JSONDecodeError:
        print(f"error: input file {input_file} is not valid json format")
    except Exception as e:
        print(f"process exception info: {str(e)}")


def gen_perfetto_example():
    sample_data = [
        {
            "blockIdx": 0,
            "coreType": "AICPU-SCHED",
            "freq": 50,
            "tasks": [
                {"name": "BEGIN", "end": 5236903326282},
                {"name": "ALLOC_THREAD_ID", "end": 5236903326381},
                {"name": "INIT", "end": 5236903329385},
                {"name": "HAND_SHAKE", "end": 5236903330212},
                {"name": "WAIT_ALL_TASK_FIN", "end": 5236903331821},
                {"name": "SEND_STOP", "end": 5236903332087},
                {"name": "EXIT", "end": 5236903332854},
            ],
        },
        {
            "blockIdx": 1,
            "coreType": "AICPU-SCHED",
            "freq": 50,
            "tasks": [
                {"name": "BEGIN", "end": 5236903326282},
                {"name": "ALLOC_THREAD_ID", "end": 5236903326383},
                {"name": "INIT", "end": 5236903329389},
                {"name": "HAND_SHAKE", "end": 5236903330219},
                {"name": "WAIT_SEND_FIRST_TASK", "end": 5236903331446},
                {"name": "WAIT_ALL_TASK_FIN", "end": 5236903331829},
                {"name": "SEND_STOP", "end": 5236903332102},
                {"name": "EXIT", "end": 5236903333765},
            ],
        },
    ]

    result = convert_to_perfetto_format(sample_data)
    with open('perfetto_output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("example have saved to perfetto_output.json")
    print("You can check it by upload this file to https://ui.perfetto.dev/")


def load_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def to_us(cycles: float, freq: float) -> float:
    return safe_div(cycles, freq)


def display_width(text: str) -> int:
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return width


def pad_cell(text: str, width: int) -> str:
    pad = max(width - display_width(text), 0)
    return text + (" " * pad)


def render_table_lines(headers: List[str], rows: List[List[str]]) -> List[str]:
    line_rows: List[List[str]] = [[str(x) for x in headers]]
    line_rows.extend([[str(x) for x in row] for row in rows])

    widths = [display_width(h) for h in headers]
    for row in line_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], display_width(cell))

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(pad_cell(cell, widths[i]) for i, cell in enumerate(row)) + " |"

    def fmt_border() -> str:
        return "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"

    lines = [fmt_border(), fmt_row(line_rows[0]), fmt_border()]
    for row in line_rows[1:]:
        lines.append(fmt_row(row))
    lines.append(fmt_border())
    return lines


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    for line in render_table_lines(headers, rows):
        print(line)


def print_section(title: str, total_width: Optional[int] = None) -> None:
    if total_width is None:
        term_cols = shutil.get_terminal_size(fallback=(100, 20)).columns
        total_width = max(60, term_cols - 2)
    else:
        total_width = max(total_width, 20)
    text = f" {title} "
    text_width = display_width(text)
    if text_width >= total_width:
        print(f"\n{text}")
        return
    left = (total_width - text_width) // 2
    right = total_width - text_width - left
    print("\n" + ("=" * left) + text + ("=" * right))


def parse_task_name(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    m = re.match(r"^([A-Z0-9_]+?)(?:_(\d+))?(?:\((\d+)\))?$", str(name))
    if not m:
        return str(name), None, None
    base = m.group(1)
    round_id = int(m.group(2)) if m.group(2) is not None else None
    idx = int(m.group(3)) if m.group(3) is not None else None
    return base, round_id, idx


def collect_round_ids(aicpu_dev_pref: List[Dict[str, Any]]) -> List[Optional[int]]:
    round_ids = set()
    for core in aicpu_dev_pref:
        for task in core.get("tasks", []):
            _, round_id, _ = parse_task_name(task.get("name", ""))
            if round_id is not None:
                round_ids.add(round_id)
    if not round_ids:
        return [None]
    return sorted(round_ids)


def calc_duration_from_ends(start_end: Optional[float], end_end: Optional[float]) -> Optional[float]:
    if start_end is None or end_end is None:
        return None
    return end_end - start_end


def sort_tasks_by_end(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(tasks, key=lambda x: float(x.get("end", 0)))


def normalize_cores_tasks_by_end(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    for core in aicpu_dev_pref:
        tasks = core.get("tasks", [])
        if isinstance(tasks, list):
            core["tasks"] = sort_tasks_by_end(tasks)


class CoreRoundStats:
    """Single-pass index for one core's tasks under a round filter."""

    _SCHED_POST_EVENTS: Tuple[Tuple[str, Optional[int]], ...] = (
        ("DEV_TASK_SYNC_CORE_STOP", 0),
        ("DEV_TASK_RSP", 0),
        ("WAIT_ALL_DEV_TASK_FINISH", None),
        ("WAIT_CORE_EXIT", None),
        ("EXIT", None),
    )

    def __init__(self, sorted_tasks: List[Dict[str, Any]], round_id: Optional[int]) -> None:
        self._first_duration_by_base: Dict[str, float] = {}
        self._durations: Dict[Tuple[str, Optional[int]], float] = {}
        self._duration_sums: Dict[str, float] = {}
        self._duration_sum_bases: Set[str] = set()
        self._first_end_by_base: Dict[str, float] = {}
        self._end_cycles_first: Dict[Tuple[str, Optional[int]], float] = {}
        self._end_cycle_maps: Dict[str, Dict[int, float]] = {}
        self._ordered_events: List[Tuple[str, float]] = []
        self.total_runtime: Optional[float] = None
        self.min_end: Optional[float] = None
        self.max_end: Optional[float] = None
        self._scan(sorted_tasks, round_id)

    def get_event_duration(self, event_name: str, idx: Optional[int] = None) -> Optional[float]:
        if idx is not None:
            return self._durations.get((event_name, idx))
        return self._first_duration_by_base.get(event_name)

    def get_event_duration_sum(self, event_name: str) -> Optional[float]:
        if event_name not in self._duration_sum_bases:
            return None
        return self._duration_sums[event_name]

    def get_end_cycle(self, event_name: str, idx: Optional[int] = None) -> Optional[float]:
        if idx is not None:
            return self._end_cycles_first.get((event_name, idx))
        return self._first_end_by_base.get(event_name)

    def get_end_cycle_map(self, event_name: str) -> Dict[int, float]:
        return dict(self._end_cycle_maps.get(event_name, {}))

    def calc_sched_post_process_sum(self) -> Optional[float]:
        total = 0.0
        found = False
        for event_name, idx in self._SCHED_POST_EVENTS:
            dur = self.get_event_duration(event_name, idx)
            if dur is None:
                continue
            found = True
            total += max(dur, 0.0)
        return total if found else None

    def calc_runtime_in_window(self, start_event: str, end_events: List[str]) -> Optional[float]:
        start_cycle = self.get_end_cycle(start_event)
        if start_cycle is None:
            return None

        end_candidates: List[float] = []
        for end_event in end_events:
            for base, task_end in self._ordered_events:
                if base != end_event:
                    continue
                if task_end >= float(start_cycle):
                    end_candidates.append(task_end)
                    break
        if not end_candidates:
            return None
        return max(max(end_candidates) - float(start_cycle), 0.0)

    def _scan(self, sorted_tasks: List[Dict[str, Any]], round_id: Optional[int]) -> None:
        prev_end: Optional[float] = None
        total = 0.0
        found = False
        for task in sorted_tasks:
            base, task_round, idx = parse_task_name(task.get("name", ""))
            if round_id is not None and task_round != round_id:
                continue

            task_end = float(task.get("end", 0))
            event_dur = None if prev_end is None else (task_end - prev_end)
            self._ordered_events.append((base, task_end))

            if event_dur is not None:
                if base not in self._first_duration_by_base:
                    self._first_duration_by_base[base] = event_dur
                key = (base, idx)
                if key not in self._durations:
                    self._durations[key] = event_dur
                self._duration_sums[base] = self._duration_sums.get(base, 0.0) + max(event_dur, 0.0)
                self._duration_sum_bases.add(base)
                total += max(event_dur, 0.0)

            if base not in self._first_end_by_base:
                self._first_end_by_base[base] = task_end
            key = (base, idx)
            if key not in self._end_cycles_first:
                self._end_cycles_first[key] = task_end
            if idx is not None:
                self._end_cycle_maps.setdefault(base, {})[idx] = task_end

            if self.min_end is None:
                self.min_end = task_end
            self.max_end = task_end
            prev_end = task_end
            found = True
        self.total_runtime = total if found else None


def build_core_round_stats(sorted_tasks: List[Dict[str, Any]], round_id: Optional[int]) -> CoreRoundStats:
    return CoreRoundStats(sorted_tasks, round_id)


def format_us(v: Optional[float], freq: float) -> str:
    if v is None:
        return "-"
    return f"{to_us(v, freq):.2f}"


def calc_aicore_init_preprocess_us(aicore_exec_rows: List[Dict[str, Any]]) -> Optional[float]:
    min_end_values = [float(row["min_end"]) for row in aicore_exec_rows if row.get("min_end") is not None]
    wait_first_values = [
        float(row["wait_first_cycle"]) for row in aicore_exec_rows if row.get("wait_first_cycle") is not None
    ]
    if not min_end_values or not wait_first_values:
        return None

    freq = float(aicore_exec_rows[0].get("freq", 1.0)) or 1.0
    return to_us(max(min(wait_first_values) - min(min_end_values), 0.0), freq)


def calc_aicore_last_core_exit_wait_us(
    aicore_exec_rows: List[Dict[str, Any]],
    aicore_core_bounds: List[Dict[str, Any]],
) -> Optional[float]:
    all_exec_values = [
        float(row["all_exec_cycle"]) for row in aicore_exec_rows if row.get("all_exec_cycle") is not None
    ]
    max_end_values = [float(row["max_end"]) for row in aicore_core_bounds if row.get("max_end") is not None]
    if not all_exec_values or not max_end_values:
        return None

    freq = float(aicore_core_bounds[0].get("freq", 1.0)) or 1.0
    return to_us(max(max(max_end_values) - max(all_exec_values), 0.0), freq)


def format_sched_post_process(post_dur_cycles: Optional[float], sched_freq: float) -> str:
    if post_dur_cycles is None:
        return "-"
    return f"{to_us(post_dur_cycles, sched_freq):.2f}"


def is_aicore_sched(core_type: str) -> bool:
    return core_type.startswith("SCHED") and ("-AIC" in core_type or "-AIV" in core_type)


def _collect_aicore_from_sched_core(
    core: Dict[str, Any], stats: CoreRoundStats
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (exec_row, core_bound) for one SCHED-AIC/AIV core; (None, None) if skipped."""
    core_type = str(core.get("coreType", ""))
    if not is_aicore_sched(core_type):
        return None, None

    block_idx = int(core.get("blockIdx", -1))
    freq = float(core.get("freq", 0)) or 1.0

    core_bound = None
    if stats.min_end is not None and stats.max_end is not None:
        core_bound = {
            "core_type": core_type,
            "block_idx": block_idx,
            "freq": freq,
            "min_end": stats.min_end,
            "max_end": stats.max_end,
        }

    wait_first_map = stats.get_end_cycle_map("DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK")
    all_exec_map = stats.get_end_cycle_map("DEV_TASK_ALL_LEAF_TASK_EXEC")
    if not all_exec_map:
        return None, core_bound

    wait_first_cycle = min(wait_first_map.values()) if wait_first_map else None
    all_exec_cycle = max(all_exec_map.values()) if all_exec_map else None
    first_wait_first_idx = min(wait_first_map, key=wait_first_map.get) if wait_first_map else None
    wait_exit_notify = stats.get_end_cycle("WAIT_EXIT_NOTIFY")
    exit_wait = calc_duration_from_ends(all_exec_cycle, wait_exit_notify)
    exec_row = {
        "core_type": core_type,
        "block_idx": block_idx,
        "freq": freq,
        "wait_first_map": wait_first_map,
        "all_exec_map": all_exec_map,
        "exit_wait": exit_wait,
        "wait_first_cycle": wait_first_cycle,
        "all_exec_cycle": all_exec_cycle,
        "init_dur": stats.get_event_duration("INIT"),
        "rcv_model_dur": stats.get_event_duration("DEV_TASK_RCV_MODEL"),
        "wait_first_dur": stats.get_event_duration("DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK", first_wait_first_idx),
        "min_end": stats.min_end,
        "max_end": stats.max_end,
    }
    return exec_row, core_bound


ANALYZE_TABLE_HEADERS: Tuple[str, ...] = (
    "Compute Units",
    "DEV_TASK_BUILD(us)",
    "ALLOC_THREAD_ID(us)",
    "INIT(us)",
    "CORE_HAND_SHAKE(us)",
    "DEV_TASK_RCV(us)",
    "Post-process(us)",
    "End-to-End time(us)",
    "Total run time(us)",
)


def _format_ctrl_row(ctrl: Dict[str, Any], stats: CoreRoundStats) -> List[str]:
    freq = float(ctrl.get("freq", 0)) or 1.0
    block_idx = int(ctrl.get("blockIdx", 0))
    return [
        f"AICPU-CTRL-{block_idx}",
        format_us(stats.get_event_duration_sum("DEV_TASK_BUILD"), freq),
        "-",
        "-",
        "-",
        "-",
        format_us(stats.get_event_duration("EXIT"), freq),
        "-",
        format_us(stats.total_runtime, freq),
    ]


def _format_sched_row(sched: Dict[str, Any], stats: CoreRoundStats) -> List[str]:
    block_idx = int(sched.get("blockIdx", -1))
    freq = float(sched.get("freq", 0)) or 1.0
    sched_total_dur = stats.calc_runtime_in_window("BEGIN", ["WAIT_CORE_EXIT", "EXIT"])
    if sched_total_dur is None:
        sched_total_dur = stats.total_runtime
    return [
        f"AICPU-SCHED-{block_idx}",
        "-",
        format_us(stats.get_event_duration("ALLOC_THREAD_ID"), freq),
        format_us(stats.get_event_duration("INIT"), freq),
        format_us(stats.get_event_duration("CORE_HAND_SHAKE"), freq),
        format_us(stats.get_event_duration("DEV_TASK_RCV", 0), freq),
        format_sched_post_process(stats.calc_sched_post_process_sum(), freq),
        "-",
        format_us(sched_total_dur, freq),
    ]


def _format_aicore_summary_row(
    aicore_exec_rows: List[Dict[str, Any]],
    aicore_core_bounds: List[Dict[str, Any]],
) -> List[str]:
    aicore_post_process_us = calc_aicore_last_core_exit_wait_us(aicore_exec_rows, aicore_core_bounds)
    aicore_init_us = calc_aicore_init_preprocess_us(aicore_exec_rows)
    if not aicore_exec_rows:
        return ["AICore", "-", "-", "-", "-", "-", "-", "-", "-"]
    e2e_time, total_runtime_e2e = calc_aicore_timing_summary(aicore_exec_rows, aicore_core_bounds)
    return [
        "AICore",
        "-",
        "-",
        "-" if aicore_init_us is None else f"{aicore_init_us:.2f}",
        "-",
        "-",
        "-" if aicore_post_process_us is None else f"{aicore_post_process_us:.2f}",
        e2e_time,
        total_runtime_e2e,
    ]


class PerfTraceAnalyzer:
    """Read trace once; cache CoreRoundStats per (coreType, blockIdx, round_id)."""

    def __init__(self, cores: List[Dict[str, Any]]) -> None:
        if not isinstance(cores, list):
            raise TypeError("perf trace data must be a list of core dicts")
        self.cores = cores
        normalize_cores_tasks_by_end(self.cores)
        self.round_ids = collect_round_ids(self.cores)
        self._stats_cache: Dict[Tuple[str, int, Optional[int]], CoreRoundStats] = {}
        self._build_stats_cache()

    @staticmethod
    def _core_key(core: Dict[str, Any]) -> Tuple[str, int]:
        return (str(core.get("coreType", "")), int(core.get("blockIdx", -1)))

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "PerfTraceAnalyzer":
        return cls(load_json(Path(file_path)))

    def stats_for_core(self, core: Dict[str, Any], round_id: Optional[int]) -> CoreRoundStats:
        core_type, block_idx = self._core_key(core)
        return self._stats_cache[(core_type, block_idx, round_id)]

    def collect_aicore_rows(self, round_id: Optional[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        exec_rows: List[Dict[str, Any]] = []
        core_bounds: List[Dict[str, Any]] = []
        for core in self.cores:
            exec_row, core_bound = _collect_aicore_from_sched_core(
                core,
                self.stats_for_core(core, round_id),
            )
            if core_bound is not None:
                core_bounds.append(core_bound)
            if exec_row is not None:
                exec_rows.append(exec_row)
        exec_rows.sort(key=lambda x: x["block_idx"])
        core_bounds.sort(key=lambda x: x["block_idx"])
        return exec_rows, core_bounds

    def build_ctrl_row(self, round_id: Optional[int]) -> Optional[List[str]]:
        ctrl = next((x for x in self.cores if str(x.get("coreType")) == "AICPU-CTRL"), None)
        if ctrl is None:
            return None
        return _format_ctrl_row(ctrl, self.stats_for_core(ctrl, round_id))

    def build_sched_rows(self, round_id: Optional[int]) -> List[List[str]]:
        scheds = [x for x in self.cores if str(x.get("coreType")) == "AICPU-SCHED"]
        return [
            _format_sched_row(sched, self.stats_for_core(sched, round_id))
            for sched in sorted(scheds, key=lambda x:int(x.get("blockIdx", 0)))
        ]

    def build_aicore_row(self, round_id: Optional[int]) -> List[str]:
        exec_rows, core_bounds = self.collect_aicore_rows(round_id)
        return _format_aicore_summary_row(exec_rows, core_bounds)

    def build_round_combined_rows(self, round_id: Optional[int]) -> List[List[str]]:
        rows: List[List[str]] = []
        ctrl_row = self.build_ctrl_row(round_id)
        if ctrl_row is not None:
            rows.append(ctrl_row)
        rows.extend(self.build_sched_rows(round_id))
        rows.append(self.build_aicore_row(round_id))
        return rows

    def _build_stats_cache(self) -> None:
        for core in self.cores:
            core_type, block_idx = self._core_key(core)
            tasks = core.get("tasks", [])
            for round_id in self.round_ids:
                self._stats_cache[(core_type, block_idx, round_id)] = CoreRoundStats(tasks, round_id)


def collect_aicore_rows(
    aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return PerfTraceAnalyzer(aicpu_dev_pref).collect_aicore_rows(round_id)


def _min_optional(current: Optional[float], value: float) -> float:
    if current is None:
        return value
    return min(current, value)


def _max_optional(current: Optional[float], value: float) -> float:
    if current is None:
        return value
    return max(current, value)


def _accumulate_aicore_exec_timing(
    aicore_exec_rows: List[Dict[str, Any]],
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    freq = 1.0
    min_end: Optional[float] = None
    max_end: Optional[float] = None
    first_wait_cycle: Optional[float] = None
    last_all_exec_cycle: Optional[float] = None
    for row in aicore_exec_rows:
        row_min_end = row.get("min_end")
        row_max_end = row.get("max_end")
        if row_min_end is None or row_max_end is None:
            continue
        freq = float(row.get("freq", 1.0)) or 1.0
        min_end = _min_optional(min_end, float(row_min_end))
        max_end = _max_optional(max_end, float(row_max_end))
        wait_first_cycle = row.get("wait_first_cycle")
        if wait_first_cycle is not None:
            first_wait_cycle = _min_optional(first_wait_cycle, float(wait_first_cycle))
        all_exec_cycle = row.get("all_exec_cycle")
        if all_exec_cycle is not None:
            last_all_exec_cycle = _max_optional(last_all_exec_cycle, float(all_exec_cycle))
    return freq, min_end, max_end, first_wait_cycle, last_all_exec_cycle


def _extend_aicore_bounds_max_end(
    aicore_core_bounds: List[Dict[str, Any]],
    freq: float,
    max_end: Optional[float],
) -> Tuple[float, Optional[float]]:
    for row in aicore_core_bounds:
        row_max_end = row.get("max_end")
        if row_max_end is None:
            continue
        freq = float(row.get("freq", 1.0)) or 1.0
        max_end = _max_optional(max_end, float(row_max_end))
    return freq, max_end


def _format_aicore_timing_e2e(
    freq: float,
    first_wait_cycle: Optional[float],
    last_all_exec_cycle: Optional[float],
    min_end: Optional[float],
    max_end: Optional[float],
) -> Tuple[str, str]:
    e2e_time = "-"
    total_runtime_e2e = "-"
    if first_wait_cycle is not None and last_all_exec_cycle is not None:
        e2e_cycles = max(last_all_exec_cycle - first_wait_cycle, 0.0)
        e2e_time = f"{to_us(e2e_cycles, freq):.2f}"
    if min_end is not None and max_end is not None:
        total_runtime_e2e_cycles = max(max_end - min_end, 0.0)
        total_runtime_e2e = f"{to_us(total_runtime_e2e_cycles, freq):.2f}"
    return e2e_time, total_runtime_e2e


def calc_aicore_timing_summary(
    aicore_exec_rows: List[Dict[str, Any]],
    aicore_core_bounds: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if not aicore_exec_rows:
        return "-", "-"
    freq, min_end, max_end, first_wait_cycle, last_all_exec_cycle = _accumulate_aicore_exec_timing(aicore_exec_rows)
    freq, max_end = _extend_aicore_bounds_max_end(aicore_core_bounds, freq, max_end)
    return _format_aicore_timing_e2e(freq, first_wait_cycle, last_all_exec_cycle, min_end, max_end)


def build_ctrl_row(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> Optional[List[str]]:
    return PerfTraceAnalyzer(aicpu_dev_pref).build_ctrl_row(round_id)


def build_sched_rows(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[List[str]]:
    return PerfTraceAnalyzer(aicpu_dev_pref).build_sched_rows(round_id)


def build_aicore_row(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[str]:
    return PerfTraceAnalyzer(aicpu_dev_pref).build_aicore_row(round_id)


def build_round_combined_rows(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[List[str]]:
    return PerfTraceAnalyzer(aicpu_dev_pref).build_round_combined_rows(round_id)


def analyze_output_command(output_dir_arg: Optional[str]) -> None:
    if not output_dir_arg:
        print("Error: analyze requires an input json path")
        return

    input_path = Path(output_dir_arg)
    if not input_path.exists():
        print(f"Error: path does not exist: {input_path}")
        return

    print(f"Analyzing input: {input_path}")
    analyzer = PerfTraceAnalyzer.from_file(input_path)
    headers = list(ANALYZE_TABLE_HEADERS)
    for round_id in analyzer.round_ids:
        display_round = 1 if round_id is None else (round_id + 1)
        round_name = f"round{display_round}"
        rows = analyzer.build_round_combined_rows(round_id)
        table_lines = render_table_lines(headers, rows)
        table_width = max(display_width(line) for line in table_lines) if table_lines else None
        print_section(round_name, table_width)
        print_table(headers, rows)
    print()


def main():
    parser = argparse.ArgumentParser(description='Performance data processing tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # parse_log 子命令
    parse_parser = subparsers.add_parser('parse_log', help='Parse device log and generate performance JSON')
    parse_parser.add_argument('input_file', help='Path to input log file')
    parse_parser.add_argument('output_file', help='Path to output JSON file')

    # gen_perfetto 子命令
    perfetto_parser = subparsers.add_parser('gen_perfetto', help='Convert performance JSON to Perfetto format')
    perfetto_parser.add_argument('input_file', help='Input JSON file path')
    perfetto_parser.add_argument('output_file', help='Output Perfetto JSON file path')
    perfetto_parser.add_argument('kernel_file', help='aicore kernel Perfetto JSON file path', default="", nargs='?')

    # gen_perfetto_example 子命令
    _example_parser = subparsers.add_parser('gen_perfetto_example', help='Generate example Perfetto data')
    # analyze 子命令
    analyze_parser = subparsers.add_parser('analyze', help='Analyze perf json by output dir or json file path')
    analyze_parser.add_argument(
        'output_dir',
        nargs='?',
        help='Output directory or perf json file path; latest output_* if omitted',
    )
    args = parser.parse_args()

    if args.command == 'parse_log':
        parse_log_command(args.input_file, args.output_file)
    elif args.command == 'gen_perfetto':
        gen_perfetto_command(args.input_file, args.output_file, args.kernel_file)
    elif args.command == 'gen_perfetto_example':
        gen_perfetto_example()
    elif args.command == 'analyze':
        analyze_output_command(args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
