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
DUMP_DEVICE_PERF 落盘与 machine_perf_trace analyze 汇总语义看护。
"""

import json
import multiprocessing as mp
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch_npu

import pypto

_MULTI_STITCH_NUM_CHUNKS = 128
_MULTI_STITCH_CHUNK_SIZE = 64
_MULTI_STITCH_TOTAL_SIZE = _MULTI_STITCH_NUM_CHUNKS * _MULTI_STITCH_CHUNK_SIZE
_MULTI_STITCH_RUN_COUNT = 3


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 64})
def multi_stitch_kernel(
    x: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(_MULTI_STITCH_CHUNK_SIZE)

    for i in pypto.loop(_MULTI_STITCH_NUM_CHUNKS, name="stitch_loop", idx_name="i"):
        chunk = pypto.view(x, [_MULTI_STITCH_CHUNK_SIZE], [i * _MULTI_STITCH_CHUNK_SIZE])
        result = pypto.add(chunk, 1.0)
        pypto.assemble(result, [i * _MULTI_STITCH_CHUNK_SIZE], out)


def _device_run_multi_stitch(queue):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    os.environ["DUMP_DEVICE_PERF"] = "true"

    x = torch.arange(_MULTI_STITCH_TOTAL_SIZE, dtype=torch.float32, device=f'npu:{device_id}')

    for _ in range(_MULTI_STITCH_RUN_COUNT):
        out = torch.zeros(_MULTI_STITCH_TOTAL_SIZE, dtype=torch.float32, device=f'npu:{device_id}')
        multi_stitch_kernel(x, out)

    torch_npu.npu.synchronize()
    pref_path = pypto.pypto_impl.LogTopFolder()
    queue.put(pref_path)


_MIN_E2E_US = 0.01
_MAX_US = 1e9
_AICORE_ROW_NAME = "AICore"
# (col_idx, metric_name, hint, require_positive) — require_positive 时须 > _MIN_E2E_US
_SUMMARY_ROW_REQUIREMENTS: Dict[str, Tuple[Tuple[int, str, str, bool], ...]] = {
    "AICPU-CTRL": (
        (1, "DEV_TASK_BUILD", "analyze could not match DEV_TASK_BUILD events", False),
        (6, "Post-process", "analyze could not match EXIT event", False),
        (8, "Total run time", "analyze could not compute CTRL total_runtime", True),
    ),
    "AICPU-SCHED": (
        (2, "ALLOC_THREAD_ID", "analyze could not match ALLOC_THREAD_ID event", False),
        (3, "INIT", "analyze could not match INIT event", False),
        (4, "CORE_HAND_SHAKE", "analyze could not match CORE_HAND_SHAKE event", False),
        (5, "DEV_TASK_RCV", "analyze could not match DEV_TASK_RCV event", False),
        (6, "Post-process", "analyze could not match SCHED post-process events", False),
        (8, "Total run time", "analyze could not compute SCHED total_runtime", True),
    ),
    _AICORE_ROW_NAME: (
        (3, "INIT(preprocess)", "analyze could not match WAIT_RCV_FIRST_LEAF_TASK / min_end events", False),
        (6, "Post-process", "analyze could not match ALL_LEAF_TASK_EXEC / max_end events", False),
        (7, "End-to-End time", "analyze could not match key AICore perf events in dump", True),
        (8, "Total run time", "analyze could not compute AICore total_runtime", True),
    ),
}
_MACHINE_PERF_TRACE_MODULE: Optional[Any] = None


def _repo_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def _load_machine_perf_trace_module():
    """懒加载 analyze 脚本，避免与 python/tests/st/operator 包名冲突。"""
    global _MACHINE_PERF_TRACE_MODULE
    if _MACHINE_PERF_TRACE_MODULE is not None:
        return _MACHINE_PERF_TRACE_MODULE
    import importlib.util

    script_path = os.path.join(_repo_root(), "tools", "scripts", "machine_perf_trace.py")
    spec = importlib.util.spec_from_file_location("machine_perf_trace_st_guard", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MACHINE_PERF_TRACE_MODULE = module
    return module


def _parse_us_cell(cell: str) -> Optional[float]:
    if cell is None or cell == "-":
        return None
    return float(cell)


def _assert_us_cell_in_range(round_id: int, row_name: str, col_idx: int, value_us: float) -> None:
    assert value_us >= 0, f"round={round_id} {row_name} col={col_idx}: negative value {value_us} us"
    assert value_us <= _MAX_US, f"round={round_id} {row_name} col={col_idx}: abnormally large value {value_us} us"


def _validate_analyze_row_cells(round_id: int, row: List) -> None:
    row_name = row[0]
    for col_idx in range(1, 9):
        value_us = _parse_us_cell(row[col_idx])
        if value_us is None:
            continue
        _assert_us_cell_in_range(round_id, row_name, col_idx, value_us)


class _RequiredUsCell(NamedTuple):
    col_idx: int
    metric_name: str
    hint: str = ""


def _assert_required_us_cell(
    round_id: int,
    row_name: str,
    value_us: Optional[float],
    cell: _RequiredUsCell,
) -> None:
    suffix = f" ({cell.hint})" if cell.hint else ""
    assert value_us is not None, f"round={round_id} {row_name} col={cell.col_idx} ({cell.metric_name}) is '-'{suffix}"
    assert value_us >= 0, (
        f"round={round_id} {row_name} col={cell.col_idx} ({cell.metric_name})={value_us} us (expected >= 0)"
    )


def _validate_summary_row(round_id: int, row: List, row_kind: str) -> None:
    row_name = row[0]
    for col_idx, metric_name, hint, require_positive in _SUMMARY_ROW_REQUIREMENTS[row_kind]:
        value_us = _parse_us_cell(row[col_idx])
        _assert_required_us_cell(
            round_id,
            row_name,
            value_us,
            _RequiredUsCell(col_idx, metric_name, hint),
        )
        if require_positive:
            assert value_us > _MIN_E2E_US, (
                f"round={round_id} {row_name} col={col_idx} ({metric_name})={value_us} us (expected > {_MIN_E2E_US})"
            )

    if row_kind == _AICORE_ROW_NAME:
        e2e_us = _parse_us_cell(row[7])
        total_us = _parse_us_cell(row[8])
        assert e2e_us <= total_us * 1.01, f"round={round_id}: e2e({e2e_us}) > total({total_us}), aggregate inconsistent"


def _validate_round_summary_rows(round_id: int, rows: List[List]) -> None:
    ctrl_row = next((row for row in rows if str(row[0]).startswith("AICPU-CTRL")), None)
    assert ctrl_row is not None, f"round={round_id}: missing AICPU-CTRL summary row"
    _validate_summary_row(round_id, ctrl_row, "AICPU-CTRL")

    sched_rows = [row for row in rows if str(row[0]).startswith("AICPU-SCHED")]
    assert sched_rows, f"round={round_id}: missing AICPU-SCHED summary row(s)"
    for sched_row in sched_rows:
        _validate_summary_row(round_id, sched_row, "AICPU-SCHED")

    aicore_row = next((row for row in rows if row[0] == _AICORE_ROW_NAME), None)
    assert aicore_row is not None, f"round={round_id}: missing AICore summary row"
    _validate_summary_row(round_id, aicore_row, _AICORE_ROW_NAME)


def _validate_analyze_semantics(core_list: List[Dict]) -> None:
    """对真实 dump JSON 跑 analyze 逻辑，校验汇总语义。"""
    mpt = _load_machine_perf_trace_module()
    trace_data = json.loads(json.dumps(core_list))
    analyzer = mpt.PerfTraceAnalyzer(trace_data)
    assert analyzer.round_ids, "no rounds found for analyze"

    for round_id in analyzer.round_ids:
        rows = analyzer.build_round_combined_rows(round_id)
        assert rows, f"round={round_id}: analyze produced empty rows"

        for row in rows:
            _validate_analyze_row_cells(round_id, row)

        _validate_round_summary_rows(round_id, rows)


@pypto.options(pass_options={"enable_slice": True})
def test_dump_perf():
    """
    看护用例：DUMP_DEVICE_PERF 落盘后，对 trace JSON 跑 analyze 并校验汇总语义。

    使用 multi_stitch_kernel（stitch_function_max_num=64, 128 loop iterations =>
    2 stitch batches/轮, RUN_COUNT=3）触发多 stitch 场景，验证 machine 框架在多 stitch
    batch 下的 perf analyze 正确性（CTRL/SCHED/AICore 汇总列非 '-'、范围与 e2e 一致性）。
    machine 框架改动导致 perf 数据错乱（事件缺失、e2e=0、聚合不一致）时，本用例应 fail。
    """
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    proc = mp.Process(target=_device_run_multi_stitch, args=(result_queue,))
    proc.start()
    proc.join()
    assert not result_queue.empty(), "Could not get perf output path"

    pref_path = result_queue.get()
    trace_json_path = os.path.join(pref_path, "machine_trace_perf_data_0.json")
    assert os.path.exists(trace_json_path), f"missing machine trace json: {trace_json_path}"

    with open(trace_json_path, "r", encoding="utf-8") as f:
        core_list: List[Dict] = json.load(f)

    _validate_analyze_semantics(core_list)
