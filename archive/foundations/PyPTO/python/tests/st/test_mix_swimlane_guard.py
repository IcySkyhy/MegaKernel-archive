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
"""
MIX swimlane guard test for 950 platform.

Validates structural integrity and alignment of:
  - tilefwk_L1_prof_data.json  (runtime device perf data with syncEvents)
  - mix_event_info.json        (compile-time sync event metadata with syncMsg)
  - dyn_topo.txt               (task-to-leafHash mapping)
  - merged_swimlane.json       (merged swimlane output)

Guards the fix in ef024ca3 (mix_info.cpp: FFTS_CROSS_CORE_SYNC / WAIT_FLAG_DEV
filtering) and the PMU trace event-id scheme in bf32be10.

Uses the proven GLM attention kernel (models/glm_v4_5/glm_attention.py) which
produces sufficient cross-core sync events for full validation.
"""

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
import torch
import torch_npu

import pypto


def _load_dyn_topo(path):
    tasks = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        col = {name: i for i, name in enumerate(next(reader))}
        for row in reader:
            if not row or not row[0].strip():
                continue
            tasks.append(
                {
                    "seqNo": int(row[col["seqNo"]]),
                    "taskId": int(row[col["taskId"]]),
                    "leafIndex": int(row[col["leafIndex"]]),
                    "leafHash": row[col["leafHash"]].strip(),
                    "coreType": int(row[col["coreType"]]),
                }
            )
    return tasks


def _validate_tilefwk_sync_event(errors, block_i, task_j, evt_idx, evt):
    prefix = f"tilefwk: block[{block_i}].tasks[{task_j}].syncEvents[{evt_idx}]"
    if "type" not in evt or "time" not in evt:
        errors.append(f"{prefix} missing 'type' or 'time'")
        return
    if evt["type"] not in ("CV_SYNC_SET", "CV_SYNC_WAIT"):
        errors.append(f"{prefix} invalid type: {evt['type']}")


def _validate_tilefwk_task(errors, block_i, task_j, task):
    for key in ("seqNo", "taskId", "leafIndex", "execStart", "execEnd"):
        if key not in task:
            errors.append(f"tilefwk: block[{block_i}].tasks[{task_j}] missing '{key}'")
    sync_events = task.get("syncEvents", [])
    if not sync_events:
        return 0
    for k, evt in enumerate(sync_events):
        _validate_tilefwk_sync_event(errors, block_i, task_j, k, evt)
    return 1


def _validate_tilefwk_block(errors, block_i, block):
    if not isinstance(block, dict):
        errors.append(f"tilefwk: block[{block_i}] must be a dict")
        return 0, 0
    for key in ("blockIdx", "coreType", "tasks"):
        if key not in block:
            errors.append(f"tilefwk: block[{block_i}] missing '{key}'")
    total_tasks = 0
    tasks_with_sync = 0
    for task_j, task in enumerate(block.get("tasks", [])):
        total_tasks += 1
        tasks_with_sync += _validate_tilefwk_task(errors, block_i, task_j, task)
    return total_tasks, tasks_with_sync


def _validate_tilefwk_structure(data):
    errors = []
    if not isinstance(data, list):
        errors.append("tilefwk: root must be a list")
        return errors, 0, 0
    total_tasks = 0
    tasks_with_sync = 0
    for i, block in enumerate(data):
        block_tasks, block_sync = _validate_tilefwk_block(errors, i, block)
        total_tasks += block_tasks
        tasks_with_sync += block_sync
    if total_tasks == 0:
        errors.append("tilefwk: no tasks found")
    return errors, total_tasks, tasks_with_sync


@dataclass(frozen=True)
class _MixSyncMsgCtx:
    mix_i: int
    wrap_j: int
    task_k: int
    msg_idx: int

    def prefix(self) -> str:
        return f"mix_event: [{self.mix_i}].wrapInfos[{self.wrap_j}].coreTask[{self.task_k}].syncMsg[{self.msg_idx}]"


def _validate_mix_sync_msg(errors, ctx: _MixSyncMsgCtx, msg):
    prefix = ctx.prefix()
    if "isSet" not in msg or "eventID" not in msg:
        errors.append(f"{prefix} missing 'isSet' or 'eventID'")
    if not isinstance(msg.get("isSet"), bool):
        errors.append(f"{prefix} 'isSet' must be bool, got {type(msg.get('isSet')).__name__}")


def _validate_mix_core_task(errors, mix_i, wrap_j, task_k, task):
    if "hashValue" not in task:
        errors.append(f"mix_event: [{mix_i}].wrapInfos[{wrap_j}].coreTask[{task_k}] missing 'hashValue'")
    sync_msgs = task.get("syncMsg", [])
    for m, msg in enumerate(sync_msgs):
        _validate_mix_sync_msg(errors, _MixSyncMsgCtx(mix_i, wrap_j, task_k, m), msg)
    return 1, len(sync_msgs)


def _validate_mix_wrap(errors, mix_i, wrap_j, wrap):
    if "wrapID" not in wrap:
        errors.append(f"mix_event: [{mix_i}].wrapInfos[{wrap_j}] missing 'wrapID'")
    total_tasks = 0
    total_sync_msgs = 0
    for task_k, task in enumerate(wrap.get("coreTask", [])):
        total_tasks += 1
        _, msg_count = _validate_mix_core_task(errors, mix_i, wrap_j, task_k, task)
        total_sync_msgs += msg_count
    return total_tasks, total_sync_msgs


def _validate_mix_event_structure(data):
    errors = []
    if not isinstance(data, list):
        errors.append("mix_event: root must be a list")
        return errors, 0, 0
    total_sync_msgs = 0
    total_tasks = 0
    for i, mix in enumerate(data):
        if not isinstance(mix, dict):
            errors.append(f"mix_event: [{i}] must be a dict")
            continue
        if "mixId" not in mix:
            errors.append(f"mix_event: [{i}] missing 'mixId'")
        for j, wrap in enumerate(mix.get("wrapInfos", [])):
            wrap_tasks, wrap_msgs = _validate_mix_wrap(errors, i, j, wrap)
            total_tasks += wrap_tasks
            total_sync_msgs += wrap_msgs
    return errors, total_tasks, total_sync_msgs


def _validate_dyn_topo_structure(tasks):
    errors = []
    if not tasks:
        errors.append("dyn_topo: no tasks found")
        return errors
    for i, t in enumerate(tasks):
        for key in ("seqNo", "taskId", "leafIndex", "leafHash", "coreType"):
            if key not in t:
                errors.append(f"dyn_topo: task[{i}] missing '{key}'")
    return errors


def _validate_sync_alignment(tilefwk_data, dyn_lookup, mix_lookup):
    errors = []
    aligned_count = 0
    skipped_no_hash = 0

    for block in tilefwk_data:
        for task in block.get("tasks", []):
            sync_events = task.get("syncEvents", [])
            if not sync_events:
                continue

            leaf_hash = dyn_lookup.get(task["taskId"])
            if leaf_hash is None:
                errors.append(f"alignment: taskId={task['taskId']} not found in dyn_topo.txt")
                continue

            if leaf_hash == "0":
                skipped_no_hash += 1
                continue

            sync_msg = mix_lookup.get(leaf_hash)
            if sync_msg is None:
                skipped_no_hash += 1
                continue

            aligned_count += 1
            runtime_set = sum(1 for e in sync_events if e["type"] == "CV_SYNC_SET")
            runtime_wait = sum(1 for e in sync_events if e["type"] == "CV_SYNC_WAIT")
            compile_set = sum(1 for e in sync_msg if e["isSet"])
            compile_wait = sum(1 for e in sync_msg if not e["isSet"])

            if runtime_set != compile_set or runtime_wait != compile_wait:
                errors.append(
                    f"alignment: taskId={task['taskId']} leafHash={leaf_hash}: "
                    f"runtime(set={runtime_set},wait={runtime_wait}) vs "
                    f"compile(set={compile_set},wait={compile_wait})"
                )

    return errors, aligned_count, skipped_no_hash


def _run_mix_swimlane_guard_kernel():
    output_name = "temp_mix_swimlane_guard"
    output_dir = str(Path.cwd() / output_name)
    os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    pypto.set_debug_options(runtime_debug_mode=1)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "glm_v4_5"))
    from glm_attention import build_ifa_config, get_case_config, ifa

    case_config = get_case_config("ifa_950_b16_s1_1_s2_8k")
    atten_cfg, tile_config = build_ifa_config(case_config)
    ifa(atten_cfg, tile_config, is_950=True, is_high_through=False, is_high_precision=False)
    torch_npu.npu.synchronize()
    return pypto.pypto_impl.LogTopFolder(), output_dir


def _cleanup_guard_artifacts(perf_dir, output_dir):
    if perf_dir:
        shutil.rmtree(perf_dir, ignore_errors=True)
    if output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)


def _build_mix_lookup(mix_data):
    mix_lookup = {}
    for mix in mix_data:
        for wrap in mix.get("wrapInfos", []):
            for task in wrap.get("coreTask", []):
                mix_lookup[str(task["hashValue"])] = task.get("syncMsg", [])
    return mix_lookup


def _validate_swimlane(swimlane_path):
    if not os.path.exists(swimlane_path):
        return
    with open(swimlane_path, encoding="utf-8") as f:
        swimlane_data = json.load(f)
    assert isinstance(swimlane_data, dict), "merged_swimlane.json: root must be a dict"


def _collect_guard_errors(perf_dir):
    all_errors = []
    tilefwk_path = os.path.join(perf_dir, "tilefwk_L1_prof_data.json")
    dyn_topo_path = os.path.join(perf_dir, "dyn_topo.txt")
    mix_event_path = os.path.join(perf_dir, "mix_event_info.json")
    swimlane_path = os.path.join(perf_dir, "merged_swimlane.json")

    for p in [tilefwk_path, dyn_topo_path, mix_event_path]:
        assert os.path.exists(p), f"missing output file: {p}"

    with open(tilefwk_path, encoding="utf-8") as f:
        tilefwk_data = json.load(f)
    errs, total_tasks, tasks_with_sync = _validate_tilefwk_structure(tilefwk_data)
    all_errors.extend(f"[tilefwk] {e}" for e in errs)
    assert total_tasks > 0, "tilefwk: no tasks found"
    assert tasks_with_sync > 0, (
        f"tilefwk: {total_tasks} tasks found but 0 have syncEvents. "
        "The GLM kernel should produce cross-core sync events."
    )

    dyn_tasks = _load_dyn_topo(dyn_topo_path)
    all_errors.extend(f"[dyn_topo] {e}" for e in _validate_dyn_topo_structure(dyn_tasks))

    with open(mix_event_path, encoding="utf-8") as f:
        mix_data = json.load(f)
    errs, _, mix_total_msgs = _validate_mix_event_structure(mix_data)
    all_errors.extend(f"[mix_event] {e}" for e in errs)
    assert mix_total_msgs > 0, (
        "mix_event_info.json has 0 syncMsg entries. The GLM kernel should produce cross-core sync metadata."
    )

    _validate_swimlane(swimlane_path)

    if not all_errors:
        dyn_lookup = {t["taskId"]: t["leafHash"] for t in dyn_tasks}
        errs, aligned_count, skipped = _validate_sync_alignment(tilefwk_data, dyn_lookup, _build_mix_lookup(mix_data))
        all_errors.extend(f"[alignment] {e}" for e in errs)
        assert aligned_count > 0, (
            f"alignment: {tasks_with_sync} tasks have syncEvents but none could be "
            f"matched to mix_event_info.json via dyn_topo leafHash (skipped={skipped})"
        )

    return all_errors


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mix_swimlane_guard():
    perf_dir = None
    output_dir = None
    try:
        perf_dir, output_dir = _run_mix_swimlane_guard_kernel()
        all_errors = _collect_guard_errors(perf_dir)
        assert not all_errors, f"mix swimlane guard failed ({len(all_errors)} issue(s)):\n" + "\n".join(all_errors[:20])
    finally:
        _cleanup_guard_artifacts(perf_dir, output_dir)
