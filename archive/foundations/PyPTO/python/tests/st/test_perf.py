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
BMM Kernel Performance Test

This test validates the performance and correctness of PyPTO BMM (Batch Matrix Multiplication) kernel
by comparing profiling data against expected baselines.

Test Components:
    - CoreType consistency: Validates that core types match between dyn_topo.txt and tilefwk JSON
    - Kernel details consistency: Validates kernel metadata matches expected baseline
"""

import csv
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import shutil
from typing import Any, Dict, List

import torch
import torch_npu

import pypto

FP32 = pypto.DT_FP32
FP16 = pypto.DT_FP16
INT32 = pypto.DT_INT32
INT8 = pypto.DT_INT8
UINT64 = pypto.DT_UINT64
UINT32 = pypto.DT_UINT32


KERNEL_DETAILS_BASELINE = [
    {
        "Name": "PYPTO_bmm_kernel_with_no_mn_split",
        "Type": "PyPTO",
        "Accelerator Core": "AI_CPU",
        "Input Shapes": "\"3,64,64;3,64,64\"",
        "Input Formats": "ND;ND",
        "Output Shapes": "\"3,64,64\"",
        "Output Formats": "ND",
    },
    {
        "Name": "PYPTO_bmm_kernel_with_no_mn_split",
        "Type": "PyPTO",
        "Accelerator Core": "MIX_AIC",
        "Input Shapes": "\"3,64,64;3,64,64\"",
        "Input Formats": "ND;ND",
        "Output Shapes": "\"3,64,64\"",
        "Output Formats": "ND",
    },
]

CORETYPE_MAP = {"AIV": 0, "AIC": 1}


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    gm_acc: bool = False


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    pass_options={"enable_slice": False},
    runtime_options={"device_sched_mode": 2, "stitch_function_max_num": 32},
)
def bmm_kernel_with_no_mn_split(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    pypto.set_cube_tile_shapes(
        shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape, enable_split_k=shape_info.gm_acc
    )
    result = pypto.matmul(
        a_tensor, b_tensor, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans, out_dtype=shape_info.out_dtype
    )
    pypto.set_vec_tile_shapes(3, 4, 4)
    add_result = pypto.add(result, result)
    out_tensor.move(add_result)


def bmm_with_mn_split(queue):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    b = 3
    m = 64
    k = 64
    n = 64
    tile_m = 16
    tile_k = 16
    tile_n = 16
    shape_info = ShapeConfig(
        [b, m, k, n],
        [tile_m, tile_m],
        [tile_k, tile_k],
        [tile_n, tile_n],
        [-1, -1],
        FP16,
        FP32,
        True,
        False,
        False,
        False,
        False,
        False,
    )
    a1_tensor = torch.rand([b, k, m], dtype=torch.float16, device=f'npu:{device_id}')
    b1_tensor = torch.rand([b, k, n], dtype=torch.float16, device=f'npu:{device_id}')
    c1_tensor = torch.zeros([b, m, n], dtype=torch.float32, device=f'npu:{device_id}')
    experimental_config = _build_experimental_config()
    root_dir = _get_root_dir()
    profiler_output_dir = os.path.join(root_dir, "ifa_profiler_output")
    shutil.rmtree(profiler_output_dir, ignore_errors=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiler_output_dir, analyse_flag=True),
    ) as prof:
        bmm_kernel_with_no_mn_split(a1_tensor, b1_tensor, c1_tensor, shape_info)
        torch_npu.npu.synchronize()
        prof.step()

    perf_path = pypto.pypto_impl.LogTopFolder()
    queue.put(perf_path)


def _get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def _build_experimental_config():
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return experimental_config


@pypto.options(pass_options={"enable_slice": True})
def test_perf():
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    p = mp.Process(target=bmm_with_mn_split, args=(result_queue,))
    p.start()
    p.join()

    perf_path = ""
    if not result_queue.empty():
        perf_path = result_queue.get()
    else:
        assert False, "Could not get profiler output path"

    assert os.path.exists(perf_path), f"Profiler output path not found: {perf_path}"

    tilefwk_json_path = os.path.join(perf_path, "tilefwk_L1_prof_data.json")
    assert os.path.exists(tilefwk_json_path), f"Could not find tilefwk_L1_prof_data.json in {perf_path}"

    dyn_topo_path = os.path.join(perf_path, "dyn_topo.txt")
    assert os.path.exists(dyn_topo_path), f"Could not find dyn_topo.txt in {perf_path}"

    root_dir = _get_root_dir()
    profiler_output_dir = os.path.join(root_dir, "ifa_profiler_output")

    try:
        verify_coretype_consistency(dyn_topo_path, tilefwk_json_path)

        kernel_details_path = None
        for root, _, files in os.walk(profiler_output_dir):
            if "kernel_details.csv" in files:
                kernel_details_path = os.path.join(root, "kernel_details.csv")
                break

        if kernel_details_path and os.path.exists(kernel_details_path):
            verify_kernel_details_consistency(kernel_details_path)
    finally:
        shutil.rmtree(profiler_output_dir, ignore_errors=True)


def load_dyn_topo_data(dyn_topo_path):
    dyn_topo_data = {}
    with open(dyn_topo_path, 'r') as f:
        reader = csv.reader(f)
        _header = next(reader)

        for row in reader:
            seq_no = int(row[0])
            task_id = int(row[1])
            leaf_index = int(row[5])
            core_type = int(row[7])

            key = (seq_no, task_id, leaf_index)
            dyn_topo_data[key] = core_type

    return dyn_topo_data


def load_and_validate_json_data(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    json_coretype_data = {}
    missing_field_errors = []

    for block in json_data:
        block_idx = block['blockIdx']
        core_type_str = block['coreType']
        core_type_num = CORETYPE_MAP.get(core_type_str, -1)

        if 'tasks' not in block:
            missing_field_errors.append(f"block_idx={block_idx}: missing 'tasks' field")
            continue

        for task in block['tasks']:
            required_fields = ['seqNo', 'taskId', 'leafIndex']
            missing_fields = [f for f in required_fields if f not in task]

            if missing_fields:
                missing_field_errors.append(
                    f"block_idx={block_idx}, task_id={task.get('taskId', 'unknown')}: missing fields {missing_fields}"
                )
                continue

            seq_no = task['seqNo']
            task_id = task['taskId']
            leaf_index = task['leafIndex']

            key = (seq_no, task_id, leaf_index)
            json_coretype_data[key] = {
                'core_type_num': core_type_num,
                'core_type_str': core_type_str,
                'block_idx': block_idx,
            }

    return json_data, json_coretype_data, missing_field_errors


def verify_coretype_consistency(dyn_topo_path, json_path):
    dyn_topo_data = load_dyn_topo_data(dyn_topo_path)
    json_data, json_coretype_data, missing_field_errors = load_and_validate_json_data(json_path)

    assert len(missing_field_errors) == 0, (
        f"JSON missing required fields, expected: "
        f"each record contains seq_no/task_id/leaf_index, actual: {len(missing_field_errors)} missing"
    )
    for key, json_info in json_coretype_data.items():
        seq_no, task_id, leaf_index = key
        json_core_type = json_info['core_type_num']

        assert key in dyn_topo_data, (
            f"JSON task (seq_no={seq_no}, task_id={task_id}, leaf_index={leaf_index}) not found in dyn_topo.txt"
        )

        dyn_core_type = dyn_topo_data[key]
        assert dyn_core_type == json_core_type, (
            f"(seq_no={seq_no}, task_id={task_id}, leaf_index={leaf_index}) "
            f"core_type mismatch, expected: {dyn_core_type} (dyn_topo), "
            f"actual: {json_core_type} (JSON)"
        )


def normalize_field_value(value: str) -> str:
    if not value:
        return ''

    normalized = value.strip()

    has_double_quotes = normalized.startswith('"') and normalized.endswith('"')
    has_single_quotes = normalized.startswith("'") and normalized.endswith("'")

    if has_double_quotes or has_single_quotes:
        normalized = normalized[1:-1].strip()

    normalized = ' '.join(normalized.split())

    return normalized


def extract_kernel_details_baseline(kernel_details_path: str) -> List[Dict[str, Any]]:
    baseline_records = []
    required_fields = [
        'Name',
        'Type',
        'Accelerator Core',
        'Input Shapes',
        'Input Formats',
        'Output Shapes',
        'Output Formats',
    ]

    with open(kernel_details_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {field: normalize_field_value(row.get(field, '')) for field in required_fields}
            if record['Type'] == 'PyPTO':
                baseline_records.append(record)

    return baseline_records


def verify_kernel_details_consistency(kernel_details_path: str):
    baseline = KERNEL_DETAILS_BASELINE

    baseline_with_duplicates = [baseline[0], baseline[0], baseline[1]]

    current_records = extract_kernel_details_baseline(kernel_details_path)

    normalized_baseline = [
        {field: normalize_field_value(value) for field, value in rec.items()} for rec in baseline_with_duplicates
    ]

    assert len(current_records) == len(normalized_baseline), (
        f"Record count mismatch\nexpected: {len(normalized_baseline)}\nactual: {len(current_records)}"
    )

    for idx, (baseline_rec, current_rec) in enumerate(zip(normalized_baseline, current_records)):
        for field in baseline_rec.keys():
            assert baseline_rec[field] == current_rec[field], (
                f"Record #{idx} field '{field}' mismatch\nexpected: {baseline_rec[field]}\nactual: {current_rec[field]}"
            )
