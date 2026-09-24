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
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import sys
import time as time_module

import function_json_convert as fcvt
import parse_pipe_time_trace as pipe_time


class TaskInfo:
    def __init__(self, task_id):
        self.task_id = task_id  # task_id is stitchedStatic << 32 + stitchedRootIndex << 20 + opIndex
        self.root_index = -1
        self.root_hash = 0
        self.opmagic = 0
        self.core_idx = 0
        self.core_type = ""
        self.psg_id_within_static = -1
        self.psg_id_in_dyn = -1
        self.exec_start = (0,)
        self.exec_end = (0,)
        self.color_label = ""
        self.func_name = ""
        self.predecessor = 0
        self.predecessors_taskid = []
        self.predecessor_ready_time = 0
        self.task_wait_schedule_time = 0
        self.successors = []
        self.swim_lane_offset = 0
        self.is_fake = False
        self.origin_seq_no = 0
        self.origin_task_id = 0  # data in log file
        self.inoperand_label = ""
        self.outoperand_label = ""
        self.in_operands = []
        self.out_operands = []
        self.leaf_total_cycles = 0
        self.leaf_total_time = 0
        self.leaf_pipe_exec_cycles = {}
        self.leaf_pipe_exec_time = {}
        self.func_hash = 0
        self.tensors = {}
        self.rawtensors = {}
        self.wrap_id = -1
        self.l1_reuse_hash_order = None
        self.cube_merge_hash_order = None
        self.vec_merge_hash_order = None
        self.l1_reuse_subgraph_count = None
        self.cube_merge_subgraph_count = None
        self.vec_merge_subgraph_count = None
        self.sync_events = []

    def formal_name(self):
        seq_no = self.task_id >> 32
        func_id = (self.task_id >> 16) & ((1 << 10) - 1)
        oper_idx = self.task_id & ((1 << 16) - 1)
        return f"{seq_no}-{func_id}-{oper_idx}"

    def get_hash_order_display(self):
        parts = []
        if self.l1_reuse_hash_order is not None:
            parts.append(
                f"l1ReuseInfo hashOrder: {self.l1_reuse_hash_order}, subGraphCount: {self.l1_reuse_subgraph_count}"
            )
        if self.cube_merge_hash_order is not None:
            parts.append(
                f"cubeMergeInfo hashOrder: {self.cube_merge_hash_order}, "
                f"subGraphCount: {self.cube_merge_subgraph_count}"
            )
        if self.vec_merge_hash_order is not None:
            parts.append(
                f"vecMergeInfo hashOrder: {self.vec_merge_hash_order}, subGraphCount: {self.vec_merge_subgraph_count}"
            )
        return "\n".join(parts) if parts else ""

    def get_task_full_name(self):
        base = (
            f"Task:[{self.formal_name()}], "
            f"rootHash:{self.root_hash}, "
            f"callOpMagic:{self.opmagic}, "
            f"leafHash:{self.func_hash}, "
            f"TaskId:{self.origin_task_id}"
        )
        ho = self.get_hash_order_display()
        if ho:
            return f"{base}\n{ho}"
        return base

    def get_task_execution_time_analysis(self):
        if self.is_fake or not self.func_hash or self.func_hash not in task_analysis:
            return ""
        psg_id_analysis = task_analysis[self.func_hash]
        average = psg_id_analysis.total_execution_time / psg_id_analysis.count
        report = (
            f"Average Execution Time: {average}\n"
            f"Max Execution Time: {psg_id_analysis.max_execution_time}, "
            f"Task: [{psg_id_analysis.max_execution_task.formal_name()}]\n"
            f"Min Execution Time: {psg_id_analysis.min_execution_time}, "
            f"Task: [{psg_id_analysis.min_execution_task.formal_name()}]\n"
        )
        return report

    def get_task_name(self):
        return f"{self.formal_name()}-{self.root_index}-{self.psg_id_within_static}({self.color_label})"

    def get_task_wait_schedule_time(self):
        if self.is_fake:
            return 0
        return self.task_wait_schedule_time

    # Get start/end event
    def get_dur_event(self, event_id, pid, tid):
        res = {}
        res["args"] = {}
        res["args"]["event-hint"] = self.get_task_full_name()
        res["args"]["ioperand-hint"] = self.inoperand_label
        res["args"]["ooperand-hint"] = self.outoperand_label
        res["args"]["execution-hint"] = self.get_task_execution_time_analysis()
        res["args"]["color"] = self.color_label
        res["args"]["taskId"] = self.origin_task_id
        res["args"]["seqNo"] = self.origin_seq_no
        res["args"]["wrapId"] = self.wrap_id
        self._add_hash_order_args(res["args"])
        if self.sync_events:
            res["args"]["syncEvents"] = self.sync_events
        res["cat"] = "event"
        res["id"] = event_id
        res["name"] = self.get_task_name()
        res["ph"] = "X"
        res["pid"] = pid
        res["tid"] = tid
        res["ts"] = self.exec_start
        res["dur"] = self.exec_end - self.exec_start
        return res

    def get_execute_json_entry(self):
        res = {}
        res["taskId"] = self.task_id
        res["oriSeqNo"] = self.origin_seq_no
        res["oriTaskId"] = self.origin_task_id
        res["nameLabel"] = self.get_task_name()
        res["args"] = {}
        res["args"]["ioperand-hint"] = self.inoperand_label
        res["args"]["ooperand-hint"] = self.outoperand_label
        res["args"]["taskId"] = self.origin_task_id
        res["args"]["seqNo"] = self.origin_seq_no
        self._add_hash_order_args(res["args"])
        if self.sync_events:
            res["args"]["syncEvents"] = self.sync_events
        if len(self.func_name) == 0:
            res["funcName"] = "Func"
        else:
            res["funcName"] = self.func_name
        res["coreType"] = self.core_type
        res["execTime"] = self.exec_end - self.exec_start
        res["successors"] = self.successors
        res["remainingPredecessors"] = self.predecessor
        return res

    def _add_hash_order_args(self, args):
        info = self.get_hash_order_display()
        if info:
            args["hashOrder-hint"] = info


class TaskAnalysisInfo:
    def __init__(self):
        self.count = 0
        self.total_execution_time = 0
        self.max_execution_time = 0
        self.max_execution_task = None
        self.min_execution_time = 0
        self.min_execution_task = None

    def add_task(self, task_info: TaskInfo):
        self.count += 1
        execution_time = task_info.exec_end - task_info.exec_start
        self.total_execution_time += execution_time
        if self.max_execution_task is None or self.max_execution_time < execution_time:
            self.max_execution_time = execution_time
            self.max_execution_task = task_info
        if self.min_execution_task is None or self.min_execution_time > execution_time:
            self.min_execution_time = execution_time
            self.min_execution_task = task_info


class CoreInfo:
    def __init__(self, core_idx, c_type):
        self.core_idx = core_idx
        self.core_type = c_type
        self.tasks = []
        self.total_time = 0
        self.pipe_exec_cycles = {}
        self.pipe_exec_time = {}  # vector and cube time unchanged. MTE scale proportionally
        self.has_overlap = False
        self.last_task_end_time = 0
        self.total_wait_time = 0
        self.core_wait_schedule_time = 0
        self.core_wait_predecessor_time = 0
        self.faketask_num = 0

    def get_brief_core_type(self):
        name = ""
        if "AIC" in self.core_type:
            name += "AIC"
        elif "AIV" in self.core_type:
            name += "AIV"
        else:
            name += self.core_type
        return name

    def get_core_name(self):
        name = ""
        if "AIC" in self.core_type:
            name += "AIC"
        elif "AIV" in self.core_type:
            name += "AIV"
        else:
            name += self.core_type
        name += "_" + str(self.core_idx)
        return name

    def get_execute_task_num(self):
        return len(self.tasks) - self.faketask_num

    def trans_cycles_to_time(self):
        for task in self.tasks:
            freq_convert = 1800
            task.leaf_total_time = task.leaf_total_cycles / freq_convert
            proportion = (task.exec_end - task.exec_start) / task.leaf_total_time
            print((task.exec_end - task.exec_start), task.leaf_total_time)
            for pipe, cycles in task.leaf_pipe_exec_cycles.items():
                real_time = 0
                if 'MTE' in pipe:
                    real_time = (cycles / freq_convert) * proportion
                else:
                    real_time = cycles / freq_convert
                self.pipe_exec_time[pipe] = self.pipe_exec_time.get(pipe, 0) + real_time


# key: task_id, value: TaskInfo
total_tasks = {}
# key: subgraph_id, value: TaskAnalysisInfo
task_analysis = defaultdict(lambda: TaskAnalysisInfo())
total_cores = {}  # key: core_idx, value: [CoreInfo]
mininum_start_time = sys.maxsize
max_end_time = 0
fake_task_start_time_alloc = sys.maxsize


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process two JSON files.")
    parser.add_argument("swim_json_file", type=str, help="Path to the first JSON file")
    parser.add_argument("topo_json_file", type=str, help="Path to the second JSON file")
    parser.add_argument("func_table_file", type=str, nargs="?", help="Path to the second JSON file")
    parser.add_argument("pipe_exec_time", type=str, nargs="?", help="Path to the second JSON file")
    parser.add_argument(
        "--time_convert_denominator",
        type=int,
        default=1,
        help="Log time covert denominator,default=1",
    )
    parser.add_argument(
        "--label_type",
        type=int,
        default=0,
        help="Choose the color label type,default=0",
    )
    parser.add_argument(
        "-g",
        "--gen_exe_topo_json",
        action="store_true",
        help="Generate executable json",
    )
    parser.add_argument(
        "--mix_event_info",
        type=str,
        default="",
        help="Path to mix_event_info.json",
    )
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def decimal_to_26(num):
    num = abs(num)
    number_system = 26
    if num == 0:
        return "a"
    result = []
    while num > 0:
        remainder = num % number_system
        result.append(chr(ord("a") + remainder))
        num = num // number_system
    return "".join(reversed(result))


def build_fake_entry(task_id):
    global total_tasks
    global total_cores

    entry = TaskInfo(task_id)
    entry.core_idx = 0
    entry.psg_id_within_static = 0
    entry.psg_id_in_dyn = 0
    entry.color_label = "fake"
    entry.is_fake = True
    total_tasks[task_id] = entry
    total_cores[0].tasks.append(entry)
    total_cores[0].faketask_num += 1
    return entry


def get_fake_task_start_end_cycles(fake_task_id):
    global total_tasks
    global fake_task_start_time_alloc

    entry = total_tasks[fake_task_id]
    find = False
    # 根据successor 对Fake Task 赋值
    if len(entry.successors) > 0:
        tmp_list = []
        for succ in entry.successors:
            succ_entry = total_tasks[succ]
            if succ_entry.is_fake:
                continue
            tmp_list.append(succ_entry.exec_start)
        if len(tmp_list) > 0:
            time = min(tmp_list)
            entry.exec_start = time - 1
            entry.exec_end = time
            find = True

    # 根据predecessor 结点对Fake Task 赋值
    if not find:
        tmp_list = []
        for _, task_entry in total_tasks.items():
            if task_entry.is_fake:
                continue
            for succ in task_entry.successors:
                if succ == fake_task_id:
                    tmp_list.append(task_entry.exec_end)
        if len(tmp_list) > 0:
            time = max(tmp_list)
            entry.exec_start = time
            entry.exec_end = time + 1
            find = True

    # 默认值为 Fake Task 赋值
    if not find:
        entry.exec_start = fake_task_start_time_alloc
        entry.exec_end = fake_task_start_time_alloc + 1
        fake_task_start_time_alloc += 2


def build_leaf_hash_to_events(mix_event_data):
    leaf_hash_to_events = {}
    for wrap_info_list in mix_event_data:
        for wrap_info in wrap_info_list.get("wrapInfos"):
            for core_task in wrap_info.get("coreTask", []):
                leaf_hash = core_task.get("hashValue")
                sync_msg = core_task.get("syncMsg", [])
                leaf_hash_to_events.setdefault(leaf_hash, sync_msg)
    return leaf_hash_to_events


def enrich_sync_events_with_mix_info(mix_event_path):
    global total_tasks
    if not mix_event_path or not os.path.exists(mix_event_path):
        return
    with open(mix_event_path, "r") as f:
        mix_event_data = json.load(f)
    leaf_hash_to_events = build_leaf_hash_to_events(mix_event_data)
    for _, task in total_tasks.items():
        if task.func_hash and task.func_hash in leaf_hash_to_events:
            sync_msg = leaf_hash_to_events[task.func_hash]
            idx = 0
            for event in task.sync_events:
                event["eventid"] = sync_msg[idx].get("eventID")
                idx = idx + 1


def process_sync_events(sync_events):
    for event in sync_events:
        if "time" in event:
            event["time"] = event["time"] / args.time_convert_denominator


def parse_swim_data(swim_data, label_type):
    global total_tasks
    global task_analysis
    global total_cores
    global mininum_start_time
    global max_end_time
    # 在日志信息中不存在但是在topo 信息中存在的task_id，为其构建虚拟task结点
    fake_task_list = []
    core_idx = 0
    core_type = "Fake Core"
    core_entry = CoreInfo(core_idx, core_type)
    total_cores[core_idx] = core_entry

    # 解析swim.json 文件中的信息
    for core in swim_data:
        # 构建Core
        core_idx = core["blockIdx"] + 1
        core_type = core.get("coreType", "Core")
        core_entry = CoreInfo(core_idx, core_type)
        last_valid = False
        last_task_id = 0
        # 解析每个AICORE 执行的task
        for task in core["tasks"]:
            seq_no = task.get("seqNo", 0)
            task_id = (seq_no << 32) | task["taskId"]
            entry = TaskInfo(task_id)
            entry.origin_seq_no = seq_no
            entry.origin_task_id = task["taskId"]
            entry.core_idx = core_idx
            entry.psg_id_in_dyn = task.get("subGraphId", -1)
            entry.exec_start = task.get("execStart", 0) / args.time_convert_denominator
            entry.exec_end = task.get("execEnd", 0) / args.time_convert_denominator
            entry.core_type = core_entry.get_brief_core_type()
            entry.sync_events = task.get("syncEvents", [])
            process_sync_events(entry.sync_events)
            # 判断task 间是否存在时间交叠
            if (
                last_valid
                and entry.exec_start < total_tasks[last_task_id].exec_end
                and total_tasks[last_task_id].swim_lane_offset == 0
            ):
                entry.swim_lane_offset = 1
                core_entry.has_overlap = True

            mininum_start_time = min(mininum_start_time, entry.exec_start)
            max_end_time = max(max_end_time, entry.exec_end)
            # 记录当前Core 的上一次执行的Task
            _last_task_end_time = entry.exec_end
            last_task_id = task_id
            last_valid = True

            total_tasks[task_id] = entry
            core_entry.tasks.append(entry)
        total_cores[core_idx] = core_entry
    return fake_task_list


def _assign_color_label(entry, label_type, sematic_label):
    if label_type == 1:
        entry.color_label += sematic_label
    elif label_type == 2:
        entry.color_label = decimal_to_26(entry.psg_id_in_dyn)
        entry.color_label += " " + sematic_label
    else:
        entry.color_label = decimal_to_26(entry.psg_id_in_dyn)


def _assign_hash_order_info(entry, topo_task):
    for prefix, attr_hash, attr_count in [
        ("l1Reuse", "l1_reuse_hash_order", "l1_reuse_subgraph_count"),
        ("cubeNBuffer", "cube_merge_hash_order", "cube_merge_subgraph_count"),
        ("vecNBuffer", "vec_merge_hash_order", "vec_merge_subgraph_count"),
    ]:
        info = topo_task.get(f"{prefix}HashOrderInfo")
        setattr(entry, attr_hash, info.get("hashOrder") if info else None)
        setattr(entry, attr_count, info.get("subgraphCount") if info else None)


def parse_topo_data(topo_data, label_type, fake_task_list):
    if topo_data is None:
        return
    for topo_task in topo_data:
        task_id = topo_task["taskId"]
        if task_id not in total_tasks:
            build_fake_entry(task_id)
            fake_task_list.append(task_id)
        func_name = topo_task.get("funcName", "")
        sematic_label = topo_task.get("semanticLabel", "")
        entry = total_tasks[task_id]
        entry.root_index = topo_task.get("rootIndex", -1)
        entry.root_hash = topo_task.get("rootHash", -1)
        entry.opmagic = topo_task.get("opMagic", -1)
        entry.origin_task_id = topo_task.get("oriTaskId", 0)
        entry.origin_seq_no = topo_task.get("oriSeqNo", 0)
        _assign_color_label(entry, label_type, sematic_label)
        entry.func_name = func_name
        entry.psg_id_within_static = topo_task.get("psgId", entry.psg_id_in_dyn)
        entry.wrap_id = topo_task.get("wrapId", -1)
        _assign_hash_order_info(entry, topo_task)
        entry.inoperand_label = f"{topo_task.get('inoperands', [])}"
        entry.outoperand_label = f"{topo_task.get('outoperands', [])}"
        entry.successors = topo_task["successors"]
        entry.in_operands = topo_task.get('in_operands') if topo_task.get('in_operands') else []
        entry.out_operands = topo_task.get('out_operands') if topo_task.get('out_operands') else []
        entry.func_hash = topo_task.get('funcHash')
        entry.tensors = topo_task.get('tensors')
        entry.rawtensors = topo_task.get('rawtensors')


def build_swim_info(swim_data, topo_data, label_type: int = 0, dir_name: str = "", mix_event_path: str = ""):
    global fake_task_start_time_alloc
    global mininum_start_time
    fake_task_list = parse_swim_data(swim_data, label_type)
    parse_topo_data(topo_data, label_type, fake_task_list)
    # Get Predecessors for each task
    get_predecessors()

    # 为fake task 设置开始和结束时间
    fake_task_start_time_alloc = mininum_start_time
    for fake_task_id in fake_task_list:
        get_fake_task_start_end_cycles(fake_task_id)
    build_task_analysis()
    print(f"Total Core:{len(total_cores) - 1}")
    print(f"Total Task Count:{len(total_tasks)}")
    print(f"|--Fake Task Count:{len(fake_task_list)}")
    enrich_sync_events_with_mix_info(mix_event_path)
    print("Parse Swim json and Topo json Data End")


# Get process metadata
def get_process_metadata(name, pid):
    res = {}
    res["args"] = {}
    res["args"]["name"] = name
    res["cat"] = "__metadata"
    res["name"] = "process_name"
    res["ph"] = "M"
    res["pid"] = pid
    return res


# Get thread metadata
def get_thread_metadata(name, pid, tid):
    res = {}
    res["args"] = {}
    res["args"]["name"] = name
    res["cat"] = "__metadata"
    res["name"] = "thread_name"
    res["ph"] = "M"
    res["pid"] = pid
    res["tid"] = tid
    return res


def get_flow_src(event_id, pid, tid, time):
    res = {}
    res["cat"] = "machine-view-last-dep"
    res["id"] = event_id
    res["name"] = "machine-view-last-dep"
    res["ph"] = "s"
    res["pid"] = pid
    res["tid"] = tid
    res["ts"] = time
    return res


def get_flow_dst(event_id, pid, tid, time):
    res = {}
    res["bp"] = "e"
    res["cat"] = "machine-view-last-dep"
    res["id"] = event_id
    res["name"] = "machine-view-last-dep"
    res["ph"] = "f"
    res["pid"] = pid
    res["tid"] = tid
    res["ts"] = time
    return res


def process_ready_count(outjson):
    global total_tasks
    global mininum_start_time
    dpd_step = 5
    time_events = {}
    for task_id, task in total_tasks.items():
        t = task.exec_start
        if t not in time_events:
            time_events[t] = []
        time_events[t].append((task_id, "S"))
        t = task.exec_end
        if t not in time_events:
            time_events[t] = []
        time_events[t].append((task_id, "E"))

    ready_start_aic, ready_start_aiv = {}, {}
    for _, task in total_tasks.items():
        if task.predecessor > 0:
            continue
        seq_no = task.task_id >> 32
        if task.core_type == "AIC":
            ready_start_aic[seq_no] = ready_start_aic.get(seq_no, 0) + 1
        elif task.core_type == "AIV":
            ready_start_aiv[seq_no] = ready_start_aiv.get(seq_no, 0) + 1

    ready_aic, ready_aiv = 0, 0
    task_ind = {i: j.predecessor for i, j in total_tasks.items()}
    curr_seq_no = -1
    dpd_time, dpd_count = mininum_start_time, 0
    for t, events in sorted(time_events.items()):
        diff_aic, diff_aiv = 0, 0
        # update new devtask readycount
        if curr_seq_no != events[0][0] >> 32:
            curr_seq_no = events[0][0] >> 32
            diff_aic += ready_start_aic.get(curr_seq_no, 0)
            diff_aiv += ready_start_aiv.get(curr_seq_no, 0)
        # process readycount events
        for task_id, event_type in events:
            task = total_tasks[task_id]
            if event_type == "S":
                if task.core_type == "AIC":
                    diff_aic -= 1
                elif task.core_type == "AIV":
                    diff_aiv -= 1
            elif event_type == "E":
                dpd_count += len(task.successors)
                for s in task.successors:
                    task_s = total_tasks[s]
                    task_ind[s] -= 1
                    if task_ind[s] == 0:
                        if task_s.core_type == "AIC":
                            diff_aic += 1
                        elif task_s.core_type == "AIV":
                            diff_aiv += 1
        if diff_aic != 0:
            ready_aic += diff_aic
            outjson["traceEvents"].append(
                {
                    "name": "ReadyCount_AIC",
                    "pid": 1,
                    "tid": 1,
                    "ph": "C",
                    "ts": t,
                    "args": {
                        "size": ready_aic,
                    },
                }
            )
        if diff_aiv != 0:
            ready_aiv += diff_aiv
            outjson["traceEvents"].append(
                {
                    "name": "ReadyCount_AIV",
                    "pid": 1,
                    "tid": 1,
                    "ph": "C",
                    "ts": t,
                    "args": {
                        "size": ready_aiv,
                    },
                }
            )
        if diff_aic + diff_aiv != 0:
            outjson["traceEvents"].append(
                {
                    "name": "ReadyCount_Total",
                    "pid": 1,
                    "tid": 1,
                    "ph": "C",
                    "ts": t,
                    "args": {
                        "size": ready_aic + ready_aiv,
                    },
                }
            )
        if t - dpd_time > dpd_step:
            outjson["traceEvents"].append(
                {
                    "name": "Dependence Solving (MHz)",
                    "pid": 1,
                    "tid": 1,
                    "ph": "C",
                    "ts": dpd_time,
                    "args": {
                        "size": dpd_count / (t - dpd_time),
                    },
                }
            )
            dpd_time = t
            dpd_count = 0


def convert_to_chrome_trace_json(out_path, is_dyn):
    global total_tasks
    global total_cores
    global mininum_start_time

    machine_view_pid = 1
    machine_view_thread_offset = 1000
    event_id = 0

    res = {}
    res["traceEvents"] = []
    # 设置进程名称
    res["traceEvents"].append(get_process_metadata("Machine View", machine_view_pid))

    # 设置线程名称
    for core_idx, core_entry in total_cores.items():
        core_name = core_entry.get_core_name()
        pid = machine_view_pid
        tid = core_idx * 2 + machine_view_thread_offset
        res["traceEvents"].append(get_thread_metadata(core_name, pid, tid))
        if core_entry.has_overlap:
            res["traceEvents"].append(get_thread_metadata(core_name, pid, tid + 1))

    # 输出每个task 的开始和结束时间
    for _, task_entry in total_tasks.items():
        pid = machine_view_pid
        tid = task_entry.core_idx * 2 + machine_view_thread_offset + task_entry.swim_lane_offset
        res["traceEvents"].append(task_entry.get_dur_event(event_id, pid, tid))
        event_id += 1

    # 输出task 间的依赖
    for _, task_entry in total_tasks.items():
        pid = machine_view_pid
        src_tid = task_entry.core_idx * 2 + machine_view_thread_offset + task_entry.swim_lane_offset
        src_time = task_entry.exec_end - 0.0001

        for dst in task_entry.successors:
            if dst not in total_tasks:
                print(f"WARNING: successor {dst} of [task:{task_entry.task_id}] is not in LOG INFO\n")
                continue
            dst_task_entry = total_tasks[dst]
            dst_tid = dst_task_entry.core_idx * 2 + machine_view_thread_offset + dst_task_entry.swim_lane_offset
            dst_time = dst_task_entry.exec_start
            res["traceEvents"].append(get_flow_src(event_id, pid, src_tid, src_time))
            res["traceEvents"].append(get_flow_dst(event_id, pid, dst_tid, dst_time))
            event_id += 1

    # add readycount & dpd solving events
    process_ready_count(res)

    # 构建chrome trace json 文件
    # 写入到JSON文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print("Convert To perfetto trace End")


def generate_execute_json(path):
    global total_tasks
    global total_cores

    res = []

    for _, task_entry in total_tasks.items():
        res.append(task_entry.get_execute_json_entry())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print("Generate Executable Json:", path)


def get_func_index(func_hash, func_data):
    i = 0
    while i < len(func_data):
        if str(func_hash) == func_data[i].get("hash", "0"):
            return i
        i += 1
    return i


def load_dyn_topo(file_path, func_data):
    # 输入文件由Tensor_Main_2.json改为program.json，处理json数据接入
    func_hash_data = {}
    for _, func in enumerate(func_data):
        func_hash_data[func['hash']] = func
    for _, func in enumerate(func_data):
        if func['graphtype'] != 2:
            continue
        func_tensors = dict()
        func_rawtensors = dict()
        for tensor in func['tensors']:
            func_tensors[tensor['magic']] = tensor
        for rawtensor in func['rawtensors']:
            func_rawtensors[rawtensor['rawmagic']] = rawtensor
        for operation in func['operations']:
            ioperands = []
            ioperands_added = set()
            for ioperand in operation['ioperands']:
                if not isinstance(func_tensors[ioperand]['rawtensor'], dict):
                    func_tensors[ioperand]['rawtensor'] = func_rawtensors[func_tensors[ioperand]['rawtensor']]
                if func_tensors[ioperand]['magic'] not in ioperands_added:
                    ioperands.append(func_tensors[ioperand])
                    ioperands_added.add(func_tensors[ioperand]['magic'])
            operation['ioperands'] = ioperands
            ooperands = []
            ooperands_added = set()
            for ooperand in operation['ooperands']:
                if not isinstance(func_tensors[ooperand]['rawtensor'], dict):
                    func_tensors[ooperand]['rawtensor'] = func_rawtensors[func_tensors[ooperand]['rawtensor']]
                if func_tensors[ooperand]['magic'] not in ooperands_added:
                    ooperands.append(func_tensors[ooperand])
                    ooperands_added.add(func_tensors[ooperand]['magic'])
            operation['ooperands'] = ooperands
            operation['funcName'] = func_hash_data.get(operation['calleehash']).get('func_magicname')
    topo = []
    with open(file_path) as file:
        for line in file:
            if len(line) > 0 and line[0].isalpha():
                continue
            fields = [int(x) for x in line.strip().split(",") if x.strip()]
            (
                seq_no,
                task_id,
                root_index,
                root_hash,
                opmagic,
                leaf_index,
                func_hash,
                core_type,
                psg_id_within_root,
                wrap_id,
                static_succ_count,
            ) = fields[:11]
            root_index = get_func_index(root_hash, func_data)
            leaf_index = get_func_index(func_hash, func_data)
            succs = fields[11:]
            l1_info, cube_info, vec_info = fcvt.get_hash_order_info_for_task(root_index, opmagic, leaf_index, func_data)
            topo.append(
                {
                    "taskId": seq_no << 32 | task_id,
                    "oriTaskId": task_id,
                    "oriSeqNo": seq_no,
                    "successors": [seq_no << 32 | x for x in succs],
                    "coreType": core_type,
                    "rootIndex": root_index,
                    "rootHash": root_hash,
                    "opMagic": opmagic,
                    "leafIndex": leaf_index,
                    "psgId": psg_id_within_root,
                    "wrapId": wrap_id,
                    "funcHash": func_hash,
                    "l1ReuseHashOrderInfo": l1_info,
                    "cubeNBufferHashOrderInfo": cube_info,
                    "vecNBufferHashOrderInfo": vec_info,
                    "semanticLabel": fcvt.get_sematic(root_index, opmagic, func_data),
                    "inoperands": fcvt.get_in_out_operand_str(True, root_index, opmagic, func_data),
                    "outoperands": fcvt.get_in_out_operand_str(False, root_index, opmagic, func_data),
                    "in_operands": fcvt.get_in_out_operands_data(True, root_index, opmagic, func_data),
                    "out_operands": fcvt.get_in_out_operands_data(False, root_index, opmagic, func_data),
                    "tensors": fcvt.get_tensors(str(func_hash), func_hash_data),
                    "rawtensors": fcvt.get_rawtensors(str(func_hash), func_hash_data),
                }
            )
    return topo


def get_predecessor_ready_time(task_id):
    global total_tasks
    global mininum_start_time

    task_entry = total_tasks[task_id]
    task_entry.predecessor_ready_time = mininum_start_time
    for pre in task_entry.predecessors_taskid:
        pre_task_entry = total_tasks[pre]
        task_entry.predecessor_ready_time = max(task_entry.predecessor_ready_time, pre_task_entry.exec_end)


def get_predecessors():
    global total_tasks

    for _, task_entry in total_tasks.items():
        for succ in task_entry.successors:
            total_tasks[succ].predecessor += 1
            total_tasks[succ].predecessors_taskid.append(task_entry.task_id)


def build_task_analysis():
    global total_tasks
    global task_analysis
    for _, task in total_tasks.items():
        if task.func_hash and not task.is_fake:
            task_analysis[task.func_hash].add_task(task)


def analysis_wait_cycles(path):
    global total_tasks
    global total_cores
    global mininum_start_time

    res = []

    sorted_cores = sorted(total_cores.items(), key=lambda x: x[1].core_idx)
    for _, core_entry in sorted_cores:
        core_entry.last_task_end_time = mininum_start_time
        for task in core_entry.tasks:
            if task.is_fake:
                continue
            get_predecessor_ready_time(task.task_id)
            max_ready_time = max(core_entry.last_task_end_time, task.predecessor_ready_time)
            task.task_wait_schedule_time = task.exec_start - max_ready_time
            core_entry.core_wait_schedule_time += task.task_wait_schedule_time
            core_entry.total_wait_time += task.exec_start - core_entry.last_task_end_time

            core_entry.last_task_end_time = task.exec_end
        fake_num = ", Fake Task: " + str(core_entry.faketask_num) if core_entry.faketask_num > 0 else ""
        res.append(f"[{core_entry.get_core_name()}] Execute task num:{core_entry.get_execute_task_num()}{fake_num}")
        res.append(f"    Core Total Work Time: {core_entry.last_task_end_time - mininum_start_time}")
        res.append(f"    Total Wait Time: {core_entry.total_wait_time}")
        res.append(f"    Wait Schedule Time: {core_entry.core_wait_schedule_time}")
        core_entry.core_wait_predecessor_time = core_entry.total_wait_time - core_entry.core_wait_schedule_time
        res.append(f"    Wait Predecessor Time: {core_entry.core_wait_predecessor_time}")
        sorted_tasks = sorted(
            core_entry.tasks,
            key=lambda s: s.get_task_wait_schedule_time(),
            reverse=True,
        )
        if len(sorted_tasks) > 0:
            res.append("    Top 3 tasks in waiting schedule time")
            for top_task in sorted_tasks[:3]:
                if top_task.is_fake:
                    continue
                res.append(
                    f"    Task:{top_task.task_id}, label:{top_task.get_task_name()}, wait: \
                    {top_task.get_task_wait_schedule_time()}"
                )

    sorted_tasks = sorted(
        total_tasks.items(),
        key=lambda s: s[1].get_task_wait_schedule_time(),
        reverse=True,
    )
    top = []
    if len(sorted_tasks) > 0:
        top.append("Top 10 tasks in waiting schedule")
        for _, top_task in sorted_tasks[:10]:
            if top_task.is_fake:
                continue
            top.append(
                f"    Task:{top_task.task_id}, label:{top_task.get_task_name()}, wait: \
                {top_task.get_task_wait_schedule_time()}"
            )
    top.append("\n")
    res = top + res

    with open(path, "w", encoding="utf-8") as f:
        for line in res:
            f.write(line + "\n")
    print("Generate Bubble Analysis Report:", path)


def get_total_time(tasks):
    if len(tasks) == 0:
        return 1
    if tasks[0].is_fake:
        return 1
    return tasks[-1].exec_end - tasks[0].exec_start


def print_aicore_summary():
    global total_cores

    total_aicore_time = 0.0
    lane_count = 0
    global_min_start = None
    global_max_end = None

    for _, core_entry in total_cores.items():
        core_type = core_entry.get_brief_core_type()
        if core_type not in ("AIC", "AIV"):
            continue
        lane_count += 1
        for task in core_entry.tasks:
            if task.is_fake:
                continue
            exec_start = task.exec_start
            exec_end = task.exec_end
            if exec_end < exec_start:
                continue
            total_aicore_time += exec_end - exec_start
            if global_min_start is None or exec_start < global_min_start:
                global_min_start = exec_start
            if global_max_end is None or exec_end > global_max_end:
                global_max_end = exec_end

    title = "-------- AICORE Prof Summary --------"
    if lane_count <= 0 or global_min_start is None or global_max_end is None:
        print(f"\n{title}")
        print("AICore End-to-End Time: 0.00")
        print("AICore Utilization:     0.00%")
        print("-" * len(title))
        return

    lane_duration = max(global_max_end - global_min_start, 0.0)
    utilization = 0.0
    if lane_duration > 0:
        utilization = total_aicore_time / (lane_count * lane_duration)

    print(f"\n{title}")
    print(f"AICore End-to-End Time: {lane_duration:.2f}")
    print(f"AICore Utilization:     {utilization * 100:.2f}%")
    print("-" * len(title))


def calculate_pipe_usage(path):
    global total_cores
    global total_tasks

    leaf_data = load_json(args.pipe_exec_time)
    leaf_funcs, pipe_list = pipe_time.get_leaf_funcs(leaf_data)

    total_execute_time = max_end_time - mininum_start_time
    total_pipe_use_time = {}
    aic_num = 0
    aiv_num = 0
    for _, value in total_cores.items():
        value.total_time = get_total_time(value.tasks)
        if value.total_time == 1:
            continue
        if 'AIC' in value.core_type:
            aic_num += 1
        elif 'AIV' in value.core_type:
            aiv_num += 1
        for entry in value.tasks:
            func_name = entry.func_name()
            if func_name in leaf_funcs:
                leaf_func = leaf_funcs[func_name]
                entry.leaf_total_cycles = leaf_func.leaf_total_time
                entry.leaf_pipe_exec_cycles = leaf_func.pipe_exe_time
        # Convert cycle number to time
        value.trans_cycles_to_time()

        # Accumulate pipe execute time
        for pipe, time in value.pipe_exec_time.items():
            total_pipe_use_time[pipe] = total_pipe_use_time.get(pipe, 0) + time

    sorted_cores = sorted(total_cores.items(), key=lambda x: x[1].core_idx)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Total Core Num:{aic_num + aiv_num}\n")
        f.write(f"AIC: {aic_num}\n")
        f.write(f"AIV: {aiv_num}\n")
        f.write("Total Pipe Usage\n")
        f.write("Pipe, AverageTime, TotalExecuteTime, AverageUsage\n")
        avg_time = total_pipe_use_time.get('CUBE', 0) / aic_num
        f.write(f"CUBE, {avg_time}, {total_execute_time}, {(avg_time / total_execute_time) * 100:.{4}}%\n")
        avg_time = total_pipe_use_time.get('VECTOR_ALU', 0) / aiv_num
        f.write(f"VECTOR_ALU, {avg_time}, {total_execute_time}, {(avg_time / total_execute_time) * 100:.{4}}%\n")
        avg_time = total_pipe_use_time.get('MTE_IN', 0) / (aic_num + aiv_num)
        f.write(f"MTE_IN, {avg_time}, {total_execute_time}, {(avg_time / total_execute_time) * 100:.{4}}%\n")
        avg_time = total_pipe_use_time.get('MTE1', 0) / aic_num
        f.write(f"MTE1, {avg_time}, {total_execute_time}, {(avg_time / total_execute_time) * 100:.{4}}%\n")
        avg_time = total_pipe_use_time.get('MTE_OUT', 0) / (aic_num + aiv_num)
        f.write(f"MTE_OUT, {avg_time}, {total_execute_time}, {(avg_time / total_execute_time) * 100:.{4}}%\n")

        pipe_list = ['MTE_IN', 'MTE1', 'MTE_OUT', 'CUBE', 'VECTOR_ALU']
        formatted_pipe = [f"{x}_Time, {x}_Usage" for x in pipe_list]
        head_str = ", ".join(formatted_pipe)
        f.write("\n\n")
        f.write("AICore Pipe Usage\n")
        f.write(f"Core, TotalTime, {head_str}\n")
        for _, value in sorted_cores:
            if value.total_time == 1:
                continue
            info = ''
            for pipe in pipe_list:
                info += f", {value.pipe_exec_time[pipe]}, {(value.pipe_exec_time[pipe] / value.total_time) * 100:.{4}}%"
            res = f"{value.get_core_name()}, {total_execute_time}{info}\n"
            f.write(res)
    return


if __name__ == "__main__":
    start_time = time_module.time()
    start_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Start time: {start_str}")

    args = parse_arguments()
    input_swim_data = load_json(args.swim_json_file)

    is_dyn = "dyn" in os.path.basename(args.topo_json_file)
    if is_dyn:
        assert args.func_table_file is not None, "For dynamic topo, program.json is required"
        if not os.path.exists(args.func_table_file):
            sys.exit(0)
        program_data = load_json(args.func_table_file)
        func_data = program_data["functions"]
        input_topo_data = load_dyn_topo(args.topo_json_file, func_data)
    else:
        input_topo_data = load_json(args.topo_json_file)

    dir_name = os.path.dirname(args.swim_json_file)
    # 根据日志信息和topo 信息构建total_cores 和 total_tasks
    build_swim_info(input_swim_data, input_topo_data, args.label_type, dir_name, args.mix_event_info)
    print_aicore_summary()

    # 输出核分析日志
    ana_path = os.path.join(dir_name, "bubble_analysis.log")
    analysis_wait_cycles(ana_path)
    if args.gen_exe_topo_json:
        pipe_usage_path = os.path.join(dir_name, "pipe_usage.csv")
        calculate_pipe_usage(pipe_usage_path)
        execute_json_path = os.path.join(dir_name, "execute.json")
        generate_execute_json(execute_json_path)

    output_path = os.path.join(dir_name, "merged_swimlane.json")
    convert_to_chrome_trace_json(output_path, is_dyn)
    print("Open the trace at https://ui.perfetto.dev/ \nOutput: ", output_path)

    end_time = time_module.time()
    end_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f"End time: {end_str}")

    duration = int(end_time - start_time)
    print(f"Time taken: {duration} secs")
