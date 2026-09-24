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

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo


class PipeEntryInfo:
    def __init__(self, task_id):
        self.task_id = task_id
        self.core_idx = 0
        self.core_type = ""
        self.subgraph_id = -1
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
        self.pipe_name = ""
        self.exec_info = ""


class CoreInfo:
    def __init__(self, core_id, c_type):
        self.core_idx = core_id
        self.core_type = c_type
        self.tasks = []
        self.has_overlap = False
        self.last_task_end_time = 0
        self.total_wait_time = 0
        self.core_wait_schedule_time = 0
        self.core_wait_predecessor_time = 0
        self.faketask_num = 0
        self.pipes = {}


total_pipe_events = {}  # key: task_id, value: [TaskInfo]
total_cores = {}  # key: core_idx, value: [CoreInfo]
pipe_event_alloc = 0
max_end_time = 0
work_data = []
pipe_name_map = {5: 'PIPE_S', 4: 'MTE_IN', 3: 'MTE1', 2: 'VECTOR_ALU', 1: 'CUBE', 0: 'MTE_OUT'}
pipe_name_revers_map = {'PIPE_S': 5, 'MTE_IN': 4, 'MTE1': 3, 'VECTOR_ALU': 2, 'CUBE': 1, 'MTE_OUT': 0}
colors = ['#83639F', '#FAC03D', '#449945', '#1F70A9', '#C22F2F']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process two JSON files.')
    parser.add_argument('swim_json_file', type=str, help='Path to the first JSON file')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Sample interval time,default=1')
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def build_swim_info(swim_data):
    global total_cores
    global total_pipe_events
    global pipe_event_alloc
    global max_end_time

    for core in swim_data:
        core_idx = core['blockIdx']
        core_type = core.get('coreType', 'Core')
        core_entry = CoreInfo(core_idx, core_type)

        pipe_logs = core.get('pipeLogs', {})

        for pipe_name, logs in pipe_logs.items():
            core_entry.pipes[pipe_name] = []
            if pipe_name not in total_pipe_events.keys():
                total_pipe_events[pipe_name] = []
            for event in logs:
                event_id = pipe_event_alloc
                pipe_event_alloc += 1

                entry = PipeEntryInfo(event_id)
                entry.core_idx = core_idx
                entry.exec_start = event.get('execStart', 0)
                entry.exec_end = event.get('execEnd', 0)
                max_end_time = max(max_end_time, entry.exec_end)
                entry.pipe_name = pipe_name
                entry.exec_info = event.get('tileOp', "")
                core_entry.pipes[pipe_name].append(entry)
                total_pipe_events[pipe_name].append(entry)

        total_cores[core_idx] = core_entry


def sample_work_data():
    global total_cores
    global total_pipe_events
    global max_end_time
    global work_data

    # 参数配置
    time_convert = 1
    sample_interval = args.sample_interval
    num_cores = len(total_cores)
    stages_per_core = len(pipe_name_map)
    total_time = ((max_end_time // time_convert) // sample_interval) + 1  # 时间轴长度

    # 数据准备：生成三维数据矩阵 [core][stage][time]
    work_data = np.zeros((num_cores, stages_per_core, total_time))
    for core_idx, core in total_cores.items():
        if len(core.pipes) == 0:
            continue

        for pipe_name, pipe_events in core.pipes.items():
            if len(pipe_events) == 0:
                continue
            pipe_swim_line_idx = pipe_name_revers_map[pipe_name]
            for event in pipe_events:
                start_time = int(event.exec_start // time_convert)
                end_time = int(event.exec_end // time_convert)
                while start_time < end_time:
                    sample_time = start_time // sample_interval
                    work_data[core_idx, pipe_swim_line_idx, sample_time] = 1  # 标记活跃时间段
                    start_time += sample_interval


def draw_pipe_swim_lane_png(path):
    global total_cores
    global total_pipe_events
    global max_end_time
    global work_data

    # 参数配置
    time_convert = 1
    sample_interval = args.sample_interval
    num_cores = len(total_cores)
    stages_per_core = len(pipe_name_map)
    total_time = ((max_end_time // time_convert) // sample_interval) + 1  # 时间轴长度

    # 创建图表
    width = max(total_time * 0.5 + 1, 18)
    height = max(num_cores * 2, 9)
    fig, ax = plt.subplots(figsize=(width, height))

    # 绘制泳道栅格
    for core_idx in range(num_cores):
        base_stage = core_idx * stages_per_core  # 当前核心的基础阶段偏移量

        for stage in range(stages_per_core):
            current_color = colors[stage % len(colors)]  # 循环使用颜色

            # 绘制该阶段的时间序列
            for t in range(total_time):
                if work_data[core_idx, stage, t]:  # 如果该时刻工作
                    rect = plt.Rectangle(
                        xy=(t, base_stage + stage),  # 左下角坐标(t, y)
                        width=1,  # 时间跨度1单位
                        height=1,  # 泳道高度1单位
                        facecolor=current_color,  # 填充颜色
                        edgecolor='#A0A0A0',  # 边框灰边
                        linewidth=0.5,  # 边框宽度
                    )
                    ax.add_patch(rect)  # 添加到图表

    # 设置坐标系
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, num_cores * stages_per_core)

    # 添加横向分割线（每5个泳道）
    for split_line in range(stages_per_core, num_cores * stages_per_core + 1, stages_per_core):
        ax.axhline(y=split_line, color='black', linestyle='-', linewidth=0.8)

    # 设置Y轴标签
    core_centers = [(core_id * stages_per_core) + stages_per_core / 2 for core_id in range(num_cores)]
    ax.set_yticks(core_centers)
    ax.set_yticklabels([f'{total_cores[i].core_type}_{i}' for i in sorted(total_cores.keys())])

    legend_elements = []
    for i, color in enumerate(colors):
        legend_element = Patch(facecolor=color, edgecolor='black', label=f"{pipe_name_map[i]}")
        legend_elements.append(legend_element)

    # 添加图例和视觉修饰
    ax.legend(handles=legend_elements, title="Pipe Types", loc='upper center', ncol=15, frameon=False)

    # 其他美化设置
    plt.title("Multi-Core Task Scheduling Visualization", pad=20)
    plt.xlabel("Time Units")
    plt.ylabel("AICores")
    ax.grid(axis='x', alpha=0.3)  # 只显示横向辅助线
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig(path)


def draw_pipe_swim_lane_html(path):
    global total_cores
    global total_pipe_events
    global max_end_time
    global work_data

    # 参数配置
    time_convert = 1
    sample_interval = args.sample_interval
    num_cores = len(total_cores)
    stages_per_core = len(pipe_name_map)
    total_time = ((max_end_time // time_convert) // sample_interval) + 1  # 时间轴长度

    # 创建 Plotly 图表
    fig = go.Figure()
    # 添加泳道条
    for core_idx in range(num_cores):
        base_stage = core_idx * stages_per_core  # 当前核心的基础阶段偏移量
        for stage in range(stages_per_core):
            current_color = colors[stage % len(colors)]  # 循环使用颜色
            # 绘制该阶段的时间序列
            for t in range(total_time):
                if work_data[core_idx, stage, t]:  # 如果该时刻工作
                    level = base_stage + stage
                    fig.add_trace(
                        go.Scatter(
                            x=[t, t + 1, t + 1, t, t],
                            y=[level, level, level + 0.9, level + 0.9, level],
                            fill='toself',
                            fillcolor=current_color,
                            mode='lines',
                            line=dict(color=current_color),
                            showlegend=False,
                        )
                    )

    # 设置布局
    fig.update_layout(
        title="Multi-Core Task Scheduling Visualization",
        xaxis_title="Time",
        yaxis_title="AICores",
        yaxis=dict(
            tickmode='array',
            range=[0, num_cores * stages_per_core],
            tickvals=list(range(0, num_cores * stages_per_core, stages_per_core)),
            ticktext=[f'{total_cores[i].core_type}_{i}' for i in sorted(total_cores.keys())],
            showgrid=False,
        ),
        xaxis=dict(
            tickmode='auto',
            range=[0, total_time],
        ),
    )

    # 添加图例
    legend_items = []
    for index in range(len(colors) - 1, -1, -1):
        legend_items.append(
            go.Scatter(
                x=[0], y=[0], mode='markers', marker=dict(color=colors[index], size=10), name=pipe_name_map[index]
            )
        )

    fig.add_traces(legend_items)
    pyo.plot(fig, filename=path, auto_open=False)


if __name__ == '__main__':
    args = parse_arguments()
    input_swim_data = load_json(args.swim_json_file)
    dir_name = os.path.dirname(args.swim_json_file)

    # 根据日志信息构建total_cores 和 total_pipe_events
    build_swim_info(input_swim_data)

    # 采样构建work_data
    sample_work_data()

    # 绘制包含pipe信息的泳道图
    output_path = os.path.join(dir_name, 'pipe_swimlane.png')
    draw_pipe_swim_lane_png(output_path)

    output_path = os.path.join(dir_name, 'pipe_swimlane.html')
    draw_pipe_swim_lane_html(output_path)
