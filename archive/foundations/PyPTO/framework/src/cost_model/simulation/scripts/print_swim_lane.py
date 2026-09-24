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
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

args = None


def plot_workflow(ndata, task_ids, labels, core_type):
    color_id = 0
    core_nr, cols_nr = ndata.shape

    _, ax = plt.subplots(figsize=(cols_nr * 1.2 + 1, core_nr * 0.3))

    # 统计每个核心的任务数量
    tasks_count = {}
    for i in range(core_nr):
        # 计算非零任务数（compute部分）
        task_count = np.count_nonzero(ndata[i, 3::2])  # 从第4列开始，每隔2列统计compute部分
        tasks_count[i] = task_count

    # 修改y轴标签，添加任务数量信息
    ticklabels = []
    for i in range(core_nr):
        if 'AIV' in core_type[i]:
            ticklabels.append(f'AIV_{i + 1} ({tasks_count[i]})')
        elif 'MIX' in core_type[i]:
            ticklabels.append(f'AICORE_{i + 1} ({tasks_count[i]})')
        else:
            ticklabels.append(f'AIC_{i + 1} ({tasks_count[i]})')

    start_time = np.zeros(core_nr)
    for i in range(cols_nr):
        color_id += 1
        if color_id % 10 == 3:
            color_id += 1
        if 'delay' in labels[i]:
            color = (0.9, 0.9, 0.9)  # white color
        else:
            color = plt.cm.tab10(color_id % 10)

        ax.barh(range(core_nr), ndata[:, i], left=start_time, height=0.8, label=labels[i], color=color)
        if 'compute' in labels[i]:
            for j, left in enumerate(start_time):
                ax.text(left + ndata[j, i] / 2, j, task_ids[j][i], va='center', ha='center')
        start_time += ndata[:, i]

    ax.set_xlabel('Cycles' if args.cycles else 'Time (us)')

    ax.set_yticks(range(core_nr))
    ax.set_yticklabels(ticklabels)  # 使用新的带有任务数量的标签
    x_ticks = ax.get_xticks()
    for x in x_ticks:
        ax.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

    plt.xlim(0, None)
    plt.tight_layout()
    if args.output == '':
        plt.savefig(f"{os.path.splitext(args.infile)[0]}.png")
    else:
        plt.savefig(f"{args.output}/{args.op}.png")


def prepare_workflow_data(infile):
    with open(infile) as file:
        jdata = json.load(file)

    fdata = list(filter(lambda x: x["tasks"], jdata))
    # 如果全部为空，直接返回空场景
    if not fdata:
        return [], [], [], []  # 与下游变量个数保持一致
    max_task_nr = max([len(data["tasks"]) for data in fdata])

    labels = ["start", "handshake"]
    for i in range(max_task_nr):
        labels += [f"delay{i}", f"compute{i}"]

    cols_nr, core_nr = len(labels), len(jdata)
    ndata = np.zeros((core_nr, cols_nr))
    task_ids = np.zeros_like(ndata, dtype=np.int32)
    core_type = {}
    for i, data in enumerate(jdata):
        core_type[i] = data['coreType']
        if not data["tasks"]:
            continue
        ndata[i][0] = 0
        ndata[i][1] = 0

        for j, task in enumerate(data["tasks"]):
            if j == 0:
                ndata[i][j * 2 + 2] = task["execStart"]
            else:
                ndata[i][j * 2 + 2] = task["execStart"] - data["tasks"][j - 1]["execEnd"]
            ndata[i][j * 2 + 3] = task["execEnd"] - task["execStart"]
            if args.task_id:
                task_ids[i][j * 2 + 3] = task["taskId"]
            else:
                task_ids[i][j * 2 + 3] = task["subGraphId"]

    if not args.cycles:
        ndata /= args.frequency * 1000
    return ndata, task_ids, labels, core_type


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument('-t', '--task-id', action='store_true', help="show task id in workflow")
    parser.add_argument('--output', default='', help="png csv output directory")
    parser.add_argument('--op', help="op name", default='op')
    parser.add_argument('-c', '--cycles', action='store_true', help="use cycles as unit of x-label")
    parser.add_argument('-f', '--frequency', type=float, default=2.0, help="clock frequency in GHz (default: 2.0 GHz)")
    args = parser.parse_args()

    ndata, task_ids, labels, core_type = prepare_workflow_data(args.infile)
    length = len(ndata) if isinstance(ndata, list) else ndata.size

    # 2. 空场景直接退出
    if length == 0:
        sys.exit(0)
    if args.output == '':
        np.savetxt(f"{os.path.splitext(args.infile)[0]}.csv", ndata, fmt='%.2f', delimiter=',', header=','.join(labels))
    else:
        np.savetxt(f"{args.output}/{args.op}.csv", ndata, fmt='%.2f', delimiter=',', header=','.join(labels))
    plot_workflow(ndata, task_ids, labels, core_type)
