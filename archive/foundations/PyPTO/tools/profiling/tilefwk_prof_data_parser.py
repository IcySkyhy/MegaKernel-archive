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
import copy
import json
import os
from pathlib import Path
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# MsprofAdditonalInfo: 256 byte
TILEFWK_STARS_LOG_FMT_DYN = "HHIIIQ" + "HHIHBB" + "I" * 7 + "HHIQQ" * 8
MSPROF_REPORT_AICPU_HCCL_OP_INFO = 10
PROF_DATATYPE_LOG = 2
CORE_TYPE = ["AIV", "AIC", "MIX", "AICPU"]


class TileFwkLogStructBean:
    def __init__(self: any, isdyn: bool, *args) -> None:  # 利用struct 的unpack的特性传递 解压后数据
        filed = args[0]
        self._magic_number = filed[0]
        self._level = filed[1]
        self._type = filed[2]
        self._thread_id = filed[3]
        self._data_len = filed[4]
        self._time_stamp = filed[5]
        self._head = filed[6]
        self._core_id = filed[7] & 127  # 取低7位core_id
        self._core_type = (filed[7] >> 7) & 7  # 中间3位 core_type AIV = 0, AIC = 1, MIX = 2, AICPU = 3
        self._data_type = (filed[7] >> 10) & 63  # 取高6位data_type
        self._task_id = filed[8]
        self._stream_id = filed[9]
        self._task_count = filed[10]
        self._res0 = filed[11]
        if self._type != MSPROF_REPORT_AICPU_HCCL_OP_INFO or self._data_type != PROF_DATATYPE_LOG:
            return
        self._task_list = []
        self._task_count = min(8, self._task_count)

        for i in range(0, self._task_count):
            if isdyn:
                self._task_list.append(
                    {
                        "seqNo": filed[19 + i * 0x5],
                        "subGraphId": filed[20 + i * 0x5],
                        "taskId": filed[21 + i * 0x5],
                        "execStart": filed[22 + i * 0x5],
                        "execEnd": filed[23 + i * 0x5],
                    }
                )
            else:
                self._task_list.append(
                    {
                        "subGraphId": filed[20 + i * 0x5],
                        "taskId": filed[21 + i * 0x5],
                        "execStart": filed[22 + i * 0x5],
                        "execEnd": filed[23 + i * 0x5],
                    }
                )

    @property
    def data(self: any) -> dict():
        if self._type != MSPROF_REPORT_AICPU_HCCL_OP_INFO or self._data_type != PROF_DATATYPE_LOG:
            return {}
        m = 0
        if m == 1:
            print("----------------------------------------------------------")
            print("TileFwk log get level: ", self._level)
            print("TileFwk log get type: ", self._type)
            print("TileFwk log get data len: ", self._data_len)
            print("TileFwk log get time stamp: ", self._time_stamp)
            print("TileFwk log get head: ", self._head)
            print("TileFwk log get core id: ", self._core_id)
            print("TileFwk log get data type: ", self._data_type)
            print("TileFwk log get task count: ", self._task_count)
            print("----------------------------------------------------------")
        if self._core_type > len(CORE_TYPE):
            assert False, f"{self._core_type} is invalid, only supprt [0~3]"
        task_dict = {"blockIdx": self._core_id, "coreType": CORE_TYPE[self._core_type], "tasks": self._task_list}
        return task_dict


def parse(file_path: str, isdyn: bool = False) -> list():
    bean_list = []
    print("start to parse")
    for name in os.listdir(file_path):
        if "aicpu.data" not in name:
            continue
        print("support data: " + name)
        # 构建完整的文件或目录路径
        file_name = os.path.join(file_path, name)
        # 如果是文件，则打印其路径
        print("support file path: ", file_name)
        file_size = os.path.getsize(file_name)
        if not file_size and "complete" not in file_name:
            print(file_name + " file size is empty.")
            return bean_list

        struct_fmt = TILEFWK_STARS_LOG_FMT_DYN
        struct_size = struct.calcsize(struct_fmt)
        with open(file_name, "rb") as fd:
            context = fd.read()
            for _index in range(file_size // struct_size):
                data = context[_index * struct_size:(_index + 1) * struct_size]
                append_dict = TileFwkLogStructBean(isdyn, struct.unpack(struct_fmt, data)).data
                if not append_dict:
                    continue
                for block_dict in bean_list:
                    if block_dict["blockIdx"] == append_dict["blockIdx"]:
                        block_dict["tasks"] += append_dict["tasks"]
                        append_dict.clear()
                        break
                if append_dict:
                    bean_list.append(append_dict)
    return bean_list


def plot_workflow(ndata, task_ids, labels, core_data, output=''):
    color_id = 0
    core_nr, cols_nr = ndata.shape

    _, ax = plt.subplots(figsize=(cols_nr * 0.3 + 1, core_nr * 0.3))

    start_time = np.zeros(core_nr)
    for i in range(cols_nr):  # 按列轮询
        color_id += 1
        if color_id % 10 == 3:
            color_id += 1
        if 'delay' in labels[i]:
            color = (0.9, 0.9, 0.9)  # white color
        else:
            color = plt.cm.tab10(color_id % 10)
        # 生成条状图，起始时间为start_time表格对应time，行数core_nr，多行数值ndata[:, i]
        ax.barh(range(core_nr), ndata[:, i], left=start_time, height=0.8, label=labels[i], color=color)
        if 'compute' in labels[i]:
            for j, left in enumerate(start_time):
                if ndata[j, i] == 0:  # compute时间为0时无需添加taskid信息
                    continue
                # 添加文本注释，（left + ndata[j, i]/2，j）位置添加注释task_ids[j][[i]]
                ax.text(left + ndata[j, i] / 2, j, task_ids[j][i], va='center', ha='center')
        start_time += ndata[:, i]  # 更新start_time表格

    ax.set_xlabel('Durations[us]')  # 设置x轴name
    ax.set_ylabel('[Aic:0-24, Aiv:25-74]')  # 设置x轴name
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=15)

    ax.set_yticks(range(core_nr))
    ax.set_yticklabels(core_data)  # 设置y轴name

    x_ticks = ax.get_xticks()
    for x in x_ticks:
        ax.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)  # 添加垂直线

    plt.xlim(0, None)
    if output == '':
        plt.savefig("tilefwk_prof_data.png", bbox_inches='tight')
    else:
        plt.savefig(f"{output}/tilefwk_prof_data.png", bbox_inches='tight')


def prepare_workflow_data(infile, task_id_flag, output):
    with open(infile) as file:
        jdata = json.load(file)  # 读取json格式的json文件

    fdata = list(filter(lambda x: x["tasks"], jdata))  # json格式转list

    core_data = []
    for x in fdata:
        core_data.append(x["blockIdx"])
    print("Core id list: ", core_data)
    max_task_nr = max([len(data["tasks"]) for data in fdata])  # 获取所有json对象中最多的task个数
    min_start_time = float('inf')  # 获取最小的task.execStart时间戳
    for _, data in enumerate(jdata):  # 遍历json格式数据
        min_task_time = min([task["execStart"] for task in data["tasks"]])
        min_start_time = min(min_task_time, min_start_time)
    labels = []
    for i in range(max_task_nr):
        labels += [f"delay{i}", f"compute{i}"]  # label list添加max task个数的延迟字符串f"delay{i}"和f"compute{i}"

    labels_task_info = ["coreId", "seqNo", "subgraphId", "taskId", "startCycle", "endCycle"]
    task_info_data = np.empty((0, len(labels_task_info)), np.uint64)

    cols_nr, core_nr = len(labels), len(jdata)
    ndata = np.zeros((core_nr, cols_nr))  # 创建一个形状为(len(labels), len(jdata))的全0数组
    task_ids = np.zeros_like(ndata, dtype=np.int32)  # 创建一个和ndata有相同形状的int32数据类型的全0数组

    task_info = np.zeros(len(labels_task_info), np.uint64)
    core_type_list = []
    for i, data in enumerate(jdata):  # 遍历json格式数据
        if not data["tasks"]:  # 排除没有task的json对象
            continue
        task_info[0] = data['blockIdx']
        last_start_time = min_start_time
        for j, task in enumerate(data["tasks"]):  # 遍历task数据
            ndata[i][j * 2] = task["execStart"] - last_start_time  # 计算execStart - (上一个)execEnd，写入ndata i行
            ndata[i][j * 2 + 1] = task["execEnd"] - task["execStart"]  # 计算execEnd - execStart，写入ndata i行
            last_start_time = task["execEnd"]  # 更新上一个时间戳
            if task_id_flag:  # 若输入task-id，写入task id到task_ids i行
                task_ids[i][j * 2 + 1] = task["taskId"]
            else:
                task_ids[i][j * 2 + 1] = task["subGraphId"]
            task_info[1] = task.get("seqNo", 0)
            task_info[2] = task["subGraphId"]
            task_info[3] = task["taskId"]
            task_info[4] = task["execStart"]
            task_info[5] = task["execEnd"]
            task_info_data = np.vstack((task_info_data, task_info))
            core_type_list.append(data['coreType'])
    df = pd.DataFrame(task_info_data, columns=labels_task_info)
    df.insert(1, "coreType", core_type_list)
    sort_df = df.sort_values(by="coreId", ascending=True)

    if output == '':
        sort_df.to_csv("tilefwk_task_info.csv", index=False, encoding="utf-8")
    else:
        sort_df.to_csv(f"{output}/tilefwk_task_info.csv", index=False, encoding="utf-8")
    ndata /= 50  # trans cycle to us
    return ndata, task_ids, labels, core_data


def data_supple(ndata, task_ids, core_data):
    cols_nr = ndata.shape[1]
    all_data = np.zeros((75, cols_nr))
    all_core_tasks = np.zeros((75, cols_nr), np.int32)
    all_core_list = list(range(75))
    for i in range(75):
        if i in core_data:
            index = core_data.index(i)
            all_data[i] = ndata[index]
            all_core_tasks[i] = task_ids[index]
    return all_data, all_core_tasks, all_core_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="请输入要解析的绝对路径")
    parser.add_argument('-t', '--task-id', action='store_true', help="show task id in workflow")
    parser.add_argument('--output', default='', help="png csv output directory")
    parser.add_argument('--op', help="op name")
    parser.add_argument('-d', "--dyn", action='store_true', help="parse dynamic perf data")
    parser.add_argument('-c', '--cycles', action='store_true', help="use cycles as unit of x-label")
    args = parser.parse_args()
    print("start parser log data:" + args.path)
    task_log_list = parse(args.path, args.dyn)
    for task in task_log_list:
        task["tasks"].sort(key=lambda x: x['execStart'])
    print("end parser log data")
    json_path = 'tilefwk_prof_data.json'
    if args.output != '':
        json_path = f'{args.output}/tilefwk_prof_data.json'
        if not Path(args.output).exists():
            os.makedirs(args.output)
    with open(json_path, 'w') as json_file:
        json.dump(task_log_list, json_file, indent=4)

    task_id_flag = False
    if args.task_id:
        task_id_flag = True
    ndata, task_ids, labels, core_data = prepare_workflow_data(json_path, task_id_flag, args.output)
    all_data, all_core_tasks, all_core_list = data_supple(ndata, task_ids, core_data)
    core_info = copy.deepcopy(all_data)
    core_info = np.insert(core_info, 0, all_core_list, axis=1)
    labels_core = copy.deepcopy(labels)
    labels_core.insert(0, "core_id")

    if args.output == '':
        np.savetxt("tilefwk_prof_data.csv", core_info, fmt='%.2f', delimiter=',', header=','.join(labels_core))
    else:
        np.savetxt(
            f"{args.output}/tilefwk_prof_data.csv", core_info, fmt='%.2f', delimiter=',', header=','.join(labels_core)
        )
    plot_workflow(all_data, all_core_tasks, labels, all_core_list, args.output)


if __name__ == '__main__':
    main()
