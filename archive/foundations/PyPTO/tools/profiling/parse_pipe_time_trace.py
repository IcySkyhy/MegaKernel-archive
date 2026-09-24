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

import json


class LeafFunc:
    def __init__(self, name):
        self.name = name
        self.func_hash = 0
        self.leaf_total_time = 0
        self.pipe_exe_time = {}


leaf_funcs = {}
pipe_list = []


def get_leaf_funcs(json_data):
    global leaf_funcs
    global pipe_list
    for leaf in json_data:
        name = leaf["FuncName"]
        func = LeafFunc(name)
        func.func_hash = leaf["FuncHash"]
        func.leaf_total_time = leaf["TotalCycles"]
        func.pipe_exe_time = leaf["pipes"]
        leaf_funcs[name] = func
    if len(json_data) > 0:
        for pipe, _ in json_data[0]["pipes"].items():
            pipe_list.append(pipe)
    return leaf_funcs, pipe_list


if __name__ == "__main__":
    file_path = "_simulate.leafFuncs.executetime.json"
    data = {}
    with open(file_path, "r") as file:
        data = json.load(file)
    res = get_leaf_funcs(data)
