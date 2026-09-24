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
from collections import namedtuple
import json


def find_block_with_idx(data, idx):
    for i, block in enumerate(data):
        if block['blockIdx'] == idx:
            return i
    return -1


def create_or_copy_block(data, global_idx, new_global_idx, core_type):
    curr_idx = find_block_with_idx(data, global_idx)
    if curr_idx != -1:
        new_block = data[curr_idx].copy()
        new_block['blockIdx'] = new_global_idx
    else:
        new_block = {
            "blockIdx": new_global_idx,
            "coreType": core_type,
            "tasks": [],
        }
    return new_block


def process_time_and_blocks(filename, aic_config, aiv_config, time_config):
    with open(filename, 'r') as f:
        data = json.load(f)

    min_start = float('inf')
    for block in data:
        for task in block['tasks']:
            min_start = min(min_start, task['execStart'])

    init_time = int(time_config.init_time / time_config.cycle_ratio)
    offset = min_start - init_time

    for block in data:
        for task in block['tasks']:
            task['execStart'] = (task['execStart'] - offset) * time_config.cycle_ratio
            task['execEnd'] = (task['execEnd'] - offset) * time_config.cycle_ratio

    new_data = []
    global_idx = 0
    new_global_idx = 0

    if aic_config.block_num is not None and aic_config.group_num is not None:
        base_blocks = aic_config.block_num // aic_config.group_num
        extra_blocks = aic_config.block_num % aic_config.group_num

        for group in range(aic_config.group_num):
            blocks_count = base_blocks + (1 if group < extra_blocks else 0)
            for _ in range(blocks_count):
                new_block = create_or_copy_block(data, global_idx, new_global_idx, "AIC")
                new_data.append(new_block)
                global_idx += 1
                new_global_idx += 1

            if group >= extra_blocks:
                new_data.append(
                    {
                        "blockIdx": new_global_idx,
                        "coreType": "AIC",
                        "tasks": [],
                    }
                )
                new_global_idx += 1

    if aiv_config.block_num is not None and aiv_config.group_num is not None:
        base_blocks = aiv_config.block_num // aiv_config.group_num
        extra_blocks = aiv_config.block_num % aiv_config.group_num

        for group in range(aiv_config.group_num):
            blocks_count = base_blocks + (1 if group < extra_blocks else 0)
            for _ in range(blocks_count):
                new_block = create_or_copy_block(data, global_idx, new_global_idx, "AIV")
                new_data.append(new_block)
                global_idx += 1
                new_global_idx += 1

            if group >= extra_blocks:
                new_data.append(
                    {
                        "blockIdx": new_global_idx,
                        "coreType": "AIV",
                        "tasks": [],
                    }
                )
                new_global_idx += 1

    new_data.sort(key=lambda x: x['blockIdx'])

    with open('processed_tilefwk_prof_data.json', 'w') as f:
        json.dump(new_data, f, indent=2)

    print("Onboard Json Processed!")


def main():
    parser = argparse.ArgumentParser(description='Process profile log')
    parser.add_argument('--input_file', default="tilefwk_prof_data.json", help='Input JSON file path')
    parser.add_argument('--aic_block_num', type=int, default=25, help='AIC block number')
    parser.add_argument('--aic_group_num', type=int, default=3, help='AIC group number')
    parser.add_argument('--aiv_block_num', type=int, default=50, help='AIV block number')
    parser.add_argument('--aiv_group_num', type=int, default=3, help='AIV group number')
    parser.add_argument('--init_time', type=int, default=3600, help='Initial time')
    parser.add_argument('--cycle_ratio', type=int, default=36, help='Cycle ratio')

    args = parser.parse_args()

    BlockConfig = namedtuple('BlockConfig', ['block_num', 'group_num'])
    TimeConfig = namedtuple('TimeConfig', ['init_time', 'cycle_ratio'])

    aic_config = BlockConfig(args.aic_block_num, args.aic_group_num)
    aiv_config = BlockConfig(args.aiv_block_num, args.aiv_group_num)
    time_config = TimeConfig(args.init_time, args.cycle_ratio)

    process_time_and_blocks(args.input_file, aic_config, aiv_config, time_config)


if __name__ == '__main__':
    main()
