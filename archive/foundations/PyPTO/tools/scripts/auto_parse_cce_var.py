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
import re


def format_list(input_list):
    return [item for item in input_list if item != ""]


def int_list(input_list, base=10):
    return [int(item, base) for item in input_list if item != ""]


def extract_func(func_str_list, raw_tensor_addrs):
    func_hash_tuple_list = []
    for idx in range(0, len(func_str_list), 2):
        func_hash = func_str_list[idx].split("#funcHash:")[1].split(' #')[0].strip()
        if int(func_hash) == 0:
            continue

        invoke_attrs = format_list(func_str_list[idx + 1].split("#invokeAttrs : ")[1].split("@"))
        tensor_info = []
        for tensor_info_str in invoke_attrs:
            tensor_info_list = int_list(tensor_info_str.replace(',', '').split(' '), 10)

            shape_dim = (len(tensor_info_list) - 1) // 4
            tensor_addr = raw_tensor_addrs[int(tensor_info_list[0])]
            offset = tensor_info_list[1:1 + shape_dim]
            shape = tensor_info_list[1 + shape_dim:1 + shape_dim * 2]
            raw_shape = tensor_info_list[1 + shape_dim * 2:1 + shape_dim * 3]
            valid_shape = tensor_info_list[1 + shape_dim * 3:1 + shape_dim * 4]
            tensor_info.append(
                [
                    tensor_addr,  # tensorAddr
                    offset,  # offset
                    shape,  # shape
                    raw_shape,  # rawShape
                    valid_shape,  # validShape
                ]
            )
        func_hash_tuple_list.append((func_hash, tensor_info))
    _func_hash_map_json = json.dumps(func_hash_tuple_list, indent=4, ensure_ascii=False)
    return func_hash_tuple_list


def extract_kernel_params(kernel_path):
    """
    遍历kernel_meta,获取func_hash与kernel的对应关系

    数据结构
    kernel_params = {
        "func_hash":(cce文件名, var_code_list)
    }

    """
    hash_pattern = re.compile(
        r'// funcHash:\s*(.*?)\n',  # 匹配// funcHash: 这一行的内容
        re.DOTALL | re.IGNORECASE,  # DOTALL允许.匹配换行符,IGNORECASE忽略大小写
    )

    var_pattern = re.compile(
        r'(uint64_t sym\s*.*?)\n',  # 匹配// int32_t sym 这一行的内容
        re.DOTALL | re.IGNORECASE,  # DOTALL允许.匹配换行符,IGNORECASE忽略大小写
    )

    kernel_params_map_result = {}
    for entry in os.listdir(kernel_path):
        entry_path = os.path.join(kernel_path, entry)
        if os.path.isfile(entry_path):
            if entry.lower().endswith(".cpp"):
                # 解析funcHash
                with open(entry_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                func_hash = hash_pattern.findall(content)

                with open(entry_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                kernel_var = var_pattern.findall(content)
                kernel_params_map_result.setdefault(func_hash[0], (entry, kernel_var))

    return kernel_params_map_result


def extract_func_data(file_path):
    """
    从指定文件中提取所有#funcData块的内容(从[开始到]结束)
    数据结构
    func_data = {
        "func_data_0": [(func_hash, tensor_info), (func_hash, tensor_info)...],
        "func_data_1": [(func_hash, tensor_info), (func_hash, tensor_info)...],
    }
    tensor_info = [tensorAddr, offset, shape, rawShape, validShape]

    """
    pattern = re.compile(
        r'#funcData:\s*\[(.*?)\n\]',  # 匹配#funcData: [ 到 ] 之间的内容(非贪婪模式)
        re.DOTALL | re.IGNORECASE,  # DOTALL允许.匹配换行符,IGNORECASE忽略大小写
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找所有匹配的funcData块
    func_datas = pattern.findall(content)

    func_data_result = {}
    for i, func_data in enumerate(func_datas):
        # 去除#funcData: [和]之间的多余空白,保留内部换行
        func_data_str = re.sub(r'^\s+|\s+$', '', func_data)  # 去除首尾空白
        func_str = func_data_str.split('#rawTensorAddrs:')[0].strip()

        raw_tensor_addrs = (
            func_data_str.replace('\n', '')
            .replace('\r', '')
            .replace(' ', '')
            .split('#rawTensorAddrs:')[1]
            .strip()
            .split(',')
        )
        raw_tensor_addr_list = format_list(raw_tensor_addrs)

        func_str_list = func_str.splitlines()
        # 解析aicpu3中的每一个funcData
        func_data_tuple = extract_func(func_str_list, raw_tensor_addr_list)
        func_data_result.setdefault(f"func_data_{i}", func_data_tuple)

    return func_data_result


def parse_cce_var(kernel_params, func_data):
    """
    解析cce_var
    func_data_kernel_var_map = {
        "func_data_0": [(func_hash, kernel_var), (func_hash, kernel_var)...],
        "func_data_1": [(func_hash, kernel_var), (func_hash, kernel_var)...],
    }
    kernel_params = {
        "func_hash":(cce文件名, var_code_list)
    }
    func_data = {
        "func_data_0": [(func_hash, tensor_info), (func_hash, tensor_info)...],
        "func_data_1": [(func_hash, tensor_info), (func_hash, tensor_info)...],
    }
    tensor_info = [tensorAddr, offset, shape, rawShape, validShape]

    """
    func_data_kernel_var_map = {}
    for func_data_key, func_data_val in func_data.items():
        func_hash_kernel_var_list = []
        for func_hash, tensor_info in func_data_val:
            if func_hash not in kernel_params.keys():
                raise RuntimeError(f"not find func_hash{func_hash} in kernel_meta")
            kernel_bin = kernel_params[func_hash][0]
            kernel_var_code = kernel_params[func_hash][1]
            kernel_var_value = []
            for sym_var in kernel_var_code:
                var_info = int_list(sym_var.split(");")[0].split("param, ")[1].split(", "))
                tensor_id = var_info[0]
                valid_shape_dim = var_info[3]
                var_value = tensor_info[tensor_id][4][valid_shape_dim]
                tensor_addr = tensor_info[tensor_id][0]
                new_sym_var_code = (
                    sym_var.replace(" = GET", f" = {var_value};  // GET") + f"  // tensorAddr: {tensor_addr}"
                )
                kernel_var_value.append(new_sym_var_code)
            func_hash_kernel_var_list.append((func_hash, kernel_bin, kernel_var_value))
        func_data_kernel_var_map.setdefault(func_data_key, func_hash_kernel_var_list)

    return func_data_kernel_var_map


if __name__ == "__main__":
    """
    脚本说明:
    脚本用于将动态cce中表征validShape的变量, 根据aicpu3日志中的funcData进行真实值计算。
    使用方法:python3 tools/scripts/auto_parse_cce_var.py -f ./aicpu3.txt -k ./build/output/bin/kernel_meta/

    输入:
    aicpu3.txt-----funcData代表了运行时的真实参数
    kernel_meta ---cce文件

    输出:
    func_data_kernel_var.json

    json生成结果如下:
    [func_data_kernel_var_map, func_data_map, kernel_params_map]

    func_data_kernel_var_map:
    {
        "func_data_0": [funcHash, cce文件名, kernel_var], # 已替换后的真实值
        "func_data_1": [funcHash, cce文件名, kernel_var],
        ...
    }
    func_data_map:
    {
        "func_data_0": [funcHash, [tensorAddr, offset, shape, rawshape, validshape]]
        ...
    }
    kernel_params_map:
    {
        "funcHash": [cce文件名, kernel_var_code_list]
        ...
    }
    """

    parser = argparse.ArgumentParser(description="根据aicpu3中的funcData, 自动还原cce中sym_var的真值")
    parser.add_argument("-f", "--func_data_file", required=True, help="输入aicpu3文件路径")
    parser.add_argument("-k", "--kernel_meta_path", required=True, help="输入kernel_meta路径")

    args = parser.parse_args()
    func_data_file_path = args.func_data_file
    kernel_path = args.kernel_meta_path

    # 记录kernel中的信息,包括funcHash和变量
    kernel_params_map = extract_kernel_params(kernel_path)
    # 记录aicpu3.txt中funcData的信息,包括funcHash和tensorInfo
    func_data_map = extract_func_data(func_data_file_path)
    # 根据funcData中的信息,将kernel中var变量的进行替换
    func_data_kernel_var_map = parse_cce_var(kernel_params_map, func_data_map)

    with open("./func_data_kernel_var.json", "w", encoding="utf-8") as f:
        json_list = [func_data_kernel_var_map, func_data_map, kernel_params_map]
        json.dump(json_list, f, ensure_ascii=False, indent=4)

    print("生成 ./func_data_kernel_var.json 成功,包含 func_data_kernel_var_map,func_data_map,kernel_params_map 信息")
