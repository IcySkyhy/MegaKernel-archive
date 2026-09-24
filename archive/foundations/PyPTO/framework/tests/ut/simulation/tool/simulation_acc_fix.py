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
import os
import subprocess


def run_command_and_get_output(command):
    # 执行命令并捕获标准输出和错误输出（文本模式）
    result = subprocess.run(
        command,
        shell=True,  # 使用shell解析命令（如需路径扩展或通配符可保留）
        check=True,  # 命令执行失败（非零返回码）时抛出异常
        text=True,  # 以文本模式返回输出（非字节流）
        capture_output=True,  # 捕获stdout和stderr
    )

    # 合并标准输出和错误输出（根据需求选择）
    output = f"标准输出:\n{result.stdout}\n"
    return {
        "成功": True,
        "返回码": result.returncode,
        "输出": output,
    }


def execute():
    # 要执行的目标指令
    target_command = "build/output/bin/tile_fwk_utest --gtest_filter=CostModelTest.TestAttentionPostFunctional"

    # 执行并获取结果
    execution_result = run_command_and_get_output(target_command)

    # print(execution_result["输出"])

    _lines = []
    mp = {}
    for line in execution_result['输出'].split('\n'):
        if 'latency' in line:
            c_line = line.split(' ')[0]
            latency = line.split(' ')[1]
            op = c_line.split(':')[-1].strip(']')
            if op not in mp:
                mp[op] = []
            mp[op].append(latency)

    return mp


def compile():
    # 要执行的目标指令
    target_command = "python3 build_ci.py -c -u --disable_auto_execute"

    # 执行并获取结果
    _execution_result = run_command_and_get_output(target_command)


def change_acc_level(new_value):
    """
    :param file_path: 配置文件路径（如'tile_fwk_config.json'）
    :param key: 要修改的键（支持嵌套，如'log.level'）
    :param new_value: 新值
    :return: 成功状态及提示信息
    """
    file_path = "src/configs/tile_fwk_config.json"
    key = "global.simulation.ACCURACY_LEVEL"
    try:
        # 读取原始配置
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 处理嵌套键（如key为'log.level'时，拆分为['log', 'level']）
        keys = key.split('.')
        current = config
        for i, k in enumerate(keys):
            if i == len(keys) - 1:  # 最后一级键，修改值
                current[k] = new_value
            else:  # 中间级键，确保存在（否则会抛出KeyError）
                current = current[k]

        # 写回修改后的配置（缩进保持原格式，确保可读性）
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        return {"success": True, "message": f"配置修改成功！键 '{key}' 已更新为: {new_value}"}
    except FileNotFoundError:
        return {"success": False, "message": f"错误：文件 {file_path} 不存在"}
    except json.JSONDecodeError:
        return {"success": False, "message": f"错误：{file_path} 不是有效的JSON格式"}
    except KeyError as e:
        return {"success": False, "message": f"错误：键 '{e}' 不存在于配置中"}
    except Exception as e:
        return {"success": False, "message": f"未知错误：{str(e)}"}


def checkout_work_dir():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    target_dir = script_dir
    for _ in range(4):
        target_dir = os.path.dirname(target_dir)
    os.chdir(target_dir)


if __name__ == "__main__":
    checkout_work_dir()
    compile()
    change_acc_level(1)
    res1 = execute()
    change_acc_level(2)
    res2 = execute()

    for key in res1.keys():
        print(key)
        print(res1[key])
        print(res2[key])
        print("!")
