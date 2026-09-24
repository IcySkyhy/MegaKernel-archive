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
"""可执行文件执行辅助模块

本模块提供可执行文件(主要是GTest测试程序)的执行, 测试用例列表获取(按耗时预估刷新顺序)功能,
主要用于测试框架的用例管理和执行加速.

主要功能:
- 可执行文件的封装与执行
- GTest测试用例列表的自动获取
- 测试用例耗时预估获取与用例列表重排(从可执行文件元数据或JSON缓存)
- 环境变量配置与管理
- 执行超时控制

主要类:
- Exec: 可执行文件封装类, 提供完整的执行和管理功能
- CaseDesc: 测试用例描述, 包含名称和预估耗时

使用示例:
    exec_obj = Exec(file=Path("test_executable"), envs={"ENV_VAR": "value"}, timeout=300)
    case_list, case_dict = exec_obj.get_case_name_info(case_duration_json=Path("duration.json"))
    ret, cmd, duration = exec_obj.run(params=["--gtest_filter=TestSuite.TestCase"])
"""

import dataclasses
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple


class Exec:
    """可执行文件封装类

    提供可执行文件的执行, 测试用例列表获取等功能.
    """

    @dataclasses.dataclass
    class CaseDesc:
        """测试用例描述

        包含用例名称和预估/实际执行耗时
        """

        name: Optional[str] = None
        duration: Optional[float] = None

        def __init__(self, name: str, duration: Optional[float] = None):
            self.name = name
            self.duration = duration

    def __init__(self, file: Path, envs: Optional[Dict[str, str]] = None, timeout: Optional[int] = None):
        """
        :param file: 可执行文件路径
        :type file: Path
        :param envs: 可执行文件执行时额外指定环境变量
        :type envs: Optional[Dict[str, str]]
        :param timeout: 执行超时时间(秒), 为空时从环境变量 PYPTO_TESTS_CASE_EXECUTE_TIMEOUT 中获取, 未指定时无超时限制
        :type timeout: Optional[int]
        """
        self.file: Path = Path(file).resolve()
        self.envs: Dict[str, str] = envs if envs is not None else {}
        self.timeout: Optional[int] = None
        env_timeout = os.environ.get("PYPTO_TESTS_CASE_EXECUTE_TIMEOUT", None)
        if env_timeout:
            self.timeout = int(env_timeout)
        if timeout and timeout > 0:
            self.timeout = timeout  # 参数设置优先级高于环境变量

    @property
    def brief(self) -> str:
        """获取可执行过程简要描述

        包含文件名和ASAN/UBSAN状态信息

        :return: 简要描述字符串
        """
        asan = "ON" if "ASAN_OPTIONS" in self.envs.keys() else "OFF"
        ubsan = "ON" if "UBSAN_OPTIONS" in self.envs.keys() else "OFF"
        return f"({self.file.name}) XSAN(ASAN:{asan} UBSAN:{ubsan})"

    def get_case_name_info(
        self, case_name_list: Optional[List[str]] = None, duration_json: Optional[Path] = None
    ) -> Tuple[int, List[CaseDesc], Dict[str, CaseDesc]]:
        """获取测试用例信息并排序

        根据指定的用例列表或可执行文件获取用例列表, 补充耗时预估信息, 并按耗时预估降序排列.

        :param case_name_list: 指定的用例名称列表, 为空或包含"*"时从可执行文件获取全部用例
        :type case_name_list: Optional[List[str]]
        :param case_duration_json: 用例耗时缓存JSON文件路径, 用于补充预估耗时
        :type case_duration_json: Optional[Path]
        :return: 排序后的用例描述列表和用例描述字典
        :rtype: Tuple[List[CaseDesc], Dict[str, CaseDesc]]
        """
        # 确定待执行用例名列表
        if case_name_list is None or len(case_name_list) == 0 or "*" in case_name_list:
            case_name_list = self._get_case_name_list_origin()
            logging.info("Determine TestCase from file, get %s cases", len(case_name_list))
        else:
            logging.info("Determine TestCase from args, get %s cases", len(case_name_list))
        desc_dict = {name: self.CaseDesc(name=name) for name in case_name_list}

        # 补充刷新预估耗时信息(先由用例内定义耗时, 后根据 json 刷新, 确保其更贴近真实耗时)
        self._mdf_case_desc_dict(case_desc_dict=desc_dict, path=duration_json)

        # 重排用例
        desc_list = desc_dict.values()
        desc_list = sorted(
            desc_list, key=lambda x: x.duration if x.duration is not None else float('-inf'), reverse=True
        )
        ordered_cnt = 0
        for desc in desc_list:
            if desc.duration:
                ordered_cnt += 1
            else:
                break  # 排序后, 有耗时预估的会排在前面
        normal_cnt = len(case_name_list) - ordered_cnt
        logging.info("Determine TestCase Order, OrderdCase(%s), NormalCase(%s)", ordered_cnt, normal_cnt)
        return ordered_cnt, desc_list, desc_dict

    def run(
        self,
        params: Optional[List[str]] = None,
        check: bool = False,
        capture_output: bool = True,
        envs: Optional[Dict[str, str]] = None,
    ) -> Tuple[subprocess.CompletedProcess, str, timedelta]:
        """执行可执行文件

        :param params: 额外配置的命令参数
        :type params: Optional[List[str]]
        :param check: 透传至 subprocess.run 的 check 参数
        :type check: bool
        :param capture_output: 透传至 subprocess.run 的 capture_output 参数
        :type capture_output: bool
        :param envs: 运行时额外需配置的环境变量
        :type envs: Optional[Dict[str, str]]
        :return: 返回值, 执行命令, 执行耗时
        :rtype: Tuple[CompletedProcess, str, timedelta]
        """
        cmd = self._get_run_cmd(params=params)
        # 环境变量优先级: 函数参数指定 > 类内环境变量(命令行参数指定) > 系统内已有的
        envs = envs if envs is not None else {}
        act_env = os.environ.copy()  # 系统环境变量
        act_env.update(self.envs)  # 额外指定环境变量
        act_env.update(envs)  # 函数调用时指定的环境变量
        cwd = str(self.file.parent)
        ts = datetime.now(tz=timezone.utc)
        ret = subprocess.run(
            shlex.split(cmd),
            env=act_env,
            cwd=cwd,
            timeout=self.timeout,
            capture_output=capture_output,
            check=check,
            text=True,
            encoding='utf-8',
        )
        return ret, cmd, datetime.now(tz=timezone.utc) - ts

    def _get_run_cmd(self, params: Optional[List[str]] = None) -> str:
        """构建执行命令字符串

        :param params: 命令行参数列表
        :return: 完整的执行命令字符串
        """
        cmd = f"./{self.file.name}"
        if params:
            cmd += " " + " ".join(params)
        return cmd

    def _get_case_name_list_origin(self) -> List[str]:
        """
        从可执行文件中获取原始测试用例列表:

        :return: 用例名列表
        :rtype: List[str]
        """
        case_name_list = []
        (
            ret,
            _,
            _,
        ) = self.run(params=["--gtest_list_tests"], check=True)  # GoogleTest 原生参数
        for line in ret.stdout.split('\n'):
            line = line.rstrip()
            if not line or line.startswith('#') or "GoogleTestVerification" in line:
                continue
            if line.endswith('.'):
                current_suite = line[:-1]
            elif line.startswith('  '):
                test_name = line.strip()
                full_name = f"{current_suite}.{test_name}"
                case_name_list.append(full_name)
        return case_name_list

    def _get_case_desc_list_origin(self) -> List[CaseDesc]:
        """从可执行文件中获取包含耗时的测试用例列表

        使用自定义参数 --gtest_list_tests_with_meta 获取用例名称及预估耗时

        :return: 用例描述列表, 包含用例名和预估耗时
        """
        case_desc_list = []
        (
            ret,
            _,
            _,
        ) = self.run(params=["--gtest_list_tests_with_meta"], check=True)
        pattern = re.compile(r'^([\w\.]+)\|(\d+\.?\d*)$', re.MULTILINE)
        matches = pattern.findall(ret.stdout)
        for test_name, cost_str in matches:
            case_desc_list.append(self.CaseDesc(name=test_name, duration=float(cost_str.strip())))
        return case_desc_list

    def _mdf_case_desc_dict(self, case_desc_dict: Dict[str, CaseDesc], path: Optional[Path] = None):
        """刷新用例描述字典中的耗时预估

        优先从可执行文件元数据获取, 再从JSON缓存文件刷新

        :param case_desc_dict: 用例描述字典, key为用例名, value为CaseDesc对象
        :param path: 用例耗时缓存JSON文件路径
        """
        # 根据用例内定义的耗时预估, 刷新用例耗时预估
        update_cnt = 0
        case_desc_list = self._get_case_desc_list_origin()
        for item in case_desc_list:
            desc = case_desc_dict.get(item.name, None)
            if not desc:
                continue
            desc.duration = item.duration
            update_cnt += 1
        logging.info("Determine TestCase Order, %s case's estimate update by local define", update_cnt)

        # 根据 json 缓存文件, 刷新用例耗时预估
        case_duration_dict = {}
        if path is not None and path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    case_duration_dict = json.load(f)
            except json.JSONDecodeError:
                case_duration_dict = {}

        update_cnt = 0
        for case_name, duration in case_duration_dict.items():
            desc = case_desc_dict.get(case_name, None)
            if not desc:
                continue
            desc.duration = float(duration)
            update_cnt += 1
        logging.info("Determine TestCase Order, %s case's estimate update by cache file", update_cnt)
