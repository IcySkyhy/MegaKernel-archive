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
"""STest Golden 处理函数注册管理."""

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union


@dataclasses.dataclass
class GoldenRegInfo:
    func: Optional[Callable]
    version: int = 0  # Golden 实现版本
    timeout: Optional[int] = None  # Golden 超时时间


class GoldenRegister:
    # 全局回调函数注册表
    _REG_MAP: Dict[str, GoldenRegInfo] = {}

    @classmethod
    def reg_golden_func(
        cls, case_names: Union[str, List[str]], version: int = 0, timeout: Optional[int] = None
    ) -> Callable:
        """
        注册回调函数, 支持两种函数原型:
            func(case_name: str, output: Path)
            func(case_name: str, output: Path, case_index: int)

        :param case_names: CaseName
        :param version: 实现版本, 由 Golden 脚本控制. 当框架感知 version 大于缓存内的 version, 会触发重新生成 Golden
        :param timeout: 超时时长(单位秒), 当框架感知 Golden 文件已超过指定时长, 会触发重新生成 Golden
        """

        def decorator(func: Callable) -> Callable:
            case_name_list = [case_names] if isinstance(case_names, str) else case_names
            for name in case_name_list:
                ori_func = cls._REG_MAP.get(name, None)
                if ori_func:
                    logging.debug("Case(%s) update func %s -> %s to %s", name, ori_func, func, hex(id(cls._REG_MAP)))
                else:
                    logging.debug("Case(%s) register func %s to %s", name, func, hex(id(cls._REG_MAP)))
                cls._REG_MAP[name] = GoldenRegInfo(func=func, version=version, timeout=timeout)
            return func

        return decorator

    @classmethod
    def get_golden_func(cls, case_name: str) -> Tuple[Optional[GoldenRegInfo], Optional[int]]:
        """根据名称获取回调函数

        支持以下用例名传入

        1. TEST/TEST_F 场景下:
            TestSuiteName.TestCaseName
        2. TEST_P 场景下:
            TestInstanceName/TestSuiteName.TestCaseName
            TestInstanceName/TestSuiteName.TestCaseName/
            TestInstanceName/TestSuiteName.TestCaseName/*
            TestInstanceName/TestSuiteName.TestCaseName*
            TestInstanceName/TestSuiteName.TestCaseName/数字标号

        对应注册用例名支持以下场景

        1. TEST/TEST_F 场景下:
            TestSuiteName.TestCaseName
        2. TEST_P 场景下:
            TestInstanceName/TestSuiteName.TestCaseName

        :param case_name: CaseName
        """
        # 用例名归一化
        #   TestSuiteName.TestCaseName
        #   TestInstanceName/TestSuiteName.TestCaseName
        #   TestInstanceName/TestSuiteName.TestCaseName/{int}
        cs = case_name.replace("*", "")
        cs = cs[:-1] if cs.endswith("/") else cs

        # 提取用例编号(可选)
        cs_idx = None
        cs_split = cs.split("/")
        if cs_split[-1].isdigit():
            cs_idx = int(cs_split[-1])
            cs_split = cs_split[:-1]

        # 用例名再归一
        #   TestSuiteName.TestCaseName
        #   TestInstanceName/TestSuiteName.TestCaseName
        cs = "/".join(cs_split)

        return cls._REG_MAP.get(cs, None), cs_idx

    @classmethod
    def get_golden_func_num(cls) -> int:
        return len(cls._REG_MAP)
