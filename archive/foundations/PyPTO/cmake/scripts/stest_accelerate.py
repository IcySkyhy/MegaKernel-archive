#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""STest 用例并行执行."""

import argparse
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from accelerate.tests_accelerate import TestsAccelerate


class STestAccelerate(TestsAccelerate):
    """STest 执行加速

    通过多进程并行执行, 以提升 STest 执行效率.
    """

    def __init__(self, args: argparse.Namespace, scene_mark: str = "STest", cntr_name: str = "Device"):
        """
        :param args: 命令行参数
        :param scene_mark: 场景标识
        :param cntr_name: 容器名称
        """
        # 在调用父类初始化之前，从二进制文件获取 meta 信息并重排序用例列表
        # 二进制文件路径通过 -t/--target 参数传入，存储在 args.target[0] 中
        # 但需要检查 args.target 是否存在，因为它是 required=True 的参数
        binary_path = None
        if hasattr(args, 'target') and args.target and len(args.target) > 0:
            binary_path = args.target[0]
        elif hasattr(args, 'exe') and hasattr(args.exe, 'file'):
            # 如果 args.target 不存在，尝试从其他地方获取
            binary_path = args.exe.file

        if args.cases and binary_path:
            # 尝试从二进制文件获取 meta 信息并重排序
            reordered_cases = self._reorder_cases_with_binary_meta(args.cases, binary_path)
            # 修改 args.cases，这样父类初始化时会使用重排序后的用例列表
            args.cases = reordered_cases
        elif args.cases and not binary_path:
            logging.warning("Binary path not found, skipping meta-based reordering")

        # 调用父类初始化
        super().__init__(args, scene_mark=scene_mark, cntr_name=cntr_name)

        self.device_list: List[int] = self._init_get_device_list(args=args)

    @staticmethod
    def reg_args(parser: argparse.ArgumentParser) -> None:
        """注册STest加速器参数

        先调用父类(TestsAccelerate)的参数注册, 再添加STest特有参数
        """
        TestsAccelerate.reg_args(parser)
        parser.add_argument(
            "-d",
            "--device",
            nargs="?",
            type=int,
            action="append",
            help="Specific parallel accelerate device, "
            "If this parameter is not specified, 0 device will be used by default.",
        )

    @staticmethod
    def main() -> bool:
        """主处理流程"""
        # 参数注册
        parser = argparse.ArgumentParser(description="STest Execute Accelerate", epilog="Best Regards!")
        STestAccelerate.reg_args(parser=parser)
        # 流程处理
        args = parser.parse_args()
        ctrl = STestAccelerate(args=args)
        ctrl.prepare()
        ctrl.process()
        return ctrl.post()

    @staticmethod
    def get_case_exec_update_envs(p: Any) -> Optional[Dict[str, str]]:
        self = p
        return {"TILE_FWK_DEVICE_ID": f"{self.cntr_id}"}

    @staticmethod
    def _init_get_device_list(args) -> List[int]:
        device_list = [0]
        if args.device is not None:
            device_list = [int(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        return device_list

    @staticmethod
    def _get_test_costs(binary: str) -> Dict[str, float]:
        """
        获取所有带耗时信息的测试用例(通过自定义参数--gtest_list_tests_with_meta)
        返回格式: { "TestCaseName.TestName": cost_seconds, ... }
        """
        cost_map = {}
        if not binary or not os.path.exists(binary):
            logging.warning("Binary file not found: %s", binary)
            return cost_map

        try:
            result = subprocess.run(
                [binary, '--gtest_list_tests_with_meta'], capture_output=True, text=True, encoding='utf-8'
            )
            if result.returncode != 0:
                logging.warning("Failed to get test costs from binary %s: %s", binary, result.stderr)
                return cost_map

            # 仅解析stdout(格式:TestCaseName.TestName|cost_seconds)
            pattern = re.compile(r'^([\w.]+)\|(\d+\.?\d*)$', re.MULTILINE)
            matches = pattern.findall(result.stdout)
            for test_name, cost_str in matches:
                try:
                    cost_map[test_name.strip()] = float(cost_str.strip())
                except ValueError:
                    continue
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.warning("Failed to run binary %s to get meta info: %s", binary, e)
        return cost_map

    @staticmethod
    def _reorder_cases_with_binary_meta(cases: List[str], binary: str) -> List[str]:
        """
        基于 binary meta 耗时对 stest 用例进行重排：
          - 有耗时信息的用例排前面，按耗时降序
          - 无耗时信息的用例排后面，保持原有顺序
        """
        if not cases or not binary:
            return cases

        cost_map = STestAccelerate._get_test_costs(binary)
        if not cost_map:
            # 未获取到耗时信息，保持原序
            logging.debug("No cost meta found for %s, keep original cases order", binary)
            return cases

        cost_cases: List[str] = []
        no_cost_cases: List[str] = []
        for cs in cases:
            if cs in cost_map:
                cost_cases.append(cs)
            else:
                no_cost_cases.append(cs)

        # 有耗时信息的用例按耗时降序重排
        cost_cases_sorted = sorted(cost_cases, key=lambda x: cost_map[x], reverse=True)

        logging.info(
            "STest(meta): Found %d tests with cost info, %d tests without.", len(cost_cases_sorted), len(no_cost_cases)
        )
        if cost_cases_sorted:
            logging.info("STest(meta): First few cost-aware tests(desc): %s", cost_cases_sorted[:5])

        return cost_cases_sorted + no_cost_cases

    def _prepare_get_params(self) -> List[TestsAccelerate.ExecParam]:
        params = []
        for _id in self.device_list:
            p = TestsAccelerate.ExecParam(cntr_id=_id, envs_func=STestAccelerate.get_case_exec_update_envs)
            params.append(p)
        return params


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    exit(0 if STestAccelerate.main() else 1)
