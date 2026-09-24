#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""STest 分布式 STest 并性执行."""

import argparse
import datetime
import logging
import os
import subprocess
from typing import Any, Dict, List, Tuple

from stest_accelerate import STestAccelerate


class DistributedSTestAccelerate(STestAccelerate):
    """分布式STest执行加速

    支持多卡并行执行 通过设备分组实现分布式测试.
    继承自STestAccelerate, 使用父类的device_list参数, 按照rank_size进行设备分组.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args, scene_mark="Distributed STest", cntr_name="Device Group")
        device_list = self._init_get_device_list(args)
        self.rank_size = args.rank_size
        self.device_groups = self._group_devices_by_rank_size(device_list, self.rank_size)

    @staticmethod
    def reg_args(parser: argparse.ArgumentParser) -> None:
        """注册分布式STest参数
        先调用父类(STestAccelerate)的参数注册，再添加分布式特有参数
        """
        STestAccelerate.reg_args(parser)
        parser.add_argument("--rank_size", type=int, required=True, help="Number of devices per test group")

    @staticmethod
    def main() -> bool:
        """分布式主处理流程"""
        parser = argparse.ArgumentParser(
            description="Distributed STest Execute Accelerate",
            epilog="Best Regards!",
        )

        DistributedSTestAccelerate.reg_args(parser=parser)
        args = parser.parse_args()

        ctrl = DistributedSTestAccelerate(args=args)
        ctrl.prepare()
        ctrl.process()
        return ctrl.post()

    @staticmethod
    def set_distributed_device_envs(p: Any) -> Dict[str, str]:
        """设置分布式设备环境变量

        多卡用例通过TILE_FWK_DEVICE_ID_LIST环境变量指定使用的设备组
        """
        custom_data = p.custom
        device_group = custom_data["device_group"]

        device_list_str = ",".join(str(device_id) for device_id in device_group)

        return {
            "TILE_FWK_DEVICE_ID_LIST": device_list_str,
        }

    @staticmethod
    def _group_devices_by_rank_size(devices: List[int], rank_size: int) -> List[List[int]]:
        """按照rank_size对设备进行顺序分组

        :param devices: 设备列表
        :param rank_size: 每组设备数量
        :return: 设备分组列表
        """
        if len(devices) < rank_size:
            raise ValueError(f"Available devices ({len(devices)}) are less than required rank_size ({rank_size})")

        device_groups = []
        sorted_devices = sorted(devices)

        for i in range(0, len(sorted_devices), rank_size):
            group = sorted_devices[i:i + rank_size]
            if len(group) == rank_size:
                device_groups.append(group)

        return device_groups

    def _prepare_get_params(self) -> List[STestAccelerate.ExecParam]:
        params = []
        for group_id, device_group in enumerate(self.device_groups):
            param = STestAccelerate.ExecParam(
                cntr_id=group_id,
                envs_func=DistributedSTestAccelerate.set_distributed_device_envs,
                custom={
                    "device_group": device_group,
                    "rank_size": self.rank_size,
                    "group_id": group_id,
                },
            )
            params.append(param)
        return params

    def _execute_case(
        self, ctx: STestAccelerate.CaseContext, param: STestAccelerate.ExecParam, case_name: str
    ) -> Tuple[subprocess.CompletedProcess, str, datetime.timedelta]:
        """多卡模式执行 - 重写父类方法"""
        rank_size = param.custom.get("rank_size")
        if rank_size is None:
            raise ValueError("Missing rank_size in custom config, run distribute case failed.")
        if rank_size <= 1:
            raise ValueError("Distribute case rank size need greater than 1, run distribute case failed.")
        device_group = param.custom.get("device_group", [param.cntr_id])
        return self._run_multi_device_case(ctx, device_group, rank_size, case_name)

    def _run_multi_device_case(
        self, ctx: STestAccelerate.CaseContext, device_group: List[int], rank_size: int, case_name: str
    ) -> Tuple[subprocess.CompletedProcess, str, datetime.timedelta]:
        """执行多卡分布式测试用例

        :param ctx: Case上下文
        :param device_group: 设备组列表
        :param rank_size: 设备组大小
        :return: 执行结果，命令行，时间
        """
        env_vars = os.environ.copy()
        env_vars.update(self.exe.envs)
        env_vars.update(ctx.exec_param.get_envs())
        command = [
            "mpirun",
            "-n",
            str(rank_size),
            str(self.exe.file),
            f"--gtest_filter={case_name}",
        ]
        device_info = f"DeviceGroup{device_group}"
        logging.info(f"Executing {case_name} on {device_info} with rank_size {rank_size}.")
        ts = datetime.datetime.now(tz=datetime.timezone.utc)
        completed_process = subprocess.run(
            command,
            env=env_vars,
            capture_output=True,
            text=True,
        )
        return completed_process, ' '.join(command), datetime.datetime.now(tz=datetime.timezone.utc) - ts


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    exit(0 if DistributedSTestAccelerate.main() else 1)
