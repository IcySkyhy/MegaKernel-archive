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
"""Tests 执行加速

本模块提供测试用例并行执行加速功能, 支持多进程并发运行测试用例, 主要功能包括:

- 多容器/进程并行执行测试用例, 提高测试效率
- 智能用例排序, 基于历史耗时预估进行负载均衡
- 用例耗时缓存机制, 优化重复执行场景
- 实时执行状态监控与异常处理
- 详细的执行报告与统计信息(包括容器执行摘要, 用例耗时统计, 异常信息等)
- CPU亲和性配置, 优化多CPU环境下的性能

主要类:
- TestsAccelerate: 测试加速的主类, 提供完整的并行执行框架
- CaseDesc: 测试用例描述, 包含名称和预估耗时
- ExecParam: 执行参数配置
- ExecResult: 执行结果统计与报告生成
- CntrContext: 容器/进程执行上下文
- CaseContext: 用例执行上下文

使用示例:
1. 继承 TestsAccelerate 类并实现 _prepare_get_params 方法
2. 调用 prepare() 进行准备工作
3. 调用 process() 执行测试
4. 调用 post() 获取执行结果
"""

from abc import ABC
import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
import json
import logging
import multiprocessing
from multiprocessing import Event, JoinableQueue, Process, Value, cpu_count
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.args_action import ArgsEnvDictAction
from utils.executable import Exec
from utils.table import Table


class ArgsCaseListAction(argparse.Action):
    """解析命令行参数传入的 cases 字段(适配自定义元信息参数)"""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: List[str],
        option_string: Optional[str] = None,
    ) -> None:
        # 解析每个字符串, 按冒号分隔并展平
        case_list = []
        for value in values:
            cases = [cs.strip() for cs in value.split(':') if cs.strip()]  # 分割每个字符串, 并过滤空字符串
            case_list.extend(cases)
        # 将结果设置到命名空间
        setattr(namespace, self.dest, case_list)


class TestsAccelerate(ABC):
    """Tests 加速"""

    @dataclasses.dataclass
    class ExecParam:
        """执行参数"""

        cntr_id: Optional[int] = None
        envs_func: Optional[Callable] = None
        custom: Optional[Any] = None

        def __init__(self, cntr_id: int, envs_func: Optional[Callable] = None, custom: Optional[Any] = None):
            self.cntr_id = cntr_id
            self.envs_func = envs_func
            self.custom = custom

        def get_envs(self) -> Optional[Dict[str, str]]:
            """获取额外的环境变量配置"""
            if self.envs_func:
                return self.envs_func(self)
            return None

    @dataclasses.dataclass
    class ExecResult:
        """执行结果"""

        cntr_name: str = "Cntr"
        act_duration: Optional[timedelta] = None  # 实际总耗时
        ori_duration: Optional[timedelta] = None  # 原始总耗时(预估)
        cntr_max_duration: Optional[timedelta] = None  # 各 Cntr 中最长的耗时
        cntr_min_duration: Optional[timedelta] = None  # 各 Cntr 中最短的耗时
        cntr_execution_details: JoinableQueue = JoinableQueue()
        cntr_duration_dict: Dict[int, timedelta] = dataclasses.field(default_factory=dict)
        case_execution_details: JoinableQueue = JoinableQueue()
        case_exception_details: JoinableQueue = JoinableQueue()
        case_terminate_details: JoinableQueue = JoinableQueue()

        @property
        def revenue_desc(self) -> str:
            diff = self.ori_duration - self.act_duration
            rate = float(diff / self.act_duration) * 100
            desc = f"Revenue(Act/Ori, {self.act_duration.total_seconds():.2f}/"
            desc += f"{self.ori_duration.total_seconds():.2f}) {rate:.2f}%"
            return desc

        @property
        def cntr_latency_desc(self) -> str:
            diff = self.cntr_max_duration - self.cntr_min_duration
            rate = float(diff / self.cntr_min_duration) * 100
            desc = f"Latency(Max/Min/Diff, {self.cntr_max_duration.total_seconds():.2f}/"
            desc += f"{self.cntr_min_duration.total_seconds():.2f}/{diff.total_seconds():.2f}) {rate:.2f}%"
            return desc

        @staticmethod
        def save_case_duration_to_json(
            sorted_datas: List[List[Any]],
            dump_item_num: int = 100,
            dump_min_duration: float = 5,
            path: Optional[Path] = None,
        ):
            # 路径处理
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            # 数据处理
            item_num = 0
            case_name_idx = 1
            duration_idx = 2
            duration_dict = {}
            for item in sorted_datas:
                case_name = item[case_name_idx]
                duration = float(item[duration_idx])
                duration_dict[case_name] = duration
                item_num += 1
                if item_num >= dump_item_num:
                    break
                if duration <= dump_min_duration:
                    break
            # 数据落盘
            with path.open("w", encoding="utf-8") as f:
                json.dump(duration_dict, f, indent=4)

        def get_cntr_exec_info(self) -> Tuple[str, str]:
            """获取 Container 执行信息统计.

            :returns:
                Tuple[str, str]:
                    - Container 执行信息统计表(str)
                    - Container 并行执行收益描述(str)
            """
            heads = [self.cntr_name, "Total", "Success", "Failed", "Duration"]
            datas = []
            self.ori_duration = timedelta()
            while not self.cntr_execution_details.empty():
                _brief = self.cntr_execution_details.get()
                devs_id = int(_brief[0])
                case_total = int(_brief[1])
                case_pass = int(_brief[2])
                case_fail = int(_brief[3])
                devs_duration = _brief[-1]
                # 耗时统计
                if self.cntr_max_duration is None:
                    self.cntr_max_duration = devs_duration
                self.cntr_max_duration = max(self.cntr_max_duration, devs_duration)
                if self.cntr_min_duration is None:
                    self.cntr_min_duration = devs_duration
                self.cntr_min_duration = min(self.cntr_min_duration, devs_duration)
                # 结果保存
                self.cntr_duration_dict[devs_id] = devs_duration
                self.ori_duration += devs_duration
                datas.append([devs_id, case_total, case_pass, case_fail, f"{devs_duration.total_seconds():.2f}"])
                self.cntr_execution_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                brief = Table.table(datas=datas, headers=heads)
            # 并行执行收益计算
            desc = f"Duration {self.act_duration.total_seconds():.2f} secs, {self.revenue_desc}"
            return f"\n\n{self.cntr_name} Execution Brief:{brief}", desc

        def get_case_exec_terminate_info(self) -> Tuple[str, int]:
            """获取 Case 执行终止信息.

            :returns:
                Tuple[str, int]:
                    - Case 终止执行情况信息
                    - Case 终止执行数量
            """
            heads = ["Idx", self.cntr_name, "CaseName", "Duration"]
            datas = []
            while not self.case_terminate_details.empty():
                _brief = self.case_terminate_details.get()
                cntr_id = int(_brief[0])
                case_name = str(_brief[1])
                case_duration = _brief[2]
                datas.append([cntr_id, case_name, f"{case_duration.total_seconds():.2f}"])
                self.case_terminate_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                datas = [[f"{idx}/{len(datas)}"] + ele for idx, ele in enumerate(datas, start=1)]
                brief = Table.table(datas=datas, headers=heads)
            return f"\n\nCase Terminate Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_exception_info(self) -> Tuple[str, int]:
            """获取 Case 执行异常信息.

            :returns:
                Tuple[str, int]:
                    - Case 异常执行情况信息
                    - Case 异常执行数量
            """
            datas = []
            brief = ""
            while not self.case_exception_details.empty():
                chunk = self.case_exception_details.get()
                if len(chunk) != 0:
                    brief += chunk
                else:
                    datas.append(str(brief))
                    brief = ""
                self.case_exception_details.task_done()
            brief = "\nNone" if len(datas) == 0 else ""
            for idx, data in enumerate(datas, start=1):
                brief += f"\nIdx:{idx}/{len(datas)}\n{data}"
            return f"\n\nCase Exception Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_duration_info(
            self,
            case_dict: Dict[str, Exec.CaseDesc],
            min_print_cnt: Optional[int] = None,
            dump_json_path: Optional[Path] = None,
            dump_item_num: int = 100,
            dump_min_duration: float = 5,
        ) -> str:
            """获取 Case 执行耗时统计信息.

            :return: Case 执行耗时统计信息.
            """
            heads = [self.cntr_name, "CaseName", "Duration", "Estimate", f"Ratio({self.cntr_name})", "Ratio(Total)"]
            datas = []
            while not self.case_execution_details.empty():
                _brief = self.case_execution_details.get()
                cntr_id = _brief[0]
                case_name = str(_brief[1])
                case_duration = _brief[2]
                case_desc = case_dict.get(case_name, None)
                case_estimate = ""
                if case_desc and case_desc.duration:
                    case_estimate = timedelta(seconds=case_desc.duration).total_seconds()
                cntr_duration = self.cntr_duration_dict[cntr_id]
                ratio_cntr = float(case_duration / cntr_duration) * 100
                ratio_process = float(case_duration / self.act_duration) * 100
                datas.append(
                    [
                        cntr_id,
                        case_name,
                        case_duration.total_seconds(),
                        case_estimate,
                        f"{case_duration.total_seconds():.2f}/{cntr_duration.total_seconds():.2f} {ratio_cntr:.2f}%",
                        f"{case_duration.total_seconds():.2f}/{self.act_duration.total_seconds():.2f} "
                        f"{ratio_process:.2f}%",
                    ]
                )
                self.case_execution_details.task_done()
            brief = "\nNone"
            add_desc = ""
            if len(datas) != 0:
                # 把 data 按耗时降序重排, 重排后转换格式
                duration_idx = 2  # 2 is idx of duration
                datas = sorted(datas, key=lambda x: x[duration_idx], reverse=True)
                for item in datas:
                    item[duration_idx] = f"{item[duration_idx]:.2f}"
                # 结果落盘, 常用于本地重复执行时加速
                self.save_case_duration_to_json(
                    sorted_datas=datas,
                    path=dump_json_path,
                    dump_item_num=dump_item_num,
                    dump_min_duration=dump_min_duration,
                )
                # 缩略功能
                if min_print_cnt:
                    print_cnt = min_print_cnt + 50  # 除已配置预估耗时的用例外, 再额外打印 50 个用例
                    ori_len = len(datas)
                    datas = datas[:print_cnt]
                    cur_len = len(datas)
                    if ori_len > cur_len:
                        hidden_cnt = ori_len - cur_len
                        hidden_first_data = datas[-1]  # 取切片后的最后一个用例
                        add_desc = f"\n({hidden_cnt} durations <= {hidden_first_data[2]}s hidden.)"  # 2 Duration
                # 结果汇总
                brief = Table.table(datas=datas, headers=heads, auto_sort=False)
            return f"\n\nCase Duration Brief:{brief}" + add_desc

    @dataclasses.dataclass
    class CntrContext:
        """Cntr处理上下文"""

        cntr_id: int = 0
        exec_param: Optional[Any] = None
        success: int = 0
        failed: int = 0
        ts: Optional[datetime] = None
        exit_code: int = 0

        def __init__(self, cntr_id: int, exec_param):
            self.cntr_id = cntr_id
            self.exec_param = exec_param
            self.ts = datetime.now(tz=timezone.utc)

        @property
        def total(self) -> int:
            return self.success + self.failed

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.total, self.success, self.failed, (datetime.now(tz=timezone.utc) - self.ts)]

    @dataclasses.dataclass
    class CaseContext:
        """Case处理上下文"""

        cntr_id: int = 0
        exec_param: Optional[Any] = None
        ts: Optional[datetime] = None
        case_name: str = ""

        def __init__(self, cntr_id: int, exec_param, case_name):
            self.cntr_id = cntr_id
            self.exec_param = exec_param
            self.case_name = case_name
            self.ts = datetime.now(tz=timezone.utc)

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.case_name, (datetime.now(tz=timezone.utc) - self.ts)]

    @dataclasses.dataclass
    class MoveContext:
        """Move进程处理上下文"""

        ele_count: int
        src_queue: JoinableQueue
        dst_queue: JoinableQueue

        def __init__(self, src: JoinableQueue, dst: JoinableQueue):
            self.ele_count = 0
            self.src_queue = src
            self.dst_queue = dst

        def move(self, timeout: int = 1) -> bool:
            try:
                ele = self.src_queue.get(timeout=timeout)
                self.src_queue.task_done()
                if ele is None:
                    return False
                if isinstance(ele, str):
                    if len(ele) == 0:
                        self.ele_count += 1
                else:
                    self.ele_count += 1
                self.dst_queue.put(ele)
            except (queue.Empty, KeyboardInterrupt):
                pass
            return True

    def __init__(self, args, scene_mark: str, cntr_name: str):
        """
        :param args: 命令行参数
        :param cntr_name: 容器名称, 用于回显内容
        """
        # 场景标识
        self.mark: str = scene_mark

        # 用例执行参数, 执行行为控制参数
        self.exe: Exec = Exec(file=args.target[0], envs=args.envs, timeout=args.timeout_case)
        self.exe_params: List[TestsAccelerate.ExecParam] = []
        self.exe_result: TestsAccelerate.ExecResult = TestsAccelerate.ExecResult(cntr_name=cntr_name)
        self.exe_timeout: Optional[int] = args.timeout
        self.exe_halt_on_error: bool = args.halt_on_error  # 失败时终止后续 Case 执行

        # 用例管理
        self.case_duration_json: Path = self._init_get_case_duration_json(args=args)
        self.case_duration_max_num: int = self._init_get_case_duration_max_num(args=args)
        self.case_duration_min_sec: float = self._init_get_case_duration_min_sec(args=args)
        self.case_list: List[Exec.CaseDesc] = []
        self.case_dict: Dict[str, Exec.CaseDesc] = {}
        self.case_ordered_cnt: int = 0
        # GCOV 并发写隔离: 在获取用例列表前设置 GCOV_PREFIX, 避免主进程 --gtest_list_tests 产生并发写
        self._init_gcov_prefix()
        self.case_ordered_cnt, self.case_list, self.case_dict = self.exe.get_case_name_info(
            case_name_list=args.cases, duration_json=self.case_duration_json
        )

        self.case_queue: JoinableQueue = JoinableQueue()
        self.case_execution_queue: JoinableQueue = JoinableQueue()  # Case 正常执行结束时, 收集相关信息
        self.case_exception_queue: JoinableQueue = JoinableQueue()  # Case 执行失败时, 用于收集错误信息
        self.case_terminate_queue: JoinableQueue = JoinableQueue()  # Case 被终止执行时, 收集相关信息
        self.case_exec_count = Value('i', 0)  # DFX, 统计 Case 完成进度

        # 容器管理
        self.cntr_name: str = cntr_name
        self.cntr_execution_queue: JoinableQueue = JoinableQueue()  # Container 执行结果统计上报
        self.cntr_terminate_event = Event()  # 用于通知其他 Container 进程结束运行
        self.cntr_exit_count = Value('i', 0)  # DFX, 统计 Container 退出进度

        # CPU 亲和性管理
        self.cpu_rank_size: Optional[int] = self._init_get_cpu_rank_size(args=args)
        self.cpu_affinity_policy: Optional[int] = None

    @property
    def brief(self) -> List[Any]:
        ver = sys.version_info
        lst = [
            ["Python3", f"{sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"],
            ["Timeout", self.exe_timeout],
            ["HaltOnError", self.exe_halt_on_error],
            [f"{self.cntr_name}Num", self.cntr_num],
            [f"{self.cntr_name}List", [p.cntr_id for p in self.exe_params]],
            ["CaseNum", self.case_num],
            ["CaseTimeout", self.exe.timeout],
            ["CaseDurationFile", self.case_duration_json],
            ["CaseDurationMaxNum", self.case_duration_max_num],
            ["CaseDurationMinSecs", self.case_duration_min_sec],
            ["Executable", self.exe.file],
        ]
        if self.cpu_rank_size:
            lst.append(["CpuRankSize", self.cpu_rank_size])
            lst.append(["CpuAffinityPolicy", f"{self.cpu_affinity_policy_str}({self.cpu_affinity_policy})"])
        return lst

    @property
    def cntr_num(self) -> int:
        return len(self.exe_params)

    @property
    def case_num(self) -> int:
        return len(self.case_list)

    @property
    def cpu_affinity_policy_str(self) -> str:
        if not self.cpu_affinity_policy:
            return "Disable"
        elif self.cpu_affinity_policy == 1:
            return "Even Allocation"  # 均匀分配
        elif self.cpu_affinity_policy == 2:
            return "Cyclic Reuse Allocation"  # 循环再利用分配
        else:
            return "Unknown"

    @staticmethod
    def reg_args(parser: argparse.ArgumentParser):
        """注册命令行参数

        注意事项:
            1. 本函数应与 get_container_manager 函数协同使用;
            2. 本函数注册了 'cases' 字段, 但 get_container_manager 内不会解析处理, 该字段应由使用者解析处理;

        :param parser: ArgumentParser 外部创建
        """
        # 执行所需参数
        parser.add_argument(
            "-t", "--target", nargs=1, type=str, required=True, help="Specific target executable file path."
        )
        parser.add_argument(
            "-e",
            "--env",
            nargs="+",
            action=ArgsEnvDictAction,
            default={},
            dest="envs",
            help="Specify additional environment variables to set when executing the target.",
        )
        parser.add_argument("--timeout", nargs="?", type=int, default=None, help="Timeout for executing all cases.")
        parser.add_argument(
            "--timeout_case", nargs="?", type=int, default=None, help="Timeout for executing single case."
        )
        parser.add_argument(
            "--halt_on_error",
            action="store_true",
            default=False,
            help="If any case failed, subsequent cases are not executed.",
        )
        # 用例参数
        parser.add_argument(
            "-c",
            "--cases",
            nargs='*',
            action=ArgsCaseListAction,
            default=[],
            required=False,
            help="Cases, multiple cases are separated by ':'",
        )
        # 其他
        parser.add_argument(
            "--cpu_rank_size",
            nargs="?",
            type=int,
            default=None,
            help="Specify the rank size for CPU affinity grouping.",
        )
        # 用例耗时缓存相关参数
        parser.add_argument(
            "--dump_case_duration_json",
            nargs="?",
            type=Path,
            default=None,
            help="Specify the path to the case duration json cache file.",
        )
        parser.add_argument(
            "--dump_case_duration_max_num",
            nargs="?",
            type=int,
            default=None,
            help="Maximum number of cases to dump to duration json cache.",
        )
        parser.add_argument(
            "--dump_case_duration_min_secends",
            nargs="?",
            type=int,
            default=None,
            help="Minimum duration (in seconds) for cases to dump to duration json cache.",
        )

    @staticmethod
    def _init_get_cpu_rank_size(args) -> Optional[int]:
        cpu_rank_size = None
        if args.cpu_rank_size:
            cpu_rank_size = args.cpu_rank_size
        else:
            cpu_rank_size_str = os.environ.get("PYPTO_TESTS_CASE_EXECUTE_CPU_RANK_SIZE", None)
            if cpu_rank_size_str:
                cpu_rank_size = int(cpu_rank_size_str)
        if cpu_rank_size and cpu_rank_size > 0:
            return cpu_rank_size
        return None

    @staticmethod
    def _init_get_case_duration_json(args) -> Path:
        """初始化 case_duration_json

        命令行参数优先, 然后是环境变量, 最后是默认值
        """
        if args.dump_case_duration_json:
            return args.dump_case_duration_json.resolve()

        # 从环境变量获取
        env_json_path = os.environ.get("PYPTO_TESTS_DUMP_CASE_DURATION_JSON", None)
        if env_json_path:
            return Path(env_json_path).resolve()

        # 默认值
        tagert = Path(args.target[0])
        return tagert.parent / f"{tagert.stem}_duration.json"

    @staticmethod
    def _init_get_case_duration_max_num(args) -> int:
        """初始化 case_duration_max_num

        命令行参数优先, 然后是环境变量, 最后是默认值
        """
        if args.dump_case_duration_max_num is not None:
            return args.dump_case_duration_max_num

        # 从环境变量获取
        env_max_num = os.environ.get("PYPTO_TESTS_DUMP_CASE_DURATION_MAX_NUM", None)
        if env_max_num:
            return int(env_max_num)

        # 默认值
        return 500

    @staticmethod
    def _init_get_case_duration_min_sec(args) -> float:
        """初始化 case_duration_min_sec

        命令行参数优先, 然后是环境变量, 最后是默认值
        """
        if args.dump_case_duration_min_secends is not None:
            return float(args.dump_case_duration_min_secends)

        # 从环境变量获取
        env_min_sec = os.environ.get("PYPTO_TESTS_DUMP_CASE_DURATION_MIN_SECONDS", None)
        if env_min_sec:
            return float(env_min_sec)

        # 默认值
        return 5.0

    def _init_gcov_prefix(self):
        """初始化 GCOV_PREFIX 隔离配置

        设置 GCOV_PREFIX_STRIP, 由 Cntr 进程通过 fork 继承.
        主进程不设置 GCOV_PREFIX (.gcda 写入 build 目录, 无并发写问题), 减少不必要的 .gcda 副本.
        """
        gcov_data_dir = self.exe.envs.get("PYPTO_GCOV_DATA_DIR")
        if not gcov_data_dir:
            return
        gcov_root = Path(gcov_data_dir)
        if gcov_root.exists():
            shutil.rmtree(gcov_root)
        gcov_root.mkdir(parents=True, exist_ok=True)
        build_dir = str(gcov_root.parent.resolve())
        strip_count = len(Path(build_dir).parts) - 1
        os.environ["GCOV_PREFIX_STRIP"] = str(strip_count)

    def _cntr_set_gcov_prefix(self, cntr_id: int):
        """为 Cntr 进程设置独立 GCOV_PREFIX

        Case 子进程继承 Cntr 的 os.environ, GTest 退出时 .gcda 写入隔离目录, 避免并发写冲突.
        GCOV_PREFIX_STRIP 由主进程 _init_gcov_prefix 设置, 通过 fork 继承, 无需重复设置.
        """
        gcov_data_dir = self.exe.envs.get("PYPTO_GCOV_DATA_DIR")
        if not gcov_data_dir:
            return
        cntr_dir = Path(gcov_data_dir) / f"cntr_{cntr_id}"
        cntr_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GCOV_PREFIX"] = str(cntr_dir)

    @staticmethod
    def _move(src: JoinableQueue, dst: JoinableQueue):
        TestsAccelerate._set_process_desc()
        ctx = TestsAccelerate.MoveContext(src=src, dst=dst)
        while True:
            if not ctx.move():
                break
        logging.info("%s Exist, Move %s elements.", TestsAccelerate._get_process_desc(), ctx.ele_count)

    @staticmethod
    def _get_process_desc() -> str:
        cur_process = multiprocessing.current_process()
        return f"{cur_process.name}"

    @staticmethod
    def _set_process_desc():
        try:
            import setproctitle

            setproctitle.setproctitle(TestsAccelerate._get_process_desc())
        except ModuleNotFoundError:
            pass

    def prepare(self):
        """执行准备"""
        self.exe_params = self._prepare_get_params()
        if self.cntr_num == 0:
            raise ValueError("ExecParams is empty, won't run any task.")
        if self.cntr_num > self.case_num:
            logging.info(
                "CaseNum(%s) less than len(ExecParams)=%s, will only start the first %s %s.",
                self.case_num,
                self.cntr_num,
                self.case_num,
                self.cntr_name,
            )
            self.exe_params = self.exe_params[:self.case_num]
        # CPU 亲和性设置
        self._prepare_determine_cpu_affinity_policy()

    def process(self):
        """执行任务"""
        logging.info("\n\n%s Accelerate Args:%s", self.mark, Table.table(datas=self.brief))
        # 执行流程
        ts = datetime.now(tz=timezone.utc)
        self._main()
        self.exe_result.act_duration = datetime.now(tz=timezone.utc) - ts

    def post(self) -> bool:
        """后处理, 获得执行结果汇总"""
        # Cntr 执行信息收集汇总
        cntr_exec_brief, cntr_revenue_desc = self.exe_result.get_cntr_exec_info()

        # Case 执行信息收集汇总
        case_exec_brief, case_exec_result = self._post_case_exec_info()

        out = f"{self.mark}, HaltOnError({self.exe_halt_on_error}), {cntr_revenue_desc}"
        out += cntr_exec_brief
        out += case_exec_brief

        if case_exec_result:
            logging.info(out)
            logging.info(
                "Use %s %s | Exec %s case | %s | %s",
                self.cntr_num,
                self.cntr_name,
                self.case_num,
                self.exe_result.revenue_desc,
                self.exe_result.cntr_latency_desc,
            )
        else:
            logging.error(out)
        return case_exec_result

    def _prepare_determine_cpu_affinity_policy(self):
        """初始化 CPU 亲和性策略

        策略确定需要依赖的 CntrNum 等参数无法在类构造阶段确定, 故本流程延迟到 prepare 阶段处理
        """
        self.cpu_affinity_policy = None
        if self.cpu_rank_size and self.cpu_rank_size > 0:
            if self.cntr_num * self.cpu_rank_size <= cpu_count():
                self.cpu_affinity_policy = 1  # 策略1: 均匀分配(每 CPU 组对应 1 个 cntr)
            else:
                self.cpu_affinity_policy = 2  # 策略2: 循环复用核心组(期望 CPU 数超出 CPU 总数场景)
        logging.info(
            "Determine CpuAffinity, Policy=%s(%s), CntrNum=%s, CpuNum=%s, CpuRankSize=%s",
            self.cpu_affinity_policy_str,
            self.cpu_affinity_policy,
            self.cntr_num,
            cpu_count(),
            self.cpu_rank_size,
        )

    def _prepare_get_params(self) -> List[ExecParam]:
        return []

    def _post_case_exec_info(self) -> Tuple[str, bool]:
        """获取 Case 执行信息.

        :returns:
            Tuple[str, bool]:
                - Case 执行情况信息
                - Case 执行成功与否判定结果
        """
        terminate_brief, terminate_count = self.exe_result.get_case_exec_terminate_info()
        exception_brief, exception_count = self.exe_result.get_case_exec_exception_info()
        duration_brief = self.exe_result.get_case_exec_duration_info(
            case_dict=self.case_dict,
            min_print_cnt=self.case_ordered_cnt,
            dump_json_path=self.case_duration_json,
            dump_item_num=self.case_duration_max_num,
            dump_min_duration=self.case_duration_min_sec,
        )

        # Case 执行总体情况汇总
        remaining_count = 0
        while not self.case_queue.empty():
            cs = self.case_queue.get()
            if cs is not None:
                remaining_count += 1
            self.case_queue.task_done()
        success_count = self.case_num - remaining_count - terminate_count - exception_count
        execution_heads = ["Total", "Success", "Failed", "Terminate", "Remaining"]
        execution_datas = [[self.case_num, success_count, exception_count, terminate_count, remaining_count]]
        execution_brief = Table.table(datas=execution_datas, headers=execution_heads)
        execution_brief = f"\n\nCase Execution Brief:{execution_brief}"

        rst = (terminate_count + exception_count + remaining_count) == 0
        out = execution_brief + duration_brief + terminate_brief + exception_brief
        return out, rst

    def _main(self):
        """用例执行, 管理执行状态(主进程)

        :return: 执行成功与否
        """
        # 创建并启动子进程, 进行任务处理
        cntr_step = 1
        cntr_process_group = []
        try:
            # 任务准备
            self._push_all_case_sync()
            # 启动上报监控进程
            self._start_move_process_grp()
            # 创建并启动任务子进程, 进行任务处理
            cntr_process_group = self._start_cntr_process_grp()
            # 等待任务处理结束
            self._join_cntr_process_grp(cntr_process_grp=cntr_process_group, step=cntr_step)
        except KeyboardInterrupt:
            logging.info("MainProcess Recv download terminate event.")
        finally:
            self._stop_cntr_process_grp(cntr_process_grp=cntr_process_group, timeout=cntr_step)
            self._stop_move_process_grp()

    def _push_all_case_sync(self):
        """以同步方式将待执行用例插入待执行队列, 按 Container 数量插入终止信号"""
        for cs in self.case_list:
            self.case_queue.put(cs.name)
        for _ in range(self.cntr_num):
            self.case_queue.put(None)

    def _start_move_process_grp(self) -> List[Process]:
        """启动 Move 进程组

        :return: Move 进程列表
        """
        move_grp = []
        desc_list = self._get_move_process_grp_desc_list()
        for name, src_queue, dst_queue in desc_list:
            process = Process(
                name=f"MoveProcess({name})",
                target=self._move,
                args=(
                    src_queue,
                    dst_queue,
                ),
            )
            process.start()
            move_grp.append(process)
        return move_grp

    def _stop_move_process_grp(self):
        """停止 Move 进程组"""
        desc_list = self._get_move_process_grp_desc_list()
        for _, src_queue, _ in desc_list:
            src_queue.put(None)
            src_queue.join()

    def _get_move_process_grp_desc_list(self) -> List[Tuple[str, JoinableQueue, JoinableQueue]]:
        pairs = [
            ("CaseExecution", self.case_execution_queue, self.exe_result.case_execution_details),
            ("CaseException", self.case_exception_queue, self.exe_result.case_exception_details),
            ("CaseTerminate", self.case_terminate_queue, self.exe_result.case_terminate_details),
            (f"{self.cntr_name}Execution", self.cntr_execution_queue, self.exe_result.cntr_execution_details),
        ]
        return pairs

    def _start_cntr_process_grp(self, delay: int = 2) -> List[Process]:
        """启动 Cntr 进程组

        :param delay: 各 Cntr 启动后, 处理具体 Case 前延迟时长, 在多消费者模式下, 各消费者启动时增加一定延迟,
            等待所有消费者启动完成
        :return: Cntr 进程组
        """
        process_group: List[Process] = []
        for exec_param in self.exe_params:
            process = Process(
                name=f"{self.cntr_name}Process({self.cntr_name}[{exec_param.cntr_id}])",
                target=self._cntr,
                args=(
                    exec_param.cntr_id,
                    exec_param,
                    delay,
                ),
            )
            process_group.append(process)
            process.start()
        return process_group

    def _join_cntr_process_grp(self, cntr_process_grp: List[Process], step: int = 1):
        """以同步方式等待 Cntr 进程组完成

        :param cntr_process_grp: Cntr 进程组
        :param step: 内部检测步长, 单位为秒
        """
        s_time = time.time()
        while True:
            if not self._wait_cntr_one_step(cntr_process_grp=cntr_process_grp, s_time=s_time, step=step):
                break

    def _wait_cntr_one_step(self, cntr_process_grp: List[Process], s_time, step: int = 1) -> bool:
        """阻塞当前进程, 检测 Cntr 进程组完成情况

        :param cntr_process_grp: Cntr 进程组
        :param s_time: 进程组启动时间
        :param step: 检测步长
        :return: 是否要继续检测
        """
        time.sleep(step)
        need_next_step = True
        timeout = int(time.time() - s_time) > self.exe_timeout if self.exe_timeout else False
        if timeout:
            # 停止所有子进程对新任务的处理
            self.cntr_terminate_event.set()
            need_next_step = False
            time.sleep(step)
        alive_process_count = 0
        for process in cntr_process_grp:
            if process.is_alive():
                if timeout:
                    logging.info("%s timeout, terminate it.", process.name)
                    os.kill(process.pid, signal.SIGINT)  # 停止对应子进程当前处理的任务
                alive_process_count += 1
                continue
            if process.exitcode != 0 and self.exe_halt_on_error:
                need_next_step = False
                logging.info("MainProcess Recv %s upload terminate event", process.name)
                break
        need_next_step = False if alive_process_count == 0 else need_next_step
        if not need_next_step:
            self._stop_cntr_process_grp(cntr_process_grp=cntr_process_grp, timeout=step)
        return need_next_step

    def _stop_cntr_process_grp(self, cntr_process_grp: List[Process], timeout: int = 1):
        """停止 Cntr 进程组

        :param cntr_process_grp: Cntr 进程组
        :param timeout: 等待退出超时时长
        """
        self.cntr_terminate_event.set()  # 停止所有子进程对新任务的处理
        for process in cntr_process_grp:
            # 当通过 build_ci.py 经 CMake 调用本脚本时, build_ci.py 会向整个进程组(包括 Cntr/Case 子进程)
            # 发送 SIGINT 信号.
            # 此时优先等待子进程自主退出.
            if process.is_alive():
                process.join(timeout=timeout)
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)  # 停止对应子进程当前处理的任务
                logging.info("MainProcess Send download terminate event to %s.", process.name)
                process.join(timeout=timeout)

    def _cntr(self, cntr_id: int, exec_param, delay: int):
        """Container 进程

        说明:
            1. Container 进程执行时, 不会产生 Exception, 用例执行异常信息会上报至异常信息队列;
            2. Container 进程在任务队列为空, 或异常终止事件被设置时退出;

        :param cntr_id: ContainerId
        :param exec_param: ContainerParam
        """
        self._set_process_desc()
        self._cntr_set_cpu_affinity(cntr_id=cntr_id)
        self._cntr_set_gcov_prefix(cntr_id=cntr_id)
        ctx = TestsAccelerate.CntrContext(cntr_id=cntr_id, exec_param=exec_param)
        try:
            time.sleep(delay)
            while not self.cntr_terminate_event.is_set():
                # 用例获取
                case_name = self._cntr_get_case()
                if case_name is None:
                    break
                # 用例处理
                need_next = self._cntr_deal_case(case_name=case_name, ctx=ctx)
                if not need_next:
                    break  # 不需处理下一个 Case, 退出处理
        except KeyboardInterrupt:
            pass
        # Container 执行结果统计与上报
        self._put_cntr_execution_info(info=ctx.brief)
        if not ctx.exit_code:
            logging.info("%s Send terminate event upload.", self._get_process_desc())
        logging.info(
            "%s Exist[%s] %s %s",
            self._get_process_desc(),
            ctx.exit_code,
            self._cntr_progress(update=True),
            self._case_progress(update=False),
        )
        exit(ctx.exit_code)  # 通过 exit_code 传递 Container 执行结果, 触发上层感知

    def _cntr_get_case(self) -> Optional[str]:
        """获取待执行用例

        :return: 待执行用例名, None 表示无待执行用例
        """
        try:
            case_name = self.case_queue.get()
            self.case_queue.task_done()
        except queue.Empty:
            case_name = None  # 队列为空, 正常退出
        except KeyboardInterrupt:
            case_name = None  # 等待获取待执行用例过程中, 强制终止时, 正常退出
        return case_name

    def _cntr_deal_case(self, case_name: str, ctx: CntrContext) -> Optional[bool]:
        """处理单个 Case

        :param case_name: 用例名称
        :param ctx: Cntr 处理上下文
        :return: 需要继续处理下个 Case
        """
        process = None
        try:
            # 用例进程启动
            process = Process(
                name=f"CaseProcess({self.cntr_name}[{ctx.cntr_id}] Case[{case_name}])",
                target=self._case,
                args=(
                    ctx.cntr_id,
                    ctx.exec_param,
                    case_name,
                ),
            )
            process.start()
            process.join()
        except KeyboardInterrupt:
            if process and process.is_alive():
                # 用例执行过程中, 强制终止时, 杀停子进程
                logging.info(
                    "%s Recv terminate event download, stop running Case[%s]", self._get_process_desc(), case_name
                )
                os.kill(process.pid, signal.SIGINT)
                process.join()  # 等待 Case 进程结束
        finally:
            need_next = self._cntr_deal_case_finally(process=process, case_name=case_name, ctx=ctx)
        return need_next

    def _cntr_deal_case_finally(self, process: Process, case_name: str, ctx: CntrContext) -> bool:
        """处理单个 Case 结束

        :param process: CaseProcess
        :param case_name: 用例名称
        :param ctx: Cntr 处理上下文
        :return: 需要继续处理下个 Case
        """
        if process is None:
            return False
        if process.exitcode == 0:
            ctx.success += 1
            return True
        ctx.failed += 1
        if not self.exe_halt_on_error:
            return True
        self.cntr_terminate_event.set()
        ctx.exit_code = process.exitcode
        logging.info("%s Recv Case[%s] upload terminate event.", self._get_process_desc(), case_name)
        return False

    def _execute_case(
        self, ctx: CaseContext, param: ExecParam, case_name: str
    ) -> Tuple[subprocess.CompletedProcess, str, timedelta]:
        """统一的用例执行入口 - 由子类重写此方法实现不同模式"""
        return self.exe.run(params=[f"--gtest_filter={case_name}"], envs=param.get_envs())

    def _cntr_set_cpu_affinity(self, cntr_id: int):
        """在 Cntr 启动初期, 设置 CPU 亲和性

        将 CPU 亲和性配置在 Cntr 进程, 则该 Cntr 所执行的 Case 都会继承该配置
        """
        if not self.cpu_affinity_policy:
            return
        # 确定 CPU 分组索引
        if self.cpu_affinity_policy == 1:
            group_idx = cntr_id
        else:
            cpu_rank_num = cpu_count() // self.cpu_rank_size
            group_idx = cntr_id % cpu_rank_num
        # 计算 CPU 分组内容
        start_core = group_idx * self.cpu_rank_size
        end_core = min(start_core + self.cpu_rank_size, cpu_count())  # 防止超出 CPU 总数
        cpu_core_list = [int(i) for i in range(start_core, end_core)]
        try:
            os.sched_setaffinity(0, cpu_core_list)  # 0代表当前进程PID
            # 验证设置结果(可选)
        except OSError as e:
            # CPU 亲和性设置失败不影响用例执行
            logging.error("%s[%s] Failed to set CPU affinity: %s", self.cntr_name, cntr_id, e)
        current_affinity = os.sched_getaffinity(0)  # 0代表当前进程PID
        logging.debug("%s[%s] cpu affinity cores: %s", self.cntr_name, cntr_id, current_affinity)

    def _case(self, cntr_id: int, param: ExecParam, case_name: str):
        """具体用例执行进程

        通过子进程实现各 Case 执行上下文隔离, 避免 Case 间相互影响

        :param cntr_id: Container ID
        :param case_name: 用例名称
        """
        self._set_process_desc()
        ctx = TestsAccelerate.CaseContext(cntr_id=cntr_id, exec_param=param, case_name=case_name)
        run_desc = f"Run {self.mark}{self.exe.brief} Case({case_name})"
        try:
            logging.info("%s[%s] [BGN] %s", self.cntr_name, cntr_id, run_desc)
            ret, cmd, _ = self._execute_case(ctx, param, case_name)
            if ret.returncode:
                self._case_exception_exit(
                    cntr_id=cntr_id, cmd=cmd, ret_code=ret.returncode, out=ret.stdout, err=ret.stderr
                )
            else:
                msg = f"{ret.stdout}\n{ret.stderr}"
                logging.info(
                    "%s[%s] [END] %s %s Output Below:\n%s",
                    self.cntr_name,
                    cntr_id,
                    run_desc,
                    self._case_progress(update=True),
                    msg,
                )
                self._put_case_execution_info(info=ctx.brief)
        except subprocess.TimeoutExpired as e:
            self._put_case_terminate_info(info=ctx.brief)  # 执行超时时, 主动退出执行, 上报已运行时长
            self._case_exception_exit(cntr_id=cntr_id, cmd=str(e), ret_code=1, out=None, err=str(e.output))
        except KeyboardInterrupt:
            self._put_case_terminate_info(info=ctx.brief)  # 强制终止时, 主动退出执行, 上报已运行时长
            logging.info("%s Recv terminate event download, stop running.", self._get_process_desc())

    def _case_exception_exit(
        self, cntr_id: int, cmd: str, ret_code: int, out: Optional[str] = None, err: Optional[str] = None
    ):
        """用例执行进程异常退出处理

        :param cntr_id: CntrId
        :param cmd: 失败命令行
        :param ret_code: 进程退出码
        :param out: 输出信息
        :param err: 异常信息
        """
        # 收集错误现场信息并上报
        msg = f"{self.cntr_name} : {cntr_id}\nCmd : {cmd}\nRetCode : {ret_code}\nstdout :\n{out}\nstderr :\n{err}"
        self._put_case_exception_info(info=msg)
        # 异常后处理
        if self.exe_halt_on_error:
            self.cntr_terminate_event.set()
            logging.info("%s Send terminate event upload.", self._get_process_desc())
            exit(ret_code)  # 触发 Container 执行进程感知 Case 执行异常

    def _cntr_progress(self, update=True) -> str:
        """获取 Container 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)"""
        if update:
            with self.cntr_exit_count.get_lock():
                self.cntr_exit_count.value += 1
        cnt = int(self.cntr_exit_count.value)
        pgs = cnt / self.cntr_num * 100
        return f"{self.cntr_name}Progress[{cnt}/{self.cntr_num} {pgs:.2f}%]"

    def _case_progress(self, update=True) -> str:
        """获取 Case 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)"""
        if update:
            with self.case_exec_count.get_lock():
                self.case_exec_count.value += 1
        cnt = int(self.case_exec_count.value)
        pgs = cnt / self.case_num * 100
        return f"CaseProgress[{cnt}/{self.case_num} {pgs:.2f}%]"

    def _put_case_execution_info(self, info: List[Any]):
        self.case_execution_queue.put(info)

    def _put_case_exception_info(self, info: str, chunk_size: int = 4096):
        for i in range(0, len(info), chunk_size):
            self.case_exception_queue.put(info[i:i + chunk_size])
        self.case_exception_queue.put("")  # 插入分隔符

    def _put_case_terminate_info(self, info: List[Any]):
        self.case_terminate_queue.put(info)

    def _put_cntr_execution_info(self, info: List[Any]):
        self.cntr_execution_queue.put(info)
