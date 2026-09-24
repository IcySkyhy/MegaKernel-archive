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
"""STest Golden 生成总入口.

在执行 STest 前需要生成用例所需的 Golden 数据并保存在文件中, 以供用例使用. 设计本入口脚本以统一其处理逻辑.
本脚本在 CMake 中识别需要执行 STest 时, 由 CMake 调用.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, List, Tuple

from golden_register import GoldenRegInfo, GoldenRegister
from utils.table import Table


class GoldenCtrl:
    """STest Golden 生成逻辑控制."""

    _MAX_JOB_NUM: int = 16  # 16 控制 Golden 产生时多进程最大并行度

    def __init__(self, args):
        self.sys_paths: List[Path] = []
        # 命令行参数处理
        self.cases: List[str] = str(args.cases).split(":")
        self.output: Path = Path(args.output).resolve()
        self.impl_dirs: List[Path] = [Path(p).resolve() for p in args.path]
        self.impl_dirs = list(set(self.impl_dirs))
        self.impl_dirs.sort()
        self.clean: bool = args.clean
        self.job_num: int = min(max(args.job_num, 1), multiprocessing.cpu_count(), self._MAX_JOB_NUM, len(self.cases))
        #
        self.json_file_name: str = "golden_desc.json"
        logging.info("\n\nGolden Ctrl Args:\n%s", Table.table(datas=self.brief))

    @property
    def brief(self) -> List[Any]:
        ver = sys.version_info
        datas = [
            ["Python3", f"{sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"],
            ["CaseNum", len(self.cases)],
            ["OutputDir", self.output],
            ["CleanFlag", self.clean],
            ["JobNum", self.job_num],
            ["ImplDirsNum", len(self.impl_dirs)],
        ]
        for impl_i, impl_d in enumerate(self.impl_dirs, start=1):
            datas.append([f"ImplDir[{impl_i}]", str(impl_d)])
        return datas

    @staticmethod
    def default_golden_path(base_dir) -> List:
        golden_paths = []
        for root, dirs, _ in os.walk(base_dir):
            for d in dirs:
                if '__pycache__' in d:
                    continue
                golden_paths.append(os.path.join(root, d))
        return golden_paths

    @staticmethod
    def main() -> bool:
        """主处理流程"""
        parser = argparse.ArgumentParser(description="STest Golden Ctrl", epilog="Best Regards!")
        parser.add_argument(
            "-c",
            "--cases",
            type=str,
            default="",
            required=True,
            help="STest Cases, multiple test cases are separated by ':'",
        )
        parser.add_argument("-o", "--output", type=str, default="golden", help="Golden output path.")
        parser.add_argument(
            "-p",
            "--path",
            nargs="?",
            type=str,
            action="append",
            help="Golden impl path, relative path to the source root directory.",
        )
        parser.add_argument("--clean", action="store_true", default=False, help="clean, clean before generate.")
        parser.add_argument(
            "-j",
            "--job_num",
            nargs="?",
            type=int,
            # Golden 生成不确定是否 CPU Bound, 默认使用 0.8 倍 CPU 数进程
            default=int(math.ceil(float(multiprocessing.cpu_count()) * 0.8)),
            help="Specific parallel accelerate job num.",
        )
        args = parser.parse_args()
        if not args.path:
            base_dir = os.path.join(os.path.dirname(sys.argv[0]), "golden")
            args.path = GoldenCtrl.default_golden_path(base_dir)

        ctrl = GoldenCtrl(args)
        ret = ctrl.prepare()
        ret = ret and ctrl.process()
        return ret

    def prepare(self) -> bool:
        """执行 Golden 生成任务前准备"""
        return self.prepare_module()

    def prepare_module(self) -> bool:
        """执行 Golden 生成任务前准备

        将需 import module 在主进程完成 import, 子进程继承 import 关系
        """
        for impl_d in self.impl_dirs:
            if not impl_d.exists():
                logging.error("ImplDir(%s) not exist.", impl_d)
                return False
            if not impl_d.is_dir():
                logging.error("ImplDir(%s) is not directory.", impl_d)
                return False
            if impl_d not in self.sys_paths and impl_d not in sys.path:
                sys.path.append(str(impl_d))
                self.sys_paths.append(impl_d)
            for impl_f in impl_d.glob("*.py"):
                if impl_f.stem == "__init__":
                    continue
                module_name = f"{impl_f.stem}"
                importlib.import_module(module_name)
        logging.info("Register golden func finish, get %s func", GoldenRegister.get_golden_func_num())
        return True

    def process(self) -> bool:
        """执行 Golden 生成任务, 生成 Cases 所需 Golden"""
        # 输出路径处理
        if self.clean and self.output.exists():
            shutil.rmtree(self.output)
        self.output.mkdir(parents=True, exist_ok=True)
        # 任务执行
        ts = datetime.now(tz=timezone.utc)
        if self.job_num <= 1:
            ret = self.run_all_task_single_process()
        else:
            ret = self.run_all_task_multi_process()
        logging.info(
            "Generate golden finish[%s], Duration %s secs, Return(%s)",
            len(self.cases),
            (datetime.now(tz=timezone.utc) - ts).seconds,
            ret,
        )
        return ret

    def run_all_task_multi_process(self) -> bool:
        with ProcessPoolExecutor(max_workers=self.job_num) as executor:
            # 提交多个任务
            futures = [executor.submit(self.run_task, c, i + 1) for i, c in enumerate(self.cases)]
            # 按完成顺序获取结果
            for future in as_completed(futures):
                ret = False if not future.result() else True
                if not ret:
                    return False
        return True

    def run_all_task_single_process(self) -> bool:
        for i, c in enumerate(self.cases):
            ret = self.run_task(c=c, idx=i + 1)
            if not ret:
                return False
        return True

    def run_task(self, c: str, idx: int = 0) -> bool:
        ts = datetime.now(tz=timezone.utc)
        # 获取 Golden 生成函数
        reg_info, case_idx = GoldenRegister.get_golden_func(case_name=c)
        if reg_info is None:
            logging.debug("Generate golden failed Idx[%s/%s] Case(%s) Can't find generator.", idx, len(self.cases), c)
            return True

        # 用例 Golden 路径处理
        case_output, need_gen = self._prepare_output(case=c, reg_info=reg_info)
        if not need_gen:
            logging.info("Generate golden skip Idx[%s/%s] Case(%s).", idx, len(self.cases), c)
            return True
        try:
            if case_idx is None:
                ret = bool(reg_info.func(case_name=c, output=case_output))
            else:
                ret = bool(reg_info.func(case_name=c, output=case_output, case_index=case_idx))
        except Exception as e:
            raise RuntimeError(f"Error in Case[{c}]") from e
        if ret:
            self._dump_golden_desc(case_output=case_output, reg_info=reg_info)

        msg = "success" if ret else "failed"
        logging.info(
            "Generate golden %s Idx[%s/%s] Case(%s) Duration %s secs.",
            msg,
            idx,
            len(self.cases),
            c,
            (datetime.now(tz=timezone.utc) - ts).seconds,
        )
        return ret

    def _prepare_output(self, case: str, reg_info: GoldenRegInfo) -> Tuple[Path, bool]:
        case_output = Path(self.output, case.replace("*", ""))
        # 获取原始控制信息(Version, TimeStamp)
        ori_ver = 0
        ori_time = time.time()
        ver_file = Path(case_output, self.json_file_name)
        if ver_file.exists():
            with open(ver_file, 'r', encoding='utf-8') as fh:
                datas = json.load(fh)
            ori_ver = datas["version"]
            ori_time = datas["timestamp"]

        # 若版本变化, 或已过期, 需要提前删除
        now_time = time.time()
        need_del_version = reg_info.version > ori_ver
        need_del_time = False if reg_info.timeout is None else int(now_time - ori_time) > reg_info.timeout
        if (need_del_version or need_del_time) and case_output.exists():
            logging.info(
                "Remove Case(%s)'s golden, VersionFlg(%s), TimeFlag(%s)", case, need_del_version, need_del_time
            )
            shutil.rmtree(case_output)

        # 创建 Golden 目录
        case_output.mkdir(parents=True, exist_ok=True)
        return case_output, not ver_file.exists()

    def _dump_golden_desc(self, case_output: Path, reg_info: GoldenRegInfo):
        # 刷新控制信息
        now_time = time.time()
        desc = {"version": reg_info.version, "timestamp": now_time}
        ver_file = Path(case_output, self.json_file_name)
        with open(ver_file, 'w', encoding='utf-8') as fh:
            json.dump(desc, fh)
        return case_output


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    exit(0 if GoldenCtrl.main() else 1)
