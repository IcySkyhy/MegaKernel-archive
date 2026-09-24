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
"""构建产物二进制符号分析."""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import shlex
import subprocess
from typing import List


class Analysis:
    def __init__(self, args):
        self.file: Path = Path(args.file[0])
        self.print_defined_relations: bool = args.print_defined_relations
        self.check_undefined_symbols_self: bool = not args.ignore_undefined_symbols_self
        self.check_undefined_symbols_pass: bool = not args.ignore_undefined_symbols_pass
        self.defined_relations: List[str] = []
        self.undefined_symbols_self: List[str] = []
        self.undefined_symbols_pass: List[str] = []

    def __str__(self):
        # 结果
        str_1 = ""
        if self.print_defined_relations:
            str_1 = "\nDefined:\n\t"
            str_1 += "\n\t".join(self.defined_relations)
        str_2 = ""
        if len(self.undefined_symbols_self) != 0:
            str_2 += f"\nUnDefined-Self({len(self.undefined_symbols_self)}):\n\t"
            str_2 += "\n\t".join(self.undefined_symbols_self)
        str_3 = ""
        if len(self.undefined_symbols_pass) != 0:
            str_3 += f"\nUnDefined-Pass({len(self.undefined_symbols_pass)}):\n\t"
            str_3 += "\n\t".join(self.undefined_symbols_pass)
        return str_1 + str_2 + str_3

    @staticmethod
    def _norma_list(lst: List[str], reverse=False) -> List[str]:
        st = set(lst)
        lst = list(st)
        lst.sort(reverse=reverse)
        return lst

    def analysis(self) -> bool:
        ts = datetime.now(tz=timezone.utc)
        # 获取二进制自身未定义原始符号范围
        ori_undefined_symbols_self = self._analysis_ori_undefined_symbols_self()

        # 解析二进制符号
        self._analysis_symbols(ori_undefined_symbols_self=ori_undefined_symbols_self)

        # 结果确定
        return self._analysis_result(ts=ts)

    def _analysis_ori_undefined_symbols_self(self) -> List[str]:
        cmd = f"nm -u {self.file}"
        ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
        ret.check_returncode()
        lines = str(ret.stdout)
        ori_undefined_symbols_self = []
        for line in lines.splitlines():
            ori_symbol = line.strip().split()[1]
            ori_undefined_symbols_self.append(ori_symbol)
        ori_undefined_symbols_self = self._norma_list(lst=ori_undefined_symbols_self)
        return ori_undefined_symbols_self

    def _analysis_symbols(self, ori_undefined_symbols_self: List[str]):
        cmd = f"ldd -r {self.file}"
        ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
        ret.check_returncode()
        lines = str(ret.stdout)
        for line in lines.splitlines():
            if line.startswith("\t"):
                self.defined_relations.append(f"{line.strip()}")
            elif line.startswith("undefined symbol: "):
                line = line[18:]  # 跳过 'undefined symbol: '
                ori_symbol = line.split("\t")[0]  # 提取符号
                cmd = f"c++filt {ori_symbol}"
                ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
                ret.check_returncode()
                symbol = str(ret.stdout).strip()
                if ori_symbol in ori_undefined_symbols_self:
                    self.undefined_symbols_self.append(symbol)
                else:
                    self.undefined_symbols_pass.append(symbol)
            else:
                continue
        self.defined_relations = self._norma_list(lst=self.defined_relations)
        self.undefined_symbols_self = self._norma_list(lst=self.undefined_symbols_self)
        self.undefined_symbols_pass = self._norma_list(lst=self.undefined_symbols_pass)

    def _analysis_result(self, ts) -> bool:
        rst_str = str(self)
        if len(rst_str) != 0:
            logging.info("%s", rst_str)
        ret = True
        cnt1 = len(self.undefined_symbols_self)
        cnt2 = len(self.undefined_symbols_pass)
        if self.check_undefined_symbols_self and cnt1 != 0:
            ret = False
            logging.error(
                "%s has %s undefined symbols self, Duration %s secs.",
                self.file.name,
                cnt1,
                (datetime.now(tz=timezone.utc) - ts).seconds,
            )
        if self.check_undefined_symbols_pass and cnt2 != 0:
            ret = False
            logging.error(
                "%s has %s undefined symbols pass, Duration %s secs.",
                self.file.name,
                cnt2,
                (datetime.now(tz=timezone.utc) - ts).seconds,
            )
        if ret:
            logging.info(
                "%s symbols check success, Duration %s secs.",
                self.file.name,
                (datetime.now(tz=timezone.utc) - ts).seconds,
            )
        return ret

    @staticmethod
    def main():
        """主处理流程"""
        # 参数注册
        parser = argparse.ArgumentParser(description="Symbol Analysis.", epilog="Best Regards!")
        parser.add_argument("-f", "--file", nargs=1, type=str, required=True, help="Specific binary file path.")
        parser.add_argument(
            "--print_defined_relations", action="store_true", default=False, help="Print defined relations."
        )
        parser.add_argument(
            "--ignore_undefined_symbols_self",
            action="store_true",
            default=False,
            help="Ignore undefined symbols self-contained.",
        )
        parser.add_argument(
            "--ignore_undefined_symbols_pass",
            action="store_true",
            default=False,
            help="Ignore undefined symbols passed between binaries.",
        )
        # 流程处理
        ctrl = Analysis(args=parser.parse_args())
        return ctrl.analysis()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    exit(0 if Analysis.main() else 1)
