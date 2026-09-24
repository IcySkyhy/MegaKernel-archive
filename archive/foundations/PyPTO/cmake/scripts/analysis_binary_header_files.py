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
"""构建产物二进制头文件分析."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, List


class Analysis:
    def __init__(self, args):
        # 参数预处理
        self.source: Path = Path(args.source[0])
        self.binary: Path = Path(args.binary[0])
        self.target_file: Path = Path(args.target[0])
        self.target_name: str = self.target_file.name
        self.target_binary_dir: Path = Path(args.target_binary_dir[0])
        self.target_objects: List[Path] = [Path(p) for p in str(args.objects[0]).split(';')]
        self.target_include_filters: List[Path] = [] if args.filters is None else [i for r in args.filters for i in r]
        self.target_include_blacks: List[Path] = []
        if args.json is not None:
            self.update_by_json(f=args.json[0])
        self.target_include_filters = list(set(self.target_include_filters))
        self.target_include_filters.sort()
        self.target_include_blacks = list(set(self.target_include_blacks))
        self.target_include_blacks.sort()
        self.target_include_illegal: Dict[Path, List[Path]] = {}

    def __str__(self) -> str:
        desc = "\nBinaryHeaderAnalysis:"
        desc += f"\n    Source          : {self.source}"
        desc += f"\n    Binary          : {self.binary}"
        desc += f"\n    Target          : {self.target_file}"
        desc += f"\n    TargetBinaryDir : {self.target_binary_dir}"
        return desc

    def update_by_json(self, f: Path):
        with open(str(f), 'r', encoding='utf-8') as fh:
            desc = json.load(fh)
        target_src_inc_cfg = desc.get(self.target_name, {}).get("Source-Header", {})
        for typ, lst in target_src_inc_cfg.items():
            for sub_dir in lst:
                sub_path = Path(self.source, sub_dir)
                if str(typ).lower() == "blacklist":
                    self.target_include_blacks.append(sub_path)
                else:
                    self.target_include_filters.append(sub_path)

    def analysis(self) -> bool:
        ts = datetime.now(tz=timezone.utc)
        for o in self.target_objects:
            if not self.analysis_object(o=o):
                return False
        if len(self.target_include_illegal) != 0:
            cnt = 0
            for o, lst in self.target_include_illegal.items():
                for d in lst:
                    logging.info("%s : %-40s dependency %s", self.target_name, self.get_o_sub_path(o=o), d)
                    cnt += 1
            logging.error(
                "%s has %s illegal header-file dependence, Duration %s secs.",
                self.target_name,
                cnt,
                (datetime.now(tz=timezone.utc) - ts).seconds,
            )
            return False
        logging.info(
            "%s header-file dependence check success, Duration %s secs.",
            self.target_name,
            (datetime.now(tz=timezone.utc) - ts).seconds,
        )
        return True

    def analysis_object(self, o: Path) -> bool:
        # 获取 .o 对应 .d 文件
        if not o.exists():
            logging.error("%s object-file not exist, %s", self.target_name, o)
            return False
        d = Path(str(o) + ".d")
        d = d if d.exists() else o.with_suffix(".d")
        if not d.exists():
            logging.error("%s dependence-file not exist, %s", self.target_name, d)
            return False
        # 解析 .d 获取具体 .h 依赖列表
        h_lst = []
        with open(d, mode='r', encoding='utf-8', errors="ignore") as fh:
            for _, line in enumerate(fh, 1):
                # 去除首尾空白字符, 规范化路径
                line_str = line.split("\\")[0].replace(":", "").strip()
                # 存在一行内包含多个路径的情况, 此时路径间以空格分割
                line_lst = line_str.split(" ")
                for cur_line_str in line_lst:
                    cur_line_str = cur_line_str.strip()
                    cur_path = Path(cur_line_str)
                    if not cur_path.is_absolute():
                        cur_path = Path(self.target_binary_dir, cur_path).resolve(strict=False)
                    h_lst.append(cur_path)
        h_lst = h_lst[1:]  # 去除 cpp.o 描述
        h_lst.sort()
        # 判断 .h 依赖列表内容合法合理性
        for h in h_lst:
            legal = self.in_filters(h=h) and not self.in_blacks(h=h)
            if not legal:
                lst = self.target_include_illegal.get(o, [])
                lst.append(h)
                lst = list(set(lst))
                lst.sort()
                self.target_include_illegal[o] = lst
        return True

    def get_o_sub_path(self, o: Path) -> str:
        var = o.parts[len(self.binary.parts):]
        var = [var[0]] + list(var[3:-1]) + [str(var[-1]).rsplit(".", 1)[0]]
        return "/".join(var)

    def in_filters(self, h: Path) -> bool:
        for p in self.target_include_filters:
            if self.is_sub_path(child=h, parent=p):
                return True
        return False

    def in_blacks(self, h: Path) -> bool:
        for p in self.target_include_blacks:
            if self.is_sub_path(child=h, parent=p):
                return True
        return False

    @staticmethod
    def is_sub_path(child: Path, parent: Path) -> bool:
        child = child.absolute()
        parent = parent.absolute()
        # 比较路径部分
        parent_parts = parent.parts
        child_parts = child.parts
        # 子路径需完全包含父路径的前缀
        return parent_parts == child_parts[:len(parent_parts)]

    @staticmethod
    def main() -> bool:
        """主处理流程"""
        # 参数注册
        parser = argparse.ArgumentParser(description="Header-File Analysis.", epilog="Best Regards!")
        parser.add_argument("-s", "--source", nargs=1, type=Path, required=True, help="Specific source root path.")
        parser.add_argument("-b", "--binary", nargs=1, type=Path, required=True, help="Specific binary root path.")
        parser.add_argument(
            "-t", "--target", nargs=1, type=str, required=True, help="Specific target binary file path."
        )
        parser.add_argument("--target_binary_dir", nargs=1, type=str, required=True, help="Specific target binary dir.")
        parser.add_argument("-o", "--objects", nargs=1, type=str, required=True, help="Specific target binary objects.")
        parser.add_argument(
            "-j",
            "--json",
            nargs=1,
            type=Path,
            required=False,
            help="Specific target include filter(file/dir) json file.",
        )
        parser.add_argument(
            "-f",
            "--filters",
            required=False,
            action='append',
            nargs='*',
            type=Path,
            help="Specific target include filter(file/dir).",
        )
        # 流程处理
        ctrl = Analysis(args=parser.parse_args())
        return ctrl.analysis()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    exit(0 if Analysis.main() else 1)
