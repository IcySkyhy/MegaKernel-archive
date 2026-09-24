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
"""分析修改文件清单.

分析修改文件清单, 判断当前测试场景是否需要执行及获取用例执行范围.
"""

import argparse
import dataclasses
import fnmatch
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclasses.dataclass
class Module:
    name: str
    cases: List[str]
    write: List[Path]

    @staticmethod
    def _relative_to(s: Path, d: Path) -> bool:
        try:
            if s.relative_to(d):
                return True
        except ValueError:
            pass
        return False

    def is_trigger(self, changed: List[Path]) -> Tuple[bool, List[str]]:
        # 若无 changed, 默认触发所有用例
        if not changed:
            return True, self.cases
        # 当所有 changed 均命中白名单, 无需触发
        for c in changed:
            c_skip = False
            for w in self.write:
                if self._relative_to(c, w):
                    c_skip = True
                    logging.debug("Changed(%s) hit writeList(%s), skip Module(%s)", c, w, self.name)
                    break
            if not c_skip:
                logging.debug("Changed(%s) not hit writeList, trigger Module(%s)", c, self.name)
                return True, self.cases
        return False, []


class Analysis:
    _KEY_WRITE_LIST: str = "write_list"
    _KEY_CASES: str = "cases"

    def __init__(self, args):
        self.rule: Path = Path(args.rule[0]).resolve()
        self.type: str = str(args.type[0]).lower()
        self.group: List[str] = args.group.split(",") if args.group else []
        self.file: Optional[Path] = Path(args.file[0]).resolve() if args.file and args.file[0] else None
        # 内部对象转化
        self.modules: Dict[str, Module] = self._init_get_models()
        self.changed: List[Path] = self._init_get_changed()

    def __str__(self) -> str:
        ver = sys.version_info
        desc = f"\nPython3 : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\nRule    : {self.rule}"
        desc += f"\nType    : {self.type}"
        desc += f"\nGroup   : {self.group}"
        desc += f"\nFile    : {self.file}"
        desc += "\n"
        return desc

    @staticmethod
    def main() -> str:
        parser = argparse.ArgumentParser(description="Analysis Changed Files", epilog="Best Regards!")
        parser.add_argument("-r", "--rule", required=True, nargs=1, type=Path, help="Specific classify_rule.yaml")
        parser.add_argument(
            "-t", "--type", nargs=1, type=str, required=True, choices=["utest", "stest"], help="Specific tests type"
        )
        parser.add_argument(
            "-g",
            "--group",
            nargs='?',
            type=str,
            required=False,
            default="",
            help="Specific tests group, multiple group are separated by ','",
        )
        parser.add_argument(
            "-c", "--changed_files", nargs=1, type=Path, required=False, dest="file", help="Specific changed_files.txt"
        )
        parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode")
        args = parser.parse_args()
        # 日志级别注册, 本文件有两种调用场景:
        # 1) 由 CMake 调用, 此时需保证若正常处理无任何额外输出, 需把日志级别调整为 ERROR;
        # 2) 调试时由 Python 直调, 此时需输出较多日志, 可将日志级别设置为 DEBUG;
        logging.basicConfig(
            format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG if args.debug else logging.ERROR,
            handlers=[logging.StreamHandler()],
        )
        # 参数解析
        ctrl = Analysis(args=args)
        logging.info(ctrl)
        # 流程处理
        return ctrl.analysis()

    def analysis(self) -> str:
        cases = self._analysis_cases()
        cases_str = ",".join(cases) if cases else ""
        return cases_str

    def _get_write_list(self, _desc: Dict[str, Any]) -> List[Path]:
        _lst = _desc.get(self._KEY_WRITE_LIST, [])
        _lst = _lst if _lst else []
        _rst = [Path(_rel) for _rel in _lst]
        _desc.pop(self._KEY_WRITE_LIST, None)
        return _rst

    def _init_get_models_from_file(self, file: Path, write_list: List[Path] = None) -> Dict[str, Module]:
        modules = {}
        with open(file, 'r', encoding='utf-8') as f:
            rule_dict = yaml.safe_load(f)
        rule_dict = rule_dict.get(self.type, {})
        # 处理 type 下白名单
        type_write_list = self._get_write_list(_desc=rule_dict)
        type_write_list = write_list if write_list else type_write_list
        # 循环处理 module
        for name, desc in rule_dict.items():
            # 处理 module 下白名单
            write_list = self._get_write_list(_desc=desc)
            write_list.extend(type_write_list)
            # 获取 module 下用例列表
            cases_list = desc.get(self._KEY_CASES, [])
            mod = Module(name=name, cases=cases_list, write=write_list)
            modules[name] = mod
        return modules

    def _init_get_models(self) -> Dict[str, Module]:
        yaml_lst = self.rule.glob(pattern="classify_rule_*.yaml")
        modules = {}
        rule_file = self.rule.joinpath(f"classify_rule_{self.type}.yaml")
        with open(rule_file, 'r', encoding='utf-8') as f:
            rule_dict = yaml.safe_load(f)
        write_list = self._get_write_list(_desc=rule_dict.get(self.type, {}))
        for file in yaml_lst:
            file_module = self._init_get_models_from_file(file=file, write_list=write_list)
            modules.update(file_module)
        return modules

    def _init_get_changed(self) -> List[Path]:
        changed = []
        if self.file:
            with open(self.file, 'r', encoding='utf-8') as f:
                changed = [Path(line.rstrip('\n')) for line in f]
        return changed

    def _analysis_cases(self) -> List[str]:
        cases = []
        for module in self.modules.values():
            match_group = False if self.group else True
            for group in self.group:
                # 支持 group 名称模糊匹配
                if fnmatch.fnmatch(module.name, group):
                    match_group = True
                    break
            if not match_group:
                logging.debug("Module(%s) not match group %s", module.name, self.group)
                continue
            logging.debug("Module(%s) match group %s", module.name, self.group)
            trigger, module_cases = module.is_trigger(changed=self.changed)
            if not trigger:
                continue
            if module_cases is not None:
                cases.extend(module_cases)
        return cases


if __name__ == "__main__":
    print(Analysis.main(), end='')
