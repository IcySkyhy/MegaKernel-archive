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
"""生成 gcov_config.json 配置文件

由 CMake 在 POST_BUILD 阶段调用, 收集所有 filter 目录并生成 JSON 配置文件.
生成的配置文件供外部在 Python 用例覆盖率生成时使用.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List


class GenCoverageConfig:
    """生成 gcov 配置文件的控制器类"""

    class FilterPathAction(argparse.Action):
        """自定义 Action: 解析 filter 参数时校验路径并格式化"""

        def __call__(self, parser, namespace, values, option_string=None):
            # 获取当前已收集的列表(初始为 None)
            cur_values = getattr(namespace, self.dest, None) or []
            # 处理分号分隔的多个路径 (VERBATIM 模式下，生成器表达式展开为分号分隔字符串)
            for path_str in values.split(';'):
                path_str = path_str.strip()
                if not path_str:
                    continue
                path = Path(path_str)
                cur_values.append(str(path))
            # 更新命名空间的值
            setattr(namespace, self.dest, cur_values)

    def __init__(self, args):
        """初始化控制器实例"""
        self.binary_dir: Path = Path(args.binary_dir).resolve()
        self.filter_lst: List[str] = args.filter

    def __str__(self) -> str:
        """返回配置信息字符串"""
        desc = "\nGenCoverageConfig"
        desc += f"\n    BinaryDir    : {self.binary_dir}"
        desc += f"\n    FilterDirs   : {self.filter_lst}"
        desc += "\n"
        return desc

    @classmethod
    def main(cls):
        """主入口函数"""
        # 参数注册
        parser = argparse.ArgumentParser(description="Generate Coverage Config")
        parser.add_argument(
            "-d", "--binary_dir", required=True, type=Path, help="CMake binary directory (PTO_FWK_BIN_ROOT)"
        )
        parser.add_argument(
            "-f",
            "--filter",
            required=False,
            action=cls.FilterPathAction,
            type=str,
            help="Specify filter file/dir in coverage info.",
        )
        # 参数处理
        ctrl = cls(args=parser.parse_args())
        logging.info("%s", ctrl)
        # 流程处理
        ctrl.process()

    def process(self):
        """生成配置文件"""
        # 构造配置内容
        config = {
            'cmake_binary_dir': str(self.binary_dir),
            'filter_dirs': [str(p) for p in self.filter_lst],
        }

        # 写入配置文件 (覆盖式)
        config_file = self.binary_dir / 'gcov_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        logging.info("Generated gcov_config.json: %s", config_file)
        logging.info("  filter_lst: %s", self.filter_lst)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    GenCoverageConfig.main()
