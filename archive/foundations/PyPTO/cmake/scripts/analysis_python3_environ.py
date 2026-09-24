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
"""Python3环境分析."""

import argparse
import logging
import os
from pathlib import Path
import sys


class Analysis:
    def __init__(self, args):
        self.output: Path = Path(args.output[0]).resolve()
        self.interpreter_version: str = self._get_interpreter_version()
        self.py_mod_pybind11_cmake_dir: str = self._get_py_mod_pybind11_dir()
        self.py_mod_torch_version: str = ""
        self.py_mod_torch_root_dir: str = ""
        self.py_mod_torch_cmake_dir: str = ""
        self.py_mod_torch_c_use_cxx11_abi: int = 1
        self._init_torch_param()

    def __str__(self) -> str:
        ver = sys.version_info
        desc = "\n"
        desc += "\nEnviron"
        desc += f"\n  Output  : {self.output}"
        desc += f"\n  Python3 : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += "\n    pybind11"
        desc += f"\n      CMake_Dir                 : {self.py_mod_pybind11_cmake_dir}"
        desc += "\n    torch"
        desc += f"\n      Version                   : {self.py_mod_torch_version}"
        desc += f"\n      CMake_Dir                 : {self.py_mod_torch_cmake_dir}"
        desc += f"\n      Root_Dir                  : {self.py_mod_torch_root_dir}"
        desc += f"\n      _C._GLIBCXX_USE_CXX11_ABI : {self.py_mod_torch_c_use_cxx11_abi}"
        return desc

    @staticmethod
    def main():
        """主处理流程"""
        # 参数注册
        parser = argparse.ArgumentParser(description="Python3-Environ Analysis.", epilog="Best Regards!")
        parser.add_argument(
            "-o", "--output", nargs=1, type=Path, default=None, required=True, help="Specify output file path."
        )
        # 流程处理
        ctrl = Analysis(args=parser.parse_args())
        ctrl.analysis()

    @staticmethod
    def _get_interpreter_version():
        """Init python3 version(major.minor)

        :return: python3 version(major.minor)
        """
        ver = sys.version_info
        return f"{ver.major}.{ver.minor}"

    @staticmethod
    def _get_py_mod_pybind11_dir() -> str:
        """获取 pybind11_DIR, 以便外层 CMake 处理

        :return: pybind11_DIR
        """
        pybind11_dir = None
        try:
            import pybind11

            pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
        except ModuleNotFoundError or ImportError:
            pass
        return str(pybind11_dir) if pybind11_dir else ""

    def analysis(self):
        lines = [
            '\n# Python3 Version',
            f'\nset(PYTHON3_VERSION_ID "{self.interpreter_version}")',
            '\nmessage(STATUS "PYTHON3_VERSION_ID=${PYTHON3_VERSION_ID}")',
            '\n',
        ]
        if self.py_mod_pybind11_cmake_dir:
            lines += [
                '\n# Python3 module pybind11',
                f'\nget_filename_component(PY3_MOD_PYBIND11_CMAKE_DIR "{self.py_mod_pybind11_cmake_dir}" REALPATH)',
                '\nmessage(STATUS "PY3_MOD_PYBIND11_CMAKE_DIR=${PY3_MOD_PYBIND11_CMAKE_DIR}")',
                '\n',
            ]
        if self.py_mod_torch_version:
            lines += [
                '\n# Python3 module pybind11',
                f'\nset(PY3_MOD_TORCH_VERSION "{self.py_mod_torch_version}")',
                f'\nget_filename_component(PY3_MOD_TORCH_ROOT_PATH "{self.py_mod_torch_root_dir}" REALPATH)',
                f'\nget_filename_component(PY3_MOD_TORCH_CMAKE_DIR "{self.py_mod_torch_cmake_dir}" REALPATH)',
                f'\nset(PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI {self.py_mod_torch_c_use_cxx11_abi})',
                '\nmessage(STATUS "PY3_MOD_TORCH_VERSION=${PY3_MOD_TORCH_VERSION}")',
                '\nmessage(STATUS "PY3_MOD_TORCH_ROOT_PATH=${PY3_MOD_TORCH_ROOT_PATH}")',
                '\nmessage(STATUS "PY3_MOD_TORCH_CMAKE_DIR=${PY3_MOD_TORCH_CMAKE_DIR}")',
                '\nmessage(STATUS "PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI=${PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI}")',
                '\n',
            ]
        with open(str(self.output), "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _init_torch_param(self):
        os_env = os.environ.copy()
        try:
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            import torch

            self.py_mod_torch_version = str(torch.__version__)
            self.py_mod_torch_root_dir = str(Path(torch.__file__).parent)
            self.py_mod_torch_cmake_dir = str(Path(torch.utils.cmake_prefix_path).resolve())
            self.py_mod_torch_c_use_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        except ImportError:
            pass
        finally:
            os.environ = os_env


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    Analysis.main()
