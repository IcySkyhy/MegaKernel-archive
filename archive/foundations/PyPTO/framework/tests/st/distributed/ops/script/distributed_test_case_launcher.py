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

""" """

import os
from pathlib import Path
import sys

helper_path: Path = Path(Path(__file__).parent.parent.parent.parent.parent, "cmake/scripts/helper").resolve()
if str(helper_path) not in sys.path:
    sys.path.append(str(helper_path))
from distributed_test_case_runner import OperationTestCaseRunner  # noqa: E402
from test_case import TestCase  # noqa: E402
from test_case_shell_actuator import TestCaseShellActuator  # noqa: E402


class OperationTestCase(TestCase):
    def __init__(
        self,
        test_case_info: dict,
    ):
        super().__init__(
            test_case_info.get("index"),
            test_case_info.get("case_name"),
            test_case_info.get("operation"),
            test_case_info.get("input_tensors"),
            test_case_info.get("output_tensors"),
            test_case_info.get("view_shape"),
            test_case_info.get("tile_shape"),
            test_case_info.get("params"),
            OperationTestCaseRunner(test_case_info),
        )
        self._info = test_case_info
        self._root_path = Path(Path(__file__).parent.parent.parent.parent.parent.parent.parent).resolve()

    def run_in_dyn_func(self, _inputs, _params: dict) -> dict:
        return None

    def run_golden_ctrl(self, test_case: str):
        cmd = f"{sys.executable} {self._root_path}/cmake/scripts/golden_ctrl.py "
        cmd += f"-o={self._root_path}/build/output/bin/golden -c={test_case} "
        cmd += f"--path={self._root_path}/framework/tests/st/distributed/ops/script"
        TestCaseShellActuator.run(cmd)
        return None

    def golden_func(self, inputs, _params: dict) -> list:
        case_file = os.environ.get('JSON_PATH')
        if not case_file:
            raise ValueError("JSON_PAHT environment variable is not set.")
        case_path = Path(case_file)
        if case_path.is_file():
            test_case = f"TestDistributedOps/DistributedTest.TestOps/{self._info.get('index')}"
            self.run_golden_ctrl(test_case)
        if case_path.is_dir() and self._info.get('index') == 0:
            test_case = "TestDistributedOps/DistributedTest.TestOps"
            self.run_golden_ctrl(test_case)
        return None

    def golden_func_params(self) -> dict:
        return self._info.get("params")


def test_case_launcher(test_case_info: dict):
    test_case = OperationTestCase(test_case_info)
    test_case.exec(False)
