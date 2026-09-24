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

import logging
import os
from pathlib import Path
import shutil
import sys
from typing import NoReturn

helper_path: Path = Path(Path(__file__).parent.parent.parent.parent.parent, "cmake/scripts/helper").resolve()
if str(helper_path) not in sys.path:
    sys.path.append(str(helper_path))
from test_case_desc import TensorDesc  # noqa: E402
from test_case_runner import TestCaseRunner  # noqa: E402
from test_case_shell_actuator import TestCaseShellActuator  # noqa: E402


class OperationTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        test_case_info: dict,
    ):
        super().__init__(
            test_case_info.get("view_shape"),
            test_case_info.get("tile_shape"),
            test_case_info.get("params"),
        )
        self._index = test_case_info.get("index")
        self._name = test_case_info.get("case_name")
        self._op = test_case_info.get("operation")
        self._params = test_case_info.get("params")
        self._world_size = self._params.get("world_size")
        self._input_tensors = [
            TensorDesc.from_dict(tensor) if isinstance(tensor, dict) else tensor
            for tensor in test_case_info.get("input_tensors")
        ]
        self._output_tensors = [
            TensorDesc.from_dict(tensor) if isinstance(tensor, dict) else tensor
            for tensor in test_case_info.get("output_tensors")
        ]
        self._root_path = Path(Path(__file__).parent.parent.parent.parent.parent.parent.parent).resolve()
        self._log_file = test_case_info.get("log_file")

    def input_tensors(self):
        return self._input_tensors

    def input_data(self):
        return []

    def output_tensors(self):
        return self._output_tensors

    def output_data(self):
        return []

    def exec_dyn_func(self, _input_tensors: list, _output_tensors: list):
        pass

    def tear_up(self) -> NoReturn:
        os.environ["TILE_FWK_STEST_GOLDEN_PATH"] = f"{str(self._root_path)}/build/output/bin/golden"
        os.chdir(f"{str(self._root_path)}/build/output/bin")

    def tear_down(self) -> NoReturn:
        os.chdir(f"{str(self._root_path)}")
        golden_path = os.getenv("TILE_FWK_STEST_GOLDEN_PATH")
        if not golden_path:
            return
        golden_path = Path(golden_path)
        if golden_path.is_dir():
            golden_path = golden_path / "TestDistributedOps" / "DistributedTest.TestOps"
        else:
            golden_path = golden_path / "TestDistributedOps" / "DistributedTest.TestOps" / str(self._index)
        if golden_path.exists():
            shutil.rmtree(golden_path)

    def run_on_device(self, inputs: list) -> list:
        test_case = f"TestDistributedOps/DistributedTest.TestOps/{self._index}"
        cmd = f"mpirun -n {self._world_size} ./tile_fwk_stest_distributed run "
        cmd += f"--gtest_filter={test_case} --frontend=cpp 2>&1 | tee {self._log_file}"
        TestCaseShellActuator.run(cmd)
        logging.info("Execution finished for test case '%s' of operation '%s'.", self._name, self._op)
        return None
