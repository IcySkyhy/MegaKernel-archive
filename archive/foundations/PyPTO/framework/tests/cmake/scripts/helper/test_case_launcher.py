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

from glob import glob
import json
import os
from pathlib import Path
import pkgutil
import sys

import pytest
from test_case_loader import TestCaseLoader
from test_case_log_analyzer import TestCaseLogAnalyzer
from test_case_logger import TestCaseLogger
from test_case_shell_actuator import TestCaseShellActuator


class TestCaseLauncher:
    def __init__(self, config):
        self.work_path = os.getcwd()
        self.input_file = os.path.abspath(config.input_file)
        self.op = config.op
        self.index = [config.start_index, config.end_index]
        self.report_file = os.path.abspath(config.report)
        self.device = config.device
        self.python = config.python
        self.model = config.model
        self.json_only = config.json_only
        self.clean = config.clean
        self.save_data = config.save_data
        self.log_path = os.path.dirname(self.report_file) + "/test_case_log"
        self.plog_cache_path = f"{self.work_path}/plog"
        self.golden_script = Path(config.golden_script).resolve()
        self.json_path = os.path.abspath(config.json_path)
        self.distributed_op = config.distributed_op  # 通信算子标识
        self.executable_path = config.executable_path

    def tear_up(self):
        if os.path.exists(self.log_path):
            os.system(f"rm -rf {self.log_path}")
        os.mkdir(self.log_path)
        if os.path.exists(self.plog_cache_path):
            os.system(f"rm -rf {self.plog_cache_path}")
        os.mkdir(self.plog_cache_path)
        if os.path.exists(self.report_file):
            os.remove(self.report_file)

        os.environ["TILE_FWK_DEVICE_ID"] = f"{self.device}"
        os.environ["ASCEND_PROCESS_LOG_PATH"] = self.plog_cache_path
        os.environ["JSON_PATH"] = self.json_path

        if not self.python:
            golden_dir = str(self.golden_script.parent)
            if golden_dir not in sys.path:
                sys.path.append(golden_dir)
            module_name = f"{self.golden_script.stem}"
            import importlib

            importlib.import_module(module_name)
            sys.path.remove(golden_dir)

    def tear_down(self):
        del os.environ["TILE_FWK_DEVICE_ID"]
        del os.environ["ASCEND_PROCESS_LOG_PATH"]
        del os.environ["JSON_PATH"]

    def compile_if_need(self):
        clean_str = "-c" if self.clean else ""
        cmd = f"{sys.executable} build_ci.py {clean_str}"
        if self.python:
            cmd += " -f=python3"
        elif not self.distributed_op:
            cmd += " -f=cpp -s='TestAdd/AddOperationTest.TestAdd/*' --disable_auto_execute"
        else:
            cmd += " -f=cpp --stest_distributed='TestDistributedOps/DistributedTest.TestOps/*' --disable_auto_execute"
        TestCaseShellActuator.run(cmd)

        if self.python:
            pypto_pkg = f"{self.work_path}/build_out/pypto-*.whl"
            if len(glob(pypto_pkg)) == 0:
                raise FileNotFoundError("Not found pypto install package.")
            os.system(f"{sys.executable} -m pip install --upgrade --no-deps --force-reinstall {pypto_pkg}")

    def run_test_case(self, test_case_info):
        log_file = f"{self.log_path}/{test_case_info['case_name']}.log"
        test_case_info["log_file"] = log_file
        base_path = Path(__file__).parents[3]
        launcher_patch = base_path / ("st/operation/python" if not self.distributed_op else "st/distributed/ops/script")
        launcher_patch = launcher_patch.resolve()
        if str(launcher_patch) not in sys.path:
            sys.path.append(str(launcher_patch))
        if not self.distributed_op:
            from vector_operation_test_case_launcher import test_case_launcher
        else:
            from distributed_test_case_launcher import test_case_launcher

        test_case_launcher(test_case_info)

    def gen_test_report(self, test_case_info: dict):
        case_index = test_case_info["case_index"]
        case_name = test_case_info["case_name"]
        case_op = test_case_info["operation"]
        # test report(excel file)
        log_file = f"{self.log_path}/{case_name}.log"
        log_path = f"{self.log_path}/{case_op}/{case_index}"
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        analyzer = TestCaseLogAnalyzer(case_index, case_name, case_op, log_file, self.report_file)
        is_pass = analyzer.run()
        if not is_pass or self.save_data:
            os.system(f"cp -rf {self.plog_cache_path} {log_path}")
        os.system(f"mv {log_file} {log_path}")
        os.system(f"rm -rf {self.plog_cache_path}")

    def run_pto_test_case(self, test_case_info):
        case_name = test_case_info["case_name"]
        logger = TestCaseLogger(f"{self.log_path}/{case_name}.log")
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = logger
        sys.stderr = logger
        pytest.main(
            [
                "-vs",
                "--test_case_info",
                json.dumps(test_case_info),
                "python/tests/st/operation/vector_operation_test_case_launcher.py",
            ]
        )
        sys.stdout = stdout
        sys.stderr = stderr

    def run(self):
        self.tear_up()
        test_case_info_list = TestCaseLoader(self.input_file, self.op, self.index, self.model, self.json_path).run()
        if self.json_only:
            return

        self.compile_if_need()
        is_package_ready = self.python and pkgutil.find_loader("pypto")
        if not self.distributed_op:
            stest_exec_file = f"{self.work_path}/build/output/bin/tile_fwk_stest"
        else:
            stest_exec_file = f"{self.work_path}/build/output/bin/tile_fwk_stest_distributed"
        if not self.executable_path:
            self.executable_path = stest_exec_file
        is_exec_ready = not self.python and os.path.exists(self.executable_path)
        if not is_package_ready and not is_exec_ready:
            raise RuntimeError("Runtime time is not ready, Not found package pypto or tile_fwk_stest.")
        for test_case_info in test_case_info_list:
            # run test
            (self.run_pto_test_case(test_case_info) if self.python else self.run_test_case(test_case_info))
            # generate test report
            self.gen_test_report(test_case_info)
            # clear golden data
            index = test_case_info["index"]
            case_op = test_case_info["operation"]
            if not self.distributed_op:
                test_case = f"Test{case_op}/{case_op}OperationTest.Test{case_op}/{index}"
            else:
                test_case = f"TestDistributedOps/{case_op}"
            golden_path = f"{self.work_path}/build/output/bin/golden/{test_case}"
            if os.path.exists(golden_path + "/golden_desc.json"):
                os.remove(golden_path + "/golden_desc.json")
            if not self.save_data:
                os.system(f"rm -rf {golden_path}")
        self.tear_down()
