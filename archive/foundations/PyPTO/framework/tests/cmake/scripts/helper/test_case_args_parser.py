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

import argparse
import logging
import os
import pathlib


class TestCaseArgsParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="Run operation st test case.", epilog="Best Regards!")

    @staticmethod
    def dump_test_case_args(args):
        logging.info("Run operation test case args :")
        logging.info("Op : %s", args.op)
        logging.info("Input data : %s", args.input_file)
        logging.info("Start index : %s", args.start_index)
        logging.info("End index : %s", args.end_index)
        logging.info("Device : %s", args.device)
        logging.info("Clean : %s", args.clean)
        logging.info("Is python case : %s", args.python)
        logging.info("Save data : %s", args.save_data)
        logging.info("Is json only : %s", args.json_only)
        logging.info("Report : %s", args.report)
        logging.info("json_path : %s", args.json_path)
        logging.info("Is distributed op : %s", args.distributed_op)

    @staticmethod
    def update_default_value(args):
        if args.op == "distributed_op":
            args.distributed_op = True
        if args.distributed_op:
            op_path = f"{os.getcwd()}/framework/tests/st/distributed/ops"
            default_golden = f"{op_path}/script/distributed_golden.py"
            if args.input_file is None:
                raise ValueError("The input_file argument must be provided for distributed_op.")
        else:
            op_path = f"{os.getcwd()}/framework/tests/st/operation"
            default_golden = f"{op_path}/python/vector_operator_golden.py"
            if args.input_file is None:
                args.input_file = f"{op_path}/test_case/{args.op}_st_test_cases.csv"
        if not os.path.exists(args.input_file):
            raise ValueError(args.input_file + " is not exists.")
        if args.golden_script is None:
            args.golden_script = default_golden
        if args.json_path is None:
            if args.distributed_op:
                input_file_path = pathlib.Path(args.input_file)
                if input_file_path.is_file():
                    args.json_path = input_file_path.with_suffix(".json")
                else:
                    args.json_path = input_file_path
            else:
                args.json_path = f"{op_path}/test_case/"

    def add_test_case_args(self):
        # 参数注册
        self._parser.add_argument("op", type=str, help="The operation will be test.")
        self._parser.add_argument(
            "-i",
            "--input_file",
            type=str,
            default=None,
            help="The input test case data file.",
        )
        self._parser.add_argument(
            "-s",
            "--start_index",
            nargs="?",
            type=int,
            default=0,
            help="The start index of test case, it will be row id sub 2 for csv or excel.",
        )
        self._parser.add_argument(
            "-e",
            "--end_index",
            nargs="?",
            type=int,
            default=-1,
            help="The end index of test case,"
            " it will be reset to the max index when it is less than 0 or greater than the max.",
        )
        self._parser.add_argument(
            "--report",
            nargs="?",
            type=str,
            default="test_result_report.xlsx",
            help="The report of test case result.",
        )
        self._parser.add_argument(
            "--golden_script",
            nargs="?",
            type=str,
            default=None,
            help="The report of test case result.",
        )

    def add_build_args(self):
        self._parser.add_argument(
            "-c",
            "--clean",
            action="store_true",
            help="clean the compile result.",
        )

    def add_run_args(self):
        self._parser.add_argument(
            "-d",
            "--device",
            nargs="?",
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            help="Select npu id.",
        )

        self._parser.add_argument("--python", action="store_true", help="Test python test case.")
        self._parser.add_argument("--model", action="store_true", help="Run test case with model.")
        self._parser.add_argument(
            "--json_only",
            action="store_true",
            help="Convert csv to json only, not run.",
        )
        self._parser.add_argument("--save_data", action="store_true", help="Save golden and plog etc.")
        self._parser.add_argument("--distributed_op", action="store_true", help="corresponding distributed operator.")
        self._parser.add_argument(
            "--json_path",
            type=str,
            default=None,
            help="Path to save the converted JSON or path to the input JSON file.",
        )
        self._parser.add_argument(
            "--executable_path",
            type=str,
            default=None,
            help="Path to executable file.",
        )

    def run(self):
        """主处理流程"""
        self.add_test_case_args()
        self.add_build_args()
        self.add_run_args()

        args = self._parser.parse_args()
        TestCaseArgsParser.update_default_value(args)
        TestCaseArgsParser.dump_test_case_args(args)
        return args
