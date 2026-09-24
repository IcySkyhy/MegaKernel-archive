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

import logging
import os
import re

import pandas as pd


class TestCaseResult:
    def __init__(self, index: int, name: str, op: str):
        self._index = index
        self._name = name
        self._is_pass = False
        self._duration = ""
        self._op = op
        self._detail = ""

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_pass(self) -> bool:
        return self._is_pass

    @is_pass.setter
    def is_pass(self, is_pass: bool):
        self._is_pass = is_pass

    @property
    def status(self) -> str:
        return "PASS" if self._is_pass else "FAILED"

    @property
    def duration(self) -> str:
        return self._duration

    @duration.setter
    def duration(self, duration: str):
        self._duration = duration

    @property
    def op(self) -> str:
        return self._op

    @property
    def detail(self) -> str:
        return self._detail

    @detail.setter
    def detail(self, detail: str):
        self._detail = detail

    def dump_to_excel(self):
        return {
            "index": [self.index],
            "case_name": [self.name],
            "status": [self.status],
            "duration": [self.duration],
            "operation": [self.op],
            "detail": [self.detail],
        }


class TestCaseLogAnalyzer:
    def __init__(self, case_index: str, case_name: str, op: str, log_file: str, report_file: str):
        self.case_index = case_index
        self.case_name = case_name
        self.case_op = op
        self._log_file = log_file
        self._report_file = report_file

    @staticmethod
    def get_value_by_pattern(pattern, line: str):
        match = re.search(pattern, line)
        if match:
            return match["value"].strip()
        else:
            logging.error(f"{pattern} is no match in {line}.")
            return ""

    @staticmethod
    def get_duration(line: str):
        pattern = r"\((?P<value>\d+?) ms total\)"
        return TestCaseLogAnalyzer.get_value_by_pattern(pattern, line)

    def parse_log_file(self):
        test_result = TestCaseResult(self.case_index, self.case_name, self.case_op)
        if not os.path.exists(self._log_file):
            return test_result
        with open(self._log_file, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if "[  PASSED  ] 1 test" in line or "1 passed" in line:
                    test_result.is_pass = True
                elif "[  FAILED  ]" in line or "1 failed" in line:
                    test_result.is_pass = False
                    break
                elif "ms total" in line:
                    test_result.duration = TestCaseLogAnalyzer.get_duration(line)

        return test_result

    def generate_excel_report(self, result: TestCaseResult) -> bool:
        report_data = result.dump_to_excel()
        data_frame = pd.DataFrame(report_data)
        if not os.path.exists(self._report_file):
            df = pd.DataFrame(list(report_data.keys()))
            df.to_excel(self._report_file, sheet_name=result.op)
        else:
            with pd.ExcelFile(self._report_file) as excel_file:
                if result.op in excel_file.sheet_names:
                    data_frame = pd.concat(
                        [pd.read_excel(excel_file, sheet_name=result.op), data_frame],
                        ignore_index=True,
                    )

        with pd.ExcelWriter(self._report_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            data_frame.to_excel(
                writer,
                sheet_name=result.op,
                index=False,
            )
        return result.is_pass

    def run(self) -> bool:
        test_case_result = self.parse_log_file()
        return self.generate_excel_report(test_case_result)
