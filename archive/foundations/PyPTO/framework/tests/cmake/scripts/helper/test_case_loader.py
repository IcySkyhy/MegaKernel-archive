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

from dataclasses import dataclass
import json
import logging
import os
import pathlib
from typing import Callable, Dict, List, Union

import pandas as pd
from test_case_desc import TensorDesc, TestCaseDesc
from test_case_tools import parse_list_str, str_to_bool


@dataclass
class MatmulParam:
    trans_list: list
    input_format_list: list
    output_format_list: list
    row_data: dict
    output_dtype: str
    is_k_split: bool


class TestCaseCreator:
    def __init__(self, case_index: int, case_data, json_path: str):
        self._case_index = case_index
        self._case_data = case_data
        self._json_path = json_path

    @staticmethod
    def extend_matmul_param(matmulparam: MatmulParam, params: dict):
        if matmulparam.row_data.get("operation") not in (
            "Matmul",
            "BatchMatmul",
            "MatmulVerify",
            "BatchMatmulVerify",
        ):
            return
        params["transA"] = matmulparam.trans_list[0]
        params["transB"] = matmulparam.trans_list[1]
        params["isAMatrixNz"] = matmulparam.input_format_list[0] == "NZ"
        params["isBMatrixNz"] = matmulparam.input_format_list[1] == "NZ"
        params["isCMatrixNz"] = matmulparam.output_format_list[0] == "NZ"
        output_dtype_str = matmulparam.output_dtype
        params["outDtype"] = str(output_dtype_str).strip()
        params["func_id"] = 0
        params["enableKSplit"] = matmulparam.is_k_split

    @staticmethod
    def parse_input_tensors(data: dict) -> list:
        input_shape = parse_list_str(data.pop("input_shape"))
        if not isinstance(input_shape[0], (list, tuple)):
            input_shape = [input_shape]
        input_dtype = parse_list_str(data.pop("input_dtype"))
        data_range = parse_list_str(data.pop("input_datarange"))
        if not isinstance(data_range[0], (list, tuple)):
            data_range = [data_range]
        assert len(input_shape) == len(input_dtype)
        assert len(input_shape) == len(data_range)

        input_format_list = parse_list_str(data.pop("input_format"))
        assert len(input_format_list) == len(input_shape)

        input_trans = data.pop("input_trans", str([False] * len(input_shape)))
        input_trans = parse_list_str(input_trans)
        input_trans = [str_to_bool(item) for item in input_trans]

        input_tensors = []
        for idx, dim in enumerate(input_shape):
            input_tensors.append(
                TensorDesc(
                    "input" + str(idx),
                    dim,
                    input_dtype[idx],
                    data_range=data_range[idx],
                    tensor_format=input_format_list[idx],
                    need_trans=input_trans[idx],
                )
            )

        return input_tensors

    @staticmethod
    def parse_output_tensors(data: dict) -> list:
        output_shape = parse_list_str(data.pop("output_shape"))
        if not isinstance(output_shape[0], (list, tuple)):
            output_shape = [output_shape]
        output_dtype = parse_list_str(data.pop("output_dtype"))
        output_format_list = parse_list_str(data.pop("output_format"))
        assert len(output_format_list) == len(output_shape)

        output_tensors = []
        for idx, dim in enumerate(output_shape):
            output_tensors.append(
                TensorDesc(
                    "output" + str(idx),
                    dim,
                    output_dtype[idx],
                    data_range=None,
                    tensor_format=output_format_list[idx],
                    need_trans=False,
                )
            )

        return output_tensors

    def convert_row_data(self, row):
        row_data = row.to_dict()
        input_tensors = self.parse_input_tensors(row_data)
        output_tensors = self.parse_output_tensors(row_data)

        view_shape = parse_list_str(row_data.pop("view_shape"))
        if isinstance(view_shape[0], (list, tuple)) and len(view_shape[0]) > 1:
            view_shape = view_shape[0]
        tile_shape = parse_list_str(row_data.pop("tile_shape"))
        params = {k: None if pd.isna(v) or pd.isnull(v) else v for k, v in row_data.items()}
        # case_index, case_name, operation not need
        params.pop("case_index")
        params.pop("case_name")
        params.pop("operation")
        params["func_id"] = int(params.pop("func_id", "-1"))
        TestCaseLoader.get_params_handler(row_data.get("operation"))(params)

        is_k_split = False
        enable_k_split = row_data.pop("enableKSplit", None)
        if enable_k_split is not None:
            is_k_split = str_to_bool(enable_k_split)
        matmulparam = MatmulParam(
            [tensor.need_trans for tensor in input_tensors],
            [tensor.tensor_format for tensor in input_tensors],
            [tensor.tensor_format for tensor in output_tensors],
            row_data,
            output_tensors[0].dtype,
            is_k_split,
        )
        TestCaseCreator.extend_matmul_param(matmulparam, params)

        return TestCaseDesc(
            row_data.get("case_index"),
            row_data.get("case_name"),
            row_data.get("operation"),
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
        )

    def dump_to_json(self, write_to_json: bool = True):
        row_data = self.convert_row_data(self._case_data).dump_to_json()
        test_case = {"test_case": row_data}
        if write_to_json:
            json_file = f"{self._json_path}/{self._case_data['case_name']}.json"
            try:
                with open(json_file, "w", encoding="utf-8") as outfile:
                    json.dump(row_data, outfile, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error("Exception occur when writing %s, exception is %s.", json_file, e)
            test_case["json_file"] = json_file
        return test_case


class FileReader:
    def __init__(self, file_name: str, op: str, index_range: list, json_path: str):
        self._file_name = file_name
        self._op = None if op == "*" or op.lower() == "all" else [op]
        self._start_index = index_range[0]
        self._end_index = index_range[1]
        self._json_path = json_path
        self._data_frames = []

    def run(self) -> list:
        if not os.path.exists(self._file_name):
            logging.error(f"Process File {self._file_name} failed, file not exist.")
            return None

        data_frames = (
            self.load_test_cases_from_csv() if self._file_name.endswith(".csv") else self.load_test_cases_from_excel()
        )

        if data_frames is None or len(data_frames) == 0:
            return []

        data_frames = [self.test_case_data_cleaning(data_frame, self._op[0]) for data_frame in data_frames]
        return pd.concat(
            data_frames,
            ignore_index=True,
        )

    def load_test_cases_from_csv(self) -> list:
        data_frame = pd.read_csv(self._file_name)
        if "operation" not in data_frame.columns:
            if not isinstance(self._op, list) or len(self._op) != 1:
                raise ValueError("Must set operation for test cases.")
            data_frame["operation"] = self._op[0]
        return [data_frame]

    def load_test_cases_from_excel(self) -> list:
        data_frames = []
        file_handler = pd.ExcelFile(self._file_name)
        sheet_names = self._op if self._op is not None else list(file_handler.sheet_names)
        for sheet_name in sheet_names:
            df = pd.read_excel(file_handler, sheet_name=sheet_name)
            if "operation" not in df.columns:
                df["operation"] = sheet_name
            data_frames.append(df)
        file_handler.close()
        return data_frames

    def test_case_data_cleaning(
        self,
        data_frame: pd.DataFrame,
        op: str,
    ) -> pd.DataFrame:
        if "case_index" not in data_frame.columns:
            data_frame.loc[:, "case_index"] = data_frame.index
        if self._start_index < 0:
            self._start_index = 0
        case_cnt = len(data_frame)
        if self._start_index >= case_cnt:
            logging.info(f"The start index [{self._start_index}] exceeds the max index[{case_cnt - 1}].")
            return False
        if self._end_index < 0 or self._end_index >= case_cnt:
            self._end_index = case_cnt

        data_frame = data_frame.iloc[self._start_index:self._end_index + 1]
        if "skip" in data_frame.columns:
            data_frame = data_frame.query("skip != 1 and skip != '1' and skip != True and skip != 'TRUE'")
        if "enable" in data_frame.columns:
            data_frame = data_frame.query("(enable == 1 or enable == '1' or enable == True or enable == 'TRUE')")
        data_frame.query(f"operation == '{op}'")
        return data_frame


class JsonWriter:
    def __init__(self, data_frame: pd.DataFrame, json_path: str, cur_index: int):
        self._data = data_frame
        self._json = json_path
        self._cur_index = cur_index

    def run(self) -> list:
        if len(self._data) == 0:
            return []

        test_cases = []
        for index, row_data in self._data.iterrows():
            creator = TestCaseCreator(row_data["case_index"], row_data, self._json)
            case_info = creator.dump_to_json(False)
            case_info["test_case"]["index"] = self._cur_index + index
            test_cases.append(case_info["test_case"])
        test_cases.sort(key=lambda x: (x["operation"], x["case_index"]))
        path = pathlib.Path(self._json)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            json_file = path / f"{test_cases[0]['operation']}_st_test_cases.json"
        else:
            json_file = path
        row_data = {"test_cases": test_cases}
        with open(json_file, "w", encoding="utf-8") as outfile:
            json.dump(row_data, outfile, ensure_ascii=False, indent=4)
        return test_cases


class TestCaseLoader:
    def __init__(self, file_path: str, op: str, index_range: list, model: bool, json_path: str):
        self._path = file_path
        self._op = op
        self._index_range = index_range
        self._model = model
        self._json_path = json_path

    # 全局回调函数注册表
    _REG_MAP: Dict[str, callable] = {}

    @classmethod
    def reg_params_handler(cls, ops: Union[str, List[str]]) -> Callable:
        def decorator(func: Callable) -> Callable:
            op_list = [ops] if isinstance(ops, str) else ops
            for op in op_list:
                cls._REG_MAP[op] = func
            return func

        return decorator

    @classmethod
    def get_params_handler(cls, op: str) -> Callable:
        """根据名称获取回调函数"""
        return cls._REG_MAP.get(op, lambda params: params)

    def run(self) -> list:
        all_test_cases = []
        cur_index = 0
        if os.path.isdir(self._path):
            files = sorted(
                [f for f in os.listdir(self._path) if f.endswith((".csv", ".xlsx", ".xls"))],
                key=lambda x: x.lower(),
            )
            for file in files:
                file_path = os.path.join(self._path, file)
                json_path = os.path.join(self._json_path, f"{os.path.splitext(os.path.basename(file_path))[0]}.json")
                test_cases = self.__process_file_to_json(file_path, json_path, cur_index)
                all_test_cases.extend(test_cases)
                cur_index += len(test_cases)
        else:
            test_cases = self.__process_file_to_json(self._path, self._json_path, cur_index)
            all_test_cases.extend(test_cases)
        return all_test_cases

    def __process_file_to_json(self, file_path: str, json_path: str, cur_index: int) -> List[dict]:
        data_frame = FileReader(file_path, self._op, self._index_range, self._json_path).run()
        if data_frame is None or len(data_frame) == 0:
            return []
        data_frame["on_board"] = not self._model
        test_cases = JsonWriter(data_frame, json_path, cur_index).run()
        return test_cases
