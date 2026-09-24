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


class DataRange:
    def __init__(self, min, max):
        self._min = min
        self._max = max

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def dump_to_json(self) -> dict:
        return {"min": self._min, "max": self._max}


class TensorDesc:
    def __init__(
        self,
        name: str,
        shape: tuple,
        dtype: str,
        data_range: list,
        tensor_format: str,
        need_trans: bool = False,
    ):
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._data_range = None if data_range is None else DataRange(data_range[0], data_range[1])
        self._tensor_format = tensor_format
        self._need_trans = need_trans

    @classmethod
    def from_dict(cls, params: dict):
        data_range = params.get("data_range", None)
        data_range = data_range if not isinstance(data_range, dict) else [data_range.get("min"), data_range.get("max")]
        return cls(
            params.get("name"),
            params.get("shape"),
            params.get("dtype"),
            data_range,
            params.get("format", "ND"),
            params.get("need_trans", False),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def tensor_format(self) -> str:
        return self._tensor_format

    @property
    def data_range(self) -> DataRange:
        return self._data_range

    @property
    def need_trans(self) -> bool:
        return self._need_trans

    def dump_to_json(self) -> dict:
        json_content = {
            "name": self._name,
            "shape": self._shape,
            "dtype": self._dtype,
            "format": self._tensor_format,
            "need_trans": self._need_trans,
        }
        if self._data_range is not None:
            json_content["data_range"] = self._data_range.dump_to_json()
        return json_content


class TestCaseDesc:
    def __init__(
        self,
        case_index: str,
        case_name: str,
        operation: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        self._case_index = case_index
        self._case_name = case_name
        self._operation = operation
        self._input_tensors = [
            TensorDesc.from_dict(tensor) if isinstance(tensor, dict) else tensor for tensor in input_tensors
        ]
        self._output_tensors = [
            TensorDesc.from_dict(tensor) if isinstance(tensor, dict) else tensor for tensor in output_tensors
        ]
        self._view_shape = view_shape
        self._tile_shape = tile_shape
        self._params = params

    @classmethod
    def from_dict(cls, params: dict):
        return cls(
            params.get("case_index"),
            params.get("case_name"),
            params.get("operation"),
            params.get("input_tensors"),
            params.get("output_tensors"),
            params.get("view_shape"),
            params.get("tile_shape"),
            params.get("params", {}),
        )

    @property
    def index(self) -> str:
        return self._case_index

    @property
    def name(self) -> str:
        return self._case_name

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def input_tensors(self) -> list:
        return self._input_tensors

    @property
    def output_tensors(self) -> list:
        return self._output_tensors

    @property
    def view_shape(self) -> tuple:
        return self._view_shape

    @property
    def tile_shape(self) -> tuple:
        return self._tile_shape

    @property
    def params(self) -> dict:
        return self._params

    def dump_to_json(self) -> dict:
        return {
            "case_index": self._case_index,
            "case_name": self._case_name,
            "operation": self._operation,
            "input_tensors": tuple(map(lambda x: x.dump_to_json(), self._input_tensors)),
            "output_tensors": tuple(map(lambda x: x.dump_to_json(), self._output_tensors)),
            "view_shape": self._view_shape,
            "tile_shape": self._tile_shape,
            "params": self._params,
        }
