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
from math import ceil, prod
from pathlib import Path
import sys

import numpy as np
import torch

import pypto

helper_path: Path = Path(
    Path(__file__).parent.parent.parent.parent.parent.parent,
    "framework/tests/cmake/scripts/helper",
).resolve()
if str(helper_path) not in sys.path:
    sys.path.append(str(helper_path))
from test_case_desc import TensorDesc  # noqa: E402
from test_case_runner import TestCaseRunner  # noqa: E402
from test_case_tools import get_dtype_by_name  # noqa: E402


def get_pto_dtype_by_name(name: str):
    str_to_dtype = {
        "int4": pypto.DT_INT4,
        "int8": pypto.DT_INT8,
        "int16": pypto.DT_INT16,
        "int32": pypto.DT_INT32,
        "int64": pypto.DT_INT64,
        "fp8": pypto.DT_FP8,
        "fp16": pypto.DT_FP16,
        "fp32": pypto.DT_FP32,
        "hf4": pypto.DT_HF4,
        "hf8": pypto.DT_HF8,
        "fp8e4m3": pypto.DT_FP8E4M3,
        "fp8e5m2": pypto.DT_FP8E5M2,
        "fp8e8m0": pypto.DT_FP8E8M0,
        "uint8": pypto.DT_UINT8,
        "uint16": pypto.DT_UINT16,
        "uint32": pypto.DT_UINT32,
        "uint64": pypto.DT_UINT64,
        "bool": pypto.DT_BOOL,
        "double": pypto.DT_DOUBLE,
        "bf16": pypto.DT_BF16,
    }
    return str_to_dtype.get(name, pypto.DT_FP32)


class PTOTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        operation: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(view_shape, tile_shape, params)
        self._operation = operation
        self._input_tensors = [
            TensorDesc.from_dict(tensor) if isinstance(tensor, dict) else tensor for tensor in input_tensors
        ]
        self._output_tensors = [
            TensorDesc.from_dict(tensor) if isinstance(tensor, dict) else tensor for tensor in output_tensors
        ]

    def gen_loop_range_tuple(self):
        if len(self._input_tensors[0].shape) != len(self._view_shape):
            raise ValueError("The lengths of input tensors and view shape are not same.")
        return tuple(
            [
                ceil(self._input_tensors[0].shape[index] / self._view_shape[index])
                for index in list(range(len(self._view_shape)))
            ]
        )

    def input_tensors(self):
        return [
            pypto.tensor(
                input_tensor.shape,
                get_pto_dtype_by_name(input_tensor.dtype),
                input_tensor.name,
            )
            for input_tensor in self._input_tensors
        ]

    def input_data(self):
        input_data = []
        for input_tensor in self._input_tensors:
            min_value = input_tensor.data_range.min
            max_value = input_tensor.data_range.max
            data = None
            if min_value != max_value:
                data = np.random.uniform(min_value, max_value, prod(input_tensor.shape)).astype(
                    get_dtype_by_name(input_tensor.dtype)
                )
            else:
                data = np.full(
                    prod(input_tensor.shape),
                    max_value,
                    dtype=get_dtype_by_name(input_tensor.dtype),
                )

            input_data.append(torch.from_numpy(data))
        return input_data

    def output_tensors(self):
        return [
            pypto.tensor(
                output_tensor.shape,
                get_pto_dtype_by_name(output_tensor.dtype),
                output_tensor.name,
            )
            for output_tensor in self._output_tensors
        ]

    def output_data(self):
        return [
            torch.from_numpy(
                np.full(
                    prod(output_tensor.shape),
                    0,
                    dtype=get_dtype_by_name(output_tensor.dtype),
                )
            )
            for output_tensor in self._output_tensors
        ]

    def exec_dyn_func(self, input_tensors: list, output_tensors: list):
        loop_range_tuple = self.gen_loop_range_tuple()
        function = "import pypto\n"
        function += """import os\n"""
        function += """import torch\n"""
        function += """import torch_npu\n"""
        function += "\n"
        function += """device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))\n"""
        function += """torch.npu.set_device(device_id)\n"""
        function += "\n"
        function += f"with pypto.function('{self._operation}', *input_tensors, *output_tensors):\n"
        tab = "    "
        prefix = tab
        for index, value in enumerate(loop_range_tuple):
            function += prefix + (tab * index)
            function += f"for index_{index} in pypto.loop({value}):\n"
        loop_range_len = len(loop_range_tuple)
        prefix = tab * (loop_range_len + 1)
        function += prefix + "input_data = []\n"
        view_offset = [f"index_{index} * {self._view_shape[index]}" for index, _ in enumerate(loop_range_tuple)]
        for index, tensor in enumerate(input_tensors):
            function += prefix
            view_shape = [min(dim, view_dim) for dim, view_dim in zip(tensor.shape, self._view_shape)]
            view_offset = ["0" if dim == 1 else offset for dim, offset in zip(tensor.shape, view_offset)]
            function += f"input_{index} = pypto.view(input_tensors[{index}], {view_shape}, ["
            for offset in view_offset:
                function += offset + ", "
            function += "])\n"
            function += prefix + f"input_data.append(input_{index})\n"
        function += prefix + "res = []\n"
        function += prefix + "for _ in enumerate(output_tensors):\n"
        function += prefix + "    res.append(pypto.tensor())\n"
        function += prefix + "if len(res) == 1:\n"
        function += prefix + "    res[0].move(op_func(input_data, params))\n"
        function += prefix + "else:\n"
        function += prefix + "    for dst_, src_ in zip(res, op_func(input_data, params)):\n"
        function += prefix + "        dst_.move(src_)\n"
        if self._operation == "Transpose":
            (
                view_offset[self._params["first_dim"]],
                view_offset[self._params["second_dim"]],
            ) = (
                view_offset[self._params["second_dim"]],
                view_offset[self._params["first_dim"]],
            )
        function += prefix + "for dst_, src_ in zip(output_tensors, res):\n"
        function += prefix + "    pypto.assemble(src_, ["
        for offset in view_offset:
            function += offset + ", "
        function += "], dst_)\n"

        function += prefix + "for input in input_data:\n"
        function += prefix + "    del input\n"
        function += prefix + "for tmp in res:\n"
        function += prefix + "    del tmp\n"
        logging.info(function)
        pypto.set_vec_tile_shapes(*self.tile_shape)
        exec(
            function,
            {
                "input_tensors": input_tensors,
                "output_tensors": output_tensors,
                "op_func": self._op_func,
                "params": self._params,
            },
        )

    def tear_up(self):
        pypto.runtime._device_init()

    def tear_down(self):
        pypto.runtime._device_fini()

    def run_on_device(self, inputs: list) -> list:
        output = self.output_data()

        pto_inputs_tensor = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_output_tensor = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(output)]

        pypto.runtime._device_run_once_data_from_host(*pto_inputs_tensor, *pto_output_tensor)
        return [
            torch.tensor(
                output[index],
                dtype=get_dtype_by_name(self._output_tensors[index].dtype, True),
            ).reshape(self._output_tensors[index].shape)
            for index in list(range(len(output)))
        ]
