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

from abc import ABC, abstractmethod
from typing import NoReturn

import torch


class TestCaseRunner(ABC):
    def __init__(self, view_shape: tuple, tile_shape: tuple, params: dict):
        self._view_shape = view_shape
        self._tile_shape = tile_shape
        self._params = params

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params.get("view_shape"), params.get("tile_shape"), params.get("params", {}))

    @abstractmethod
    def input_tensors(self) -> list:
        """return the tensors used as the inputs of dynamic function"""
        return None

    @abstractmethod
    def input_data(self) -> list:
        """return the input data used to run on device"""
        return None

    @abstractmethod
    def output_tensors(self) -> list:
        """return the tensors used as the outputs of dynamic function"""
        return None

    @abstractmethod
    def exec_dyn_func(self, input_tensors: list, output_tensors: list) -> NoReturn:
        """build the dynamic function and exec"""
        pass

    @abstractmethod
    def run_on_device(self, inputs: list) -> list:
        """call the func to obtain output data from the device"""
        return None

    @property
    def view_shape(self) -> tuple:
        """return the view shape of test case"""
        return self._view_shape

    @property
    def tile_shape(self) -> tuple:
        """return the tile shape of test case"""
        return self._tile_shape

    def reg_op_func(self, op_func: callable):
        """the parameters of op_func are list[Tensor] and params(dict)"""
        self._op_func = op_func

    def reg_golden_func(self, golden_func, binary_compare: bool = False) -> NoReturn:
        self._golden_func = golden_func
        self.binary_compare = binary_compare

    def tear_up(self) -> NoReturn:
        pass

    def tear_down(self) -> NoReturn:
        pass

    def result_golden_compare(self, golden, result, is_binary: bool = False) -> NoReturn:
        if golden is None and result is None:
            return

        assert len(golden) == len(result)
        for g_data, res_data in zip(golden, result):
            if is_binary:
                assert torch.equal(g_data, res_data)
            else:
                assert torch.allclose(g_data, res_data, rtol=0.005, atol=0)

    def run(self) -> NoReturn:
        self.tear_up()
        input_tensors = self.input_tensors()
        inputs = self.input_data()
        inputs = [torch.tensor(input).reshape(tensor.shape) for input, tensor in zip(inputs, input_tensors)]
        golden = self._golden_func(inputs, self._params)
        output_tensors = self.output_tensors()
        self.exec_dyn_func(input_tensors, output_tensors)
        outputs = self.run_on_device(inputs)
        self.result_golden_compare(golden, outputs, self.binary_compare)
        self.tear_down()
