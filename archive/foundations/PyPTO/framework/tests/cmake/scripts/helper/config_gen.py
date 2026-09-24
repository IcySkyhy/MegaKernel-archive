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

import abc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class TestBase(abc.ABC):
    def __init__(self):
        super().__init__()
        self.name = "default"
        self.param_name = ""
        self.parameter_types = {}
        self.parameters = {}
        self.inputs = {}
        self.input_tensors = {}
        self.golden_outputs = {}
        self.save_path = ""

    def setup_parameters(self, **kwargs) -> None:
        """
        Specify the data type of parameters in the computation graph.

        Args:
            kwargs (Dict): {"param": type}, dictionary of parameters and type.

        Returns:
            None:

        Examples:
            >>> self.setup_parameters(i = int, j = int, check = bool)
            >>> self.setup_parameters(dtype = type, b = int, scale = float)
        """
        for key, dtype in kwargs.items():
            setattr(self, key, dtype)
            self.parameter_types[key] = dtype

    def load_parameters(self, pack: tuple) -> Dict[str, Any]:
        """
        Load a set of arguments into parameters defined in setup_parameters.

        Args:
            pack (tuple) : tuple of arguments, mapped 1 to 1 with parameters.

        Returns:
            parameters (Dict) : {"param": arg}, dictionary of parameters and arguments.

        Examples:
            >>> param_dict = self.load_parameters((16, 27, True, False))
            >>> return self.load_parameters(pack)
        """
        if self.parameter_types.__len__() != pack.__len__():
            raise ValueError("Parameter count does not match.")
        parameters = {}
        for arg, (name, dtype) in zip(pack, self.parameter_types.items()):
            if isinstance(arg, dtype):
                setattr(self, name, arg)
                parameters[name] = arg
            else:
                raise TypeError(f"Expected type {dtype}, but parameter {name} is type {type(arg)}.")
        return parameters

    def setup_input_tensors(self, tensors: Dict, op_formats: Dict = None) -> None:
        """
        Store input tensors defined in define_input_tensors()

        Args:
            tensors (Dict): input tensors, should be passed by locals()
            op_formats (Dict): {'input_name': op_format(int)}, tensor op format in cpp.
                                    As ND is default format, only NZ tensor need to be identified.
                                    op_format candidate: 0 for ND, 1 for NZ

        Returns:
            None:

        Examples:
            >>> self.setup_input_tensors(locals(), {'w' : 1})
        """
        for name, value in tensors.items():
            if name != 'self':
                if isinstance(value, np.ndarray) and name != 'self':
                    setattr(self, name, value)
                    self.input_tensors[name] = (value, list(value.shape), value.dtype, 0)
                elif isinstance(value, torch.Tensor) and name != 'self':
                    setattr(self, name, value)
                    self.input_tensors[name] = (value, list(value.shape), value.dtype, 0)

            elif value is None:
                pass

        # set op format of input_tensor
        if isinstance(op_formats, dict):
            for key, value in op_formats.items():
                if key in self.input_tensors:
                    tmp = list(self.input_tensors[key])
                    tmp[3] = value
                    self.input_tensors[key] = tuple(tmp)

    def setup_output(self, tensors: Dict) -> None:
        """
        Store computed output tensor in core()

        Args:
            tensors (dict) : {"output_name": tensor}, dictionary of output tensors

        Returns:
            None:

        Examples:
            >>> self.setup_output({'output': output})
        """
        for name, value in tensors.items():
            if isinstance(value, np.ndarray) and name != 'self':
                self.golden_outputs[name] = (value, list(value.shape), value.dtype)
            elif isinstance(value, torch.Tensor) and name != 'self':
                self.golden_outputs[name] = (value, list(value.shape), value.dtype)

    @abc.abstractmethod
    def define_parameters(self, param_set: tuple) -> Dict[str, Any]:
        """
        Need user overwrite!!
        Define parameters in computation graph.
        1. Use self.setup_parameters() to define parameters type.
        2. Use self.load_parameters() to load arguments.

        Args:
           param_set (tuple) : tuple of arguments, directly pass to self.load_parameters().

        Returns:
            parameters (dict) : {"param": arg}, dictionary of parameters and arguments

        Examples:
            >>> self.setup_parameters(i = int, j = int, check = bool)
                return self.load_parameters(param_set)
        """
        pass

    @abc.abstractmethod
    def define_input_tensors(self) -> tuple:
        """
        Need user overwrite!!
        Define input tensors and set op format in cpp
        1. Define all input tensor in computation graph.
           Conditional input should be defined as None when the condition is false and returned in the function end.
        2. Use self.setup_input_tensors() to store input tensors. The default tensor op format is ND (0).
           User can set op format to NZ by passing dict {'input_name': 1} to op_formats parameter
           of self.setup_input_tensors()
        3. Tensors return order must be consistent with the parameters order of core()

        Returns:
            inputs (tuple) : all input tensors.

        Examples:
            >>> x = np.array([1, 2, 3, 4])
                w = np.array([5, 6, 7, 8])
                self.setup_input_tensors(locals(), op_formats={'w': 1})
                return x, w
        """
        pass

    @abc.abstractmethod
    def core(self, tensors) -> None:
        """
        Need user overwrite!!
        Define comuptation to generate golden.
        1. Define computational logic to compute output tensors with input tensors.
        2. Use self.setup_output() to store output tensors and golden data.

        Args:
           tensors : all input tensors, order must be consistent with define_input_tensors() return.

        Returns:
            None:

        Examples:
            >>> def core(self, x, w):
                    output = x * w
                    self.setup_output({'output': output})
        """
        pass

    # set save path of json config
    def set_save_path(self, save_path=None):
        if save_path:
            root_dir = save_path
        else:
            root_dir = os.path.dirname(os.path.abspath(__file__))

        # process dtype in parameters
        dtype_keys = [key for key, value in self.parameters.items() if type(value) in [type, torch.dtype]]
        for key in dtype_keys:
            if hasattr(self.parameters[key], __name__) or isinstance(self.parameters[key], type):
                self.parameters[key] = self.dtype_name(self.parameters[key].__name__)
            elif isinstance(self.parameters[key], torch.dtype):
                self.parameters[key] = self.dtype_name(str(self.parameters[key]))
            else:
                self.parameters[key] = str(self.parameters[key])

        if not self.parameters:
            self.save_path = os.path.join(root_dir, "default_save")
            return self.save_path

        params = []

        for key in self.parameters.keys():
            value = self.parameters[key]
            if type(value) is int:
                params.append(f"{key}{value}")

        params_str = "_".join(params)
        self.param_name = params_str

        self.save_path = os.path.join(root_dir, self.name, params_str)
        os.makedirs(self.save_path, exist_ok=True)
        logging.debug("Golden data save path is:")
        logging.debug(self.save_path)
        return self.save_path

    def tensor_tofile(self, t, output):
        dtype = t.dtype
        if isinstance(t, np.ndarray):
            t.tofile(output)
        elif isinstance(t, torch.Tensor):
            with open(str(output), "wb") as f:
                if dtype == torch.bfloat16:
                    dtype = torch.int16
                for each in t:
                    f.write(each.view(dtype).numpy().tobytes())

    # export metadata to json and tensor data bin
    def save(self):
        combined_data = {
            "name": self.name,
            "param_name": self.param_name,
            "parameters": self.parameters,
            "inputs": {},
            "golden_outputs": {},
        }

        # export input_tensors
        for name, (tensor, shape, dtype, op_format) in self.input_tensors.items():
            bin_filename = f"{name}.bin"
            bin_path = os.path.join(self.save_path, bin_filename)
            self.tensor_tofile(tensor, bin_path)
            entry = {"shape": shape, "dtype": self.dtype_name(str(dtype)), "opFormat": op_format}

            dir_idx = max(i for i, name in enumerate(Path(bin_path).parts) if name == self.name)
            entry["bin_file"] = str(Path(*Path(bin_path).parts[dir_idx + 1:]))
            combined_data["inputs"][name] = entry

        # export golden_outputs
        for name, (tensor, shape, dtype) in self.golden_outputs.items():
            bin_filename = f"{name}.bin"
            bin_path = os.path.join(self.save_path, bin_filename)
            self.tensor_tofile(tensor, bin_path)
            entry = {"shape": shape, "dtype": self.dtype_name(str(dtype)), "opFormat": 0}

            dir_idx = max(i for i, name in enumerate(Path(bin_path).parts) if name == self.name)
            entry["bin_file"] = str(Path(*Path(bin_path).parts[dir_idx + 1:]))
            combined_data["golden_outputs"][name] = entry

        # export json
        json_path = Path(self.save_path).parent
        combined_path = os.path.join(json_path, "config.json")
        with open(combined_path, 'w') as f:
            json.dump(combined_data, f, indent=2)
        logging.debug("===Results saved===")

    def run(self, arguments: tuple, save_path: Path = None) -> None:
        """
        Api for test case execution

        Args:
            arguments (tuple): arguments of parameters defined in self.define_parameters().
            save_path (Path): golden directory.

        Returns:
            None:

        Examples:
            test_derived.run((16, 16, true), golden_dir)
        """
        self.parameters = self.define_parameters(arguments)
        self.inputs = self.define_input_tensors()
        self.set_save_path(save_path)
        self.core(*self.inputs)
        self.save()

    # dtype name transform
    def dtype_name(self, dtype_str):
        if dtype_str in ['int4', 'torch.int4']:  # Numpy not support int4
            return 'DT_INT4'
        elif dtype_str in ['int8', 'torch.int8']:
            return 'DT_INT8'
        elif dtype_str in ['int16', 'torch.int16']:
            return 'DT_INT16'
        elif dtype_str in ['int32', 'torch.int32']:
            return 'DT_INT32'
        elif dtype_str in ['int64', 'int_', 'torch.int64']:
            return 'DT_INT64'
        elif dtype_str in ['float8', 'torch.float8']:  # Numpy not support float8
            return 'DT_FP8'
        elif dtype_str in ['float16', 'half', 'torch.float16', 'numpy.float16']:
            return 'DT_FP16'
        elif dtype_str in ['float32', 'torch.float32']:
            return 'DT_FP32'
        elif dtype_str in ['float64', 'torch.float64']:
            return 'DT_DOUBLE'
        elif dtype_str in ['bfloat16', 'torch.bfloat16']:
            return 'DT_BF16'
        elif dtype_str == 'half4':  # Undefined
            return 'DT_HF4'
        elif dtype_str == 'half8':  # Undefined
            return 'DT_HF8'
        elif dtype_str in ['uint8', 'torch.uint8']:
            return 'DT_UINT8'
        elif dtype_str in ['uint16', 'torch.uint16']:
            return 'DT_UINT16'
        elif dtype_str in ['uint32', 'torch.uint32']:
            return 'DT_UINT32'
        elif dtype_str in ['uint64', 'torch.uint64']:
            return 'DT_UINT64'
        elif dtype_str in ['bool_', 'torch.bool']:
            return 'DT_BOOL'
        else:
            return 'UNDEFINED'
