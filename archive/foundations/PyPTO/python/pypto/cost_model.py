#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

from typing import List

import torch

import pypto

from . import pypto_impl
from .converter import _gen_pto_tensor

__all__ = [
    "_cost_model_run_once_data_from_host",
]


def _pto_to_tensor_data(tensors: List[pypto.Tensor]) -> List[pypto_impl.DeviceTensorData]:
    datas = []
    for t in tensors:
        if t.ori_shape is None:
            raise RuntimeError("The ori_shape of the tensor is not specified.")
        data = pypto_impl.DeviceTensorData(
            t.dtype,
            t.data_ptr,
            list(t.ori_shape),
        )
        datas.append(data)
    return datas


def _device_to_host_tensor_datas(dev_tensors):
    host_tensors, torch_tensors = _gen_pto_tensor(dev_tensors)
    host_tensor_datas = _pto_to_tensor_data(host_tensors)
    for i, dev_tensor_data in enumerate(_pto_to_tensor_data(dev_tensors)):
        pypto_impl.CopyToHost(dev_tensor_data, host_tensor_datas[i])
    return host_tensor_datas, torch_tensors


def _host_to_device_tensor_datas(dev_tensors, host_tensors):
    for i, dev_tensor_data in enumerate(_pto_to_tensor_data(dev_tensors)):
        pypto_impl.CopyToDev(dev_tensor_data, host_tensors[i])


def _cost_model_run_once_data_from_host(inputs: List[pypto.Tensor], outputs: List[pypto.Tensor]):
    is_device = False
    for t in inputs:
        if t.device != torch.device("cpu"):
            is_device = True
            break

    host_torch_tensors = []
    pypto_impl.ExchangeCaptureModeRelax()
    if is_device:
        input_datas, input_host_torch_tensors = _device_to_host_tensor_datas(inputs)
        output_datas, output_host_torch_tensors = _device_to_host_tensor_datas(outputs)
        host_torch_tensors.extend(input_host_torch_tensors)
        host_torch_tensors.extend(output_host_torch_tensors)
    else:
        input_datas = _pto_to_tensor_data(inputs)
        output_datas = _pto_to_tensor_data(outputs)

    pypto_impl.CostModelRunOnceDataFromHost(input_datas, output_datas)

    if is_device:
        _host_to_device_tensor_datas(inputs, input_datas)
        _host_to_device_tensor_datas(outputs, output_datas)
