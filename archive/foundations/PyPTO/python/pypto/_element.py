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
from . import pypto_impl
from .error import FeError


class Element:
    def __init__(self, dtype, data):
        if isinstance(data, int):
            self._base = pypto_impl.Element(dtype, data)
        elif isinstance(data, float):
            self._base = pypto_impl.Element(dtype, data)
        else:
            raise FeError(ValueError(f"Invalid data type {type(data)} for Element"))

    @property
    def dtype(self):
        """
        Returns the data type of the element.
        """
        return self._base._get_data_type()

    @property
    def value(self):
        """
        Returns the value of the element.
        """
        if self._base._is_float():
            return self._base._get_float_data()
        else:
            return self._base._get_signed_data()

    def base(self):
        return self._base
