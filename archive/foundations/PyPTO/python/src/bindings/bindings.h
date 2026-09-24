/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bindings.h
 * \brief
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace pypto {
void BindEnum(py::module_& m);
void BindElement(py::module_& m);
void BindTensor(py::module_& m);
void BindSymbolicScalar(py::module_& m);
void BindController(py::module_& m);
void BindOperation(py::module_& m);
void BindRuntime(py::module_& m);
void BindCostModelRuntime(py::module_& m);
void BindPass(py::module_& m);
void BindFunction(py::module_& m);
void BindDistributed(py::module_& m);
void BindPlatform(py::module_& m);
void BindUtils(py::module_& m);
} // namespace pypto
