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
 * \file element.cpp
 * \brief
 */

#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {

/**
 * @brief Bind the Element class.
 *
 * @param m Pybind11 module
 */
void BindElement(py::module_& m)
{
    py::class_<Element>(m, "Element")
        .def(py::init<DataType, int64_t>(), py::arg("type"), py::arg("sData"))
        .def(py::init<DataType, uint64_t>(), py::arg("type"), py::arg("uData"))
        .def(py::init<DataType, double>(), py::arg("type"), py::arg("fData"))
        .def("_get_data_type", &Element::GetDataType)
        .def("_get_signed_data", &Element::GetSignedData)
        .def("_get_unsigned_data", &Element::GetUnsignedData)
        .def("_is_float", &Element::IsFloat)
        .def("_get_float_data", &Element::GetFloatData);
}

} // namespace pypto
