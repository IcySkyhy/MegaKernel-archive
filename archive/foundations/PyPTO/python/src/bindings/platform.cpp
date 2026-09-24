/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform.cpp
 * \brief
 */

#include "pybind_common.h"

#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindPlatform(py::module_& m)
{
    m.def("GetNPUArch", []() -> std::string {
         auto npuArch = Platform::Instance().GetSoc().GetNPUArch();
         return NPUArchToString(npuArch);
     }).def("SetNPUArch", [](const std::string& value) { Platform::Instance().GetSoc().SetNPUArch(value); });

    m.def("GetAICoreNum", []() -> size_t { return Platform::Instance().GetSoc().GetAICoreNum(); });

    m.def("GetAICCoreNum", []() -> size_t { return Platform::Instance().GetSoc().GetAICCoreNum(); });

    m.def("GetAIVCoreNum", []() -> size_t { return Platform::Instance().GetSoc().GetAIVCoreNum(); });
}
} // namespace pypto
