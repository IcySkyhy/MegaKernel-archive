/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msprof_adapter_types.h
 * \brief
 */

#pragma once

#include <string>
#include <map>

namespace npu::tile_fwk {
enum class MsprofFunc {
    SysCycleTime = 0,
    GetHashId,
    ReportApi,
    ReportCompactInfo,
    ReportAdditionalInfo,
    RegisterCallback,
    Bottom
};
const std::string kMsprofLibName = "libprofapi.so";
const std::map<MsprofFunc, std::string> kMsprofFuncStrMap{
    {MsprofFunc::SysCycleTime, "MsprofSysCycleTime"},
    {MsprofFunc::GetHashId, "MsprofGetHashId"},
    {MsprofFunc::ReportApi, "MsprofReportApi"},
    {MsprofFunc::ReportCompactInfo, "MsprofReportCompactInfo"},
    {MsprofFunc::ReportAdditionalInfo, "MsprofReportAdditionalInfo"},
    {MsprofFunc::RegisterCallback, "MsprofRegisterCallback"}};
} // namespace npu::tile_fwk
