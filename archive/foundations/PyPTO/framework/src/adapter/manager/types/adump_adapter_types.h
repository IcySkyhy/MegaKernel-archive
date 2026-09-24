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
 * \file adump_adapter_types.h
 * \brief
 */

#pragma once

#include <string>
#include <map>

namespace npu::tile_fwk {
enum class AdumpFunc {
    GetDumpSwitch = 0,
    DumpTensorV2,
    DumpFailTaskExceptionCallBack,
    DataDumpServerInit,
    DataDumpServerUnInit,
    Bottom
};

const std::string kAdumpLibName = "libascend_dump.so";
const std::map<AdumpFunc, std::string> kAdumpFuncStrMap{
    {AdumpFunc::GetDumpSwitch, "_ZN3Adx18AdumpGetDumpSwitchENS_8DumpTypeE"},
    {AdumpFunc::DumpTensorV2, "_ZN3Adx17AdumpDumpTensorV2ERKSsS1_RKSt6vectorINS_12TensorInfoV2ESaIS3_EEPv"},
    {AdumpFunc::DumpFailTaskExceptionCallBack,
     "_ZN3Adx29AdumpRegExceptionDumpCallbackEPFjPvPNS_17ExceptionDumpInfoEjPjPNS_17ExceptionDumpModeEE"},
    {AdumpFunc::DataDumpServerInit, "AdxDataDumpServerInit"},
    {AdumpFunc::DataDumpServerUnInit, "AdxDataDumpServerUnInit"},
};
} // namespace npu::tile_fwk
