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
 * \file adump_stubs.cpp
 * \brief
 */

#include "adapter/stubs/adump_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
uint64_t StubDumpGetDumpSwitch(const AdxDumpType dumpType)
{
    ADAPTER_LOGD("Enter stub function of AdumpGetDumpSwitch.");
    (void)dumpType;
    return 0;
}

int32_t StubDumpDumpTensorV2(const std::string& opType, const std::string& opName,
                             const std::vector<AdxTensorInfoV2>& tensors, AclRtStream stream)
{
    ADAPTER_LOGD("Enter stub function of AdumpDumpTensorV2.");
    (void)opType;
    (void)opName;
    (void)tensors;
    (void)stream;
    return 0;
}

int32_t StubDumpDataDumpServerInit()
{
    ADAPTER_LOGD("Enter stub function of AdxDataDumpServerInit.");
    return 0;
}

int32_t StubDumpDataDumpServerUnInit()
{
    ADAPTER_LOGD("Enter stub function of AdxDataDumpServerUnInit.");
    return 0;
}
} // namespace npu::tile_fwk
