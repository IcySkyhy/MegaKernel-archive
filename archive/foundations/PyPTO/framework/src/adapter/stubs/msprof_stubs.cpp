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
 * \file msprof_stubs.cpp
 * \brief
 */

#include "adapter/stubs/msprof_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
uint64_t StubProfSysCycleTime(void)
{
    ADAPTER_LOGD("Enter stub function of MsprofSysCycleTime.");
    return 0;
}

uint64_t StubProfGetHashId(const char* hashInfo, size_t length)
{
    ADAPTER_LOGD("Enter stub function of MsprofGetHashId.");
    (void)hashInfo;
    (void)length;
    return 0;
}

int32_t StubProfReportApi(uint32_t nonPersistantFlag, const struct MspfApi* api)
{
    ADAPTER_LOGD("Enter stub function of MsprofReportApi.");
    (void)nonPersistantFlag;
    (void)api;
    return 0;
}

int32_t StubProfReportCompactInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length)
{
    ADAPTER_LOGD("Enter stub function of MsprofReportCompactInfo.");
    (void)nonPersistantFlag;
    (void)data;
    (void)length;
    return 0;
}

int32_t StubProfReportAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length)
{
    ADAPTER_LOGD("Enter stub function of MsprofReportAdditionalInfo.");
    (void)nonPersistantFlag;
    (void)data;
    (void)length;
    return 0;
}

int32_t StubProfRegisterCallback(uint32_t moduleId, MspfCommandHandleFunc handle)
{
    ADAPTER_LOGD("Enter stub function of MsprofRegisterCallback.");
    (void)moduleId;
    (void)handle;
    return 0;
}
} // namespace npu::tile_fwk
