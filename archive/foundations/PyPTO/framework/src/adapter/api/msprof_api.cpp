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
 * \file msprof_api.cpp
 * \brief
 */

#include "adapter/api/msprof_api.h"

#ifdef BUILD_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#include "profiling/aprof_pub.h"
#endif
#include "adapter/stubs/msprof_stubs.h"

namespace npu::tile_fwk {
uint64_t MspfSysCycleTime(void)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::SysCycleTime);
    if (func != nullptr) {
        uint64_t (*msprofFunc)(void) = reinterpret_cast<uint64_t (*)(void)>(func);
        return msprofFunc();
    }
#endif
    return StubProfSysCycleTime();
}

uint64_t MspfGetHashId(const char* hashInfo, size_t length)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::GetHashId);
    if (func != nullptr) {
        uint64_t (*msprofFunc)(const char*, size_t) = reinterpret_cast<uint64_t (*)(const char*, size_t)>(func);
        return msprofFunc(hashInfo, length);
    }
#endif
    return StubProfGetHashId(hashInfo, length);
}

int32_t MspfReportApi(uint32_t nonPersistantFlag, const struct MspfApi* api)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportApi);
    if (func != nullptr) {
        uint64_t (*msprofFunc)(
            uint32_t,
            const struct MsprofApi*) = reinterpret_cast<uint64_t (*)(uint32_t, const struct MsprofApi*)>(func);
        return msprofFunc(nonPersistantFlag, reinterpret_cast<const struct MsprofApi*>(api));
    }
#endif
    return StubProfReportApi(nonPersistantFlag, api);
}

int32_t MspfReportCompactInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportCompactInfo);
    if (func != nullptr) {
        uint64_t (*msprofFunc)(uint32_t, const void*,
                               uint32_t) = reinterpret_cast<uint64_t (*)(uint32_t, const void*, uint32_t)>(func);
        return msprofFunc(nonPersistantFlag, data, length);
    }
#endif
    return StubProfReportCompactInfo(nonPersistantFlag, data, length);
}

int32_t MspfReportAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportAdditionalInfo);
    if (func != nullptr) {
        uint64_t (*msprofFunc)(uint32_t, const void*,
                               uint32_t) = reinterpret_cast<uint64_t (*)(uint32_t, const void*, uint32_t)>(func);
        return msprofFunc(nonPersistantFlag, data, length);
    }
#endif
    return StubProfReportAdditionalInfo(nonPersistantFlag, data, length);
}

int32_t MspfRegisterCallback(uint32_t moduleId, MspfCommandHandleFunc handle)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::RegisterCallback);
    if (func != nullptr) {
        uint64_t (*msprofFunc)(uint32_t,
                               ProfCommandHandle) = reinterpret_cast<uint64_t (*)(uint32_t, ProfCommandHandle)>(func);
        return msprofFunc(moduleId, reinterpret_cast<ProfCommandHandle>(handle));
    }
#endif
    return StubProfRegisterCallback(moduleId, handle);
}
#ifdef BUILD_WITH_CANN
static_assert(sizeof(MspfApi) == sizeof(MsprofApi));
static_assert(sizeof(MspfTensorData) == sizeof(MsrofTensorData));
static_assert(sizeof(MspfTensorInfo) == sizeof(MsprofTensorInfo));
static_assert(sizeof(MspfContextIdInfo) == sizeof(MsprofContextIdInfo));
static_assert(sizeof(MspfAdditionalInfo) == sizeof(MsprofAdditionalInfo));
static_assert(sizeof(MspfCompactInfo) == sizeof(MsprofCompactInfo));
static_assert(sizeof(MspfCommandHandle) == sizeof(MsprofCommandHandle));
static_assert(sizeof(MspfCommandHandleFunc) == sizeof(ProfCommandHandle));
#endif
} // namespace npu::tile_fwk
