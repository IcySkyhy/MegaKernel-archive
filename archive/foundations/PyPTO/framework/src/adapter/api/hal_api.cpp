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
 * \file hal_api.cpp
 * \brief
 */

#include "adapter/api/hal_api.h"

#ifdef BUILD_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#include "driver/ascend_hal_error.h"
#include "driver/ascend_hal_define.h"
#endif
#include "adapter/stubs/hal_stubs.h"

namespace npu::tile_fwk {
HalError HalMemCtl(int type, void* paramValue, size_t paramValueSize, void* outValue, size_t* outSizeRet)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::MemCtl);
    if (func != nullptr) {
        drvError_t (*halFunc)(int, void*, size_t, void*,
                              size_t*) = reinterpret_cast<drvError_t (*)(int, void*, size_t, void*, size_t*)>(func);
        return static_cast<HalError>(halFunc(type, paramValue, paramValueSize, outValue, outSizeRet));
    }
#endif
    return StubHalMemCtl(type, paramValue, paramValueSize, outValue, outSizeRet);
}

HalError HalResMap(unsigned int devId, struct ResMapInfo* resInfo, unsigned long* va, unsigned int* len)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::ResMap);
    if (func != nullptr) {
        drvError_t (*halFunc)(unsigned int, struct res_map_info*, unsigned long*,
                              unsigned int*) = reinterpret_cast<drvError_t (*)(unsigned int, struct res_map_info*,
                                                                               unsigned long*, unsigned int*)>(func);
        return static_cast<HalError>(halFunc(devId, reinterpret_cast<res_map_info*>(resInfo), va, len));
    }
#endif
    return StubHalResMap(devId, resInfo, va, len);
}

HalError HalGetDeviceInfoByBuff(uint32_t devId, int32_t moduleType, int32_t infoType, void* buf, int32_t* size)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::GetDeviceInfoByBuff);
    if (func != nullptr) {
        drvError_t (*halFunc)(
            uint32_t, int32_t, int32_t, void*,
            int32_t*) = reinterpret_cast<drvError_t (*)(uint32_t, int32_t, int32_t, void*, int32_t*)>(func);
        return static_cast<HalError>(halFunc(devId, moduleType, infoType, buf, size));
    }
#endif
    return StubHalGetDeviceInfoByBuff(devId, moduleType, infoType, buf, size);
}
#ifdef BUILD_WITH_CANN
static_assert(static_cast<int32_t>(HAL_ERROR_NONE) == static_cast<int32_t>(DRV_ERROR_NONE));
static_assert(sizeof(ResMapInfo) == sizeof(res_map_info));
#endif
} // namespace npu::tile_fwk
