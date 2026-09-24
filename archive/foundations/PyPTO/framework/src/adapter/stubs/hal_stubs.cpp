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
 * \file hal_stubs.cpp
 * \brief
 */

#include "adapter/stubs/hal_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
HalError StubHalMemCtl(int type, void* paramValue, size_t paramValueSize, void* outValue, size_t* outSizeRet)
{
    ADAPTER_LOGD("Enter stub function of HalMemCtl.");
    (void)type;
    (void)paramValue;
    (void)paramValueSize;
    (void)outValue;
    (void)outSizeRet;
    return HAL_ERROR_NONE;
}
HalError StubHalResMap(unsigned int devId, struct ResMapInfo* resInfo, unsigned long* va, unsigned int* len)
{
    ADAPTER_LOGD("Enter stub function of HalResMap.");
    (void)devId;
    (void)resInfo;
    (void)va;
    (void)len;
    return HAL_ERROR_NONE;
}
HalError StubHalGetDeviceInfoByBuff(uint32_t devId, int32_t moduleType, int32_t infoType, void* buf, int32_t* size)
{
    ADAPTER_LOGD("Enter stub function of HalGetDeviceInfoByBuff.");
    (void)devId;
    (void)moduleType;
    (void)infoType;
    (void)buf;
    (void)size;
    return HAL_ERROR_NONE;
}
} // namespace npu::tile_fwk
