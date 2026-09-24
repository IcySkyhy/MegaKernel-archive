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
 * \file hcomm_stubs.cpp
 * \brief
 */

#include "adapter/stubs/hcomm_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
HcommResult StubGetCommName(HcommHandle comm, char* commName)
{
    ADAPTER_LOGD("Enter stub function of GetCommName.");
    (void)comm;
    (void)commName;
    return HCOMM_SUCCESS;
}

HcommResult StubGetL0TopoTypeEx(const char* group, HCommTopo* topoType, uint32_t flag)
{
    ADAPTER_LOGD("Enter stub function of GetL0TopoTypeEx.");
    (void)group;
    (void)topoType;
    (void)flag;
    return HCOMM_SUCCESS;
}

HcommResult StubGetCommHandleByGroup(const char* group, HcommHandle* commHandle)
{
    ADAPTER_LOGD("Enter stub function of GetCommHandleByGroup.");
    (void)group;
    (void)commHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubGetRootInfo(HcommRootInfo* rootInfo)
{
    ADAPTER_LOGD("Enter stub function of GetRootInfo.");
    (void)rootInfo;
    return HCOMM_SUCCESS;
}

HcommResult StubCommInitRootInfo(uint32_t nRanks, const HcommRootInfo* rootInfo, uint32_t rank, HcommHandle* comm)
{
    ADAPTER_LOGD("Enter stub function of CommInitRootInfo.");
    (void)nRanks;
    (void)rootInfo;
    (void)rank;
    (void)comm;
    return HCOMM_SUCCESS;
}

HcommResult StubCommDestroy(HcommHandle comm)
{
    ADAPTER_LOGD("Enter stub function of CommDestroy.");
    (void)comm;
    return HCOMM_SUCCESS;
}

HcommResult StubAllocComResourceByTiling(HcommHandle comm, void* stream, void* Mc2Tiling, void** commContext)
{
    ADAPTER_LOGD("Enter stub function of AllocComResourceByTiling.");
    (void)comm;
    (void)stream;
    (void)Mc2Tiling;
    (void)commContext;
    return HCOMM_SUCCESS;
}
} // namespace npu::tile_fwk
