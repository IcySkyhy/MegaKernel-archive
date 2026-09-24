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
 * \file hcomm_api.h
 * \brief
 */

#pragma once

#include "adapter/api/hcomm_define.h"

namespace npu::tile_fwk {
HcommResult HcommGetCommName(HcommHandle comm, char* commName);

HcommResult HcommGetL0TopoTypeEx(const char* group, HCommTopo* topoType, uint32_t flag);

HcommResult HcommGetCommHandleByGroup(const char* group, HcommHandle* commHandle);

HcommResult HcommGetRootInfo(HcommRootInfo* rootInfo);

HcommResult HcommCommDestroy(HcommHandle comm);

HcommResult HcommCommInitRootInfo(uint32_t nRanks, const HcommRootInfo* rootInfo, uint32_t rank, HcommHandle* comm);

HcommResult HcommAllocComResourceByTiling(HcommHandle comm, void* stream, void* Mc2Tiling, void** commContext);
} // namespace npu::tile_fwk
