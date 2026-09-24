/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cell_match_dynamic.h
 * \brief Per-launch evaluation and refresh of dynamic CellMatchTableDesc.stride on host.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk::dynamic {

struct Evaluator;

std::vector<DevDynamicCellMatchStridePatch> PrepareDynamicCellMatchDescPatches(const DyndevFunctionAttribute& dynAttr,
                                                                               Evaluator& eval);

// Overload for the kernel-bundle load path (launch-meta list only, no Function).
std::vector<DevDynamicCellMatchStridePatch> PrepareDynamicCellMatchDescPatches(
    const std::vector<DyndevFunctionAttribute::DynamicCellMatchLaunchMeta>& launchMetaList, Evaluator& eval);

void PatchHostDynamicCellMatchTableDesc(DevAscendProgram* hostDevProg,
                                        const std::vector<DevDynamicCellMatchStridePatch>& patches);

void WriteDynamicCellMatchStridePatchesToLaunchArgs(int64_t* launchInputs,
                                                    const std::vector<DevDynamicCellMatchStridePatch>& patches);

void ValidateDynamicCellMatchTableMemBudget(const DyndevFunctionAttribute& dynAttr, DevAscendProgram* hostDevProg);

void RefillDynamicMemBudgets(DevAscendProgram* hostDevProg, DyndevFunctionAttribute& dynAttr, Evaluator& eval);

std::vector<DevDynamicCellMatchStridePatch> PrepareHostDynamicCellMatchForLaunch(DyndevFunctionAttribute& dynAttr,
                                                                                 Evaluator& eval,
                                                                                 DevAscendProgram* hostDevProg);

void ResetRuntimeDynamicCellMatchPoolHost(uint64_t addr, uint64_t capacityBytes, bool isDevice);

template <typename MemoryHelperTy>
inline void PatchRuntimeDynamicCellMatchMeta(MemoryHelperTy& memoryHelper, DevAscendProgram* hostProg,
                                             DeviceKernelArgs& kArgs)
{
    if (hostProg == nullptr) {
        return;
    }
    uint64_t dynamicCellMatchBytes = hostProg->memBudget.metadata.dynamicCellMatch;
    if (dynamicCellMatchBytes == 0) {
        hostProg->devArgs.dynamicCellMatchAddr = 0;
        hostProg->devArgs.dynamicCellMatchCapacity = 0;
        kArgs.runtimeDynamicCellMatchAddr = 0;
        kArgs.runtimeDynamicCellMatchCapacity = 0;
        return;
    }
    uint8_t* dynamicCellMatchAddr = memoryHelper.AllocDev(dynamicCellMatchBytes, nullptr);
    uint64_t dynamicCellMatchAddrU64 = reinterpret_cast<uint64_t>(dynamicCellMatchAddr);
    hostProg->devArgs.dynamicCellMatchAddr = dynamicCellMatchAddrU64;
    hostProg->devArgs.dynamicCellMatchCapacity = dynamicCellMatchBytes;
    kArgs.runtimeDynamicCellMatchAddr = dynamicCellMatchAddrU64;
    kArgs.runtimeDynamicCellMatchCapacity = dynamicCellMatchBytes;
    // AllocDev may return device HBM or host/sim memory depending on MemoryHelperTy.
    ResetRuntimeDynamicCellMatchPoolHost(dynamicCellMatchAddrU64, dynamicCellMatchBytes, memoryHelper.IsDevice());
}

} // namespace npu::tile_fwk::dynamic
