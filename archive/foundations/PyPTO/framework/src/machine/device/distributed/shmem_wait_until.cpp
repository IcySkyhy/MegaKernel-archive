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
 * \file shmem_wait_until.cpp
 * \brief
 */

#include "shmem_wait_until.h"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <cstring>
#include <cstdint>

#include "securec.h"

#include "machine/device/dynamic/aicore_manager.h"
#include "tileop/distributed/comm_context.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "neon_stub.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::Distributed {
constexpr int32_t AICPU_ATTR_RAW_INDEX = 3;

inline bool SignalTileOp::PollCompleted() const
{
    if constexpr (!npu::tile_fwk::dynamic::IsDeviceMode()) {
        return true;
    }
    if (addr_[0] != expectedSum_) {
        return false;
    }
    DEV_DEBUG("expectedSum_=%d, addr_[0]=%d, resolve taskId=%lu", expectedSum_, addr_[0], taskId_);
    if (resetSignal_) {
        addr_[0] = 0;
    }
    return true;
}

int32_t ShmemWaitUntilImpl::PollCompleted(npu::tile_fwk::dynamic::AiCoreManager* aicoreManager, uint32_t parallelIdx)
{
    return runingTaskQueue_[parallelIdx].PollCompleted([&](SignalTileOp* task) {
        if (aicoreManager == nullptr) {
            DEV_ERROR(DistributedErrorCode::NULLPTR, "sche.task.pre.task.poll#: AicoreManager is nullptr");
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        return aicoreManager->ProcessCompletedAicpuTask(task->taskId_);
    });
}

TensorInfo ShmemWaitUntilImpl::GetTensorInfo(uint64_t taskId,
                                             const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode,
                                             npu::tile_fwk::DynFuncData* funcDataList, int64_t* hcclContextAddr,
                                             const AicpuParamInfo& paramInfo)
{
    const uint32_t funcId = npu::tile_fwk::FuncID(taskId);
    const uint32_t opIndex = npu::tile_fwk::TaskID(taskId);
    auto& funcData = funcDataList[funcId];
    auto opAttrs = &funcData.opAttrs[funcData.opAtrrOffsets[opIndex]];
    auto expressionTable = funcData.exprTbl;

    int32_t index = aicpuCode[paramInfo.inIndex + AICPU_ATTR_RAW_INDEX];
    TensorInfo info;
    info.rawIndex = GetCoa(index, opAttrs, expressionTable);
    ++index;
    info.dim = aicpuCode[paramInfo.inIndex + AICPU_ATTR_DIM_INDEX];
    info.offset = GetCoaVector(index, info.dim, opAttrs, expressionTable);

    info.expectedSum = aicpuCode[paramInfo.attrIndex];
    info.resetSignal = aicpuCode[paramInfo.attrIndex + AICPU_ATTR_DIM_INDEX];
    auto desc = &funcData.rawTensorDesc[info.rawIndex];
    info.vaddr = funcData.rawTensorAddr[desc->offsetOrIndex];
    info.rawAddr = MapVirtualSignalAddr(hcclContextAddr, info.vaddr);
    return info;
}
} // namespace npu::tile_fwk::Distributed
