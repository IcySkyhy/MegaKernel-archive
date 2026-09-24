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
 * \file aicore_model_kernel_meta_hook.h
 * \brief Host-side AicoreModel sim: allocate DeviceArgs backing buffers after AssignMetaAddr (ring buffer stays there).
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "interface/configs/config_manager.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {

inline void InitAicoreModelDefaultDevArgs(DevAscendProgram* devProg)
{
    if (devProg == nullptr || devProg->devArgs.nrAic != 0 || devProg->devArgs.nrAiv != 0) {
        return;
    }
    constexpr uint32_t kDefaultNrAic = 24;
    constexpr uint32_t kDefaultNrAiv = 48;
    constexpr uint32_t kDefaultNrValidAic = 24;
    constexpr uint32_t kDefaultNrAicpu = 4;
    constexpr uint32_t kDefaultScheCpuNum = 3;
    devProg->devArgs.nrAic = kDefaultNrAic;
    devProg->devArgs.nrAiv = kDefaultNrAiv;
    devProg->devArgs.nrValidAic = kDefaultNrValidAic;
    devProg->devArgs.nrAicpu = kDefaultNrAicpu;
    devProg->devArgs.scheCpuNum = kDefaultScheCpuNum;
}

template <typename DeviceMemoryTy>
inline bool InitAicoreModelCoreAddrs(DeviceMemoryTy& devMem, DevAscendProgram* devProg, uint32_t totalCoreNum)
{
    auto allocAndAssign = [&](uint64_t& addr, size_t bytes, const char* name) {
        auto* ptr = devMem.AllocZero(bytes, nullptr);
        if (ptr == nullptr) {
            MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc aicore model host %s failed, size=%zu bytes", name, bytes);
            addr = 0;
            return false;
        }
        addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
        return true;
    };
    size_t sharedBufferSize = static_cast<size_t>(totalCoreNum) * static_cast<size_t>(SHARED_BUFFER_SIZE);
    return allocAndAssign(devProg->devArgs.sharedBuffer, sharedBufferSize, "sharedBuffer") &&
           allocAndAssign(devProg->devArgs.coreRegAddr, static_cast<size_t>(totalCoreNum) * sizeof(uint64_t),
                          "coreRegAddr") &&
           allocAndAssign(devProg->devArgs.corePmuRegAddr, static_cast<size_t>(totalCoreNum) * sizeof(uint64_t),
                          "corePmuRegAddr") &&
           allocAndAssign(devProg->devArgs.corePmuAddr,
                          static_cast<size_t>(totalCoreNum) * static_cast<size_t>(PMU_BUFFER_SIZE), "corePmuAddr") &&
           allocAndAssign(devProg->devArgs.taskWastTime, sizeof(uint64_t), "taskWastTime") &&
           allocAndAssign(devProg->devArgs.devDfxArgAddr, sizeof(DevDfxArgs), "devDfxArgAddr");
}

template <typename DeviceMemoryTy>
inline bool InitAicoreModelMetrics(DeviceMemoryTy& devMem, DevAscendProgram* devProg, uint32_t totalCoreNum)
{
    auto* sharedBuffer = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(devProg->devArgs.sharedBuffer));
    if (sharedBuffer == nullptr) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "AicoreModel host sharedBuffer is null");
        devProg->devArgs.sharedBuffer = 0;
        return false;
    }
    const size_t metricSize = sizeof(Metrics) + static_cast<size_t>(MAX_DFX_TASK_NUM_PER_CORE) * sizeof(TaskStat);
    for (uint32_t i = 0; i < totalCoreNum; ++i) {
        auto* args = reinterpret_cast<KernelArgs*>(sharedBuffer + static_cast<size_t>(i) * SHARED_BUFFER_SIZE);
        auto* metric = devMem.AllocZero(metricSize, nullptr);
        if (metric == nullptr) {
            MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc aicore model host metric failed, size=%zu bytes, coreId=%u",
                         metricSize, i);
            devProg->devArgs.sharedBuffer = 0;
            return false;
        }
        args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(metric));
    }
    return true;
}

template <typename DeviceMemoryTy>
inline void AicoreModelInitKernelMetaDeviceArgs(DeviceMemoryTy& devMem, DeviceKernelArgs& kArgs,
                                                const std::vector<uint8_t>& devProgData)
{
    auto* devProg = reinterpret_cast<DevAscendProgram*>(const_cast<uint8_t*>(devProgData.data()));
    InitAicoreModelDefaultDevArgs(devProg);
    constexpr uint32_t kAicpuRunTaskNum = 1;
    const uint32_t totalCoreNum = static_cast<uint32_t>(devProg->devArgs.nrAic + devProg->devArgs.nrAiv) +
                                  kAicpuRunTaskNum;
    if (totalCoreNum == 0) {
        return;
    }
    if (!InitAicoreModelCoreAddrs(devMem, devProg, totalCoreNum) ||
        !InitAicoreModelMetrics(devMem, devProg, totalCoreNum)) {
        return;
    }
    kArgs.cfgdata = (int64_t*)devMem.CopyToDev(devProgData, nullptr);
}

} // namespace npu::tile_fwk::dynamic
