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
 * \file ctrl_flow_cache_manager.cpp
 * \brief Implementation of CtrlFlowCacheManager.
 */

#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"

#include <sstream>

#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "interface/function/function.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"

namespace npu::tile_fwk::dynamic {

CtrlFlowCacheManager& CtrlFlowCacheManager::Instance()
{
    static CtrlFlowCacheManager instance;
    return instance;
}

uint8_t* CtrlFlowCacheManager::FindOrBuildDevCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors)
{
    // Device RT entry: skip find/build entirely for value-depend programs.
    // Actual Emulation build path is separately gated in EmulationLauncher.
    if (kernel->DisableHostCtrlFlowCacheBuild()) {
        COMPILER_LOGI("Skip host control flow cache build due to disableCtrlFlowCache.");
        return nullptr;
    }
    auto devCache = kernel->FindCtrlFlowCache(tensors, true);
    if (devCache == nullptr) {
        AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
        devCache = kernel->BuildControlFlowCache(tensors, true);
    }
    COMPILER_LOGD("find ctrlflow cache: %p", devCache);
    return devCache;
}

DevControlFlowCache* CtrlFlowCacheManager::GetHostCtrlFlowCache(KernelBinary* kernel,
                                                                std::vector<DeviceTensorData>& tensors,
                                                                uint8_t* devCache, std::vector<uint8_t>& hostCache)
{
    DevControlFlowCache* ctrlCache = FindHostCtrlFlowCache(kernel, tensors, hostCache);
    if (ctrlCache == nullptr && devCache != nullptr) {
        auto devProg = reinterpret_cast<DevAscendProgram*>(
            kernel->GetFunction()->GetDyndevAttribute()->devProgBinary.data());
        size_t ctrlCacheSize = devProg->ctrlFlowCacheSize;
        std::vector<uint8_t> hostCacheVec;
        hostCacheVec.resize(ctrlCacheSize);
        AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
        RuntimeMemcpy(hostCacheVec.data(), ctrlCacheSize, devCache, ctrlCacheSize, RtMemcpyKind::DEVICE_TO_HOST);
        AddHostCtrlFlowCache(kernel, tensors, std::move(hostCacheVec));
        ctrlCache = FindHostCtrlFlowCache(kernel, tensors, hostCache);
    }
    return ctrlCache;
}

DevControlFlowCache* CtrlFlowCacheManager::FindHostCtrlFlowCache(KernelBinary* kernel,
                                                                 std::vector<DeviceTensorData>& tensors,
                                                                 std::vector<uint8_t>& hostCache)
{
    int64_t hash = ControlFlowCache::Hash(tensors);
    for (auto& cache : kernel->GetHostCtrlFlowCaches()) {
        if (cache.hash == hash) {
            hostCache = cache.hostCache;
            return reinterpret_cast<DevControlFlowCache*>(hostCache.data());
        }
    }
    return nullptr;
}

void CtrlFlowCacheManager::AddHostCtrlFlowCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors,
                                                std::vector<uint8_t>&& hostCache)
{
    kernel->GetHostCtrlFlowCaches().emplace_back(tensors, std::move(hostCache));
}

} // namespace npu::tile_fwk::dynamic
