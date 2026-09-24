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
 * \file kernel_bundle_registry.cpp
 * \brief Implementation of the kernel bundle registry singleton.
 */

#include "machine/runtime/bundle/kernel_bundle_registry.h"

#include "machine/runtime/launcher/device_launcher_binding.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

KernelBundleRegistry& KernelBundleRegistry::GetInstance()
{
    static KernelBundleRegistry instance;
    return instance;
}

std::shared_ptr<LoadedBundle> KernelBundleRegistry::CacheByContent(std::shared_ptr<LoadedBundle> loaded)
{
    if (loaded == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = byContent_.find(loaded->bundleKey);
    if (it != byContent_.end()) {
        return it->second; // idempotent: byte-identical bundle already loaded
    }
    // Same op, different shape: identical hashKey but a different bundle. Supported (both live side by side,
    // each under its own bundleKey), logged because it also shows up when a stale .pyptokb is loaded by mistake.
    for (const auto& kv : byContent_) {
        if (kv.second->GetHashKey() == loaded->GetHashKey()) {
            MACHINE_LOGW("[kernel-bundle] hashKey %#lx now has a 2nd distinct bundle (bundleKey %#lx vs %#lx); "
                         "expected when the same op is packed for several shapes",
                         loaded->GetHashKey(), kv.second->bundleKey, loaded->bundleKey);
            break;
        }
    }
    byContent_[loaded->bundleKey] = loaded;
    return loaded;
}

std::shared_ptr<LoadedBundle> KernelBundleRegistry::LoadOrGet(const std::string& path)
{
    return CacheByContent(KernelBundleLoader::LoadFromFile(path));
}

std::shared_ptr<LoadedBundle> KernelBundleRegistry::LoadOrGetFromMemory(const uint8_t* data, size_t n)
{
    return CacheByContent(KernelBundleLoader::LoadFromMemory(data, n));
}

void* KernelBundleRegistry::GetOrRegisterKernel(uint64_t bundleKey, const std::vector<uint8_t>& kernelBin)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = kernelHandles_.find(bundleKey);
    if (it != kernelHandles_.end() && it->second != nullptr) {
        return it->second;
    }
    void* hdl = dynamic::RegisterKernelBinary(kernelBin);
    if (hdl == nullptr) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "[kernel-bundle] RegisterKernelBinary failed");
        return nullptr;
    }
    kernelHandles_[bundleKey] = hdl;
    return hdl;
}

void KernelBundleRegistry::Unload(uint64_t bundleKey)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto kh = kernelHandles_.find(bundleKey);
    if (kh != kernelHandles_.end()) {
        if (kh->second != nullptr) {
            dynamic::UnregisterKernelBinary(kh->second);
        }
        kernelHandles_.erase(kh);
    }
    byContent_.erase(bundleKey);
}

void KernelBundleRegistry::UnloadAll()
{
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& kv : kernelHandles_) {
        if (kv.second != nullptr) {
            dynamic::UnregisterKernelBinary(kv.second);
        }
    }
    kernelHandles_.clear();
    byContent_.clear();
}

} // namespace npu::tile_fwk::bundle
