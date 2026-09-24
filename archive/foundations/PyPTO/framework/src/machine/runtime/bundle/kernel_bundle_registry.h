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
 * \file kernel_bundle_registry.h
 * \brief Singleton that owns loaded bundles and their registered kernel handles.
 *        The AICPU .so is copied to device by KernelBinary::InitDeviceArgs() (InitAiCpuSoBin), not here.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "machine/runtime/bundle/kernel_bundle_loader.h"

namespace npu::tile_fwk::bundle {

class KernelBundleRegistry {
public:
    static KernelBundleRegistry& GetInstance();

    // Idempotent by LoadedBundle::bundleKey (content digest, see kernel_bundle_loader.h): re-loading the same
    // .pyptokb returns the cached bundle, while two shape-variants of one op -- same hashKey, different bytes --
    // are kept as separate entries instead of aliasing onto each other.
    std::shared_ptr<LoadedBundle> LoadOrGet(const std::string& path);
    // Same, but parsing from an in-memory .pyptokb image (client-loaded bytes; never touches a disk path).
    std::shared_ptr<LoadedBundle> LoadOrGetFromMemory(const uint8_t* data, size_t n);

    // Register the .o once per bundleKey; cache the returned RtBinHandle.
    void* GetOrRegisterKernel(uint64_t bundleKey, const std::vector<uint8_t>& kernelBin);

    void Unload(uint64_t bundleKey);
    void UnloadAll();

    KernelBundleRegistry(const KernelBundleRegistry&) = delete;
    KernelBundleRegistry& operator=(const KernelBundleRegistry&) = delete;

private:
    KernelBundleRegistry() = default;
    ~KernelBundleRegistry() = default;

    // Idempotent-by-bundleKey caching shared by LoadOrGet / LoadOrGetFromMemory.
    std::shared_ptr<LoadedBundle> CacheByContent(std::shared_ptr<LoadedBundle> loaded);

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<LoadedBundle>> byContent_;
    std::unordered_map<uint64_t, void*> kernelHandles_; // RtBinHandle per bundleKey
};

} // namespace npu::tile_fwk::bundle
