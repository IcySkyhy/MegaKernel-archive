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
 * \file pypto_bundle_api.cpp
 * \brief Implementation of the two-interface path-based kernel-bundle C ABI.
 */

#include "tile_fwk_bundle/pypto_bundle_api.h"

#include <memory>
#include <vector>

#include "machine/runtime/bundle/kernel_bundle_registry.h"
#include "machine/runtime/bundle/kernel_bundle_workspace.h"
#include "machine/runtime/bundle/kernel_bundle_launch.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace {
using namespace npu::tile_fwk;
using namespace npu::tile_fwk::bundle;
using npu::tile_fwk::dynamic::DeviceLauncher;
using npu::tile_fwk::dynamic::DeviceTensorData;

std::vector<DeviceTensorData> ToDeviceTensors(const PyptoTensorDesc* descs, uint32_t n)
{
    std::vector<DeviceTensorData> out;
    if (descs == nullptr) {
        return out;
    }
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const PyptoTensorDesc& d = descs[i];
        const int32_t rank = (d.rank < 0) ? 0 : (d.rank > 8 ? 8 : d.rank);
        std::vector<int64_t> shape(d.shape, d.shape + rank);
        out.emplace_back(static_cast<DataType>(d.dataType), d.addr, shape);
    }
    return out;
}

// Post-load prep shared by the path and in-memory entries: seed the AiCpu backend .so from the bundle's AICPU_SO
// segment before any KernelBinary::InitDeviceArgs(), arch-check, and register the aicore .o. Stays pure host.
std::shared_ptr<LoadedBundle> PrepareBundle(std::shared_ptr<LoadedBundle> bundle)
{
    if (bundle == nullptr) {
        return nullptr;
    }
    if (!bundle->aicpuSo.empty()) {
        // Harmless to set repeatedly; only the first Get() (inside the first Launch) consumes it.
        npu::tile_fwk::dynamic::KernelBinary::SetAiCpuSoOverride(bundle->aicpuSo);
    } else {
        MACHINE_LOGW("[kernel-bundle] bundle has no AICPU_SO segment; will fall back to on-disk backend .so");
    }
    const uint32_t targetArch = static_cast<uint32_t>(Platform::Instance().GetSoc().GetNPUArch());
    if (bundle->GetArchInfo() != targetArch) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "[kernel-bundle] arch mismatch: bundle=%u target=%u",
                     bundle->GetArchInfo(), targetArch);
        return nullptr;
    }
    if (KernelBundleRegistry::GetInstance().GetOrRegisterKernel(bundle->bundleKey, bundle->aicoreKernel) == nullptr) {
        return nullptr;
    }
    return bundle;
}

// Idempotent load-or-get from a disk path.
std::shared_ptr<LoadedBundle> EnsureLoaded(const char* path)
{
    if (path == nullptr) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "[kernel-bundle] null path");
        return nullptr;
    }
    return PrepareBundle(KernelBundleRegistry::GetInstance().LoadOrGet(path));
}

// Idempotent load-or-get from an in-memory .pyptokb image (client already loaded the bytes; no disk path).
std::shared_ptr<LoadedBundle> EnsureLoadedFromMemory(const void* data, size_t size)
{
    if (data == nullptr || size == 0) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "[kernel-bundle] null/empty bundle image");
        return nullptr;
    }
    return PrepareBundle(
        KernelBundleRegistry::GetInstance().LoadOrGetFromMemory(reinterpret_cast<const uint8_t*>(data), size));
}

// Shared query/launch bodies (bundle already prepared). `tensors` is the unified operand list.
uint64_t WorkspaceImpl(const std::shared_ptr<LoadedBundle>& bundle, const PyptoTensorDesc* tensors, uint32_t n)
{
    if (bundle == nullptr) {
        return 0;
    }
    // Pure host workspace evaluation; patches the bundle's in-memory memBudget in place. Does NOT init the device.
    std::vector<DeviceTensorData> inputs = ToDeviceTensors(tensors, n);
    static const std::vector<DeviceTensorData> kNoOutputs;
    return EvalWorkspaceForShapes(*bundle, inputs, kNoOutputs);
}

int LaunchImpl(const std::shared_ptr<LoadedBundle>& bundle, const PyptoTensorDesc* tensors, uint32_t n,
               void* workspaceAddr, void* stream, int sync)
{
    if (bundle == nullptr) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "[kernel-bundle] EnsureLoaded failed for launch");
        return -1;
    }
    void* binHandle = KernelBundleRegistry::GetInstance().GetOrRegisterKernel(bundle->bundleKey, bundle->aicoreKernel);
    if (binHandle == nullptr) {
        return -1;
    }

    std::vector<DeviceTensorData> inputs = ToDeviceTensors(tensors, n);
    static const std::vector<DeviceTensorData> kNoOutputs;

    // Re-evaluate + patch memBudget for THIS shape before building kArgs. Idempotent if Workspace already ran on
    // the same shape; required when Launch is called standalone.
    (void)EvalWorkspaceForShapes(*bundle, inputs, kNoOutputs);

    // blockdim = op parallelism baked into the packed program (ctrlBlockDim / nrValidAic).
    const auto* head = bundle->Head();
    dynamic::DeviceLauncherConfig config;
    config.blockdim = head->ctrlBlockDim != 0 ? static_cast<int>(head->ctrlBlockDim) :
                                                static_cast<int>(head->devArgs.nrValidAic);

    return dynamic::LaunchBundleKernelOnce(bundle->devProgram, binHandle, bundle->bundleKey, inputs,
                                           bundle->ctrlFlowCache, bundle->cellMatchStridePatches, workspaceAddr,
                                           static_cast<RtStream>(stream), sync != 0, config);
}
} // namespace

extern "C" {

uint64_t PyptoWorkspace(const char* path, const PyptoTensorDesc* tensors, uint32_t nTensors)
{
    return WorkspaceImpl(EnsureLoaded(path), tensors, nTensors);
}

int PyptoLaunch(const char* path, const PyptoTensorDesc* tensors, uint32_t nTensors, void* workspaceAddr, void* stream,
                int sync)
{
    return LaunchImpl(EnsureLoaded(path), tensors, nTensors, workspaceAddr, stream, sync);
}

uint64_t PyptoWorkspaceFromMemory(const void* bundleData, uint64_t bundleSize, const PyptoTensorDesc* tensors,
                                  uint32_t nTensors)
{
    return WorkspaceImpl(EnsureLoadedFromMemory(bundleData, static_cast<size_t>(bundleSize)), tensors, nTensors);
}

int PyptoLaunchFromMemory(const void* bundleData, uint64_t bundleSize, const PyptoTensorDesc* tensors,
                          uint32_t nTensors, void* workspaceAddr, void* stream, int sync)
{
    return LaunchImpl(EnsureLoadedFromMemory(bundleData, static_cast<size_t>(bundleSize)), tensors, nTensors,
                      workspaceAddr, stream, sync);
}

} // extern "C"
