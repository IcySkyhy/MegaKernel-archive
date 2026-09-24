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
 * \file kernel_bundle_pack.cpp
 * \brief Pack side of the kernel bundle: symbol serialization + the .pyptokb pack hook (see header).
 */

#include "machine/runtime/bundle/pack/kernel_bundle_pack.h"

#include <string>

#include <nlohmann/json.hpp>

#include "machine/runtime/bundle/kernel_bundle_format.h"
#include "machine/runtime/bundle/pack/kernel_bundle_packer.h"
#include "interface/function/function.h"
#include "interface/tensor/symbolic_scalar.h"
#include "utils/file_utils.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

namespace {
constexpr int kSymbolMetaVersion = 1;
} // namespace

std::vector<uint8_t> SerializeWorkspaceSymbols(const DyndevFunctionAttribute& dynAttr)
{
    // Only emit a SymbolMeta segment when there is a dynamic tree to evaluate; a fully static op gets no segment
    // and the loader keeps reading the baked memBudget.Total() constant.
    const bool hasAssemble = dynAttr.maxDynamicAssembleOutcastMem.IsValid();
    const bool hasCellMatch = dynAttr.maxDynamicCellMatchTableMem.IsValid();
    if (!hasAssemble && !hasCellMatch) {
        return {};
    }

    nlohmann::json root = nlohmann::json::object();
    root["v"] = kSymbolMetaVersion;
    if (hasAssemble) {
        root["assembleMem"] = ToJson(dynAttr.maxDynamicAssembleOutcastMem);
    }
    if (hasCellMatch) {
        root["cellMatchMem"] = ToJson(dynAttr.maxDynamicCellMatchTableMem);
    }
    // Recording-time symbol dict: bare symbols the Evaluator can't derive from tensor shapes at runtime.
    nlohmann::json syms = nlohmann::json::object();
    for (const auto& kv : dynAttr.inputSymbolDict) {
        syms[kv.first] = static_cast<int64_t>(kv.second);
    }
    root["symbols"] = std::move(syms);

    // Dynamic cell-match launch metas: per slot, the desc offset into the base-0 devProgram plus the symbolic
    // candidate raw dims. The load side re-evaluates these against the launch shapes to rebuild the stride patches
    // (else the device reads stale strides -> AICPU execute failure, retcode 507018).
    if (!dynAttr.dynamicCellMatchLaunchMetaList.empty()) {
        nlohmann::json metas = nlohmann::json::array();
        for (const auto& lm : dynAttr.dynamicCellMatchLaunchMetaList) {
            nlohmann::json m = nlohmann::json::object();
            m["slot"] = lm.slotIndex;
            m["descOffset"] = static_cast<uint64_t>(lm.descOffset);
            m["cellShape"] = lm.cellShape;
            nlohmann::json cand = nlohmann::json::array();
            for (const auto& dims : lm.candidateRawDims) {
                nlohmann::json row = nlohmann::json::array();
                for (const auto& expr : dims) {
                    row.push_back(ToJson(expr));
                }
                cand.push_back(std::move(row));
            }
            m["cand"] = std::move(cand);
            metas.push_back(std::move(m));
        }
        root["cellMatchLaunch"] = std::move(metas);
    }

    const std::string dumped = root.dump();
    MACHINE_LOGI("[kernel-bundle] serialized SymbolMeta: %zuB (assemble=%d cellMatch=%d symbols=%zu cmLaunch=%zu)",
                 dumped.size(), static_cast<int>(hasAssemble), static_cast<int>(hasCellMatch),
                 dynAttr.inputSymbolDict.size(), dynAttr.dynamicCellMatchLaunchMetaList.size());
    return std::vector<uint8_t>(dumped.begin(), dumped.end());
}

void MaybePackKernelBundle(const DyndevFunctionAttribute& dynAttr, const std::vector<uint8_t>& ctrlFlowCache)
{
    if (!IsKernelBundleEnabled()) {
        return;
    }
    // For value-dependent control flow (empty ctrlFlowCache) the CTRL_FLOW_CACHE segment is omitted; the launch
    // passes ctrlFlowCache=nullptr so the on-device interpreter resolves loop bounds from tensor values at runtime.
    KernelBundlePacker packer;
    packer.SetAicoreKernel(dynAttr.kernelBinary);
    packer.SetAicpuSo(ReadFile(GetPyptoLibPath() + "/libtilefwk_backend_server.so"));
    packer.SetDevProgram(dynAttr.devProgBinary); // host-side base-0; workspaceSize/hashKey in header
    if (!ctrlFlowCache.empty()) {
        packer.SetCtrlFlowCache(ctrlFlowCache);
    }
    // Dynamic-workspace trees + inputSymbolDict. Empty for fully-static ops.
    packer.SetSymbolMeta(SerializeWorkspaceSymbols(dynAttr));
    packer.Pack(KernelBundleOutPath());
}

KernelBundlePackHook& KernelBundlePackHook::Instance()
{
    static KernelBundlePackHook instance;
    return instance;
}

void KernelBundlePackHook::StashCtrlFlowCache(const DyndevFunctionAttribute* dynAttr, const uint8_t* cache, size_t size)
{
    if (!IsKernelBundleEnabled() || dynAttr == nullptr || cache == nullptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    stashedFor_ = dynAttr;
    stashedCache_.assign(cache, cache + size);
    MACHINE_LOGI("[kernel-bundle] stashed ctrl-flow cache (%zuB) for pack at launch", size);
}

void KernelBundlePackHook::MaybePack(const DyndevFunctionAttribute* dynAttr)
{
    if (!IsKernelBundleEnabled() || dynAttr == nullptr) {
        return;
    }
    std::vector<uint8_t> cache;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!packed_.insert(dynAttr).second) {
            return; // already packed this op in this process
        }
        if (stashedFor_ == dynAttr) {
            cache = std::move(stashedCache_);
            stashedCache_.clear();
            stashedFor_ = nullptr;
        }
    }
    // An empty cache here is not an error: value-dependent ops never build one, and MaybePackKernelBundle omits
    // the CTRL_FLOW_CACHE segment so the device interpreter resolves control flow from tensor values instead.
    MaybePackKernelBundle(*dynAttr, cache);
}

} // namespace npu::tile_fwk::bundle
