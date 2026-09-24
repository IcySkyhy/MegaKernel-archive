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
 * \file kernel_bundle_pack.h
 * \brief Pack-side entry points for the kernel bundle (.pyptokb).
 *
 * Lives in tile_fwk_runtime, NOT in libtile_fwk_bundle.so: the pack hooks are called from the compile/emulation and
 * launch paths inside the runtime, whereas the load/query/launch side is the standalone ABI library. Keeping the
 * two halves in separate targets is what breaks the otherwise circular dependency between them.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace npu::tile_fwk {

struct DyndevFunctionAttribute;

namespace bundle {

// Serialize the dynamic-workspace symbolic trees + inputSymbolDict into a JSON byte blob.
// Returns empty when the op has no dynamic workspace trees (static bundle -> loader keeps the baked constant).
std::vector<uint8_t> SerializeWorkspaceSymbols(const DyndevFunctionAttribute& dynAttr);

// Write a .pyptokb for an already-compiled function attribute. No-op unless the bundle is enabled.
// Pass an EMPTY `ctrlFlowCache` for value-dependent (RUNTIME_FUNCKEY_CACHESTOP) ops: the bundle omits the
// CTRL_FLOW_CACHE segment and the device resolves control flow from tensor values at launch.
void MaybePackKernelBundle(const DyndevFunctionAttribute& dynAttr, const std::vector<uint8_t>& ctrlFlowCache);

// Decouples WHERE the control-flow cache exists from WHERE packing is decided.
//
// The fully relocated base-0 cache only exists for one moment, deep inside the emulation build, but packing must
// not be decided there -- it would put a feature hook in the middle of a build routine, and the value-dependent
// case (which never builds a cache) would need a second hook somewhere else. So the build path only STASHES the
// bytes, and DeviceLauncher::PrepareLaunch -- the single choke point every launch passes through -- decides and
// packs exactly once.
//
// Both entry points are no-ops when the bundle is disabled, so the non-bundle path pays one cached bool test.
class KernelBundlePackHook {
public:
    static KernelBundlePackHook& Instance();

    // Hand over the freshly built base-0 control-flow cache for `dynAttr`. Does not pack.
    void StashCtrlFlowCache(const DyndevFunctionAttribute* dynAttr, const uint8_t* cache, size_t size);

    // Pack `dynAttr` once per process. Uses the cache stashed for this op, or an empty one when nothing was
    // stashed -- which is exactly the value-dependent case, whose bundle omits the CTRL_FLOW_CACHE segment.
    void MaybePack(const DyndevFunctionAttribute* dynAttr);

    KernelBundlePackHook(const KernelBundlePackHook&) = delete;
    KernelBundlePackHook& operator=(const KernelBundlePackHook&) = delete;

private:
    KernelBundlePackHook() = default;
    ~KernelBundlePackHook() = default;

    std::mutex mutex_;
    // Single slot on purpose: the output path (PYPTO_KERNEL_BUNDLE_PATH) is one file, so a process can only ever
    // produce one meaningful bundle. Keyed so a stash from op A is never packed into op B's bundle.
    const DyndevFunctionAttribute* stashedFor_{nullptr};
    std::vector<uint8_t> stashedCache_;
    std::unordered_set<const DyndevFunctionAttribute*> packed_;
};

} // namespace bundle
} // namespace npu::tile_fwk
