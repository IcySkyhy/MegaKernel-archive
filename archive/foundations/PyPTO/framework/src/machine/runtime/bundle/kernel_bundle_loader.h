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
 * \file kernel_bundle_loader.h
 * \brief Loads a .pyptokb into memory and exposes its segments.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

namespace npu::tile_fwk::bundle {

struct LoadedBundle {
    // Content-derived identity, folded by the loader from every TLV's (type, value CRC32, value length).
    //
    // This -- NOT hashKey -- is what the registry caches on. hashKey comes from the IR structure only, so two
    // bundles of the same op built for different runtime shapes carry the SAME hashKey while holding different
    // devProgram bytes; keying the cache on it would silently hand the first bundle back for the second load
    // and run the wrong program. Two bundles share a bundleKey only when they are byte-identical.
    uint64_t bundleKey{0};

    std::vector<uint8_t> aicoreKernel;  // .o raw
    std::vector<uint8_t> aicpuSo;       // .so raw
    std::vector<uint8_t> devProgram;    // kept base-0 (device-side reloc)
    std::vector<uint8_t> ctrlFlowCache; // DevControlFlowCache blob (may be empty for old bundles)
    std::vector<uint8_t> symbolMeta;    // JSON dynamic-workspace symbols (empty -> static workspace fallback)

    // Per-shape dynamic cell-match stride patches, computed host-side by EvalWorkspaceForShapes and written into
    // the aicpu launch args. Empty for ops with no dynamic cell-match table.
    std::vector<dynamic::DevDynamicCellMatchStridePatch> cellMatchStridePatches;

    // Header scalars read directly from the base-0 buffer without reloc.
    const dynamic::DevAscendProgram* Head() const
    {
        return reinterpret_cast<const dynamic::DevAscendProgram*>(devProgram.data());
    }
    uint64_t GetWorkspaceSize() const { return Head()->memBudget.Total(); }
    uint64_t GetHashKey() const { return Head()->hashKey; }
    uint64_t GetConfigKey() const { return Head()->configKey; }
    uint32_t GetArchInfo() const { return static_cast<uint32_t>(Head()->devArgs.archInfo); }

    bool HasCtrlFlowCache() const { return !ctrlFlowCache.empty(); }
    bool HasSymbolMeta() const { return !symbolMeta.empty(); }
};

class KernelBundleLoader {
public:
    static std::shared_ptr<LoadedBundle> LoadFromFile(const std::string& path);
    static std::shared_ptr<LoadedBundle> LoadFromMemory(const uint8_t* p, size_t n);
};

} // namespace npu::tile_fwk::bundle
