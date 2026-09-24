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
 * \file kernel_bundle_packer.h
 * \brief Packs the TLV segments into a single .pyptokb bundle.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npu::tile_fwk::bundle {

class KernelBundlePacker {
public:
    void SetAicoreKernel(std::vector<uint8_t> o) { aicoreKernel_ = std::move(o); }
    void SetAicpuSo(std::vector<uint8_t> so) { aicpuSo_ = std::move(so); }
    // devProgBinary must already be base-0 (RelocProgram(this, 0)).
    void SetDevProgram(std::vector<uint8_t> prog) { devProgram_ = std::move(prog); }
    // DevControlFlowCache blob, usedCacheSize bytes, base-0. Optional (may be empty).
    void SetCtrlFlowCache(std::vector<uint8_t> cache) { ctrlFlowCache_ = std::move(cache); }
    // Serialized dynamic-workspace symbol metadata (JSON). Optional; empty -> loader falls back to static.
    void SetSymbolMeta(std::vector<uint8_t> meta) { symbolMeta_ = std::move(meta); }

    // Serialize the assembled bundle into `outPath`. Returns true on success.
    bool Pack(const std::string& outPath) const;

    // Serialize the assembled bundle into an in-memory buffer (used by round-trip tests).
    std::vector<uint8_t> Build() const;

private:
    std::vector<uint8_t> aicoreKernel_;
    std::vector<uint8_t> aicpuSo_;
    std::vector<uint8_t> devProgram_;
    std::vector<uint8_t> ctrlFlowCache_;
    std::vector<uint8_t> symbolMeta_;
};

} // namespace npu::tile_fwk::bundle
