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
 * \file kernel_bundle_packer.cpp
 * \brief Implementation of the kernel bundle packer.
 */

#include "machine/runtime/bundle/pack/kernel_bundle_packer.h"

#include <array>
#include <cstring>

#include "machine/runtime/bundle/kernel_bundle_crc32.h"
#include "machine/runtime/bundle/kernel_bundle_format.h"
#include "utils/file_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

namespace {
inline uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
} // namespace

std::vector<uint8_t> KernelBundlePacker::Build() const
{
    // Segment order matches TlvType numbering. Empty segments (e.g. ctrlFlowCache) are still emitted
    // as zero-length TLVs so the loader can distinguish "absent" from "present but empty".
    struct Seg {
        TlvType type;
        const std::vector<uint8_t>* data;
    };
    const std::array<Seg, 5> segs = {{
        {TlvType::AicoreKernel, &aicoreKernel_},
        {TlvType::AicpuSo, &aicpuSo_},
        {TlvType::DevProgram, &devProgram_},
        {TlvType::CtrlFlowCache, &ctrlFlowCache_},
        {TlvType::SymbolMeta, &symbolMeta_},
    }};
    const uint32_t tlvCount = static_cast<uint32_t>(segs.size());

    // Layout: [BundleHeader][TlvHeader * N] ...padding... [value blocks, each 4KB aligned].
    const uint64_t headerRegion = sizeof(BundleHeader) + static_cast<uint64_t>(tlvCount) * sizeof(TlvHeader);

    std::array<TlvHeader, 5> tlvs{};
    uint64_t cursor = AlignUp(headerRegion, kValueAlign);
    for (uint32_t i = 0; i < tlvCount; ++i) {
        const auto& d = *segs[i].data;
        tlvs[i].type = static_cast<uint32_t>(segs[i].type);
        tlvs[i].flags = 0;
        tlvs[i].valueOffset = cursor;
        tlvs[i].valueLength = d.size();
        tlvs[i].valueCrc32 = Crc32(d.data(), d.size());
        tlvs[i].reserved = 0;
        cursor = AlignUp(cursor + d.size(), kValueAlign);
    }
    const uint64_t totalSize = cursor;

    std::vector<uint8_t> buf(totalSize, 0);

    BundleHeader hdr{};
    hdr.magic = kBundleMagic;
    hdr.version = kBundleVersion;
    hdr.flags = 0;
    hdr.tlvCount = tlvCount;
    hdr.headerSize = static_cast<uint32_t>(headerRegion);
    hdr.totalSize = totalSize;
    hdr.headerCrc32 = Crc32(&hdr, sizeof(BundleHeader) - sizeof(uint32_t));

    std::memcpy(buf.data(), &hdr, sizeof(BundleHeader));
    std::memcpy(buf.data() + sizeof(BundleHeader), tlvs.data(), tlvCount * sizeof(TlvHeader));
    for (uint32_t i = 0; i < tlvCount; ++i) {
        const auto& d = *segs[i].data;
        if (!d.empty()) {
            std::memcpy(buf.data() + tlvs[i].valueOffset, d.data(), d.size());
        }
    }
    return buf;
}

bool KernelBundlePacker::Pack(const std::string& outPath) const
{
    std::vector<uint8_t> buf = Build();
    bool ok = SaveFile(outPath, buf.data(), buf.size());
    if (ok) {
        MACHINE_LOGI("[kernel-bundle] packed %zu bytes -> %s", buf.size(), outPath.c_str());
    } else {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] failed to write bundle to %s", outPath.c_str());
    }
    return ok;
}

} // namespace npu::tile_fwk::bundle
