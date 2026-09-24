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
 * \file kernel_bundle_loader.cpp
 * \brief Implementation of the kernel bundle loader.
 */

#include "machine/runtime/bundle/kernel_bundle_loader.h"

#include <cstring>

#include "machine/runtime/bundle/kernel_bundle_crc32.h"
#include "machine/runtime/bundle/kernel_bundle_format.h"
#include "utils/file_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

namespace {
// Copy one TLV value out of the (possibly transient) source image into a buffer the LoadedBundle owns.
//
// No host-side alignment is required or attempted here: every segment reaches the device through
// DeviceMemoryUtils::CopyToDev, which memcpy's from the host pointer, and the online path feeds the very same
// call with a plain std::vector (DyndevFunctionAttribute::devProgBinary). The 4KB alignment in the on-disk
// layout (kValueAlign) exists for the file/mmap side only. Do not "restore" a std::vector-based alignment
// claim here -- the default allocator only guarantees alignof(max_align_t).
std::vector<uint8_t> CopySegment(const uint8_t* src, size_t len)
{
    std::vector<uint8_t> v(len);
    if (len != 0) {
        std::memcpy(v.data(), src, len);
    }
    return v;
}

// Fold one TLV's identity (type, content CRC, length) into a running 64-bit FNV-1a digest. The result
// identifies the bundle by CONTENT, which is what the registry keys on -- see the bundleKey comment in
// kernel_bundle_loader.h for why hashKey cannot serve that role.
void FoldTlvIntoBundleKey(uint64_t& key, uint32_t type, uint32_t crc, uint64_t len)
{
    constexpr uint64_t kFnvPrime = 0x00000100000001B3ULL;
    for (uint64_t word : {static_cast<uint64_t>(type), static_cast<uint64_t>(crc), len}) {
        for (int byte = 0; byte < 8; ++byte) {
            key = (key ^ ((word >> (byte * 8)) & 0xFFULL)) * kFnvPrime;
        }
    }
}
} // namespace

std::shared_ptr<LoadedBundle> KernelBundleLoader::LoadFromMemory(const uint8_t* p, size_t n)
{
    if (p == nullptr || n < sizeof(BundleHeader)) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] buffer too small: %zu", n);
        return nullptr;
    }
    BundleHeader hdr{};
    std::memcpy(&hdr, p, sizeof(BundleHeader));
    if (hdr.magic != kBundleMagic) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] bad magic %#lx", hdr.magic);
        return nullptr;
    }
    if (hdr.version > kBundleVersion) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] unsupported version %u > %u", hdr.version,
                     kBundleVersion);
        return nullptr;
    }
    if (hdr.totalSize != n) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] size mismatch: header=%lu bytes actual=%zu bytes",
                     hdr.totalSize, n);
        return nullptr;
    }
    if (Crc32(p, sizeof(BundleHeader) - sizeof(uint32_t)) != hdr.headerCrc32) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] header CRC mismatch");
        return nullptr;
    }
    const uint64_t tlvRegion = sizeof(BundleHeader) + static_cast<uint64_t>(hdr.tlvCount) * sizeof(TlvHeader);
    if (tlvRegion > n) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] TLV table out of range");
        return nullptr;
    }

    auto b = std::make_shared<LoadedBundle>();
    uint64_t bundleKey = 0xCBF29CE484222325ULL; // FNV-1a offset basis
    for (uint32_t i = 0; i < hdr.tlvCount; ++i) {
        TlvHeader t{};
        std::memcpy(&t, p + sizeof(BundleHeader) + static_cast<size_t>(i) * sizeof(TlvHeader), sizeof(TlvHeader));
        if (t.valueOffset > n || t.valueLength > n - t.valueOffset) {
            MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] TLV[%u] value out of range", i);
            return nullptr;
        }
        const uint8_t* vp = p + t.valueOffset;
        if (Crc32(vp, t.valueLength) != t.valueCrc32) {
            MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] TLV[%u] value CRC mismatch", i);
            return nullptr;
        }
        FoldTlvIntoBundleKey(bundleKey, t.type, t.valueCrc32, t.valueLength);
        switch (static_cast<TlvType>(t.type)) {
            case TlvType::AicoreKernel:
                b->aicoreKernel = CopySegment(vp, t.valueLength);
                break;
            case TlvType::AicpuSo:
                b->aicpuSo = CopySegment(vp, t.valueLength);
                break;
            case TlvType::DevProgram:
                b->devProgram = CopySegment(vp, t.valueLength);
                break;
            case TlvType::CtrlFlowCache:
                b->ctrlFlowCache = CopySegment(vp, t.valueLength);
                break;
            case TlvType::SymbolMeta:
                b->symbolMeta = CopySegment(vp, t.valueLength);
                break;
            default:
                MACHINE_LOGI("[kernel-bundle] skip unknown TLV type %u", t.type);
                break;
        }
    }
    if (b->devProgram.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] missing DEV_PROGRAM segment");
        return nullptr;
    }
    b->bundleKey = bundleKey;
    MACHINE_LOGI("[kernel-bundle] loaded: bundleKey=%#lx hashKey=%#lx workspaceSize=%lu archInfo=%u ctrlCache=%zuB "
                 "symbolMeta=%zuB",
                 b->bundleKey, b->GetHashKey(), b->GetWorkspaceSize(), b->GetArchInfo(), b->ctrlFlowCache.size(),
                 b->symbolMeta.size());
    return b;
}

std::shared_ptr<LoadedBundle> KernelBundleLoader::LoadFromFile(const std::string& path)
{
    bool ok = false;
    std::vector<uint8_t> raw = ReadFile(path, &ok);
    if (!ok || raw.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] failed to read %s", path.c_str());
        return nullptr;
    }
    return LoadFromMemory(raw.data(), raw.size());
}

} // namespace npu::tile_fwk::bundle
