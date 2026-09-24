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
 * \file kernel_bundle_format.h
 * \brief TLV on-disk format for the PyPTO kernel bundle (.pyptokb).
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace npu::tile_fwk::bundle {

constexpr uint64_t kBundleMagic = 0x00424B4F54505950ULL; // "PYPTOKB\0" (little-endian)
constexpr uint32_t kBundleVersion = 1;
constexpr uint32_t kValueAlign = 4096; // 4KB page alignment (mmap + page-wise CopyDataToDevice)

enum class TlvType : uint32_t {
    AicoreKernel = 1,  // dy_kernel_<hash>_0.o, opaque ELF blob for device
    AicpuSo = 2,       // libtilefwk_backend_server.so, globally shared
    DevProgram = 3,    // EncodeDevAscendProgram flat binary, base-0 relative offsets
    CtrlFlowCache = 4, // DevControlFlowCache blob (v1: single static-shape snapshot)
    SymbolMeta = 5,    // JSON: dynamic-workspace SymbolicScalar trees + inputSymbolDict (may be empty/absent)
};

struct BundleHeader { // 64 bytes
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t tlvCount;
    uint32_t headerSize;
    uint64_t totalSize;
    uint8_t reserved[28];
    uint32_t headerCrc32; // CRC32 over the first 60 bytes
};
static_assert(sizeof(BundleHeader) == 64, "BundleHeader must be 64 bytes");

struct TlvHeader { // 32 bytes
    uint32_t type; // TlvType
    uint32_t flags;
    uint64_t valueOffset; // absolute file offset of value (4KB aligned)
    uint64_t valueLength; // value length in bytes (excluding padding)
    uint32_t valueCrc32;  // CRC32 over value bytes
    uint32_t reserved;
};
static_assert(sizeof(TlvHeader) == 32, "TlvHeader must be 32 bytes");

// Feature switch. Packing is enabled only when env PYPTO_ENABLE_KERNEL_BUNDLE == "1".
// Read from the environment exactly once and cached: the launch path queries this on every launch, so a
// getenv() per call would be pure overhead in the (far more common) bundle-disabled scenario. As a consequence
// the variable must be set before the first launch; changing it mid-process has no effect.
inline bool IsKernelBundleEnabled()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PYPTO_ENABLE_KERNEL_BUNDLE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

// Output path for the produced bundle. Overridable via PYPTO_KERNEL_BUNDLE_PATH; defaults to ./foo.pyptokb.
// Cached on first use for the same reason as the switch above.
inline const std::string& KernelBundleOutPath()
{
    static const std::string path = []() {
        const char* p = std::getenv("PYPTO_KERNEL_BUNDLE_PATH");
        return (p != nullptr && p[0] != '\0') ? std::string(p) : std::string("foo.pyptokb");
    }();
    return path;
}

} // namespace npu::tile_fwk::bundle
