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
 * \file kernel_bundle_crc32.h
 * \brief Standard IEEE-802.3 CRC32 (reflected poly 0xEDB88320). Header-only, table computed on first use.
 *        Matches zlib crc32() / `cksum -o3`, so a packed .pyptokb can be verified with off-the-shelf tools.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace npu::tile_fwk::bundle {

inline const uint32_t* Crc32Table()
{
    static uint32_t table[256];
    static bool inited = false;
    if (!inited) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1U) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        inited = true;
    }
    return table;
}

// CRC32 of a byte range. Chainable via the `crc` seed (pass 0 for a fresh computation).
inline uint32_t Crc32(const void* data, size_t len, uint32_t crc = 0)
{
    const uint32_t* table = Crc32Table();
    const uint8_t* p = static_cast<const uint8_t*>(data);
    crc = crc ^ 0xFFFFFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
}

} // namespace npu::tile_fwk::bundle
