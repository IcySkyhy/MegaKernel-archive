/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cache.h
 * \brief
 */

#pragma once

#include <stdint.h>
#ifdef __aarch64__
inline void flush_dcache_range(void* start, uint64_t size)
{
    constexpr int64_t CACHE_LINE_SIZE = 64;
    uint64_t xstart = (uint64_t)start & ~(CACHE_LINE_SIZE - 1);
    uint64_t xend = (uint64_t)start + size;
    while (xstart < xend) {
        asm volatile("dc civac, %0" : : "r"(xstart) : "memory");
        xstart += CACHE_LINE_SIZE;
    }
    asm volatile("dsb sy" : : : "memory");
}
#else
inline void flush_dcache_range(void* start, uint64_t size)
{
    (void)start;
    (void)size;
}
#endif
