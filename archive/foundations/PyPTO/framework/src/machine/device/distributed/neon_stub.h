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
 * \file neon_stub.h
 * \brief
 */

#ifndef NEON_STUB_H
#define NEON_STUB_H

// #ifdef NO_NEON_SUPPORT
#if defined(NO_NEON_SUPPORT) || !defined(__aarch64__)
#include <cstdint>
using uint8x16_t = int8_t;
using uint64x2_t = int64_t;

inline uint8x16_t vld1q_u8(uint8_t const* ptr)
{
    (void)ptr;
    return 0;
}

inline uint8x16_t vandq_u8(uint8x16_t a, uint8x16_t b)
{
    (void)a;
    (void)b;
    return 0;
}

inline uint64x2_t vreinterpretq_u64_u8(uint8x16_t a)
{
    (void)a;
    return 0;
}

inline uint64_t vgetq_lane_u64(uint64x2_t v, const int lane)
{
    (void)v;
    (void)lane;
    return 0;
}

inline void vst1q_u8(uint8_t* ptr, uint8x16_t val)
{
    (void)ptr;
    (void)val;
}
#else
#include <arm_neon.h>
#endif

#endif // NEON_STUB_H
