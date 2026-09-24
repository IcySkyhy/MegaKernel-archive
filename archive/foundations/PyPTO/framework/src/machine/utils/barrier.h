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
 * \file barrier.h
 * \brief
 */

#pragma once

#if defined(__x86_64__)
#define rmb() asm volatile("lfence" ::: "memory")
#define wmb() asm volatile("sfence" ::: "memory")
#define cpuRelax() asm volatile("rep;nop" : : : "memory")
#elif defined(__aarch64__)
#define wmb() asm volatile("dmb ishst" ::: "memory")
#define rmb() asm volatile("dmb ishld" ::: "memory")
#define cpuRelax() asm volatile("wfi")
#else
#error "unsupported arch"
#endif

#define memBarrier() asm volatile("" ::: "memory")
