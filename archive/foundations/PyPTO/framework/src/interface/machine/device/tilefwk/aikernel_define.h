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
 * \file aikernel_define.h
 * \brief
 */

#ifndef AIKERNEL_DEFINE_H
#define AIKERNEL_DEFINE_H

#ifndef __gm__

// Host or aicpu
#define IS_AICORE 0
#define __gm__
#ifndef __aicore__
#define __aicore__
#endif
#define INLINE inline
#define __TILE_FWK_HOST__
#define BLOCK_LOCAL

#else

// aicore
#define IS_AICORE 1
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#define INLINE __attribute__((always_inline)) inline __aicore__
#define BLOCK_LOCAL [[block_local]]

#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#ifndef UNUSED
#define UNUSED(n) (void)(n)
#endif

#ifndef ENABLE_AICORE_RESOLVE
#define ENABLE_AICORE_RESOLVE 0
#endif

#endif
