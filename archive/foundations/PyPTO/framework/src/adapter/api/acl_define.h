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
 * \file acl_define.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {
constexpr int ACLRT_SUCCESS = 0;
constexpr int ACLRT_ERROR_REPEAT_INITIALIZE = 100002;

#define ACL_ERROR_RT_FEATURE_NOT_SUPPORT 207000 // feature not support
#define ACL_EVENT_SYNC 0x00000001U

typedef int AclError;
typedef void* AclRtStream;
typedef void* AclRtEvent;
typedef void* AclRtBinHandle;
typedef void* AclRtFuncHandle;

enum class AclRtBinaryLoadOptionType {
    LAZY_LOAD = 1,
    MAGIC = 2,
    CPU_KERNEL_MODE = 3,
};

typedef union {
    uint32_t isLazyLoad;
    uint32_t magic;
    int32_t cpuKernelMode;
    uint32_t rsv[4];
} AclRtBinaryLoadOptionValue;

typedef struct {
    AclRtBinaryLoadOptionType type;
    AclRtBinaryLoadOptionValue value;
} AclRtBinaryLoadOption;

typedef struct {
    AclRtBinaryLoadOption* options;
    size_t numOpt;
} AclRtBinaryLoadOptions;

enum class AclRtMemcpyKind {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEFAULT,
    HOST_TO_BUF_TO_DEVICE,
    INNER_DEVICE_TO_DEVICE,
    INTER_DEVICE_TO_DEVICE,
};

enum class AclRtMemMallocPolicy {
    HUGE_FIRST,
    HUGE_ONLY,
    NORMAL_ONLY,
    HUGE_FIRST_P2P,
    HUGE_ONLY_P2P,
    NORMAL_ONLY_P2P,
    HUGE1G_ONLY,
    HUGE1G_ONLY_P2P,
    LOW_BAND_WIDTH = 0x0100,
    HIGH_BAND_WIDTH = 0x1000,
    ACCESS_USER_SPACE_READONLY = 0x100000,
};

enum class AclRtStreamAttr {
    FAILURE_MODE = 1,
    FLOAT_OVERFLOW_CHECK = 2,
    USER_CUSTOM_TAG = 3,
    CACHE_OP_INFO = 4,
};

typedef union {
    uint64_t failureMode;
    uint32_t overflowSwitch;
    uint32_t userCustomTag;
    uint32_t cacheOpInfoSwitch;
    uint32_t reserve[4];
} AclRtStreamAttrValue;

enum class AclRtDevResLimitType {
    CUBE_CORE = 0,
    VECTOR_CORE,
};

typedef struct RtExceptionInfo AclRtExceptionInfo;

typedef void (*AclRtExceptionInfoCallback)(AclRtExceptionInfo* exceptionInfo);

typedef void* AclMdlRI;

enum class AclMdlRICaptureStatus {
    NONE = 0,
    ACTIVE,
    INVALIDATED,
};

enum class AclMdlRICaptureMode {
    GLOBAL = 0,
    THREAD_LOCAL,
    RELAXED,
};
} // namespace npu::tile_fwk
