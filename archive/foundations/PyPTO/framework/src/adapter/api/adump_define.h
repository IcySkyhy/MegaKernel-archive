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
 * \file adump_define.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "acl_define.h"

namespace npu::tile_fwk {
constexpr int32_t MAX_KERNEL_BUF_LEN = 1024;
constexpr int32_t MAX_TENSOR_NUM = 128;
enum class AdxDumpType : int32_t {
    OPERATOR = 0x01,
    EXCEPTION = 0x02,
    ARGS_EXCEPTION = 0x03,
    OP_OVERFLOW = 0x04,
    AIC_ERR_DETAIL_DUMP = 0x05 // COREDUMP mode
};

enum class AdxTensorType : int32_t { INPUT, OUTPUT, WORKSPACE };

enum class AdxAddressType : int32_t { TRADITIONAL, NOTILING, RAW };

enum class AdxTensorPlacement : int32_t {
    kOnDeviceHbm, ///< Tensor位于Device上的HBM内存
    kOnHost,      ///< Tensor位于Host
    kFollowing,   ///< Tensor位于Host，且数据紧跟在结构体后面
    kOnDeviceP2p, ///< Tensor位于Device上的P2p内存
    kTensorPlacementEnd
};

enum class AdxExceptionDumpMode : uint32_t {
    ADX_DUMP_MODE_NONE = 0,
    ADX_DUMP_MODE_OVERWRITE = 1,
    ADX_DUMP_MODE_ADDITIONAL = 2,
};

struct AdxTensorInfoV2 {
    AdxTensorType type; // tensor类型
    size_t tensorSize;  // tensor内存大小
    int32_t format;
    int32_t dataType;
    int64_t* tensorAddr;     // tensor数据地址
    AdxAddressType addrType; // 地址的类型
    int32_t placement;
    uint32_t argsOffSet;              // tensor数据地址在args里的偏移
    std::vector<int64_t> shape;       // shape
    std::vector<int64_t> originShape; // originShape
};

struct AdxTensorInfo {
    AdxTensorType type; // tensor类型
    size_t tensorSize;  // tensor内存大小
    int32_t format;
    int32_t dataType;
    int64_t* tensorAddr;     // tensor数据地址
    AdxAddressType addrType; // 地址的类型
    int32_t placement;
    uint32_t argsOffSet;              // tensor数据地址在args里的偏移
    std::vector<int64_t> shape;       // shape
    std::vector<int64_t> originShape; // originShape
};

struct AdxExceptionDumpInfo {
    uint32_t coreId;
    RtCoreType coreType;
    uint32_t argssize;
    void* argAddr;
    void* bin;
    char kernelName[MAX_KERNEL_BUF_LEN];
    char kernelDisplayName[MAX_KERNEL_BUF_LEN];
    uint32_t extraTensorNum;
    AdxTensorInfo tensorInfo[MAX_TENSOR_NUM];
};

typedef int32_t (*AdumpExceptionDumpCallback)(AclRtExceptionInfo* exceptionInfo,
                                              AdxExceptionDumpInfo* exceptionDumpInfo, uint32_t exceptionDumpSize,
                                              uint32_t* exceptionDumpRealSize, AdxExceptionDumpMode* mode);
} // namespace npu::tile_fwk
