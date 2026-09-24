/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aikernel_tensor.h
 * \brief
 */

#ifndef AIKERNEL_TENSOR_H
#define AIKERNEL_TENSOR_H

#include <atomic>
#include "tilefwk/aikernel_define.h"

namespace npu::tile_fwk {

constexpr int32_t DEV_SHAPE_DIM_MAX = 6;
struct DevShape {
    int dimSize{0};
    int dim[DEV_SHAPE_DIM_MAX];

#ifdef __TILE_FWK_HOST__
    int64_t GetSize() const
    {
        int64_t size = 1;
        for (int idx = 0; idx < dimSize; idx++) {
            size *= dim[idx];
        }
        return size;
    }

    bool Equal(const DevShape& s) const
    {
        if (dimSize != s.dimSize) {
            return false;
        }
        for (int i = 0; i < dimSize; i++) {
            if (dim[i] != s.dim[i]) {
                return false;
            }
        }
        return true;
    }
#endif
};

constexpr uint32_t DEV_TENSOR_DATA_OFFSET = 2;
struct DevTensorData {
    uint64_t address{0};
    DevShape shape;
    int32_t dataType;
};

const uint32_t RAW_TENSOR_LOCATION_LOCAL = 0;
const uint32_t RAW_TENSOR_LOCATION_INCAST = 1;
const uint32_t RAW_TENSOR_LOCATION_OUTCAST = 2;
struct DevRawTensorDesc {
    uint32_t location;
    uint32_t offsetOrIndex;
};

} // namespace npu::tile_fwk

#endif
