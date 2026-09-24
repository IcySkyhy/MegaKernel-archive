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
 * \file common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include "tilefwk/aikernel_data.h"
#include "machine/utils/dynamic/dev_encode_types.h"
#include "tileop/distributed/comm_context.h"

namespace npu::tile_fwk::dynamic {
class AiCoreManager;
}

namespace npu::tile_fwk::Distributed {
constexpr uint64_t AICPU_TASK_ARRAY_SIZE = 1024;
constexpr uint64_t AICPU_TASK_ARRAY_SIZE_MOD = AICPU_TASK_ARRAY_SIZE - 1;
constexpr uint64_t OWNER_RANK_ID_INDEX = 0;
constexpr uint64_t SHMEM_DIM_ROW = 3;
constexpr uint64_t SHMEM_DIM_COL = 4;
constexpr uint64_t ATTR_STRIDE_OFFSET = 1;
constexpr uint64_t ATTR_TILESHAPE_OFFSET = 3;
constexpr uint32_t MAX_TENSOR_DIM = 5;
constexpr uint32_t MAX_SHMEM_TILE_DIMS = 4;
constexpr int32_t AICPU_ATTR_DIM_INDEX = 2;

struct TensorInfo {
    uint64_t rawAddr{0};
    uint32_t dim{0};
    uint64_t rawIndex{0};
    uint64_t vaddr{0};
    int32_t expectedSum{0};
    int32_t signalStride{0};
    bool resetSignal{false};
    std::vector<uint32_t> offset;
    std::vector<uint32_t> shape;
};

struct AicpuParamInfo {
    int32_t outIndex{0};
    int32_t inIndex{0};
    int32_t attrIndex{0};
    int32_t rawShapeIndex{0};
    int32_t shapeIndex{0};

    std::vector<uint32_t> rawShape;
    std::vector<uint32_t> shape;
    uint32_t dim{0};

    uint32_t bufferStride{0};
    std::vector<uint32_t> tileShape;
    uint32_t tileShapeDim{0};
    uint32_t rankNum{0};
    uint32_t maxTileNum{0};

    std::vector<uint32_t> viewshapes;
    std::vector<uint32_t> viewTileStrides;
    std::vector<uint32_t> viewIndexStrides;
    uint32_t viewTileNum{0};
    uint32_t totalTileNum{0};
};

inline uint64_t MapVirtualSignalAddr(int64_t* hcclContextAddr, uint64_t vaddr)
{
    uint64_t groupIndex = TileOp::Distributed::DecodeShmemAddrGroupIndex(vaddr);
    uint64_t offset = TileOp::Distributed::DecodeShmemAddrOffset(vaddr);
    auto hcclOpParam = reinterpret_cast<TileOp::CommContext*>(hcclContextAddr[groupIndex]);
    auto winAddrOffset = hcclOpParam->statusIndex + hcclOpParam->rankId;
    return hcclOpParam->winAddr[winAddrOffset] + offset;
}

inline uint64_t GetRankNum(int64_t* hcclContextAddr, uint64_t vaddr)
{
    uint64_t groupIndex = TileOp::Distributed::DecodeShmemAddrGroupIndex(vaddr);
    auto hcclOpParam = reinterpret_cast<TileOp::CommContext*>(hcclContextAddr[groupIndex]);
    return hcclOpParam->rankNum;
}

inline uint64_t GetCoa(const uint32_t index, uint64_t* opAttrs, uint64_t* expressionTable)
{
    constexpr uint64_t valueLength = 63;
    constexpr uint64_t valueMask = (1UL << valueLength) - 1;
    const uint64_t encodedValue = opAttrs[index];
    const bool isExpression = (encodedValue >> valueLength) & 1;
    const uint64_t decodedValue = encodedValue & valueMask;
    return isExpression ? expressionTable[decodedValue] : decodedValue;
}

inline std::vector<uint32_t> GetCoaVector(const uint32_t baseIndex, const uint32_t dim, uint64_t* opAttrs,
                                          uint64_t* expressionTable)
{
    std::vector<uint32_t> vec(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        vec[i] = GetCoa(baseIndex + i, opAttrs, expressionTable);
    }
    return vec;
}

inline unsigned CalcLinearOffset(unsigned GmShape1, unsigned Offset0, unsigned Offset1)
{
    return Offset1 + Offset0 * GmShape1;
}

inline AicpuParamInfo DecodeAicpuCode(const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode)
{
    AicpuParamInfo paramInfo;
    int index = 1; // aicpuCode[0]表示OpCode，paraminfo索引从1起
    paramInfo.outIndex = index + 1;

    index = index + aicpuCode[index] + 1;
    paramInfo.inIndex = index + 1;

    index = index + aicpuCode[index] + 1;
    paramInfo.rawShapeIndex = index + 1;

    paramInfo.dim = aicpuCode[paramInfo.inIndex + AICPU_ATTR_DIM_INDEX];

    paramInfo.rawShape.assign(aicpuCode.begin() + paramInfo.rawShapeIndex,
                              aicpuCode.begin() + paramInfo.rawShapeIndex + paramInfo.dim);

    paramInfo.shapeIndex = paramInfo.rawShapeIndex + paramInfo.dim;

    paramInfo.shape.assign(aicpuCode.begin() + paramInfo.shapeIndex,
                           aicpuCode.begin() + paramInfo.shapeIndex + paramInfo.dim);

    index = index + aicpuCode[index] + 1;
    if (index + 1 < static_cast<int32_t>(aicpuCode.size())) {
        paramInfo.attrIndex = index + 1;
    }
    paramInfo.bufferStride = aicpuCode[paramInfo.attrIndex + ATTR_STRIDE_OFFSET];
    uint32_t tileShapeDim = aicpuCode[paramInfo.attrIndex + ATTR_TILESHAPE_OFFSET];
    paramInfo.tileShape.assign(aicpuCode.begin() + paramInfo.attrIndex + ATTR_TILESHAPE_OFFSET + 1,
                               aicpuCode.begin() + paramInfo.attrIndex + ATTR_TILESHAPE_OFFSET + tileShapeDim + 1);

    uint32_t baseOffset = ATTR_TILESHAPE_OFFSET + tileShapeDim + 1;
    paramInfo.viewshapes.assign(aicpuCode.begin() + paramInfo.attrIndex + baseOffset,
                                aicpuCode.begin() + paramInfo.attrIndex + baseOffset + tileShapeDim);

    baseOffset += tileShapeDim;
    paramInfo.viewTileStrides.assign(aicpuCode.begin() + paramInfo.attrIndex + baseOffset,
                                     aicpuCode.begin() + paramInfo.attrIndex + baseOffset + tileShapeDim);

    baseOffset += tileShapeDim;
    paramInfo.viewIndexStrides.assign(aicpuCode.begin() + paramInfo.attrIndex + baseOffset,
                                      aicpuCode.begin() + paramInfo.attrIndex + baseOffset + tileShapeDim);

    baseOffset += tileShapeDim;
    paramInfo.viewTileNum = aicpuCode[paramInfo.attrIndex + baseOffset];
    paramInfo.totalTileNum = aicpuCode[paramInfo.attrIndex + baseOffset + 1];

    return paramInfo;
}
} // namespace npu::tile_fwk::Distributed
