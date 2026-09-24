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
 * \file cycles.cpp
 * \brief
 */

#include <cassert>
#include <algorithm>
#include "cycles.h"
#include "tilefwk/error_code.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {
constexpr const int BYTES_PER_REPEAT = 256;
constexpr const int DEFAULT_MAX_PARALLELISM = 128;
constexpr const int DEFAULT_LATENCY = 10;

// get element per cycle
int GetParallelism(const std::string& op, DataType dtype)
{
    auto iterTileOp = INTRIN_PARALLELISM_IN_OP.find(op);
    if (iterTileOp == INTRIN_PARALLELISM_IN_OP.end()) {
        return DEFAULT_MAX_PARALLELISM;
    }
    auto iterDtype = iterTileOp->second.find(dtype);
    if (iterDtype == iterTileOp->second.end()) {
        return DEFAULT_MAX_PARALLELISM;
    }
    return iterDtype->second;
}

// used to extend in future
int GetLatency(const std::string& op, DataType dtype)
{
    auto iterTileOp = INTRIN_LATENCY_IN_OP.find(op);
    if (iterTileOp == INTRIN_LATENCY_IN_OP.end()) {
        return DEFAULT_LATENCY;
    }
    auto iterDtype = iterTileOp->second.find(dtype);
    if (iterDtype == iterTileOp->second.end()) {
        return DEFAULT_LATENCY;
    }
    return iterDtype->second;
}

int64_t GetMaxShapeSize(const std::vector<std::vector<int64_t>>& shape)
{
    int64_t maxTotalSize = 0;
    for (const auto& i : shape) {
        int64_t totalSize = 1;
        for (auto dimVal : i) {
            totalSize *= dimVal;
        }
        maxTotalSize = std::max(maxTotalSize, totalSize);
    }
    return maxTotalSize;
}

int64_t GetGatherInUBResultShapeSize(const std::vector<std::vector<int64_t>>& shape)
{
    // GatherInUB fixed scenario:
    // param: [token_size, hidden_dim], indices: [1, k], block_table: [1, ...]
    // result: [k, hidden_dim]
    // Use result tile size for sparse gather estimation.
    if (shape.size() < 2 || shape[0].empty() || shape[1].empty()) {
        return GetMaxShapeSize(shape);
    }

    int64_t hiddenDim = shape[0].back();
    if (hiddenDim <= 0) {
        return GetMaxShapeSize(shape);
    }

    int64_t gatheredCount = 1;
    for (int64_t dimVal : shape[1]) {
        if (dimVal <= 0) {
            return GetMaxShapeSize(shape);
        }
        gatheredCount *= dimVal;
    }

    return gatheredCount * hiddenDim;
}

int64_t CalcCyclesCommon(const std::string& op, int64_t shapeSize, DataType dtype)
{
    int64_t totalSize = shapeSize * BytesOf(dtype);

    int64_t elePerRepeat = BYTES_PER_REPEAT / BytesOf(dtype);
    int64_t parallelism = GetParallelism(op, dtype);
    int64_t cyclePerRepeat = elePerRepeat / parallelism;
    if (cyclePerRepeat == 0) {
        cyclePerRepeat = 1;
    }

    int64_t repeatCount = (totalSize + BYTES_PER_REPEAT - 1) / BYTES_PER_REPEAT;
    int64_t latency = GetLatency(op, dtype);
    int64_t cycle = latency + (repeatCount - 1) * cyclePerRepeat;
    return cycle;
}

// according to implementation in tile op instruction
int64_t CalcUBCompactCycles(const std::vector<std::vector<int64_t>>& shape, DataType dtype)
{
    int64_t srcShape0 = shape[1][0];
    int64_t dstShape0 = shape[0][0];
    constexpr int32_t SRC_SHAPE_16 = 16;
    int64_t vnchwconvRegSetScala = 4;
    int64_t vnchwconvBytePerCycle = 512;
    if (srcShape0 < SRC_SHAPE_16) {
        return dstShape0 * vnchwconvRegSetScala;
    }
    int64_t shapeSize = GetMaxShapeSize(shape);
    int64_t totalBytes = shapeSize * BytesOf(dtype);
    if (totalBytes / vnchwconvBytePerCycle < 1) {
        return 1 + vnchwconvRegSetScala;
    }
    int64_t vnchwconvCycle = totalBytes / vnchwconvBytePerCycle + vnchwconvRegSetScala;
    int64_t copyUbToUbCycle = CalcCyclesCommon("UB_MOV", shapeSize, dtype);
    return vnchwconvCycle + copyUbToUbCycle;
}

int64_t GetCycles(const std::string& op, const std::vector<std::vector<int64_t>>& shape, DataType dtype)
{
    if (op == "NOP") {
        return 0;
    }
    // for sync op
    auto iterSyncOp = SYNC_OP_CYCLES.find(op);
    if (iterSyncOp != SYNC_OP_CYCLES.end()) {
        return iterSyncOp->second;
    }

    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, !shape.empty() && !shape[0].empty()) << "shape is invalid";

    // assume that the cycle of UB_ALLOC, L1_ALLOC, etc. is 1
    if (op.find("_ALLOC") != std::string::npos) {
        return 1;
    }

    if (op == "GATHER_IN_UB") {
        int64_t shapeSize = GetGatherInUBResultShapeSize(shape);
        return CalcCyclesCommon(op, shapeSize, dtype);
    }

    auto iterCombineIntrin = COMINE_INTRIN_CYCLES_IN_OP.find(op);
    if (iterCombineIntrin != COMINE_INTRIN_CYCLES_IN_OP.end()) {
        return iterCombineIntrin->second(shape, dtype);
    }

    int64_t shapeSize = GetMaxShapeSize(shape);
    int64_t cycle = CalcCyclesCommon(op, shapeSize, dtype);
    return cycle;
}
} // namespace npu::tile_fwk
