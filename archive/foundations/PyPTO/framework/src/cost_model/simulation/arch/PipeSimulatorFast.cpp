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
 * \file PipeSimulatorFast.cpp
 * \brief
 */

#include "PipeSimulatorFast.h"

#include <cassert>
#include <climits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "PipeFactory.h"
#include "cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h"
#include "cost_model/simulation/arch/A5/PostSimulatorA5.h"

namespace CostModel {
template class PipeSimulatorFast<PostSimulatorA2A3>;
template class PipeSimulatorFast<PostSimulatorA5>;
static const std::unordered_map<std::string, std::unordered_map<DataType, int>> INHERENT_LATENCY_IN_OP{
    // Vector
    {"UB_ADD",
     {
         {DataType::DT_FP32, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_SUB",
     {
         {DataType::DT_FP16, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_MUL",
     {
         {DataType::DT_FP16, 7},
         {DataType::DT_FP32, 7},
         {DataType::DT_INT32, 5},
         {DataType::DT_INT16, 5},
     }},
    {"UB_DIV",
     {
         {DataType::DT_FP16, 13},
         {DataType::DT_FP32, 13},
     }},
    {"UB_EXP",
     {
         {DataType::DT_FP16, 12},
         {DataType::DT_FP32, 12},
     }},
    {"UB_SQRT",
     {
         {DataType::DT_FP16, 13},
         {DataType::DT_FP32, 13},
     }},
    {"UB_RSQRT",
     {
         {DataType::DT_FP16, 5},
         {DataType::DT_FP32, 5},
     }},
    {"UB_ADDS",
     {
         {DataType::DT_FP16, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_MULS",
     {
         {DataType::DT_FP16, 7},
         {DataType::DT_FP32, 7},
         {DataType::DT_INT32, 5},
         {DataType::DT_INT16, 5},
     }},
    {"UB_PAIRSUM",
     {
         {DataType::DT_FP16, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_PAIRMAX",
     {
         {DataType::DT_FP16, 4},
         {DataType::DT_FP32, 4},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_ROWEXPSUM",
     {
         {DataType::DT_FP16, 23},
         {DataType::DT_FP32, 20},
     }},
    {"UB_ROWEXPMAX",
     {
         //  vcadd + vector_dup(no data)
         {DataType::DT_FP16, 23},
         {DataType::DT_FP32, 20},
     }},
    {"UB_MAXIMUM",
     {
         {DataType::DT_FP16, 4},
         {DataType::DT_FP32, 4},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_COMPACT",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_EXPAND",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_MOV",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_RECIPROCAL",
     {
         {DataType::DT_FP16, 5},
         {DataType::DT_FP32, 5},
     }},
    {"UB_ROWSUM",
     {
         {DataType::DT_FP16, 23},
         {DataType::DT_FP32, 20},
     }},
    {"UB_ROWMAX",
     {
         {DataType::DT_FP16, 9},
         {DataType::DT_FP32, 8},
     }},
    {"L0C_COPY_OUT",
     {
         {DataType::DT_FP16, 200},
         {DataType::DT_FP32, 200},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_COPY_IN",
     {
         {DataType::DT_FP16, 50},
         {DataType::DT_FP32, 50},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_COPY_IN",
     {
         {DataType::DT_FP16, 200},
         {DataType::DT_FP32, 200},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0A",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0At",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0B",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0Bt",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"CUBE_A_MUL_B",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
    {"CUBE_A_MUL_Bt",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
    {"CUBE_A_MULACC_B",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
    {"CUBE_A_MULACC_Bt",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }}};

constexpr const int DEFAULT_SHAPE = 256;
constexpr const int BYTES_PER_REPEAT = 256;
constexpr const int DEFAULT_LATENCY = 10;
constexpr const int CORNER_DIM_VAL = 2;
constexpr const int PARALLEL_RATIO_1 = 1;
constexpr const int PARALLEL_RATIO_2 = 2;

// used to extend in future
static int GetLatency(const std::string& op, DataType dtype)
{
    auto iterTileOp = INHERENT_LATENCY_IN_OP.find(op);
    if (iterTileOp == INHERENT_LATENCY_IN_OP.end()) {
        return DEFAULT_LATENCY;
    }
    auto iterDtype = iterTileOp->second.find(dtype);
    if (iterDtype == iterTileOp->second.end()) {
        return DEFAULT_LATENCY;
    }
    return iterDtype->second;
}

static int GetMinShapeSize(const std::vector<std::vector<int>>& shape)
{
    int minTotalSize = INT_MAX;
    for (const auto& shapeVal : shape) {
        int totalSize = 1;
        unsigned int beginVal = shapeVal.size() - CORNER_DIM_VAL;
        for (unsigned int j = beginVal; j < shapeVal.size(); j++) {
            totalSize *= shapeVal[j] > 0 ? shapeVal[j] : DEFAULT_SHAPE;
        }
        minTotalSize = std::min(minTotalSize, totalSize);
    }
    return minTotalSize;
}

static int GetShapeCntSize(const std::vector<std::vector<int>>& shape)
{
    int minTotalSize = INT_MAX;
    for (const auto& shapeVal : shape) {
        int totalSize = 1;
        if (shapeVal.size() > CORNER_DIM_VAL) {
            for (unsigned int j = 0; j < shapeVal.size() - CORNER_DIM_VAL; j++) {
                totalSize *= shapeVal[j] > 0 ? shapeVal[j] : DEFAULT_SHAPE;
            }
        }
        minTotalSize = std::min(minTotalSize, totalSize);
    }
    return minTotalSize;
}

static int CalcCyclesCommon(const std::string& op, int shapeSize, DataType dtype)
{
    int totalSize = shapeSize * BytesOf(dtype);
    int parallelism = 128;
    int cyclePerRepeat = BYTES_PER_REPEAT / parallelism;
    if (cyclePerRepeat == 0) {
        cyclePerRepeat = 1;
    }

    int repeatCount = ((totalSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT) + 1;
    int latency = GetLatency(op, dtype);
    int cycle = latency + repeatCount * cyclePerRepeat - 1;
    return cycle;
}

template <typename PostSimulator>
uint64_t PipeSimulatorFast<PostSimulator>::Simulate(const TileOpPtr& tileOp)
{
    std::string op = tileOp->opcode;
    DataType dtype = tileOp->iOperand.size() > 0 ? tileOp->iOperand[0]->dataType : tileOp->oOperand[0]->dataType;

    std::vector<std::vector<int>> shape;
    for (auto& srcTile : tileOp->iOperand) {
        shape.emplace_back(srcTile->shape);
    }
    for (auto& dstTile : tileOp->oOperand) {
        shape.emplace_back(dstTile->shape);
    }

    ASSERT(CostModel::ForwardSimErrorScene::SHAPE_INVALID, !shape.empty() && !shape[0].empty()) << "[SIMULATION]: "
                                                                                                << "shape is invalid";

    int shapeSize = GetMinShapeSize(shape);
    int shapeCnt = GetShapeCntSize(shape);
    int cycle = CalcCyclesCommon(op, shapeSize * shapeCnt, dtype);
    return cycle;
}

uint64_t GetParrelRatio(DataType t, const std::string& op)
{
    if (t == DataType::DT_INT32) {
        if (op == "MULS" || op == "MUL")
            return PARALLEL_RATIO_1;
    }
    if (t == DataType::DT_FP32) {
        if (op == "DIV" || op == "EXP" || op == "LN" || op == "SQRT")
            return PARALLEL_RATIO_1;
    }
    return PARALLEL_RATIO_2;
}

std::string GetOpCode(const TileOpPtr& tileOp)
{
    if (tileOp->opcode == "COPY_IN") {
        static const std::unordered_map<OperandType, std::string> COPY_IN_OP{
            {BUF_L1, "L1_COPY_IN"},
            {BUF_UB, "UB_COPY_IN"},
        };
        if (COPY_IN_OP.find(tileOp->bufType) != COPY_IN_OP.end()) {
            return COPY_IN_OP.at(tileOp->bufType);
        }
    } else if (tileOp->opcode == "COPY_OUT") {
        static const std::unordered_map<OperandType, std::string> COPY_OUT_OP{
            {BUF_L0C, "L0C_COPY_OUT"},
            {BUF_UB, "UB_COPY_OUT"},
        };
        if (COPY_OUT_OP.find(tileOp->bufType) != COPY_OUT_OP.end()) {
            return COPY_OUT_OP.at(tileOp->bufType);
        }
    }
    return tileOp->opcode;
}

template <typename PostSimulator>
uint64_t PipeSimulatorFast<PostSimulator>::PostSimulate(const TileOpPtr& tileOp)
{
    std::string op = GetOpCode(tileOp);
    PostSimulator psm;
    auto opLatency = psm.GetOpLatency();
    if (opLatency.find(op) == opLatency.end()) {
        return Simulate(tileOp);
    }
    std::vector<std::vector<int>> shape;
    for (auto& srcTile : tileOp->iOperand) {
        shape.emplace_back(srcTile->shape);
    }
    for (auto& dstTile : tileOp->oOperand) {
        shape.emplace_back(dstTile->shape);
    }
    int shapeSize = GetMinShapeSize(shape);
    int shapeCnt = GetShapeCntSize(shape);
    auto dataType = tileOp->iOperand.size() > 0 ? tileOp->iOperand[0]->dataType : tileOp->oOperand[0]->dataType;
    uint64_t size = BytesOf(dataType) * shapeSize;
    uint64_t parallelRatio = GetParrelRatio(dataType, op);
    auto r = opLatency.at(op);
    const int rGmLatency = 1400;
    const int wGmLatency = 300;
    int freqTrans = 2000;
    int latency = shapeCnt * (uint64_t((r[0] * size * parallelRatio * 1E-6 + r[1]) * freqTrans) + 1);
    if (tileOp->opcode == "COPY_IN") {
        latency += rGmLatency;
    } else if (tileOp->opcode == "COPY_OUT") {
        latency += wGmLatency;
    }
    return latency;
}

template <typename PostSimulator>
uint64_t PipeSimulatorFast<PostSimulator>::SimulateForPass(const std::string& op,
                                                           const std::vector<std::vector<int>>& shape, DataType dtype)
{
    ASSERT(CostModel::ForwardSimErrorScene::SHAPE_INVALID, !shape.empty() && !shape[0].empty()) << "[SIMULATION]: "
                                                                                                << "shape is invalid";
    int shapeSize = GetMinShapeSize(shape);
    int shapeCnt = GetShapeCntSize(shape);
    int cycle = CalcCyclesCommon(op, shapeSize * shapeCnt, dtype);
    return cycle;
}

template <typename PostSimulator>
uint64_t PipeSimulatorFast<PostSimulator>::PostSimulateForPass(const std::string& op,
                                                               const std::vector<std::vector<int>>& shape,
                                                               DataType dtype)
{
    PostSimulator psm;
    auto opLatency = psm.GetOpLatency();
    if (opLatency.find(op) == opLatency.end()) {
        return SimulateForPass(op, shape, dtype);
    }
    int shapeSize = GetMinShapeSize(shape);
    int shapeCnt = GetShapeCntSize(shape);
    uint64_t size = BytesOf(dtype) * shapeSize;
    uint64_t parallelRatio = GetParrelRatio(dtype, op);
    auto r = opLatency.at(op);
    const int rGmLatency = 1400;
    const int wGmLatency = 300;
    int freqTrans = 2000;
    int latency = shapeCnt * (uint64_t((r[0] * size * parallelRatio * 1E-6 + r[1]) * freqTrans) + 1);
    if (op == "COPY_IN") {
        latency += rGmLatency;
    } else if (op == "COPY_OUT") {
        latency += wGmLatency;
    }
    return latency;
}

extern "C" __attribute__((visibility("default"))) int64_t GetCyclesForPass(const std::string& op,
                                                                           const std::vector<std::vector<int>>& shape,
                                                                           DataType dtype)
{
    std::string archType = config::GetPlatformConfig("device_platform", "ASCEND_950PR_9579") == "ASCEND_950PR_9579" ?
                               "A5" :
                               "A2A3";
    auto simPtr = CreateSimulator(archType);
    return simPtr->PostSimulateForPass(op, shape, dtype);
}
} // namespace CostModel
