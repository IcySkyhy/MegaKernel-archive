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
 * \file cube_tile_setting.cpp
 * \brief Cube tile setting for matmul operations
 */

#include <climits>
#include <cmath>
#include <algorithm>
#include "set_heuristic_tile_shapes.h"
#include "interface/function/function.h"
#include "interface/operation/operation_impl.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SetHeuristicTileShapes"

namespace npu::tile_fwk {

std::map<int, int> FordBellman(const std::vector<std::pair<int, int>>& edges, Function& function)
{
    std::map<int, int> subgrDepthMap;
    for (auto& op : function.Operations()) {
        subgrDepthMap[op.GetOpMagic()] = INT_MAX;
    }
    subgrDepthMap[-1] = 0;
    bool any = true;
    size_t maxIter = subgrDepthMap.size() - 1;
    while (any == true && maxIter > 0) {
        any = false;
        maxIter--;
        for (auto elem : edges) {
            if (subgrDepthMap[elem.second] > subgrDepthMap[elem.first] - 1) {
                subgrDepthMap[elem.second] = subgrDepthMap[elem.first] - 1;
                any = true;
            }
        }
    }
    return subgrDepthMap;
}

void FindCubeTilesCombinations(std::map<std::vector<int64_t>, double>& setOfCubeTiles, int64_t m, int64_t k, int64_t n,
                               int64_t inputTypeSize)
{
    const int64_t L0A_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    const int64_t L0B_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    const int64_t L0C_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    std::vector<int64_t> tmpTile = {0, 0, 0};
    int64_t l0aReq = m * k * inputTypeSize;
    int64_t l0bReq = k * n * inputTypeSize;
    int64_t l0cReq = m * n * OUTPUT_TYPE_SIZE;
    int64_t l0aLimit = L0A_MAX_SIZE / DOUBLE_BUFFER;
    int64_t l0bLimit = L0B_MAX_SIZE / DOUBLE_BUFFER;
    int64_t l0cLimit = L0C_MAX_SIZE / DOUBLE_BUFFER_L0C;
    if ((l0aReq <= l0aLimit) && (l0bReq <= l0bLimit) && (l0cReq <= l0cLimit)) {
        tmpTile[M_DIM] = m;
        tmpTile[K_DIM] = k;
        tmpTile[N_DIM] = n;
        setOfCubeTiles[tmpTile] = 0.f;
    }
}

static void CalculateUtilizationScore(const std::vector<int64_t>& tile, double& score, int64_t M, int64_t K, int64_t N,
                                      int64_t inputTypeSize, int64_t L0A_MAX_SIZE, int64_t L0B_MAX_SIZE,
                                      int64_t L0C_MAX_SIZE)
{
    score = (tile[M_DIM] == std::max(M, MIN_TILE)) ? (score + WHOLE_M_SCORE) : score;
    score = (tile[K_DIM] == std::max(K, MIN_TILE)) ? (score + WHOLE_K_SCORE) : score;
    score = (tile[N_DIM] == std::max(N, MIN_TILE)) ? (score + WHOLE_N_SCORE) : score;
    double utilizationL0A = static_cast<double>((tile[M_DIM] * tile[K_DIM] * inputTypeSize)) /
                            (L0A_MAX_SIZE / DOUBLE_BUFFER);
    double utilizationL0B = static_cast<double>((tile[K_DIM] * tile[N_DIM] * inputTypeSize)) /
                            (L0B_MAX_SIZE / DOUBLE_BUFFER);
    double utilizationL0C = static_cast<double>((tile[M_DIM] * tile[N_DIM] * OUTPUT_TYPE_SIZE)) /
                            (L0C_MAX_SIZE / DOUBLE_BUFFER_L0C);
    std::vector<double> vectorRatio = {utilizationL0A, utilizationL0B, utilizationL0C};
    double geomeanUtilizationL0 = CalculateGeometricMean(vectorRatio);
    score += WEIGHT_L0 * geomeanUtilizationL0;
}

static void CalculateTaskScore(double& score, int64_t M, int64_t N, const std::vector<int64_t>& tile,
                               int64_t cubeL1Reuse, int64_t cubeNBuffer, int64_t numOfMatmuls, int64_t CUBE_CORES)
{
    double tasks = numOfMatmuls * (std::max(M, MIN_TILE) / static_cast<double>(tile[M_DIM])) *
                   (std::max(N, MIN_TILE) / static_cast<double>(tile[N_DIM])) / (cubeL1Reuse * cubeNBuffer);
    double tasksRatioLess = (tasks < CUBE_CORES) ? (CUBE_CORES / tasks - 1) : 0;
    double tasksRatioMore = (tasks > NUM2 * CUBE_CORES) ? (tasks / (NUM2 * CUBE_CORES) - 1) : 0;
    score -= TASKS_CUBE_WEIGHT * (tasksRatioLess + tasksRatioMore);
    int64_t residualTasks = (static_cast<int64_t>(std::ceil(tasks)) % static_cast<int64_t>(CUBE_CORES) == 0) ?
                                static_cast<int64_t>(CUBE_CORES) :
                                (static_cast<int64_t>(std::ceil(tasks)) % static_cast<int64_t>(CUBE_CORES));
    score += RESIDUAL_CUBE_TASKS_WEIGHT * residualTasks;
}

static void CalculateBalanceAndCycleScore(double& score, const std::vector<int64_t>& tile, const uint64_t inputMKN,
                                          const uint64_t mkn, const int64_t cubeL1Reuse, const int64_t cubeNBuffer,
                                          const int64_t inputTypeSize, const int64_t inputType)
{
    if (mkn == 0) {
        return;
    }
    double ratioMK = (tile[M_DIM] > tile[K_DIM]) ? static_cast<double>(tile[M_DIM]) / static_cast<double>(tile[K_DIM]) :
                                                   static_cast<double>(tile[K_DIM]) / static_cast<double>(tile[M_DIM]);
    double ratioKN = (tile[K_DIM] > tile[N_DIM]) ? static_cast<double>(tile[K_DIM]) / static_cast<double>(tile[N_DIM]) :
                                                   static_cast<double>(tile[N_DIM]) / static_cast<double>(tile[K_DIM]);
    double ratioMN = (tile[M_DIM] > tile[N_DIM]) ? static_cast<double>(tile[M_DIM]) / static_cast<double>(tile[N_DIM]) :
                                                   static_cast<double>(tile[N_DIM]) / static_cast<double>(tile[M_DIM]);
    std::vector<double> vectorRatio = {ratioMK, ratioKN, ratioMN};
    double ratioMKN = CalculateGeometricMean(vectorRatio);
    score -= BALANCE_WEIGHT * ratioMKN;
    uint64_t numL1CopyInL1A = inputMKN / (mkn * cubeL1Reuse * cubeNBuffer);
    uint64_t numL1CopyInL1B = inputMKN / mkn;
    uint64_t elePerRepeat = BYTES_PER_REPEAT / BytesOf(static_cast<DataType>(inputType));
    uint64_t parallelism = GetParallelism(static_cast<DataType>(inputType)) == 0 ?
                               1 :
                               GetParallelism(static_cast<DataType>(inputType));
    uint64_t cyclePerRepeat = elePerRepeat / parallelism;
    uint64_t latency = GetLatency(static_cast<DataType>(inputType));
    uint64_t repeatCountL1A = (tile[M_DIM] * tile[K_DIM] * inputTypeSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT + 1;
    uint64_t repeatCountL1B = (tile[K_DIM] * tile[N_DIM] * inputTypeSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT + 1;
    uint64_t cyclesL1A = numL1CopyInL1A * (latency + cyclePerRepeat * repeatCountL1A - 1);
    uint64_t cyclesL1B = numL1CopyInL1B * (latency + cyclePerRepeat * repeatCountL1B - 1);
    double cyclesLog = std::log2(cyclesL1A + cyclesL1B);
    score -= CYCLES_WEIGHT * cyclesLog;
}

void FindScoreForCubeTiles(const pairShapeType shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                           int64_t cubeL1Reuse, int64_t cubeNBuffer, int64_t numOfMatmuls)
{
    const int64_t L0A_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    const int64_t L0B_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    const int64_t L0C_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    const int64_t CUBE_CORES = Platform::Instance().GetSoc().GetAICoreNum();
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
    DataType inputType = shapeAndTypeInfo.second;
    int64_t inputTypeSize = BytesOf(inputType);
    uint64_t inputMKN = std::max(M, MIN_TILE) * std::max(K, MIN_TILE) * std::max(N, MIN_TILE);
    for (auto& [tile, score] : setOfCubeTiles) {
        uint64_t mkn = tile[M_DIM] * tile[K_DIM] * tile[N_DIM];
        CalculateUtilizationScore(tile, score, M, K, N, inputTypeSize, L0A_MAX_SIZE, L0B_MAX_SIZE, L0C_MAX_SIZE);
        CalculateTaskScore(score, M, N, tile, cubeL1Reuse, cubeNBuffer, numOfMatmuls, CUBE_CORES);
        CalculateBalanceAndCycleScore(score, tile, inputMKN, mkn, cubeL1Reuse, cubeNBuffer, inputTypeSize,
                                      static_cast<int64_t>(inputType));
    }
}

void SetPossibleCubeTiles(const pairShapeType shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles)
{
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
    DataType inputType = shapeAndTypeInfo.second;
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t newM = static_cast<int64_t>(
        std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(M, MIN_TILE))))));
    int64_t newK = static_cast<int64_t>(
        std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(K, MIN_TILE))))));
    int64_t newN = static_cast<int64_t>(
        std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(N, MIN_TILE))))));
    for (int64_t m = MIN_TILE; m <= newM; m *= FACTOR) {
        int64_t effM = m > std::max(M, MIN_TILE) ? std::max(M, MIN_TILE) : m;
        for (int64_t k = MIN_TILE; k <= newK; k *= FACTOR) {
            int64_t effK = k > std::max(K, MIN_TILE) ? std::max(K, MIN_TILE) : k;
            for (int64_t n = MIN_TILE; n <= newN; n *= FACTOR) {
                int64_t effN = n > std::max(N, MIN_TILE) ? std::max(N, MIN_TILE) : n;
                FindCubeTilesCombinations(setOfCubeTiles, effM, effK, effN, inputTypeSize);
            }
        }
    }
}

std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>> FindL1Tiles(
    const pairShapeType shapeAndTypeInfo, std::vector<int64_t> resultL0Tiles)
{
    const int64_t L1_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1);
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = std::max(shapeAndTypeInfo.first[K_DIM] / resultL0Tiles[K_DIM] * resultL0Tiles[K_DIM], MIN_TILE);
    int64_t N = shapeAndTypeInfo.first[N_DIM];
    DataType inputType = shapeAndTypeInfo.second;
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t kL1 = (resultL0Tiles[K_DIM] <= MIN_KL1_TILE) ? resultL0Tiles[K_DIM] * KL1_FACTOR : resultL0Tiles[K_DIM];
    int64_t kLA1 = kL1;
    int64_t kLB1 = kL1;
    if ((M <= resultL0Tiles[M_DIM]) && (K <= MAX_KL1)) {
        int64_t occupiedL1Memory = (resultL0Tiles[M_DIM] * K + kLB1 * resultL0Tiles[N_DIM]) * inputTypeSize;
        kLA1 = (occupiedL1Memory <= L1_MAX_SIZE) ? K : kL1;
    }
    if ((N <= resultL0Tiles[N_DIM]) && (K <= MAX_KL1)) {
        int64_t occupiedL1Memory = (resultL0Tiles[M_DIM] * kLA1 + K * resultL0Tiles[N_DIM]) * inputTypeSize;
        kLB1 = (occupiedL1Memory <= L1_MAX_SIZE) ? K : kL1;
    }
    return {{resultL0Tiles[M_DIM], resultL0Tiles[M_DIM]},
            {resultL0Tiles[K_DIM], kLA1, kLB1},
            {resultL0Tiles[N_DIM], resultL0Tiles[N_DIM]}};
}

std::vector<int64_t> FindMinimalShape(const std::vector<std::vector<int64_t>>& shapes)
{
    if (shapes.empty()) {
        return {-1, -1, -1};
    }
    size_t numDims = shapes[0].size();
    for (const auto& shape : shapes) {
        if (shape.size() != numDims) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "ERROR: Dimension inconsistency detected! Expected %zu dims, got %zu dims", numDims,
                              shape.size());
            return {-1, -1, -1};
        }
    }
    std::vector<int64_t> minimalShape(numDims, std::numeric_limits<int64_t>::max());
    for (const auto& currentShape : shapes) {
        for (size_t i = 0; i < numDims; ++i) {
            if ((currentShape[i] != -1) && (currentShape[i] < minimalShape[i])) {
                minimalShape[i] = currentShape[i];
            }
        }
    }
    return minimalShape;
}

std::array<int, NUM4> GetDimensionIndicesForTranspose(bool isTransA, bool isTransB)
{
    if (!isTransA && !isTransB) {
        return {-2, -1, 0, -1};
    } else if (!isTransA && isTransB) {
        return {-2, -1, 1, 0};
    } else if (isTransA && !isTransB) {
        return {0, 1, 0, -1};
    } else {
        return {1, 0, 1, 0};
    }
}

std::vector<std::vector<int64_t>> CollectShapesFromBackwardTraversal(const std::shared_ptr<LogicalTensor>& startTensor)
{
    std::vector<std::vector<int64_t>> collectedShapes;
    std::set<Operation*> visitedOps;
    std::queue<Operation*> worklist;
    for (auto* producer : startTensor->GetProducers()) {
        if (producer != nullptr) {
            worklist.push(producer);
        }
    }
    while (!worklist.empty()) {
        Operation* currentOp = worklist.front();
        worklist.pop();

        if (!ProcessOperation(currentOp, collectedShapes, worklist, visitedOps)) {
            break;
        }
    }
    return collectedShapes;
}

std::array<int64_t, NUM3> CalculateTileFilterForMatmul(Operation* matmulOp)
{
    std::array<int64_t, NUM3> tileFilter = {NO_FILTER_LIMIT, NO_FILTER_LIMIT, NO_FILTER_LIMIT};
    if (cubeMMOps.find(matmulOp->GetOpcode()) == cubeMMOps.end()) {
        return tileFilter;
    }
    auto leftOperand = matmulOp->GetIOperands()[0];
    auto rightOperand = matmulOp->GetIOperands()[1];
    DataType inputType = leftOperand->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    if (inputTypeSize == 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "inputTypeSize is zero in alignment calculation, cannot divide");
        return tileFilter;
    }
    int64_t alignmentMultiple = BLOCK_SIZE / inputTypeSize;
    auto leftShapes = CollectShapesFromBackwardTraversal(leftOperand);
    auto rightShapes = CollectShapesFromBackwardTraversal(rightOperand);
    auto leftMinimalShape = FindMinimalShape(leftShapes);
    auto rightMinimalShape = FindMinimalShape(rightShapes);
    bool isTransA = matmulOp->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_A);
    bool isTransB = matmulOp->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_B);
    auto dimIndices = GetDimensionIndicesForTranspose(isTransA, isTransB);
    if (leftMinimalShape.size() >= NUM2 && dimIndices[0] >= 0) {
        int64_t rawMLimit = leftMinimalShape[dimIndices[0]];
        tileFilter[M_DIM] = RoundUpToMultiple(rawMLimit, alignmentMultiple);
    }
    int64_t kLimitFromLeft = NO_FILTER_LIMIT;
    int64_t kLimitFromRight = NO_FILTER_LIMIT;
    if (leftMinimalShape.size() >= 1 && dimIndices[1] >= 0) {
        kLimitFromLeft = leftMinimalShape[dimIndices[1]];
    }
    if (rightMinimalShape.size() >= 1 && dimIndices[NUM2] >= 0) {
        kLimitFromRight = rightMinimalShape[dimIndices[NUM2]];
    }
    if (kLimitFromLeft == NO_FILTER_LIMIT && kLimitFromRight == NO_FILTER_LIMIT) {
        tileFilter[K_DIM] = NO_FILTER_LIMIT;
    } else if (kLimitFromLeft == NO_FILTER_LIMIT) {
        tileFilter[K_DIM] = RoundUpToMultiple(kLimitFromRight, alignmentMultiple);
    } else if (kLimitFromRight == NO_FILTER_LIMIT) {
        tileFilter[K_DIM] = RoundUpToMultiple(kLimitFromLeft, alignmentMultiple);
    } else {
        int64_t rawKLimit = std::min(kLimitFromLeft, kLimitFromRight);
        tileFilter[K_DIM] = RoundUpToMultiple(rawKLimit, alignmentMultiple);
    }
    if (rightMinimalShape.size() >= NUM2 && dimIndices[NUM3] >= 0) {
        int64_t rawNLimit = rightMinimalShape[dimIndices[NUM3]];
        tileFilter[N_DIM] = RoundUpToMultiple(rawNLimit, alignmentMultiple);
    }
    return tileFilter;
}

std::vector<int64_t> DetermineBestCubeTile(const std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                                           const pairShapeType& shapeInfo, const std::array<int64_t, NUM3>& tileFilter)
{
    if (setOfCubeTiles.empty()) {
        int64_t M = shapeInfo.first[M_DIM];
        int64_t K = shapeInfo.first[K_DIM];
        int64_t N = shapeInfo.first[N_DIM];
        int64_t mTile = (tileFilter[M_DIM] != NO_FILTER_LIMIT) ? std::min(M, tileFilter[M_DIM]) : M;
        int64_t kTile = (tileFilter[K_DIM] != NO_FILTER_LIMIT) ? std::min(K, tileFilter[K_DIM]) : K;
        int64_t nTile = (tileFilter[N_DIM] != NO_FILTER_LIMIT) ? std::min(N, tileFilter[N_DIM]) : N;
        return {mTile, kTile, nTile};
    }
    double maxScore = -std::numeric_limits<double>::max();
    std::vector<int64_t> bestTile(NUM3, 0);
    for (const auto& [tile, score] : setOfCubeTiles) {
        if (maxScore < score) {
            bestTile = tile;
            maxScore = score;
        }
    }
    return bestTile;
}

CubeTilesResultType FindAndSetCubeTileShapes(
    const std::pair<pairShapeType, std::array<int64_t, NUM3>> shapeAndTypeInfoFilter, int64_t numOfMatmuls,
    int64_t cubeL1Reuse, int64_t cubeNBuffer)
{
    auto shapeAndTypeInfo = shapeAndTypeInfoFilter.first;
    auto tileFilter = shapeAndTypeInfoFilter.second;
    std::map<std::vector<int64_t>, double> setOfCubeTiles;
    SetPossibleCubeTiles(shapeAndTypeInfo, setOfCubeTiles);
    FindScoreForCubeTiles(shapeAndTypeInfo, setOfCubeTiles, cubeL1Reuse, cubeNBuffer, numOfMatmuls);
    bool hasFilter = (tileFilter[M_DIM] != NO_FILTER_LIMIT || tileFilter[K_DIM] != NO_FILTER_LIMIT ||
                      tileFilter[N_DIM] != NO_FILTER_LIMIT);

    if (hasFilter) {
        FilterCubeTilesByDimensions(setOfCubeTiles, tileFilter);
    }
    std::vector<int64_t> resultL0Tiles = DetermineBestCubeTile(setOfCubeTiles, shapeAndTypeInfo, tileFilter);
#ifdef L1_TILES_SETTING
    CubeTilesResultType resultCubeTiles = FindL1Tiles(shapeAndTypeInfo, resultL0Tiles);
    if (tileFilter[K_DIM] != NO_FILTER_LIMIT) {
        int64_t adaptFilter = tileFilter[K_DIM] / resultL0Tiles[K_DIM] * resultL0Tiles[K_DIM];
        std::get<1>(resultCubeTiles)[K_DIM] = std::min(std::get<1>(resultCubeTiles)[K_DIM], adaptFilter);
        std::get<1>(resultCubeTiles)[N_DIM] = std::min(std::get<1>(resultCubeTiles)[N_DIM], adaptFilter);
    }
    return resultCubeTiles;
#endif
    return std::make_tuple(std::array<int64_t, MAX_MDIM>{resultL0Tiles[M_DIM], resultL0Tiles[M_DIM]},
                           std::array<int64_t, MAX_KDIM>{resultL0Tiles[K_DIM], resultL0Tiles[K_DIM]},
                           std::array<int64_t, MAX_NDIM>{resultL0Tiles[N_DIM], resultL0Tiles[N_DIM]});
}

ShapeDimsType ExtractNormalOpShapes(Operation* op, bool isTransA, bool isTransB)
{
    ShapeDimsType dims;
    if (!isTransA && !isTransB) {
        dims.M = op->GetIOperands()[0]->shape[0];
        dims.K = op->GetIOperands()[0]->shape[1];
        dims.N = op->GetIOperands()[1]->shape[1];
    } else if (!isTransA && isTransB) {
        dims.M = op->GetIOperands()[0]->shape[0];
        dims.K = op->GetIOperands()[0]->shape[1];
        dims.N = op->GetIOperands()[1]->shape[0];
    } else if (isTransA && !isTransB) {
        dims.M = op->GetIOperands()[0]->shape[1];
        dims.K = op->GetIOperands()[0]->shape[0];
        dims.N = op->GetIOperands()[1]->shape[1];
    } else {
        dims.M = op->GetIOperands()[0]->shape[1];
        dims.K = op->GetIOperands()[0]->shape[0];
        dims.N = op->GetIOperands()[1]->shape[0];
    }
    return dims;
}

pairShapeType ShapeAndTypeSetting(Operation* op)
{
    bool isTransA = op->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_A);
    bool isTransB = op->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_B);
    ShapeDimsType dims = ExtractNormalOpShapes(op, isTransA, isTransB);
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    return {{dims.M, dims.K, dims.N}, inputType};
}

void UniqueTilesFilling(const std::set<Operation*>& cubeOperations, std::map<pairShapeType, int64_t>& uniqueTiles,
                        std::map<uint64_t, std::pair<pairShapeType, std::array<int64_t, NUM3>>>& opShapeFilter)
{
    for (auto& op : cubeOperations) {
        auto curShapeAndType = ShapeAndTypeSetting(op);
        auto tileFilter = CalculateTileFilterForMatmul(op);
        auto it = uniqueTiles.find(curShapeAndType);
        if (it != uniqueTiles.end()) {
            uniqueTiles[curShapeAndType]++;
        } else {
            uniqueTiles[curShapeAndType] = 1;
        }
        opShapeFilter[op->GetOpMagic()] = std::pair(curShapeAndType, tileFilter);
    }
}

void SetMMTiles(Function& function, const std::set<Operation*>& cubeOperations)
{
    std::map<pairShapeType, int64_t> uniqueTiles;
    std::map<uint64_t, std::pair<pairShapeType, std::array<int64_t, NUM3>>> opShapeFilter;
    int64_t cubeL1Reuse = (function.paramConfigs_.cubeL1ReuseSetting.size() == 1 &&
                           function.paramConfigs_.cubeL1ReuseSetting.begin()->first == -1) ?
                              function.paramConfigs_.cubeL1ReuseSetting.begin()->second :
                              1;
    int64_t cubeNBuffer = (function.paramConfigs_.cubeNBufferSetting.size() == 1 &&
                           function.paramConfigs_.cubeNBufferSetting.begin()->first == -1) ?
                              function.paramConfigs_.cubeNBufferSetting.begin()->second :
                              1;
    // cubeL1Reuse = 0 means "skip L1 reuse merge" (valid semantic intent), but is used as a divisor in
    // CalculateTaskScore and CalculateBalanceAndCycleScore where division by zero would corrupt tile scores.
    // For scoring, "no reuse benefit" is equivalent to reuse factor = 1, so clamp 0 to 1 for calculations.
    if (cubeL1Reuse <= 0) {
        cubeL1Reuse = 1;
    }
    if (cubeNBuffer <= 0) {
        cubeNBuffer = 1;
    }
    UniqueTilesFilling(cubeOperations, uniqueTiles, opShapeFilter);
    for (auto* op : cubeOperations) {
        auto shapeAndTypeInfoFilter = opShapeFilter[op->GetOpMagic()];
        int64_t numOfMatmuls = uniqueTiles[shapeAndTypeInfoFilter.first];
        auto resultCubeTiles = FindAndSetCubeTileShapes(shapeAndTypeInfoFilter, numOfMatmuls, cubeL1Reuse, cubeNBuffer);
        auto m = std::get<M_DIM>(resultCubeTiles);
        auto k = std::get<K_DIM>(resultCubeTiles);
        auto n = std::get<N_DIM>(resultCubeTiles);
        op->GetTileShapeForSetting().SetCubeTile(m, k, n, false);
    }
}

std::vector<Operation*> SortCubeOpsByDepth(Function& function, const std::set<Operation*>& cubeOperations,
                                           std::map<int, int>& subgrDepthMap)
{
    std::vector<std::pair<int, int>> edges;
    for (auto& op : function.Operations()) {
        for (auto consumerOp : op.ConsumerOps()) {
            edges.push_back({consumerOp->GetOpMagic(), op.GetOpMagic()});
        }
    }
    std::vector<int> lastVertices;
    for (auto& op : function.Operations()) {
        if (op.ConsumerOps().size() == 0 && op.ProducerOps().size() != 0) {
            lastVertices.push_back(op.GetOpMagic());
        }
    }
    for (size_t idx = 0; idx < lastVertices.size(); idx++) {
        edges.push_back({-1, lastVertices[idx]});
    }
    auto d = FordBellman(edges, function);
    for (auto& [magic, depth] : d) {
        subgrDepthMap[magic] = std::max(subgrDepthMap[magic], -depth - 1);
    }
    subgrDepthMap.erase(-1);
    edges.clear();
    lastVertices.clear();
    std::vector<std::pair<Operation*, int>> cubeTmpOperations;
    for (auto cubeOp : cubeOperations) {
        cubeTmpOperations.push_back(std::make_pair(cubeOp, subgrDepthMap[cubeOp->GetOpMagic()]));
    }
    std::sort(
        cubeTmpOperations.begin(), cubeTmpOperations.end(),
        [](const std::pair<Operation*, int>& x, const std::pair<Operation*, int>& y) { return x.second < y.second; });

    std::vector<Operation*> cubeOrderedOperations;
    for (auto op : cubeTmpOperations) {
        cubeOrderedOperations.push_back(op.first);
    }
    return cubeOrderedOperations;
}

void CubeInDepsProcessing(Operation* cubeOp, Operation* opBase)
{
    std::vector<int64_t> vectorTilesCube;
    auto& cubeTile = cubeOp->GetTileShape().GetCubeTile();
    int magicA = cubeOp->GetIOperands()[0]->magic;
    std::vector<int64_t> vectorTilesA = {cubeTile.m[0], cubeTile.k[0]};
    std::vector<int64_t> vectorTilesB = {cubeTile.k[0], cubeTile.n[0]};
    bool isTransA = cubeOp->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_A);
    bool isTransB = cubeOp->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_B);
    if ((isTransA && !isTransB) || (isTransA && isTransB)) {
        std::reverse(vectorTilesA.begin(), vectorTilesA.end());
    }
    if ((!isTransA && isTransB) || (isTransA && isTransB)) {
        std::reverse(vectorTilesB.begin(), vectorTilesB.end());
    }
    vectorTilesCube = (opBase->GetOOperands()[0]->magic == magicA) ? vectorTilesA : vectorTilesB;
    cubeOp->GetTileShapeForSetting().SetVecTile(vectorTilesCube);
}

} // namespace npu::tile_fwk
