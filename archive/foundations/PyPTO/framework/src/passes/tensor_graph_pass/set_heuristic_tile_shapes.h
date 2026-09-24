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
 * \file set_heuristic_tile_shapes.h
 * \brief Heuristic tile shapes setting pass
 */

#ifndef PASS_SET_HEURISTIC_TILE_SHAPES_H_
#define PASS_SET_HEURISTIC_TILE_SHAPES_H_

#define CUBE_TILES   // comment to disable cube tiles setting
#define VECTOR_TILES // comment to disable vector tiles setting
// enable define PRINT_TILES to print tiles

// Custom defines to flexible setting
#define L1_TILES_SETTING

#include <cmath>
#include <climits>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <tuple>
#include <array>
#include "tilefwk/platform.h"
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

// Necessary params (must be before typedef)
constexpr int64_t M_DIM = 0;
constexpr int64_t K_DIM = 1;
constexpr int64_t N_DIM = 2;
constexpr int64_t FACTOR = 2;
constexpr int64_t MIN_TILE = 16;
constexpr int64_t TRANSPOSE_NUM = 16;
constexpr int64_t MAX_MDIM = 2;
constexpr int64_t MAX_KDIM = 3;
constexpr int64_t MAX_NDIM = 2;

constexpr int64_t MAX_KL1 = 2048;
constexpr int64_t MIN_KL1_TILE = 128;
constexpr int64_t KL1_FACTOR = 4;
constexpr int64_t MIN_TILE_SIZE = 2048;
constexpr int64_t DEFAULT_TILE_SIZE = 16384;
constexpr int64_t BYTES_PER_REPEAT = 256;
constexpr int64_t DEFAULT_MAX_PARALLELISM = 128;
constexpr int64_t DEFAULT_LATENCY = 10;
constexpr int64_t UINT8MAX = 255;
constexpr int64_t TRANSPOSE_VNCHWCONV_LAST_DIM = 2;
constexpr int64_t VNCHWCONV_POINTERS = 32;
constexpr int64_t MAX_GATHER_IN_L1_TILE_SIZE = 8192;
constexpr int64_t OUTPUT_TYPE_SIZE = 4;

// Type definitions
typedef std::pair<std::vector<int64_t>, DataType> pairShapeType;
typedef std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>>
    CubeTilesResultType;

struct ShapeDimsType {
    int64_t M;
    int64_t K;
    int64_t N;
};

struct NegDimResultType {
    size_t inToReplace;
    size_t outToReplace;
    bool hasNegDim;
};

// Additional cube variable parameters
constexpr int64_t DOUBLE_BUFFER = 1;
constexpr int64_t DOUBLE_BUFFER_L0C = 2;
constexpr int64_t WHOLE_M_SCORE = 2;
constexpr int64_t WHOLE_K_SCORE = 2;
constexpr int64_t WHOLE_N_SCORE = 2;
constexpr int64_t WEIGHT_L0 = 200;
constexpr double TASKS_CUBE_WEIGHT = 0.5;
constexpr double RESIDUAL_CUBE_TASKS_WEIGHT = 0.2;
constexpr int64_t BALANCE_WEIGHT = 1;
constexpr int64_t CYCLES_WEIGHT = 2;

// Additional vector variable parameters
constexpr double LAST_AXIS_WEIGHT = 0.1;
constexpr double WEIGHT_UB = 10;
constexpr double TASKS_VECTOR_WEIGHT = 0.7;
constexpr double RESIDUAL_VECTOR_TASKS_WEIGHT = 2;

// Cube tile filtering
constexpr int64_t NO_FILTER_LIMIT = -1;

// Inline const declarations for opcode sets
inline const std::unordered_map<DataType, int64_t> Latency{
    {DataType::DT_FP16, 200}, {DataType::DT_FP32, 200}, {DataType::DT_INT32, 0}, {DataType::DT_INT16, 0}};

inline const std::unordered_map<DataType, int64_t> Parallelism{
    {DataType::DT_FP16, 64}, {DataType::DT_FP32, 32}, {DataType::DT_INT32, 32}, {DataType::DT_INT16, 64}};

inline const std::set<Opcode> cubeMMOps = {Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_BT,   Opcode::OP_AT_MUL_B,
                                           Opcode::OP_AT_MUL_BT, Opcode::OP_A_MULACC_B, Opcode::OP_A_MULACC_BT};

inline const std::set<Opcode> stopOps = {
    Opcode::OP_ROWMAX,        Opcode::OP_ROWSUM,        Opcode::OP_ROWEXPMAX,       Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWSUMLINE,    Opcode::OP_ROWMAXLINE,    Opcode::OP_ROWMINLINE,      Opcode::OP_TOPK,
    Opcode::OP_TILEDMRGSORT,  Opcode::OP_BITSORT,       Opcode::OP_MRGSORT,         Opcode::OP_ARGSORT,
    Opcode::OP_TOPK_SORT,     Opcode::OP_TOPK_MERGE,    Opcode::OP_TOPK_EXTRACT,    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE, Opcode::OP_ROWSUM_SINGLE, Opcode::OP_A_MUL_B,         Opcode::OP_A_MUL_BT,
    Opcode::OP_AT_MUL_B,      Opcode::OP_AT_MUL_BT,     Opcode::OP_A_MULACC_B,      Opcode::OP_A_MULACC_BT,
    Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_PUT,     Opcode::OP_SCATTER_ELEMENT, Opcode::OP_SCATTER};

inline const std::set<Opcode> transposeOps = {Opcode::OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEOUT,
                                              Opcode::OP_TRANSPOSE_VNCHWCONV};

inline const std::set<Opcode> scatterOps = {Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_PUT, Opcode::OP_SCATTER_ELEMENT,
                                            Opcode::OP_SCATTER};

inline const std::set<Opcode> gatherVectorOps = {Opcode::OP_GATHER_ELEMENT, Opcode::OP_GATHER,
                                                 Opcode::OP_GATHER_FROM_UB};

inline const std::set<Opcode> gatherMoveOps = {Opcode::OP_GATHER_IN_UB, Opcode::OP_GATHER_IN_L1};

// Check if operation requires UB memory type based on OpcodeManager registration
inline bool IsOperationUBReq(Opcode opcode)
{
    const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
    const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);

    // Check if any input requires UB
    for (const auto& memType : inputsMemType) {
        if (memType == MemoryType::MEM_UB) {
            return true;
        }
    }

    // Check if any output requires UB
    for (const auto& memType : outputsMemType) {
        if (memType == MemoryType::MEM_UB) {
            return true;
        }
    }

    return false;
}

// Check if operation uses L1 memory type based on OpcodeManager registration
inline bool IsOperationL1Req(Opcode opcode)
{
    const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
    const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);

    // Check if any input uses L1
    for (const auto& memType : inputsMemType) {
        if (memType == MemoryType::MEM_L1) {
            return true;
        }
    }

    // Check if any output uses L1
    for (const auto& memType : outputsMemType) {
        if (memType == MemoryType::MEM_L1) {
            return true;
        }
    }

    return false;
}

// Tile setting handler function types
using TileSettingHandler = void (*)(Operation*, const std::vector<int64_t>&, std::vector<int64_t>&);
using SingletonTileSettingHandler = void (*)(Operation*, std::vector<int64_t>&);

// Forward declarations for handlers (some are inline, some in .cpp)
void ViewTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew);
void AssembleTileSettingForward(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                std::vector<int64_t>& vectorTilesNew);
void AssembleTileSettingBackward(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                 std::vector<int64_t>& vectorTilesNew);
void TransposeTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                          std::vector<int64_t>& vectorTilesNew);
void GatherTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew);
void OpWithSeveralInputsTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                    std::vector<int64_t>& vectorTilesNew);
void ReshapeTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                        std::vector<int64_t>& vectorTilesNew);
void ExpandTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew);
void DefaultTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                        std::vector<int64_t>& vectorTilesNew);

// Reduce tile setting handlers
void ReduceTileSetting(Operation* op, std::vector<int64_t>& vectorTilesNew);
void IndexInOutTileSetting(Operation* op, std::vector<int64_t>& vectorTilesNew);
void TopkTileSetting(Operation* op, std::vector<int64_t>& vectorTilesNew);

// Static maps for tile setting handlers (backward propagation)
inline const std::map<Opcode, TileSettingHandler> backwardTileHandlers = {
    {Opcode::OP_VIEW, ViewTileSetting},
    {Opcode::OP_ASSEMBLE, AssembleTileSettingBackward},
    {Opcode::OP_EXPAND, ExpandTileSetting},
    {Opcode::OP_TRANSPOSE_MOVEIN, TransposeTileSetting},
    {Opcode::OP_TRANSPOSE_MOVEOUT, TransposeTileSetting},
    {Opcode::OP_TRANSPOSE_VNCHWCONV, TransposeTileSetting}};

// Static maps for tile setting handlers (forward propagation)
inline const std::map<Opcode, TileSettingHandler> forwardTileHandlers = {
    {Opcode::OP_VIEW, ViewTileSetting},
    {Opcode::OP_ASSEMBLE, AssembleTileSettingForward},
    {Opcode::OP_RESHAPE, ReshapeTileSetting},
    {Opcode::OP_EXPAND, ExpandTileSetting},
    {Opcode::OP_TRANSPOSE_MOVEIN, TransposeTileSetting},
    {Opcode::OP_TRANSPOSE_MOVEOUT, TransposeTileSetting},
    {Opcode::OP_TRANSPOSE_VNCHWCONV, TransposeTileSetting},
    {Opcode::OP_GATHER_ELEMENT, GatherTileSetting},
    {Opcode::OP_GATHER, GatherTileSetting},
    {Opcode::OP_GATHER_FROM_UB, GatherTileSetting},
    {Opcode::OP_GATHER_IN_UB, GatherTileSetting},
    {Opcode::OP_GATHER_IN_L1, GatherTileSetting}};

// Static map for singleton tile setting handlers
inline const std::map<Opcode, SingletonTileSettingHandler> singletonTileHandlers = {
    {Opcode::OP_ROWMAX, ReduceTileSetting},
    {Opcode::OP_ROWSUM, ReduceTileSetting},
    {Opcode::OP_ROWEXPMAX, ReduceTileSetting},
    {Opcode::OP_ROWEXPSUM, ReduceTileSetting},
    {Opcode::OP_ROWSUMLINE, ReduceTileSetting},
    {Opcode::OP_ROWMAXLINE, ReduceTileSetting},
    {Opcode::OP_ROWMINLINE, ReduceTileSetting},
    {Opcode::OP_ROWMAX_SINGLE, ReduceTileSetting},
    {Opcode::OP_ROWMIN_SINGLE, ReduceTileSetting},
    {Opcode::OP_ROWSUM_SINGLE, ReduceTileSetting},
    {Opcode::OP_INDEX_OUTCAST, IndexInOutTileSetting},
    {Opcode::OP_INDEX_PUT, IndexInOutTileSetting},
    {Opcode::OP_SCATTER_ELEMENT, IndexInOutTileSetting},
    {Opcode::OP_SCATTER, IndexInOutTileSetting},
    {Opcode::OP_TOPK, TopkTileSetting},
    {Opcode::OP_TILEDMRGSORT, TopkTileSetting},
    {Opcode::OP_BITSORT, TopkTileSetting},
    {Opcode::OP_MRGSORT, TopkTileSetting},
    {Opcode::OP_ARGSORT, TopkTileSetting},
    {Opcode::OP_TOPK_SORT, TopkTileSetting},
    {Opcode::OP_TOPK_MERGE, TopkTileSetting},
    {Opcode::OP_TOPK_EXTRACT, TopkTileSetting}};

// Inline helper functions
inline int64_t FloorPowerOf2(int64_t value)
{
    if (value <= 0)
        return 0;
    return static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(value))));
}

inline uint64_t GetLatency(DataType dtype)
{
    auto iterDtype = Latency.find(dtype);
    if (iterDtype == Latency.end())
        return DEFAULT_LATENCY;
    return iterDtype->second;
}

inline uint64_t GetParallelism(DataType dtype)
{
    auto iterDtype = Parallelism.find(dtype);
    if (iterDtype == Parallelism.end())
        return DEFAULT_MAX_PARALLELISM;
    return iterDtype->second;
}

inline bool IsFloat(const std::shared_ptr<LogicalTensor> tensor)
{
    auto dataType = tensor->Datatype();
    return (dataType == DT_FP16) || (dataType == DT_FP32) || (dataType == DT_BF16);
}

inline double CalculateGeometricMean(const std::vector<double>& vectorRatio)
{
    if (vectorRatio.empty())
        return 0.0;
    double product = 1.0;
    for (double num : vectorRatio)
        product *= num;
    return std::pow(product, 1.0 / vectorRatio.size());
}

inline int64_t RoundUpToMultiple(int64_t value, int64_t multiple)
{
    if (value == -1 || multiple == 0)
        return value;
    return ((value + multiple - 1) / multiple) * multiple;
}

inline bool ShouldProcessOperation(Operation* currentOp, std::set<Operation*>& visitedOps)
{
    if (visitedOps.count(currentOp) > 0)
        return false;
    visitedOps.insert(currentOp);
    return true;
}

inline std::vector<uint32_t> FindChangedDims(Shape inShape, Shape outShape)
{
    std::vector<uint32_t> changedDims;
    for (uint32_t i = 0; i < inShape.size(); i++) {
        if (inShape[i] != outShape[i]) {
            changedDims.push_back(i);
        }
    }
    return changedDims;
}

inline void GatherTilesForFirstInput(std::vector<int64_t>& vectorTilesNew, std::vector<int64_t> inputShape,
                                     size_t outputDims)
{
    if (outputDims == NUM2) {
        vectorTilesNew[K_DIM] = FloorPowerOf2(inputShape[1]);
    } else if (outputDims == NUM3) {
        vectorTilesNew[N_DIM] = FloorPowerOf2(inputShape[1]);
    }
}

inline void GatherTilesForSecondInput(std::vector<int64_t>& vectorTilesNew, std::vector<int64_t> inputShape,
                                      size_t outputDims)
{
    if (outputDims == NUM2) {
        vectorTilesNew[0] = FloorPowerOf2(inputShape[0]);
    } else if (outputDims == NUM3) {
        vectorTilesNew[0] = FloorPowerOf2(inputShape[0]);
        vectorTilesNew[1] = FloorPowerOf2(inputShape[1]);
    }
}

inline size_t DimsCalculation(Operation* op, size_t tensorsNum, bool isInput)
{
    size_t tensorDims = 0;
    for (size_t tensor = 0; tensor < tensorsNum; tensor++) {
        if (isInput) {
            tensorDims = std::max(tensorDims, op->GetIOperands()[tensor]->shape.size());
        } else {
            tensorDims = std::max(tensorDims, op->GetOOperands()[tensor]->shape.size());
        }
    }
    return tensorDims;
}

inline std::vector<int64_t> MaxInputShapeCalculation(Operation* op, size_t inputsNum, size_t inputDims)
{
    std::vector<int64_t> maxInputShape(inputDims, LLONG_MIN);
    for (size_t input = 0; input < inputsNum; input++) {
        for (size_t inputDim = 0; inputDim < op->GetIOperands()[input]->shape.size(); inputDim++) {
            maxInputShape[inputDim] = std::max(maxInputShape[inputDim], op->GetIOperands()[input]->shape[inputDim]);
        }
    }
    return maxInputShape;
}

inline void AdaptTileToRealInputShape(Operation* op, std::vector<int64_t>& vectorTiles)
{
    if ((vectorTiles.size() == 1) && (vectorTiles[0] == -1)) {
        return;
    }
    size_t inputsNum = op->GetIOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
    for (size_t di = 0; di < vectorTiles.size(); di++) {
        vectorTiles[di] = (maxInputShape[di] != -1) ? std::min(vectorTiles[di], maxInputShape[di]) : vectorTiles[di];
    }
}

inline void SetFullShapeTiles(Operation& op)
{
    std::vector<int64_t> fullShapeTiles;
    for (auto s : op.GetIOperands()[0]->shape) {
        fullShapeTiles.push_back(std::abs(s));
    }
    op.GetTileShapeForSetting().SetVecTile(fullShapeTiles);
}

inline void UpdateBFS(Operation* op, std::queue<Operation*>& queueBFS, std::map<int, bool>& visitedBFS)
{
    if (!visitedBFS[op->GetOpMagic()]) {
        queueBFS.push(op);
    }
    visitedBFS[op->GetOpMagic()] = true;
}

inline bool DuplicateTileSetting(Operation* opInit, Operation* opNew, std::queue<Operation*>& queueBFS,
                                 std::map<int, bool>& visitedBFS)
{
    std::vector<int64_t> vectorTilesNew;
    if (opNew->GetIOperands().size() == 0) {
        vectorTilesNew = opInit->GetTileShape().GetVecTile().tile;
        opNew->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
        if (!visitedBFS[opNew->GetOpMagic()]) {
            queueBFS.push(opNew);
        }
        visitedBFS[opNew->GetOpMagic()] = true;
        return true;
    }
    return false;
}

inline void CubeOutDepsProcessing(Operation* cubeOp)
{
    auto& cubeTile = cubeOp->GetTileShape().GetCubeTile();
    std::vector<int64_t> vectorTilesOut = {cubeTile.m[0], cubeTile.n[0]}; // Output shape is [m, n]
    cubeOp->GetTileShapeForSetting().SetVecTile(vectorTilesOut);
}

inline void CollectShapesAndAddProducers(Operation* currentOp, std::vector<std::vector<int64_t>>& collectedShapes,
                                         std::queue<Operation*>& worklist, std::set<Operation*>& visitedOps)
{
    for (const auto& input : currentOp->GetIOperands()) {
        if (input->shape.size() == NUM2) {
            collectedShapes.push_back(input->shape);
            for (auto* producer : input->GetProducers()) {
                if (producer != nullptr && visitedOps.count(producer) == 0) {
                    worklist.push(producer);
                }
            }
        }
    }
    for (const auto& output : currentOp->GetOOperands()) {
        if (output->shape.size() == NUM2) {
            collectedShapes.push_back(output->shape);
        }
    }
}

inline bool ProcessOperation(Operation* currentOp, std::vector<std::vector<int64_t>>& collectedShapes,
                             std::queue<Operation*>& worklist, std::set<Operation*>& visitedOps)
{
    if (!ShouldProcessOperation(currentOp, visitedOps)) {
        return true;
    }
    if (!IsOperationUBReq(currentOp->GetOpcode()) && !IsOperationL1Req(currentOp->GetOpcode())) {
        CollectShapesAndAddProducers(currentOp, collectedShapes, worklist, visitedOps);
        return true;
    }
    // remove all gathered information because it useful only if no previous Cube/Vec op
    collectedShapes.clear();
    return false;
}

inline void TileThroughViewAssemble(Operation* op, std::vector<int64_t>& tile)
{
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    if (maxTypeSize == 0) {
        maxTypeSize = 1;
    }
    auto inShape = op->GetIOperands()[0]->shape;
    auto outShape = op->GetOOperands()[0]->shape;
    auto changedDims = FindChangedDims(inShape, outShape);
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    for (auto di : changedDims) {
        auto prevVal = tile[di];
        tile[di] = op->GetOOperands()[0]->shape[di] != -1 ? op->GetOOperands()[0]->shape[di] : tile[di];
        auto wholeSize = std::accumulate(tile.begin(), tile.end(), 1, std::multiplies<int64_t>());
        if (wholeSize > ((int64_t)UB_MAX_SIZE / maxTypeSize)) {
            tile[di] = prevVal;
        }
    }
}

// Forward declarations for functions defined in .cpp
void TileThroughReshape(Operation* op, std::vector<int64_t>& vectorTilesOld, bool isForward);
void TileThroughTranspose(Operation* op, std::vector<int64_t>& tile);
void TileThroughGather(Operation* producerOp, Operation* op, std::vector<int64_t>& vectorTilesOld);
void AdaptTileToRealOutputShape(Operation* op, int opBaseMagic, std::vector<int64_t>& vectorTiles);

// Recalculate tile through reshape/transpose/view/assemble operations
inline void RecalcTileThroughOps(Operation* op, std::vector<int64_t>& tile, bool isForward)
{
    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        TileThroughReshape(op, tile, isForward);
    } else if (transposeOps.find(op->GetOpcode()) != transposeOps.end()) {
        TileThroughTranspose(op, tile);
    } else if (isForward && ((op->GetOpcode() == Opcode::OP_VIEW) || (op->GetOpcode() == Opcode::OP_ASSEMBLE))) {
        // View/Assemble only in forward propagation
        TileThroughViewAssemble(op, tile);
    }
}

inline void ViewTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                            std::vector<int64_t>& vectorTilesNew)
{
    auto allViews = op->GetIOperands()[0]->GetConsumers();
    auto inShape = op->GetIOperands()[0]->shape;
    auto outShape = op->GetOOperands()[0]->shape;
    auto changedDims = FindChangedDims(inShape, outShape);
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    for (auto sp : changedDims) {
        auto tile = inShape[sp];
        for (auto v : allViews) {
            if (v->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            tile = (tile != -1) ? std::gcd(tile, v->GetOOperands()[0]->shape[sp]) : v->GetOOperands()[0]->shape[sp];
        }
        vectorTilesNew[sp] = (tile != -1) ? std::gcd(tile, vectorTilesNew[sp]) : vectorTilesNew[sp];
    }
}

inline bool IsTileValid(const std::vector<int64_t>& tile, const std::array<int64_t, NUM3>& filter)
{
    if (filter[M_DIM] != NO_FILTER_LIMIT && tile[M_DIM] > filter[M_DIM])
        return false;
    if (filter[K_DIM] != NO_FILTER_LIMIT && tile[K_DIM] > filter[K_DIM])
        return false;
    if (filter[N_DIM] != NO_FILTER_LIMIT && tile[N_DIM] > filter[N_DIM])
        return false;
    return true;
}

inline void FilterCubeTilesByDimensions(std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                                        const std::array<int64_t, NUM3>& filter)
{
    auto it = setOfCubeTiles.begin();
    while (it != setOfCubeTiles.end()) {
        if (!IsTileValid(it->first, filter)) {
            it = setOfCubeTiles.erase(it);
        } else {
            ++it;
        }
    }
}

class SetHeuristicTileShapes : public Pass {
public:
    SetHeuristicTileShapes() : Pass("SetHeuristicTileShapes") {}
    ~SetHeuristicTileShapes() override = default;
    Status RunOnFunction(Function& function) override;

private:
    void SetHeuristicTileShapesFunc(Function& function) const;
};

// Cube tile functions (implemented in cube_tile_setting.cpp)
std::map<int, int> FordBellman(const std::vector<std::pair<int, int>>& edges, Function& function);

void FindCubeTilesCombinations(std::map<std::vector<int64_t>, double>& setOfCubeTiles, int64_t m, int64_t k, int64_t n,
                               int64_t inputTypeSize, int64_t outputTypeSize);

void FindScoreForCubeTiles(const pairShapeType shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                           int64_t cubeL1Reuse, int64_t cubeNBuffer, int64_t numOfMatmuls);

void SetPossibleCubeTiles(const pairShapeType shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles);

CubeTilesResultType FindL1Tiles(const pairShapeType shapeAndTypeInfo, std::vector<int64_t> resultL0Tiles);

std::vector<int64_t> FindMinimalShape(const std::vector<std::vector<int64_t>>& shapes);

std::array<int, NUM4> GetDimensionIndicesForTranspose(bool isTransA, bool isTransB);

std::array<int64_t, NUM3> CalculateTileFilterForMatmul(Operation* matmulOp);

std::vector<std::vector<int64_t>> CollectShapesFromBackwardTraversal(const std::shared_ptr<LogicalTensor>& startTensor);

std::vector<int64_t> DetermineBestCubeTile(const std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                                           const pairShapeType& shapeInfo, const std::array<int64_t, NUM3>& tileFilter);

CubeTilesResultType FindAndSetCubeTileShapes(
    const std::pair<pairShapeType, std::array<int64_t, NUM3>> shapeAndTypeInfoFilter, int64_t numOfMatmuls,
    int64_t cubeL1Reuse, int64_t cubeNBuffer);

ShapeDimsType ExtractNormalOpShapes(Operation* op, bool isTransA, bool isTransB);

ShapeDimsType MapGatherVecTileToCubeTile(const std::vector<int64_t>& vecTile, bool isB, bool isTrans);

void AdjustGatherTileToL1(Operation* op, std::vector<int64_t>& vectorTilesNew);

void SetGatherInL1CubeTile(Operation* op, const std::vector<int64_t>& vectorTilesNew);

pairShapeType ShapeAndTypeSetting(Operation* op);

void UniqueTilesFilling(const std::set<Operation*>& cubeOperations, std::map<pairShapeType, int64_t>& uniqueTiles,
                        std::map<uint64_t, std::pair<pairShapeType, std::array<int64_t, NUM3>>>& opShapeFilter);

void SetMMTiles(Function& function, const std::set<Operation*>& cubeOperations);

std::vector<Operation*> SortCubeOpsByDepth(Function& function, const std::set<Operation*>& cubeOperations,
                                           std::map<int, int>& subgrDepthMap);

void CubeInDepsProcessing(Operation* cubeOp, Operation* opBase);

} // namespace npu::tile_fwk
#endif // PASS_SET_HEURISTIC_TILE_SHAPES_H_
