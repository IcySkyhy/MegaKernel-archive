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
 * \file set_heuristic_tile_shapes.cpp
 * \brief
 */

#include <climits>
#include <queue>
#include <fstream>
#include <cmath>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/tensor_graph_pass/derivation_tile_shape.h"
#include "set_heuristic_tile_shapes.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "SetHeuristicTileShapes"

using namespace npu::tile_fwk;
using json = nlohmann::json;

namespace npu::tile_fwk {
Status SetHeuristicTileShapes::RunOnFunction(Function& function)
{
    SetHeuristicTileShapesFunc(function);
    return SUCCESS;
}

void UnstableDimsCalculation(std::stack<size_t>& unstableDims, Shape opBaseInputShape, Shape opBaseOutputShape)
{
    size_t inProd = 1;
    size_t outProd = 1;
    for (size_t inPos = 0, outPos = 0; (inPos < opBaseInputShape.size()) && (outPos < opBaseOutputShape.size());) {
        if ((opBaseInputShape[inPos] == -1) || (opBaseOutputShape[outPos] == -1)) {
            unstableDims.push(inPos);
            auto ePos = inPos + 1;
            for (size_t ePos1 = opBaseInputShape.size() - 1, ePos2 = opBaseOutputShape.size() - 1;
                 ePos1 > inPos && ePos2 > inPos; ePos1--, ePos2--) {
                if ((opBaseInputShape[ePos1] == -1) || (opBaseOutputShape[ePos2] == -1) ||
                    (opBaseInputShape[ePos1] != opBaseOutputShape[ePos2])) {
                    ePos = ePos1;
                    break;
                }
            }
            for (; (inPos < ePos); inPos++) {
                unstableDims.push(inPos);
            }
            break;
        }
        if (inProd < outProd) {
            inProd *= opBaseInputShape[inPos];
            unstableDims.push(inPos);
            inPos++;
        } else if (inProd > outProd) {
            outProd *= opBaseOutputShape[outPos];
            outPos++;
        } else {
            inProd = 1;
            outProd = 1;
            if (opBaseInputShape[inPos] == opBaseOutputShape[outPos]) {
                inPos++;
                outPos++;
                continue;
            }
            inProd *= opBaseInputShape[inPos];
            outProd *= opBaseOutputShape[outPos];
            unstableDims.push(inPos);
            if (opBaseInputShape[inPos] < opBaseOutputShape[outPos]) {
                inPos++;
            } else {
                outPos++;
            }
        }
    }
}

static NegDimResultType ProcessNegativeDimensions(Shape& opBaseInputShape, Shape& opBaseOutputShape)
{
    NegDimResultType result{0, 0, false};
    int64_t inProd = 1;
    int64_t outProd = 1;
    for (size_t s = 0; s < opBaseInputShape.size(); s++) {
        if (opBaseInputShape[s] == -1) {
            opBaseInputShape[s] = 1;
            result.inToReplace = s;
            result.hasNegDim = true;
        }
        inProd *= opBaseInputShape[s];
    }
    for (size_t s = 0; s < opBaseOutputShape.size(); s++) {
        if (opBaseOutputShape[s] == -1) {
            opBaseOutputShape[s] = 1;
            result.outToReplace = s;
            result.hasNegDim = true;
        }
        outProd *= opBaseOutputShape[s];
    }
    if (result.hasNegDim && inProd > outProd) {
        opBaseOutputShape[result.outToReplace] = inProd / outProd;
    }
    if (result.hasNegDim && inProd < outProd) {
        opBaseInputShape[result.inToReplace] = outProd / inProd;
    }
    return result;
}

static bool TryDerivationWithAdjustments(Operation* op, Shape& opBaseInputShape, Shape& opBaseOutputShape,
                                         Shape& vectorTilesOld, Shape& outTileShape, std::stack<size_t>& unstableDims)
{
    DerivationTileShape derivationTileShapePass;
    Status curStatus = derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape,
                                                                          vectorTilesOld, outTileShape);
    while (curStatus != SUCCESS && unstableDims.size() != 0) {
        int32_t curDiTile = unstableDims.top();
        vectorTilesOld[curDiTile] = opBaseInputShape[curDiTile];
        unstableDims.pop();
        if (unstableDims.size() == 0) {
            size_t outIdx = opBaseOutputShape.size() - (opBaseInputShape.size() - curDiTile);
            vectorTilesOld[curDiTile] = std::min(opBaseInputShape[curDiTile], opBaseOutputShape[outIdx]);
        }
        curStatus = derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape,
                                                                       vectorTilesOld, outTileShape);
    }
    return curStatus == SUCCESS;
}

static void SetFullShapeFallback(Operation* op, Shape& opBaseInputShape, Shape& opBaseOutputShape,
                                 Shape& vectorTilesOld, Shape& outTileShape)
{
    for (size_t di = 0; di < opBaseInputShape.size(); di++) {
        vectorTilesOld[di] = opBaseInputShape[di];
    }
    DerivationTileShape derivationTileShapePass;
    Status curStatus = derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape,
                                                                          vectorTilesOld, outTileShape);
    if (curStatus != SUCCESS) {
        outTileShape.resize(opBaseOutputShape.size());
        for (size_t di = 0; di < opBaseOutputShape.size(); di++) {
            outTileShape[di] = opBaseOutputShape[di];
        }
        APASS_LOG_WARN_F(Elements::Operation, "DerivationReshapeTileShape failed. %s", GetFormatBacktrace(*op).c_str());
    }
}

void AdjustTilesToReshape(Operation* op, Shape opBaseInputShape, Shape opBaseOutputShape, Shape& vectorTilesOld,
                          Shape& outTileShape)
{
    std::stack<size_t> unstableDims;
    UnstableDimsCalculation(unstableDims, opBaseInputShape, opBaseOutputShape);
    ProcessNegativeDimensions(opBaseInputShape, opBaseOutputShape);
    if (!TryDerivationWithAdjustments(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape,
                                      unstableDims)) {
        SetFullShapeFallback(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);
        return;
    }
    int64_t outTileProd = std::accumulate(outTileShape.begin(), outTileShape.end(), 1, std::multiplies<int64_t>());
    int64_t inTileProd = std::accumulate(vectorTilesOld.begin(), vectorTilesOld.end(), 1, std::multiplies<int64_t>());
    if (inTileProd > outTileProd && inTileProd != 0 && outTileProd != 0) {
        int64_t tileRatio = inTileProd / outTileProd;
        for (size_t di = 0; di < vectorTilesOld.size(); di++) {
            if (vectorTilesOld[di] > 1) {
                vectorTilesOld[di] = vectorTilesOld[di] / tileRatio;
            }
        }
    }
    DerivationTileShape derivationTileShapePass;
    derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld,
                                                       outTileShape);
}

void TileThroughReshape(Operation* op, std::vector<int64_t>& vectorTilesOld, bool isForward)
{
    std::vector<int64_t> opBaseInputShape;
    std::vector<int64_t> opBaseOutputShape;
    if (isForward) {
        opBaseInputShape = op->GetIOperands()[0]->shape;
        opBaseOutputShape = op->GetOOperands()[0]->shape;
    } else {
        opBaseInputShape = op->GetOOperands()[0]->shape;
        opBaseOutputShape = op->GetIOperands()[0]->shape;
    }
    Shape outTileShape(opBaseOutputShape.size());
    AdjustTilesToReshape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);
    vectorTilesOld.resize(outTileShape.size());
    vectorTilesOld = outTileShape;
}

void TileThroughTranspose(Operation* op, std::vector<int64_t>& tile)
{
    auto perm = op->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    std::swap(tile[perm[0]], tile[perm[1]]);
}

uint64_t MemUsedCalculation(Operation* op, Shape& vectorTilesNew, Shape& vectorTilesOut)
{
    uint64_t memUsed = 0;
    for (auto inp : op->GetIOperands()) {
        Shape realTile(vectorTilesNew);
        size_t memUsedTmp = 1;
        for (size_t di = 0; di < std::min(vectorTilesNew.size(), inp->GetShape().size()); di++) {
            realTile[di] = (inp->GetShape()[di] != -1) ? std::min(vectorTilesNew[di], inp->GetShape()[di]) :
                                                         vectorTilesNew[di];
            if (di == vectorTilesNew.size() - 1) {
                realTile[di] = (realTile[di] + (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType())) - 1) /
                               (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType())) *
                               (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType()));
            }
            memUsedTmp *= realTile[di];
        }
        memUsedTmp *= BytesOf(inp->tensor->GetDataType());
        memUsed += memUsedTmp;
    }
    for (auto out : op->GetOOperands()) {
        Shape realTile(vectorTilesOut);
        size_t memUsedTmp = 1;
        for (size_t di = 0; di < std::min(vectorTilesOut.size(), out->GetShape().size()); di++) {
            realTile[di] = (out->GetShape()[di] != -1) ? std::min(vectorTilesOut[di], out->GetShape()[di]) :
                                                         vectorTilesOut[di];
            if (di == vectorTilesOut.size() - 1) {
                realTile[di] = (realTile[di] + (BLOCK_SIZE / BytesOf(out->tensor->GetDataType())) - 1) /
                               (BLOCK_SIZE / BytesOf(out->tensor->GetDataType())) *
                               (BLOCK_SIZE / BytesOf(out->tensor->GetDataType()));
            }
            memUsedTmp *= realTile[di];
        }
        memUsedTmp *= BytesOf(out->tensor->GetDataType());
        memUsed += memUsedTmp;
    }
    return memUsed;
}

void AdjustTileToUB(Operation* op, Shape& vectorTilesNew)
{
    if (!IsOperationUBReq(op->GetOpcode())) {
        return;
    }
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    Shape vectorTilesOut(vectorTilesNew);
    RecalcTileThroughOps(op, vectorTilesOut, true);
    uint64_t memUsed = MemUsedCalculation(op, vectorTilesNew, vectorTilesOut);
    uint64_t memUsedPrev = memUsed + 1;
    while ((memUsed > UB_MAX_SIZE) && (memUsedPrev > memUsed)) {
        size_t d = 0;
        while (d < vectorTilesNew.size() && vectorTilesNew[d] == 1) {
            d++;
        }
        if (d >= vectorTilesNew.size()) {
            return; // impossible case if UB_MAX_SIZE!=0
        }
        vectorTilesNew[d] /= NUM2;
        vectorTilesOut = vectorTilesNew;
        RecalcTileThroughOps(op, vectorTilesOut, true);
        memUsed = MemUsedCalculation(op, vectorTilesNew, vectorTilesOut);
    }
    ASSERT(memUsed <= UB_MAX_SIZE);
}

ShapeDimsType MapGatherVecTileToCubeTile(const std::vector<int64_t>& vecTile, bool isB, bool isTrans)
{
    ShapeDimsType dims;
    int64_t tileRows = vecTile[0];
    int64_t tileCols = vecTile[1];
    if (!isB && !isTrans) {
        dims.M = tileRows;
        dims.K = tileCols;
        dims.N = -1;
    } else if (!isB && isTrans) {
        dims.M = tileCols;
        dims.K = tileRows;
        dims.N = -1;
    } else if (isB && !isTrans) {
        dims.M = -1;
        dims.K = tileRows;
        dims.N = tileCols;
    } else {
        dims.M = -1;
        dims.K = tileCols;
        dims.N = tileRows;
    }
    return dims;
}

void AdjustGatherTileToL1(Operation* op, std::vector<int64_t>& vectorTilesNew)
{
    const uint64_t L1_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1);
    bool isB = op->GetBoolAttribute("isB");
    bool isTrans = op->GetBoolAttribute("isTrans");
    ShapeDimsType tileDims = MapGatherVecTileToCubeTile(vectorTilesNew, isB, isTrans);
    ASSERT(tileDims.K != -1 && ((tileDims.M == -1) != (tileDims.N == -1)));
    uint64_t memUsed;
    if (tileDims.N == -1) {
        memUsed = static_cast<uint64_t>(tileDims.M) * static_cast<uint64_t>(tileDims.K);
    } else {
        memUsed = static_cast<uint64_t>(tileDims.K) * static_cast<uint64_t>(tileDims.N);
    }
    uint64_t memUsedPrev = memUsed + 1;
    while ((memUsed > L1_MAX_SIZE) && (memUsedPrev > memUsed)) {
        size_t d = 0;
        while (d < vectorTilesNew.size() && vectorTilesNew[d] == 1) {
            d++;
        }
        if (d >= vectorTilesNew.size()) {
            return;
        }
        vectorTilesNew[d] /= NUM2;
        tileDims = MapGatherVecTileToCubeTile(vectorTilesNew, isB, isTrans);
        ASSERT(tileDims.K != -1 && ((tileDims.M == -1) != (tileDims.N == -1)));
        if (tileDims.N == -1) {
            memUsed = static_cast<uint64_t>(tileDims.M) * static_cast<uint64_t>(tileDims.K);
        } else {
            memUsed = static_cast<uint64_t>(tileDims.K) * static_cast<uint64_t>(tileDims.N);
        }
        memUsedPrev = memUsed;
    }
}

void SetGatherInL1CubeTile(Operation* op, const std::vector<int64_t>& vectorTilesNew)
{
    bool isB = op->GetBoolAttribute("isB");
    bool isTrans = op->GetBoolAttribute("isTrans");
    ShapeDimsType dims = MapGatherVecTileToCubeTile(vectorTilesNew, isB, isTrans);
    std::array<int64_t, MAX_MDIM> mArray = {dims.M, dims.M};
    std::array<int64_t, MAX_KDIM> kArray = {dims.K, dims.K, dims.K};
    std::array<int64_t, MAX_NDIM> nArray = {dims.N, dims.N};
    op->GetTileShapeForSetting().SetCubeTile(mArray, kArray, nArray);
}

static void HandleGatherVectorOps(Operation* producerOp, Operation* op, size_t outputProducerDims, size_t outputOpDims,
                                  std::vector<int64_t>& vectorTilesOld)
{
    std::vector<int64_t> vectorTilesOldTmp = vectorTilesOld;
    int magicFirst = op->GetIOperands()[0]->magic;
    int magicSecond = op->GetIOperands()[1]->magic;
    int magicProducer = producerOp->GetOOperands()[0]->magic;
    if (magicFirst == magicProducer) {
        if (outputProducerDims == NUM2) {
            vectorTilesOld.resize(outputProducerDims);
            vectorTilesOld[0] = FloorPowerOf2(producerOp->GetOOperands()[0]->shape[0]);
            vectorTilesOld[1] = FloorPowerOf2(vectorTilesOldTmp[outputOpDims - 1]);
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "Gather first input should be 2D, need to check this case");
        }
    }
    if (magicSecond == magicProducer) {
        if (outputProducerDims == 1) {
            vectorTilesOld.resize(outputProducerDims);
            vectorTilesOld[0] = FloorPowerOf2(vectorTilesOldTmp[0]);
        } else if (outputProducerDims == NUM2) {
            vectorTilesOld.resize(outputProducerDims);
            vectorTilesOld[0] = FloorPowerOf2(vectorTilesOldTmp[0]);
            vectorTilesOld[1] = FloorPowerOf2(vectorTilesOldTmp[1]);
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "Gather second input should be 1D or 2D, need to check this case");
        }
    }
    AdjustTileToUB(op, vectorTilesOld);
}

static void HandleGatherMoveOps(Operation* producerOp, Operation* op, std::vector<int64_t>& vectorTilesOld)
{
    int magicFirst = op->GetIOperands()[0]->magic;
    int magicSecond = op->GetIOperands()[1]->magic;
    int magicThird = op->GetIOperands()[NUM2]->magic;
    int magicProducer = producerOp->GetOOperands()[0]->magic;
    if (magicFirst == magicProducer) {
        vectorTilesOld[0] = FloorPowerOf2(vectorTilesOld[0]);
        vectorTilesOld[1] = FloorPowerOf2(vectorTilesOld[1]);
    }
    if ((magicSecond == magicProducer) || (magicThird == magicProducer)) {
        vectorTilesOld[0] = FloorPowerOf2(producerOp->GetOOperands()[0]->shape[0]);
        vectorTilesOld[1] = FloorPowerOf2(producerOp->GetOOperands()[0]->shape[1]);
    }
    if (op->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
        AdjustGatherTileToL1(op, vectorTilesOld);
    } else {
        AdjustTileToUB(op, vectorTilesOld);
    }
}

void TileThroughGather(Operation* producerOp, Operation* op, std::vector<int64_t>& vectorTilesOld)
{
    size_t outputsProducerNum = producerOp->GetOOperands().size();
    size_t outputsOpNum = op->GetOOperands().size();
    size_t outputProducerDims = DimsCalculation(producerOp, outputsProducerNum, false);
    size_t outputOpDims = DimsCalculation(op, outputsOpNum, false);
    if (gatherVectorOps.find(op->GetOpcode()) != gatherVectorOps.end()) {
        HandleGatherVectorOps(producerOp, op, outputProducerDims, outputOpDims, vectorTilesOld);
    } else {
        HandleGatherMoveOps(producerOp, op, vectorTilesOld);
    }
}

void ReshapeTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew)
{
    std::vector<int64_t> opBaseInputShape;
    std::vector<int64_t> opBaseOutputShape;
    opBaseInputShape = op->GetIOperands()[0]->shape;
    opBaseOutputShape = op->GetOOperands()[0]->shape;
    Shape vectorTilesWorking = vectorTilesOld; // Create local working copy
    Shape outTileShape;
    AdjustTilesToReshape(op, opBaseInputShape, opBaseOutputShape, vectorTilesWorking, outTileShape);
    vectorTilesNew.resize(vectorTilesWorking.size());
    vectorTilesNew = vectorTilesWorking;
    AdjustTileToUB(op, vectorTilesNew);
}

void TransposeTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                          std::vector<int64_t>& vectorTilesNew)
{
    ASSERT(op->GetIOperands().size() == 1) << "Transpose should have 1 input";
    ASSERT(op->GetOOperands().size() == 1) << "Transpose should have 1 output";
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    ASSERT(inputType == outputType) << "Input & output data types should be the same for Transpose";
    uint8_t typeSize = BytesOf(inputType);
    std::vector<int64_t> inputShape = op->GetIOperands()[0]->GetShape();
    size_t inputDims = inputShape.size();
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint32_t argsCount = (op->GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV) ?
                             NUM3 :
                             1; // At least IN and OUT should be placed in UB
    ASSERT(inputShape[inputDims - 1] != -1);
    ASSERT(typeSize != 0);
    ASSERT(argsCount != 0);
    int64_t maxTile = UB_MAX_SIZE / (typeSize * argsCount);
    uint8_t blockSizeForType = BLOCK_SIZE / typeSize;
    auto lastDimAligned = ((inputShape[inputDims - 1] + blockSizeForType - 1) / blockSizeForType) * blockSizeForType;
    if (lastDimAligned > maxTile) {
        APASS_LOG_ERROR_F(Elements::Operation, "Transpose have row that more than UB, doesn't support");
    }
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    vectorTilesNew[inputDims - 1] = inputShape[inputDims - 1];
    maxTile /= lastDimAligned;
    vectorTilesNew[inputDims - NUM2] = (vectorTilesNew[inputDims - NUM2] != 1) ?
                                           (vectorTilesNew[inputDims - NUM2] + VNCHWCONV_POINTERS - 1) /
                                               VNCHWCONV_POINTERS * VNCHWCONV_POINTERS :
                                           1;
    vectorTilesNew[inputDims - NUM2] = std::min(maxTile / VNCHWCONV_POINTERS * VNCHWCONV_POINTERS,
                                                vectorTilesNew[inputDims - 2]); // UB overflow condition
    maxTile /= vectorTilesNew[inputDims - NUM2];
    for (int64_t di = inputDims - NUM3; di >= 0; di--) {
        if (maxTile < vectorTilesNew[di]) {
            uint32_t coeff = (inputShape[di] + maxTile - 1) / maxTile;
            vectorTilesNew[di] = inputShape[di] / coeff;
        }
        maxTile /= vectorTilesNew[di];
    }
}

static std::vector<uint32_t> PrepareAssembleTileBase(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                                     std::vector<int64_t>& vectorTilesNew)
{
    auto inShape = op->GetIOperands()[0]->shape;
    auto outShape = op->GetOOperands()[0]->shape;
    ASSERT(inShape.size() == outShape.size()) << "VIEW have different number of dimension in input and output "
                                              << inShape.size() << " vs " << outShape.size() << "\n";
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    return FindChangedDims(inShape, outShape);
}

void AssembleTileSettingForward(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                std::vector<int64_t>& vectorTilesNew)
{
    auto changedDims = PrepareAssembleTileBase(op, vectorTilesOld, vectorTilesNew);
    auto outShape = op->GetOOperands()[0]->shape;
    auto allAssembles = op->GetOOperands()[0]->GetProducers();
    for (auto sp : changedDims) {
        auto tile = outShape[sp];
        for (auto v : allAssembles) {
            if (v->GetOpcode() == Opcode::OP_ASSEMBLE) {
                tile = (tile != -1) ? std::gcd(tile, v->GetIOperands()[0]->shape[sp]) : v->GetIOperands()[0]->shape[sp];
            }
        }
        ASSERT((size_t)sp < vectorTilesNew.size())
            << "Split dim > vectorTiles.size(): " << sp << " vs " << vectorTilesNew.size() << "\n";
        vectorTilesNew[sp] = (tile != -1) ? std::gcd(tile, vectorTilesNew[sp]) : vectorTilesNew[sp];
    }
}

void AssembleTileSettingBackward(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                 std::vector<int64_t>& vectorTilesNew)
{
    auto changedDims = PrepareAssembleTileBase(op, vectorTilesOld, vectorTilesNew);
    auto outShape = op->GetOOperands()[0]->shape;
    for (auto sp : changedDims) {
        auto tile = outShape[sp];
        vectorTilesNew[sp] = (tile != -1) ? std::min(tile, vectorTilesNew[sp]) : vectorTilesNew[sp];
    }
}

void OpWithSeveralInputsTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld,
                                    std::vector<int64_t>& vectorTilesNew)
{
    size_t inputsNum = op->GetIOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    std::vector<Operation*> toUpdate;
    for (size_t inp = 0; inp < op->GetIOperands().size(); inp++) {
        ASSERT(op->GetIOperands()[inp]->GetProducers().size() != 0);
        auto prodOp = *(op->GetIOperands()[inp]->GetProducers().begin());
        std::vector<int64_t> inpTile(inputDims, 1);
        auto consumers = op->GetOOperands()[0]->GetConsumers();
        if (((prodOp->GetIOperands().size() == 0) || (prodOp->GetIOperands()[0]->GetProducers().size() == 0))) {
            if (!consumers.empty()) {
                auto consOp = *(consumers.begin());
                if ((consOp->GetTileShape().GetVecTile().tile.size() != 1) ||
                    (consOp->GetTileShape().GetVecTile().tile[0] != -1)) {
                    inpTile = consOp->GetTileShape().GetVecTile().tile;
                } else {
                    inpTile = vectorTilesNew;
                }
            } else {
                inpTile = vectorTilesNew;
            }
            if (inpTile.back() == 1) {
                inpTile.back() = op->GetIOperands()[inp]->GetShape()[inputDims - 1];
            }
            toUpdate.push_back(prodOp);
        } else {
            inpTile = prodOp->GetTileShape().GetVecTile().tile;
        }
        // Tile not set yet
        if ((inpTile.size() == 1) && (inpTile[0] == -1)) {
            vectorTilesNew.resize(1);
            vectorTilesNew[0] = -1;
            return;
        }
        // In all other cases recalculate tile through reshape/transpose operations
        RecalcTileThroughOps(prodOp, inpTile, true);
        // Evaluate new tile according to all existing tiles
        for (size_t di = 0; di < inputDims; di++) {
            inpTile[di] = std::min(inpTile[di], op->GetIOperands()[inp]->GetShape()[di]);
            vectorTilesNew[di] = (inpTile[di] != -1) ? std::lcm(vectorTilesNew[di], inpTile[di]) : vectorTilesNew[di];
        }
    }
    AdjustTileToUB(op, vectorTilesNew);
    for (auto& opupd : toUpdate) {
        opupd->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    }
}

void GatherFromTiles(Operation* op, std::vector<int64_t>& vectorTilesNew, std::vector<int64_t> inpTile,
                     size_t outputDims, size_t inp, std::vector<bool>& inputHasTile)
{
    if (gatherVectorOps.find(op->GetOpcode()) != gatherVectorOps.end()) {
        if (inp == 0) { // Broadcast tiles from first input
            GatherTilesForFirstInput(vectorTilesNew, inpTile, outputDims);
            inputHasTile[0] = true;
        } else if (inp == 1) { // Broadcast tiles from second input
            GatherTilesForSecondInput(vectorTilesNew, inpTile, outputDims);
            inputHasTile[1] = true;
        }
    } else {
        if (inp == 0) { // Broadcast tiles from 1st input if they are
            vectorTilesNew[1] = FloorPowerOf2(inpTile[1]);
            inputHasTile[0] = true;
        } else if (inp == 1) { // Otherwise broadcast tiles from 2nd or 3rd input if they are set
            vectorTilesNew[0] = (vectorTilesNew[0] == -1) ? FloorPowerOf2(inpTile[1]) :
                                                            std::min(vectorTilesNew[0], inpTile[1]);
            int64_t inputTypeSize = BytesOf(op->GetIOperands()[0]->tensor->GetDataType());
            if (inputTypeSize == 0) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "inputTypeSize is zero in GatherFromInputTileSetting, cannot divide");
                return;
            }
            vectorTilesNew[0] = std::max(vectorTilesNew[0], BLOCK_SIZE / inputTypeSize);
            inputHasTile[1] = true;
        }
        // inp == 2 -> blockTable: is not sliced in ExpandFunction, it is always processed as a whole
    }
}

void GatherFromShapes(Operation* op, std::vector<int64_t>& vectorTilesNew, size_t outputDims,
                      const std::vector<bool>& inputHasTile)
{
    ASSERT(inputHasTile.size() >= NUM2);
    if (gatherVectorOps.find(op->GetOpcode()) != gatherVectorOps.end()) {
        if (!inputHasTile[0]) {
            auto firstInputShape = op->GetIOperands()[0]->shape;
            GatherTilesForFirstInput(vectorTilesNew, firstInputShape, outputDims);
        }
        if (!inputHasTile[1]) {
            auto secondInputShape = op->GetIOperands()[1]->shape;
            GatherTilesForSecondInput(vectorTilesNew, secondInputShape, outputDims);
        }
    } else {
        if (!inputHasTile[0]) { // If the tiles were not broadcasted, set tiles based on the first input
            auto firstInputShape = op->GetIOperands()[0]->shape;
            vectorTilesNew[1] = FloorPowerOf2(firstInputShape[1]);
        }
        if (!inputHasTile[1]) {
            auto secondInputShape = op->GetIOperands()[1]->shape;
            vectorTilesNew[0] = FloorPowerOf2(secondInputShape[1]);
        }
        int64_t inputTypeSize = BytesOf(op->GetIOperands()[0]->tensor->GetDataType());
        if (inputTypeSize == 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "inputTypeSize is zero in GatherFromShapes, cannot divide");
            return;
        }
        vectorTilesNew[0] = std::max(vectorTilesNew[0], BLOCK_SIZE / inputTypeSize);
    }
}

void GatherTileSetting(Operation* op, [[maybe_unused]] const std::vector<int64_t>& vectorTilesOld,
                       std::vector<int64_t>& vectorTilesNew)
{
    size_t inputsNum = op->GetIOperands().size();
    size_t outputsNum = op->GetOOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    size_t outputDims = DimsCalculation(op, outputsNum, false);
    vectorTilesNew.clear();
    vectorTilesNew.resize(outputDims, -1);
    if (op->GetTileShape().GetVecTile().tile.size() == outputDims) {
        vectorTilesNew = op->GetTileShape().GetVecTile().tile;
    }
    std::vector<bool> inputHasTile(inputsNum);
    // Fill vectorTilesNew based on input tiles
    for (size_t inp = 0; inp < inputsNum; inp++) {
        inputHasTile[inp] = false;
        ASSERT(op->GetIOperands()[inp]->GetProducers().size() == 1);
        auto prodOp = *(op->GetIOperands()[inp]->GetProducers().begin());
        if ((prodOp->GetIOperands().size() == 0) || (prodOp->GetIOperands()[0]->GetProducers().size() == 0)) {
            continue;
        }
        std::vector<int64_t> inpTile(inputDims, 1);
        inpTile = prodOp->GetTileShape().GetVecTile().tile;
        // Tile not set yet
        if ((inpTile.size() == 1) && (inpTile[0] == -1)) {
            continue;
        }
        // In all other cases recalculate tile through reshape/transpose operations
        RecalcTileThroughOps(prodOp, inpTile, true);
        inputHasTile[inp] = true;
        // Evaluate new tile according to all existing tiles
        GatherFromTiles(op, vectorTilesNew, inpTile, outputDims, inp, inputHasTile);
    }
    GatherFromShapes(op, vectorTilesNew, outputDims, inputHasTile);
    if (op->GetOpcode() != Opcode::OP_GATHER_IN_L1) {
        AdjustTileToUB(op, vectorTilesNew);
    } else {
        AdjustGatherTileToL1(op, vectorTilesNew);
    }
}
void DefaultTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew)
{
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    ASSERT(inputTypeSize != 0);
    uint32_t typedBlock = BLOCK_SIZE / inputTypeSize;
    if (vectorTilesNew[vectorTilesNew.size() - 1] % typedBlock != 0) {
        vectorTilesNew[vectorTilesNew.size() - 1] = (vectorTilesNew[vectorTilesNew.size() - 1] + typedBlock - 1) /
                                                    typedBlock * typedBlock;
    }
    AdjustTileToUB(op, vectorTilesNew);
}

void ExpandTileSetting(Operation* op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew)
{
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = (vectorTilesOld[vectorTilesOld.size() - 1] == 1) ? op->GetOOperands()[0]->shape : vectorTilesOld;
    ASSERT(inputTypeSize != 0);
    uint32_t typedBlock = BLOCK_SIZE / inputTypeSize;
    if (vectorTilesNew[vectorTilesNew.size() - 1] % typedBlock != 0) {
        vectorTilesNew[vectorTilesNew.size() - 1] = (vectorTilesNew[vectorTilesNew.size() - 1] + typedBlock - 1) /
                                                    typedBlock * typedBlock;
    }
    AdjustTileToUB(op, vectorTilesNew);
}

static int32_t FindReducedDimension(Operation* op, const std::vector<int64_t>& maxInputShape, size_t inputDims)
{
    int32_t reducedDim = -1;
    for (uint32_t i = 0; i < inputDims; i++) {
        if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (op->GetOOperands()[0]->shape[i] == 1) &&
            (reducedDim == -1)) {
            reducedDim = i;
        } else if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (op->GetOOperands()[0]->shape[i] == 1) &&
                   (reducedDim != -1)) {
            return -NUM2;
        }
    }
    return reducedDim;
}

static void ProcessOtherDimsForReduce(Operation* op, std::vector<int64_t>& vectorTilesNew,
                                      const std::vector<int64_t>& maxInputShape, int32_t reducedDim, size_t inputDims,
                                      int64_t tileSize, int64_t inputTypeSize)
{
    (void)op;
    if (inputTypeSize == 0) {
        return;
    }
    bool setBlock = false;
    int64_t curTile;
    int64_t blockSizePerType = BLOCK_SIZE / inputTypeSize;
    for (int64_t dim = inputDims - 1; dim >= 0; dim--) {
        if (dim == reducedDim) {
            continue;
        }
        curTile = ((reducedDim == (int32_t)(inputDims - 1))) ?
                      (((maxInputShape[dim] != 1) && (!setBlock)) ? blockSizePerType : 1) :
                      std::min(FloorPowerOf2(tileSize), FloorPowerOf2(maxInputShape[dim]));
        if (curTile == blockSizePerType) {
            if (maxInputShape[dim] < NUM2 * curTile && (tileSize > maxInputShape[dim])) {
                curTile = maxInputShape[dim];
            }
            setBlock = true;
        }
        vectorTilesNew[dim] = std::min(FloorPowerOf2(tileSize), curTile);
        tileSize /= vectorTilesNew[dim];
    }
}

void ReduceTileSetting(Operation* op, std::vector<int64_t>& vectorTilesNew)
{
    size_t inputsNum = op->GetIOperands().size();
    size_t outputsNum = op->GetOOperands().size();
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    size_t outputDims = DimsCalculation(op, outputsNum, false);
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint32_t argsCount = NUM3;
    ASSERT(maxTypeSize != 0);
    ASSERT(inputTypeSize != 0);
    ASSERT(argsCount != 0);
    int64_t tileSize = UB_MAX_SIZE / (maxTypeSize * argsCount);
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
    ASSERT(outputsNum == 1) << "Reduce have several outputs";
    ASSERT(inputDims == outputDims) << "Reduce have different shape size for input and output";
    int32_t reducedDim = FindReducedDimension(op, maxInputShape, inputDims);
    if (reducedDim == -NUM2) {
        return;
    }
    ASSERT(reducedDim >= 0) << "Not found reduced dims";
    int64_t curTile = (reducedDim == (int32_t)(inputDims - 1)) ?
                          std::max(maxInputShape[reducedDim], BLOCK_SIZE / inputTypeSize) :
                          maxInputShape[reducedDim];
    vectorTilesNew[reducedDim] = curTile;
    tileSize /= curTile;
    ProcessOtherDimsForReduce(op, vectorTilesNew, maxInputShape, reducedDim, inputDims, tileSize, inputTypeSize);
}

void IndexInOutTileSetting(Operation* op, std::vector<int64_t>& vectorTilesNew)
{
    size_t inputsNum = op->GetIOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    ASSERT(inputTypeSize != 0);
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
    int64_t curTile = std::max(maxInputShape[inputDims - 1], BLOCK_SIZE / inputTypeSize);
    curTile = (maxInputShape[inputDims - 1] >= (UINT8MAX * BLOCK_SIZE / inputTypeSize)) ?
                  std::min(static_cast<int64_t>(FloorPowerOf2(UINT8MAX * BLOCK_SIZE / inputTypeSize)),
                           curTile) :
                  curTile; // Consider additional restriction for REDUCE ops
    // temporal solution, check if it s is good
    vectorTilesNew[inputDims - 1] = curTile;
    auto curProd = curTile;
    for (size_t di = 0; di < inputDims - 1; di++) {
        if (op->GetIOperands()[0]->shape[di] != 1) {
            if (curProd > MIN_TILE_SIZE) {
                vectorTilesNew[di] = 1;
            } else {
                vectorTilesNew[di] = std::max(
                    ((MIN_TILE_SIZE / curProd) / (BLOCK_SIZE / inputTypeSize)) * (BLOCK_SIZE / inputTypeSize),
                    (long int)1);
                if (scatterOps.find(op->GetOpcode()) != scatterOps.end() && di == inputDims - NUM2) {
                    vectorTilesNew[di] = std::max<int64_t>(
                        1, std::min(vectorTilesNew[di] / TRANSPOSE_NUM * TRANSPOSE_NUM,
                                    (maxInputShape[di] - 1) / TRANSPOSE_NUM * TRANSPOSE_NUM));
                }
                vectorTilesNew[di] = std::min(vectorTilesNew[di], op->GetIOperands()[0]->shape[di]);
                curProd *= vectorTilesNew[di];
            }
        } else {
            vectorTilesNew[di] = 1;
        }
    }
}

void TopkTileSetting(Operation* op, std::vector<int64_t>& vectorTilesNew)
{
    size_t inputsNum = op->GetIOperands().size();
    size_t outputsNum = op->GetOOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    size_t outputDims = DimsCalculation(op, outputsNum, false);
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint32_t argsCount = NUM3 + NUM4; // input + 2 outputs + temporal buffer (4 inputs)
    ASSERT(inputDims == outputDims) << "Reduce have different shape size for input and output";
    ASSERT(maxTypeSize != 0);
    ASSERT(argsCount != 0);
    ASSERT(op->GetIOperands().size() == 1);
    ASSERT(op->GetOOperands().size() >= 1);
    int64_t tileSize = UB_MAX_SIZE / (maxTypeSize * argsCount);
    auto maxInputShape = op->GetIOperands()[0]->GetShape();
    int32_t reducedDim = -1;
    for (uint32_t i = 0; i < inputDims; i++) {
        if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (reducedDim == -1)) {
            reducedDim = i;
        } else if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (reducedDim != -1)) {
            return;
        }
    }
    if (reducedDim == -1) {
        vectorTilesNew = maxInputShape;
        return;
    }
    vectorTilesNew[reducedDim] = std::min(NUM2 * op->GetOOperands()[0]->GetShape()[reducedDim],
                                          maxInputShape[reducedDim]);
    vectorTilesNew[reducedDim] = std::min(tileSize, vectorTilesNew[reducedDim]);
    // Other Dims processing
    for (int64_t dim = inputDims - 1; dim >= 0; dim--) {
        if (dim == reducedDim) {
            continue;
        }
        vectorTilesNew[dim] = 1;
    }
}

void AdaptTileToRealOutputShape(Operation* op, int opBaseMagic, std::vector<int64_t>& vectorTiles)
{
    size_t outputsNum = op->GetOOperands().size();
    size_t outputDims = DimsCalculation(op, outputsNum, false);
    std::vector<int64_t> maxOutShape(outputDims, -1);
    for (auto outOp : op->GetOOperands()) {
        bool outToOpbase = false;
        for (auto c : outOp->GetConsumers()) {
            if (c->GetOpMagic() == opBaseMagic) {
                outToOpbase = true;
                break;
            }
        }
        if (!outToOpbase) {
            continue;
        }
        for (size_t di = 0; di < outOp->shape.size(); di++) {
            maxOutShape[di] = (outOp->shape[di] != -1) ? std::max(maxOutShape[di], outOp->shape[di]) : maxOutShape[di];
        }
    }
    for (size_t s = vectorTiles.size(); s < maxOutShape.size(); s++) {
        vectorTiles.insert(vectorTiles.begin(), 1);
    }
    vectorTiles.resize(maxOutShape.size()); // resize is required for case if vectorTiles.size > shape.size
    for (size_t di = 0; di < vectorTiles.size(); di++) {
        vectorTiles[di] = (maxOutShape[di] != -1) ? std::min(maxOutShape[di], vectorTiles[di]) : vectorTiles[di];
    }
}

bool IsBroadcastedShapes(Operation* op)
{
    auto inputsNum = op->GetIOperands().size();
    auto outputsNum = op->GetOOperands().size();
    if ((inputsNum <= 1) || (outputsNum == 0)) {
        return false;
    }
    Shape outShape = op->GetOOperands()[0]->GetShape();
    for (auto out : op->GetOOperands()) {
        if (outShape.size() != out->GetShape().size()) {
            return false;
        }
        for (size_t di = 0; di < outShape.size(); di++) {
            if (outShape[di] != out->GetShape()[di]) {
                return false;
            }
        }
    }
    for (auto inp : op->GetIOperands()) {
        auto inpShape = inp->GetShape();
        if (inpShape.size() != outShape.size()) {
            return false;
        }
        for (size_t di = 0; di < outShape.size(); di++) {
            if (outShape[di] % inpShape[di] != 0) {
                return false;
            }
        }
    }
    return true;
}

void AlignLastDim(Operation* op, std::vector<int64_t>& vectorTiles)
{
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    if (inputTypeSize == 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "inputTypeSize is zero in AlignLastDim, cannot divide");
        return;
    }
    int64_t lastDim = std::max(vectorTiles[vectorTiles.size() - 1], BLOCK_SIZE / inputTypeSize);
    if (lastDim != vectorTiles[vectorTiles.size() - 1]) {
        size_t inputsNum = op->GetIOperands().size();
        size_t outputsNum = op->GetOOperands().size();
        uint32_t argsCount = inputsNum + outputsNum;
        int64_t tileRatio = static_cast<int64_t>(
            std::ceil(static_cast<double>(lastDim) / static_cast<double>(vectorTiles[vectorTiles.size() - 1])));
        vectorTiles[vectorTiles.size() - 1] = lastDim;
        auto wholeSize = std::accumulate(vectorTiles.begin(), vectorTiles.end(), 1, std::multiplies<int64_t>());
        const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
        int64_t maxTile = (UB_MAX_SIZE / inputTypeSize / argsCount);
        while (wholeSize > maxTile) {
            for (size_t di = 0; di < op->GetIOperands()[0]->GetShape().size() - 1; di++) {
                int64_t curTileRatio = std::min(vectorTiles[di], tileRatio);
                if (curTileRatio == 0) {
                    APASS_LOG_ERROR_F(Elements::Operation, "curTileRatio is zero in AlignLastDim loop, cannot divide");
                    return;
                }
                vectorTiles[di] /= curTileRatio;
                tileRatio /= curTileRatio;
            }
            wholeSize = std::accumulate(vectorTiles.begin(), vectorTiles.end(), 1, std::multiplies<int64_t>());
        }
    }
}

void BackPropagate(Operation* producerOp, Operation* op)
{
    size_t inputsNum = producerOp->GetIOperands().size();
    size_t inputDims = DimsCalculation(producerOp, inputsNum, true);
    std::vector<int64_t> vectorTilesNew(inputDims, -1);
    std::vector<int64_t> vectorTilesOld = op->GetTileShape().GetVecTile().tile;
    if (producerOp->GetOpcode() == Opcode::OP_REDUCE_ACC) {
        return;
    }
    // Correct vectorTileOld to real shape of output
    if (gatherVectorOps.find(op->GetOpcode()) == gatherVectorOps.end() &&
        gatherMoveOps.find(op->GetOpcode()) == gatherMoveOps.end()) {
        AdaptTileToRealOutputShape(producerOp, op->GetOpMagic(), vectorTilesOld);
    } else {
        TileThroughGather(producerOp, op, vectorTilesOld);
    }
    // Recalculate tile from tile of output to tile of input tensor (backward)
    RecalcTileThroughOps(producerOp, vectorTilesOld, false);
    // Map lookup for backward tile setting handlers
    auto handlerIt = backwardTileHandlers.find(producerOp->GetOpcode());
    if (handlerIt != backwardTileHandlers.end()) {
        handlerIt->second(producerOp, vectorTilesOld, vectorTilesNew);
    } else {
        DefaultTileSetting(producerOp, vectorTilesOld, vectorTilesNew);
    }
    if (OpcodeManager::Inst().IsCopyInOrOut(producerOp->GetOpcode())) {
        AlignLastDim(producerOp, vectorTilesNew);
    }
    producerOp->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    if (producerOp->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
        SetGatherInL1CubeTile(producerOp, vectorTilesNew);
    }
}

void Propagate(Operation* consumerOp, Operation* op)
{
    size_t inputsNum = consumerOp->GetIOperands().size();
    size_t inputDims = DimsCalculation(consumerOp, inputsNum, true);
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(consumerOp, inputsNum, inputDims);
    std::vector<int64_t> vectorTilesNew(inputDims, -1);
    std::vector<int64_t> vectorTilesOld = op->GetTileShape().GetVecTile().tile;
    // Recalculate tile through reshape/transpose/view/assemble operations
    RecalcTileThroughOps(op, vectorTilesOld, true);
    AdaptTileToRealOutputShape(op, consumerOp->GetOpMagic(), vectorTilesOld);
    // Map lookup for forward tile setting handlers first
    auto handlerIt = forwardTileHandlers.find(consumerOp->GetOpcode());
    if (handlerIt != forwardTileHandlers.end()) {
        handlerIt->second(consumerOp, vectorTilesOld, vectorTilesNew);
    } else if (IsBroadcastedShapes(consumerOp)) {
        OpWithSeveralInputsTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else {
        DefaultTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    }
    if (gatherVectorOps.find(consumerOp->GetOpcode()) == gatherVectorOps.end() &&
        consumerOp->GetOpcode() != Opcode::OP_EXPAND) {
        AdaptTileToRealInputShape(consumerOp, vectorTilesNew);
    }
    if (OpcodeManager::Inst().IsCopyInOrOut(consumerOp->GetOpcode())) {
        AlignLastDim(consumerOp, vectorTilesNew);
    } else if (OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
        AlignLastDim(op, vectorTilesNew);
        op->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    }
    consumerOp->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    if (consumerOp->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
        SetGatherInL1CubeTile(consumerOp, vectorTilesNew);
    }
}

void BackwardPropagation(std::vector<Operation*> orderedOperations, std::queue<Operation*>& queueBFS,
                         std::map<int, bool>& visitedNodesBFS)
{
    for (auto cubeOp : orderedOperations) {
        queueBFS.push(cubeOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto producerOp : op->ProducerOps()) {
                if (cubeMMOps.find(producerOp->GetOpcode()) != cubeMMOps.end() ||
                    singletonTileHandlers.find(producerOp->GetOpcode()) != singletonTileHandlers.end()) {
                    continue;
                }
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedNodesBFS);
                if (isContinue) {
                    continue;
                }
                if (cubeMMOps.find(op->GetOpcode()) != cubeMMOps.end()) {
                    CubeInDepsProcessing(op, producerOp);
                }
                BackPropagate(producerOp, op);

                // Update queueBFS and visitedNodesBFS
                UpdateBFS(producerOp, queueBFS, visitedNodesBFS);
            }
        }
        visitedNodesBFS.clear();
    }
}

void ForwardPropagation(std::vector<Operation*> orderedOperations, std::queue<Operation*>& queueBFS,
                        std::map<int, bool>& visitedNodesBFS)
{
    for (auto cubeOp : orderedOperations) {
        queueBFS.push(cubeOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto consumerOp : op->ConsumerOps()) {
                if (cubeMMOps.find(consumerOp->GetOpcode()) != cubeMMOps.end() ||
                    singletonTileHandlers.find(consumerOp->GetOpcode()) != singletonTileHandlers.end()) {
                    continue;
                }
                // Only skip propagation if op is NOT a cube op and vecTile is unset
                if ((op->GetTileShape().GetVecTile().size() == 1) && (op->GetTileShape().GetVecTile()[0] == -1) &&
                    (cubeMMOps.find(op->GetOpcode()) == cubeMMOps.end())) {
                    UpdateBFS(consumerOp, queueBFS, visitedNodesBFS);
                    continue;
                }
                // For cube ops, set vecTile from cubeTile (m, n dimensions)
                if (cubeMMOps.find(op->GetOpcode()) != cubeMMOps.end()) {
                    CubeOutDepsProcessing(op);
                }
                auto prevTile = consumerOp->GetTileShape().GetVecTile();
                // Find max values of the shape dimensions among the outputs, to find max boundary of the tile values
                Propagate(consumerOp, op);
                if ((consumerOp->GetTileShape().GetVecTile().size() == 1) &&
                    (consumerOp->GetTileShape().GetVecTile()[0] == -1)) {
                    continue;
                }
                if (prevTile.size() != consumerOp->GetTileShape().GetVecTile().size()) {
                    visitedNodesBFS[consumerOp->GetOpMagic()] = false;
                    // Update queueBFS and visitedNodesBFS
                    UpdateBFS(consumerOp, queueBFS, visitedNodesBFS);
                    continue;
                }
                size_t di = 0;
                while ((di < prevTile.size()) && (prevTile[di] == consumerOp->GetTileShape().GetVecTile()[di])) {
                    di++;
                }
                if (di < prevTile.size()) {
                    visitedNodesBFS[consumerOp->GetOpMagic()] = false;
                }

                // Update queueBFS and visitedNodesBFS
                UpdateBFS(consumerOp, queueBFS, visitedNodesBFS);
            }
        }
        visitedNodesBFS.clear();
    }
}

void SetReduceTiles(std::vector<Operation*> reduceOrderedOperations)
{
    for (auto op : reduceOrderedOperations) {
        size_t inputsNum = op->GetIOperands().size();
        size_t inputDims = DimsCalculation(op, inputsNum, true);
        std::vector<int64_t> vectorTilesReduce(inputDims, 1);
        // Find max values of the shape dimensions among the inputs, to find max boundary of the tile values
        std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
        auto handlerIt = singletonTileHandlers.find(op->GetOpcode());
        if (handlerIt != singletonTileHandlers.end()) {
            handlerIt->second(op, vectorTilesReduce);
        }
        op->GetTileShapeForSetting().SetVecTile(vectorTilesReduce);
    }
}

static std::vector<int64_t> CalculateDefaultTileSequence(Operation& op, size_t inputDims, int64_t inputTypeSize)
{
    if (inputTypeSize == 0) {
        inputTypeSize = 1;
    }
    std::vector<int64_t> tiles;
    int64_t defaultTileSize = DEFAULT_TILE_SIZE;
    int64_t curTile = std::min(
        defaultTileSize, static_cast<int64_t>(std::pow(
                             NUM2, static_cast<int64_t>(std::log2(std::max(op.GetIOperands()[0]->shape[inputDims - 1],
                                                                           BLOCK_SIZE / inputTypeSize))))));
    curTile = (curTile == 0) ? MIN_TILE_SIZE : curTile;
    tiles.push_back(curTile);
    defaultTileSize /= curTile;
    for (size_t j = 1; j < inputDims; j++) {
        curTile = std::min(defaultTileSize,
                           static_cast<int64_t>(std::pow(
                               NUM2, static_cast<int64_t>(std::log2(op.GetIOperands()[0]->shape[inputDims - 1 - j])))));
        curTile = (curTile == 0) ? MIN_TILE_SIZE : curTile;
        tiles.push_back(curTile);
        defaultTileSize /= curTile;
    }
    std::reverse(tiles.begin(), tiles.end());
    return tiles;
}

static void SetTilesForNoConsumerOp(Operation& op)
{
    if (op.GetTileShape().GetVecTile()[0] != -1) {
        return;
    }
    if (!IsOperationUBReq(op.GetOpcode()) && !IsOperationL1Req(op.GetOpcode())) {
        SetFullShapeTiles(op);
        return;
    }
    size_t inputDims = op.GetIOperands()[0]->shape.size();
    int64_t inputTypeSize = BytesOf(op.GetIOperands()[0]->tensor->GetDataType());
    std::vector<int64_t> tiles = CalculateDefaultTileSequence(op, inputDims, inputTypeSize);
    op.GetTileShapeForSetting().SetVecTile(tiles);
}

std::vector<Operation*> FillNoConsumersOperations(Function& function)
{
    std::vector<Operation*> noConsumersOperations;
    for (auto& op : function.Operations()) {
        if (op.ConsumerOps().size() == 0) {
            SetTilesForNoConsumerOp(op);
            noConsumersOperations.push_back(&op);
        }
    }
    return noConsumersOperations;
}

void BackwardNoConsumersPropagation(std::vector<Operation*> noConsumersOperations, std::queue<Operation*>& queueBFS,
                                    std::map<int, bool>& visitedNodesBFS)
{
    for (auto noConsumerOp : noConsumersOperations) {
        queueBFS.push(noConsumerOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto producerOp : op->ProducerOps()) {
                bool isVisitedNode = (producerOp->GetTileShape().GetVecTile()[0] != -1);
                if (isVisitedNode) {
                    // Update queueBFS and visitedNodesBFS
                    UpdateBFS(producerOp, queueBFS, visitedNodesBFS);
                    continue;
                }
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedNodesBFS);
                if (isContinue) {
                    continue;
                }
                // Call propagation
                BackPropagate(producerOp, op);
                // Update queueBFS and visitedNodesBFS
                UpdateBFS(producerOp, queueBFS, visitedNodesBFS);
            }
        }
        visitedNodesBFS.clear();
    }
}

void SetHeuristicVectorTiles(Function& function, const std::set<Operation*>& cubeOperations,
                             const std::set<Operation*>& inpOperations)
{
    // Define priority levels for operation processing
    enum PriorLevel : uint8_t { CUBE_AND_INPUT_O_LEVEL = 1, REDUCE_AND_INPUT_O_LEVEL = NUM2 };
    // Define cube operations ordered by depth
    std::map<int, int> subgrDepthMap;
    std::map<uint8_t, std::vector<Operation*>> priorOps;
    priorOps[PriorLevel::CUBE_AND_INPUT_O_LEVEL] = SortCubeOpsByDepth(function, cubeOperations, subgrDepthMap);
    for (auto op : inpOperations) {
        priorOps[PriorLevel::CUBE_AND_INPUT_O_LEVEL].insert(priorOps[PriorLevel::CUBE_AND_INPUT_O_LEVEL].begin(), op);
    }
    // Define auxiliary data structures
    std::queue<Operation*> queueBFS;
    std::map<int, bool> visitedNodesBFS;
    // Sort reduce operations by depth
    std::vector<std::pair<Operation*, int>> reduceTmpOperations;
    for (auto& op : function.Operations()) {
        if (singletonTileHandlers.find(op.GetOpcode()) != singletonTileHandlers.end()) {
            reduceTmpOperations.push_back(std::make_pair(&op, subgrDepthMap[op.GetOpMagic()]));
        }
    }
    std::sort(
        reduceTmpOperations.begin(), reduceTmpOperations.end(),
        [](const std::pair<Operation*, int>& x, const std::pair<Operation*, int>& y) { return x.second < y.second; });
    for (auto op : reduceTmpOperations) {
        priorOps[PriorLevel::REDUCE_AND_INPUT_O_LEVEL].push_back(op.first);
    }
    reduceTmpOperations.clear();
    // Need to set initial vector tiles for ReduceOps
    SetReduceTiles(priorOps[PriorLevel::REDUCE_AND_INPUT_O_LEVEL]);
    for (auto op : inpOperations) {
        priorOps[PriorLevel::REDUCE_AND_INPUT_O_LEVEL].insert(priorOps[PriorLevel::REDUCE_AND_INPUT_O_LEVEL].begin(),
                                                              op);
    }
    for (auto pOps : priorOps) {
        BackwardPropagation(pOps.second, queueBFS, visitedNodesBFS);
        ForwardPropagation(pOps.second, queueBFS, visitedNodesBFS);
    }
    std::vector<int64_t> vectorTilesNew;
    // 3. Backward propagation Matmul (need to set tiles for rest -1 tiles), Start from nodes without consumers
    std::vector<Operation*> noConsumersOperations = FillNoConsumersOperations(function);
    BackwardNoConsumersPropagation(noConsumersOperations, queueBFS, visitedNodesBFS);
}

void GenerateJsonForPython(Function& function)
{
    constexpr int kJsonDumpIndent = 4; // JSON 输出缩进空格数
    json pythonJson;
    std::ofstream python_tiles(config::LogTopFolder() + "/python_tiles.json");
    int operationIdx = 0;
    if (python_tiles.is_open()) {
        for (auto& op : function.Operations()) {
            std::string opIdName = "operation_" + std::to_string(operationIdx);
            auto full_dump = op.DumpJson();
            if (full_dump["file"].is_null()) {
                continue;
            }
            if (op.GetCoreTypeStr() == "AIC") {
                pythonJson[opIdName]["type"] = "CubeTile";
                auto cubeShape = op.GetTileShape();
                auto tile = cubeShape.GetCubeTile();
                pythonJson[opIdName]["tile"] = {tile.m[0], tile.m[1], tile.k[0], tile.k[1], tile.n[0], tile.n[1]};
            } else {
                auto vecShape = op.GetTileShape();
                auto tile = vecShape.GetVecTile();
                pythonJson[opIdName]["type"] = "VecTile";
                for (size_t i = 0; i < tile.size(); ++i) {
                    pythonJson[opIdName]["tile"].push_back(tile[i]);
                }
            }
            pythonJson[opIdName]["magic"] = full_dump["opmagic"];
            pythonJson[opIdName]["opcode"] = full_dump["opcode"];
            pythonJson[opIdName]["file"] = full_dump["file"];
            pythonJson[opIdName]["line"] = full_dump["line"];
            operationIdx += 1;
        }
        python_tiles << pythonJson.dump(kJsonDumpIndent) << std::endl;
    }
}

void GenerateJsonForSemanticLabels(Function& function)
{
    constexpr int kJsonDumpIndent = 4; // JSON 输出缩进空格数
    json semanticJson;
    std::ofstream graph_tiles(config::LogTopFolder() + "/semantic_labels_tiles.json");
    if (graph_tiles.is_open()) {
        for (auto& op : function.Operations()) {
            if (op.GetSemanticLabel()) {
                auto sem_label = op.GetSemanticLabel()->label;
                semanticJson[sem_label] = {{"filename", op.GetSemanticLabel()->filename},
                                           {"line_num", op.GetSemanticLabel()->lineno}};
                if (op.GetCoreTypeStr() == "AIC") {
                    auto cubeShape = op.GetTileShape();
                    auto tile = cubeShape.GetCubeTile();
                    semanticJson[sem_label]["type"] = "CubeTile";
                    semanticJson[sem_label]["tile"] = {tile.m[0], tile.m[1], tile.k[0],
                                                       tile.k[1], tile.n[0], tile.n[1]};
                } else {
                    auto vecShape = op.GetTileShape();
                    auto tile = vecShape.GetVecTile();
                    semanticJson[sem_label]["type"] = "VecTile";
                    for (size_t i = 0; i < tile.size(); ++i) {
                        semanticJson[sem_label]["tile"].push_back(tile[i]);
                    }
                }
                semanticJson[sem_label]["operation"] = op.GetOpcodeStr();
            }
        }
        graph_tiles << semanticJson.dump(kJsonDumpIndent) << std::endl;
    }
}

void PrintTiles(Function& function, std::string fileName)
{
    int opIdx = 0;
    std::ofstream file(fileName, std::ofstream::app);
    if (file.is_open()) {
        file << "\n\n---------------------------------------BEGIN---------------------------------------\n\n"
             << std::endl;
        for (auto& op : function.Operations()) {
            if (op.GetCoreTypeStr() == "AIC") {
                file << "!Cube Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic()
                     << std::endl;
                file << op.GetTileShape().ToString(TileType::CUBE) << std::endl;
            } else if (op.GetCoreTypeStr() == "AIV") {
                file << "!Vector Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic()
                     << std::endl;
                file << op.GetTileShape().ToString(TileType::VEC) << std::endl;
            } else {
                file << "!Other Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic()
                     << std::endl;
                file << op.GetTileShape().ToString(TileType::VEC) << std::endl;
            }
            for (size_t it = 0; it < op.GetIOperands().size(); it++) {
                file << "Input " << it << " Magic = " << op.GetIOperands()[it]->magic
                     << " DataType = " << BytesOf(op.GetIOperands()[it]->tensor->GetDataType()) << " Shape = [";
                auto InputShape = op.GetIOperands()[it]->shape;
                for (size_t j = 0; j < InputShape.size(); j++) {
                    file << InputShape[j] << " ";
                }
                file << "]  ";
            }
            for (size_t it = 0; it < op.GetOOperands().size(); it++) {
                file << "Output " << it << " Magic = " << op.GetOOperands()[it]->magic
                     << " DataType = " << BytesOf(op.GetOOperands()[it]->tensor->GetDataType()) << " Shape = [";
                auto OutputShape = op.GetOOperands()[it]->shape;
                for (size_t j = 0; j < OutputShape.size(); j++) {
                    file << OutputShape[j] << " ";
                }
                file << "]" << std::endl;
            }
            file << "\n----------------------------------------------------------------------------------" << std::endl;
            opIdx++;
        }
        file << "\n\n---------------------------------------END---------------------------------------\n\n"
             << std::endl;
        file.close();
    }
}

void SetHeuristicTileShapes::SetHeuristicTileShapesFunc(Function& function) const
{
    (void)function;
    // Find all cube operations from all operations
    std::set<Operation*> cubeOperations;
    for (auto& op : function.Operations()) {
        if (op.GetCoreTypeStr() == "AIC" && cubeMMOps.find(op.GetOpcode()) != cubeMMOps.end()) {
            cubeOperations.insert(&op);
        }
    }
#ifdef PRINT_TILES
    PrintTiles(function, "main_tiles.txt");
#endif
#ifdef CUBE_TILES
    SetMMTiles(function, cubeOperations);
#endif
#ifdef VECTOR_TILES
    // Define -1 tile shapes for non-cubes operations
    std::set<Operation*> inpOperations;
    std::vector<int64_t> defTile = {-1};
    for (auto& op : function.Operations()) {
        op.GetTileShapeForSetting().SetVecTile(defTile);
        if ((op.GetIOperands().size() == 1) && (op.GetIOperands()[0]->GetProducers().size() == 0)) {
            inpOperations.insert(&op);
        }
    }
    // Set heuristic vector tiles
    SetHeuristicVectorTiles(function, cubeOperations, inpOperations);
    // Check that all tiles was defined by algorithm
    for (auto& op : function.Operations()) {
        ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, op.GetTileShape().GetVecTile()[0] != -1)
            << "Not all tiles were set";
    }
#endif
#ifdef PRINT_TILES
    PrintTiles(function, "custom_tiles.txt");
#endif
    GenerateJsonForPython(function);
    GenerateJsonForSemanticLabels(function);
}
} // namespace npu::tile_fwk
