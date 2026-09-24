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
 * \file shmem_expand_funcion.cpp
 * \brief
 */

#include "distributed_expand.h"
#include "distributed_common.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::Distributed {
namespace {
using DummyTileFunc = std::function<LogicalTensorPtr(int32_t tileIndex)>;
constexpr uint16_t UB_BUFFER_BYTE_SIZE = 16 * 1024;
constexpr uint16_t DTYPE_CAST_BYTE_SIZE = 256;
constexpr uint16_t UB_ALIGN_SIZE = 32;

LogicalTensorPtr ViewNDTile(const LogicalTensorPtr dummy, int32_t tileIndex, const std::vector<int32_t>& tileNumsDim,
                            Function& function)
{
    size_t dim = tileNumsDim.size();
    Shape dummyShape = dummy->shape;

    ASSERT(DistributedErrorCode::DIVISION_BY_ZERO, dim > 0) << "tileNumsDim cannot be empty";
    for (size_t i = 0; i < dim; ++i) {
        ASSERT(DistributedErrorCode::DIVISION_BY_ZERO, tileNumsDim[i] != 0)
            << "tileNumsDim[" << i << "] cannot be zero";
    }

    std::vector<int32_t> dimIndices(dim);
    int32_t remaining = tileIndex;

    for (size_t i = dim; i > 0; --i) {
        dimIndices[i - 1] = remaining % tileNumsDim[i - 1];
        remaining = remaining / tileNumsDim[i - 1];
    }

    std::vector<int64_t> tileShapes(dim);
    std::vector<int64_t> tileOffsets(dim);

    for (size_t i = 0; i < dim; ++i) {
        int32_t totalDimSize = static_cast<int32_t>(dummyShape[i]);
        int32_t tileNum = tileNumsDim[i];
        int32_t base = totalDimSize / tileNum;
        int32_t rem = totalDimSize % tileNum;
        int32_t idx = dimIndices[i];

        if (idx < rem) {
            tileShapes[i] = base + 1;
            tileOffsets[i] = idx * (base + 1);
        } else {
            tileShapes[i] = base;
            tileOffsets[i] = rem * (base + 1) + (idx - rem) * base;
        }
    }

    return dummy->View(function, tileShapes, tileOffsets);
}

std::vector<int32_t> CalcTileCounts(const LogicalTensorPtr shmemTensor, const VecTile& vecTile)
{
    size_t shmemDim = shmemTensor->shape.size();
    size_t vecTileDim = vecTile.size();
    size_t startDim = shmemDim - vecTileDim;

    std::vector<int32_t> tileNumsDim(vecTileDim, 1);

    for (size_t i = 0; i < vecTileDim; ++i) {
        int32_t totalShape = static_cast<int32_t>(shmemTensor->shape[startDim + i]);
        int32_t tileShape = static_cast<int32_t>(vecTile[i]);
        tileNumsDim[i] = totalShape / tileShape + (totalShape % tileShape == 0 ? 0 : 1);
    }

    return tileNumsDim;
}

bool CheckAllTileDims(const LogicalTensorPtr dummy, const std::vector<int32_t>& tileNumsDim)
{
    if (dummy->shape.size() != tileNumsDim.size()) {
        return false;
    }

    for (size_t i = 0; i < tileNumsDim.size(); ++i) {
        if (static_cast<int32_t>(dummy->shape[i]) < tileNumsDim[i]) {
            return false;
        }
    }
    return true;
}

DummyTileFunc GetDummyTileFunc(const LogicalTensorPtr dummy, const LogicalTensorPtr shmemTensor, const VecTile& vecTile,
                               Function& function)
{
    std::vector<int32_t> tileNumsDim = CalcTileCounts(shmemTensor, vecTile);

    if (CheckAllTileDims(dummy, tileNumsDim)) {
        return [dummy, tileNumsDim, &function](int32_t tileIndex) -> LogicalTensorPtr {
            return ViewNDTile(dummy, tileIndex, tileNumsDim, function);
        };
    }
    return [dummy](int32_t tileIndex) -> LogicalTensorPtr {
        (void)tileIndex;
        return dummy;
    };
}

void DfsTiling(const Shape& shmemTensorTileShape, Input& input, size_t curDim, uint32_t& tileIndex,
               std::function<void(uint32_t, Input&)> addTileOp)
{
    std::vector<int64_t>& tileShape = input.tileInfo.shape;
    std::vector<int64_t>& tileOffset = input.tileInfo.offset;
    if (curDim == tileShape.size()) {
        addTileOp(tileIndex, input);
        tileIndex++;
        return;
    }
    int64_t total = input.tensor.GetShape()[curDim];
    for (int64_t offset = 0; offset < total; offset += shmemTensorTileShape[curDim]) {
        tileShape[curDim] = std::min(total - offset, shmemTensorTileShape[curDim]);
        tileOffset[curDim] = offset;
        DfsTiling(shmemTensorTileShape, input, curDim + 1, tileIndex, addTileOp);
    }
}

void DfsTiling(const VecTile& vecTile, const LogicalTensorPtr shmemTensor,
               std::function<void(uint32_t, Input&)> addTileOp)
{
    size_t dim = shmemTensor->shape.size();
    Shape shmemTensorTileShape = shmemTensor->shape;
    Shape shmemTensorTileOffset = shmemTensor->offset;
    size_t shmemTensorStartDim = dim - vecTile.size();
    std::copy(vecTile.tile.begin(), vecTile.tile.end(), shmemTensorTileShape.begin() + shmemTensorStartDim);
    std::fill(shmemTensorTileOffset.begin() + shmemTensorStartDim, shmemTensorTileOffset.end(), 0);
    TileInfo tileInfo{shmemTensorTileShape, shmemTensorTileOffset};
    Input input{shmemTensor, tileInfo};
    uint32_t tileIndex = 0;
    DfsTiling(shmemTensorTileShape, input, shmemTensorStartDim, tileIndex, addTileOp);
}

bool shouldConvertDtype(DataType ubType, DataType castType) { return ubType != castType; }

Shape GetCopyBufferShape(DataType nonShmemDtype, DataType shmemDtype, Shape tileShape)
{
    const uint32_t copyNum = UB_BUFFER_BYTE_SIZE / BytesOf(nonShmemDtype);
    Shape copyShape;
    int64_t tileRowSize = tileShape[tileShape.size() - 2];
    int64_t tileColSize = tileShape[tileShape.size() - 1];
    int64_t alignTileColSize = AlignUp(tileColSize * BytesOf(nonShmemDtype), UB_ALIGN_SIZE) / BytesOf(nonShmemDtype);
    if ((nonShmemDtype != shmemDtype) && ((tileColSize * BytesOf(nonShmemDtype)) % UB_ALIGN_SIZE != 0)) {
        uint32_t copyColSize = copyNum > tileColSize ? tileColSize : copyNum;
        copyShape = {1, copyColSize};
    } else if (copyNum >= tileRowSize * alignTileColSize) {
        copyShape = {tileRowSize, tileColSize};
    } else if (copyNum >= tileColSize) {
        copyShape = {(copyNum + alignTileColSize - 1) / alignTileColSize, tileColSize};
    } else {
        copyShape = {1, copyNum};
    }
    return copyShape;
}

LogicalTensorPtr CreateAdaptiveUbTensor(Function& function, const Shape& shape, DataType ubType, DataType castType,
                                        bool gm2Ub = false)
{
    Shape ubShape = {0};
    int64_t ubLen = shape[0] * AlignUp(shape[1] * BytesOf(ubType), UB_ALIGN_SIZE) / BytesOf(ubType);
    if (!shouldConvertDtype(ubType, castType) && !gm2Ub) {
        ubShape = {ubLen * 2};
    } else {
        uint64_t castSize = AlignUp(ubLen * BytesOf(castType), DTYPE_CAST_BYTE_SIZE);
        if (gm2Ub) {
            ubShape = {static_cast<int64_t>(castSize / BytesOf(ubType))};
        } else {
            ubShape = {(ubLen + static_cast<int64_t>(castSize / BytesOf(ubType))) * 2};
        }
    }
    return std::make_shared<LogicalTensor>(function, ubType, ubShape);
}

std::pair<Shape, Offset> GetNonShmemDataTileShapeAndOffset(const Shape& shmemDataTileShape,
                                                           const Offset& shmemDataTileOffset, size_t nonShmemDataDim)
{
    Shape nonShmemTileShape(nonShmemDataDim);
    Offset nonShmemOffset(nonShmemDataDim);
    size_t shmemDataStartDim = shmemDataTileShape.size() - nonShmemDataDim;
    std::copy(shmemDataTileShape.begin() + shmemDataStartDim, shmemDataTileShape.end(), nonShmemTileShape.begin());
    std::copy(shmemDataTileOffset.begin() + shmemDataStartDim, shmemDataTileOffset.end(), nonShmemOffset.begin());
    return {nonShmemTileShape, nonShmemOffset};
}
} // namespace

void TiledShmemPut(Function& function, const TileShape& tileShape,
                   const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                   const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 3UL)
        << "TiledShmemPut iOperand size is not equal to 3";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemPut oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto in = iOperand[1];
    auto shmemData = iOperand[2];
    auto out = oOperand[0];

    DummyTileFunc predTokenTileFunc = GetDummyTileFunc(predToken, shmemData, tileShape.GetVecTile(), function);
    DummyTileFunc outTileFunc = GetDummyTileFunc(out, shmemData, tileShape.GetVecTile(), function);
    DfsTiling(tileShape.GetVecTile(), shmemData, [&](uint32_t tileIndex, Input& input) {
        Shape shmemDataTileShape = input.tileInfo.shape;
        Offset shmemDataTileOffset = input.tileInfo.offset;
        auto [nonShmemDataTileShape, nonShmemDataTileOffset] = GetNonShmemDataTileShapeAndOffset(
            shmemDataTileShape, shmemDataTileOffset, in->shape.size());
        auto inTile = in->View(function, nonShmemDataTileShape, nonShmemDataTileOffset);
        auto shmemDataTile = shmemData->View(function, shmemDataTileShape, shmemDataTileOffset);
        auto predTokenTile = predTokenTileFunc(tileIndex);
        auto outTile = outTileFunc(tileIndex);
        auto copyBufferShape = GetCopyBufferShape(inTile->Datatype(), shmemDataTile->Datatype(), nonShmemDataTileShape);
        auto ubTensor = CreateAdaptiveUbTensor(function, copyBufferShape, inTile->Datatype(),
                                               shmemDataTile->Datatype());

        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_PUT, {predTokenTile, inTile, shmemDataTile},
                                             {outTile, ubTensor});
        ShmemPutAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        distOpAttr.copyBufferShape = copyBufferShape;
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
        tileOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(nonShmemDataTileOffset),
            OpImmediate::Specified(nonShmemDataTileShape), OpImmediate::Specified(in->shape),
            OpImmediate::Specified(inTile->dynValidShape_)));
        tileOp.SetAttr(OpAttributeKey::isDistCopyOut, true);
    });
}

void TiledShmemStore(Function& function, const TileShape& tileShape,
                     const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                     const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 3UL)
        << "TiledShmemPut iOperand size is not equal to 3";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemPut oOperand size is not equal to 1";
    auto in = iOperand[0];
    auto shmemData = iOperand[1];
    auto barrierDummy = iOperand[2]; // operand 2
    auto dummy = oOperand[0];

    DummyTileFunc barrierDummyTileFunc = GetDummyTileFunc(barrierDummy, shmemData, tileShape.GetVecTile(), function);
    DummyTileFunc dummyTileFunc = GetDummyTileFunc(dummy, shmemData, tileShape.GetVecTile(), function);
    DfsTiling(tileShape.GetVecTile(), shmemData, [&](uint32_t tileIndex, Input& input) {
        Shape shmemDataTileShape = input.tileInfo.shape;
        Offset shmemDataTileOffset = input.tileInfo.offset;
        auto [nonShmemDataTileShape, nonShmemDataTileOffset] = GetNonShmemDataTileShapeAndOffset(
            shmemDataTileShape, shmemDataTileOffset, in->shape.size());
        auto inTile = in->View(function, nonShmemDataTileShape, nonShmemDataTileOffset);
        auto shmemDataTile = shmemData->View(function, shmemDataTileShape, shmemDataTileOffset);
        auto barrierDummyTile = barrierDummyTileFunc(tileIndex);
        auto dummyTile = dummyTileFunc(tileIndex);
        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_STORE, {inTile, shmemDataTile, barrierDummyTile},
                                             {dummyTile});
        ShmemPutAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
        tileOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_UB, OpImmediate::Specified(inTile->offset), OpImmediate::Specified(inTile->shape),
            OpImmediate::Specified(in->shape), OpImmediate::Specified(inTile->dynValidShape_)));
        tileOp.SetAttr(OpAttributeKey::isDistCopyOut, true);
    });
}

void TiledShmemSignal(Function& function, const TileShape& tileShape,
                      const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                      const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 2UL)
        << "TiledShmemSignal iOperand size is not equal to 2";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemSignal oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto shmemSignal = iOperand[1];
    auto out = oOperand[0];

    DummyTileFunc predTokenTileFunc = GetDummyTileFunc(predToken, shmemSignal, tileShape.GetVecTile(), function);
    DummyTileFunc outTileFunc = GetDummyTileFunc(out, shmemSignal, tileShape.GetVecTile(), function);

    ShmemSignalAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    distOpAttr.tileShape = tileShape.GetVecTile().tile;

    DfsTiling(tileShape.GetVecTile(), shmemSignal, [&](uint32_t tileIndex, Input& input) {
        auto predTokenTile = predTokenTileFunc(tileIndex);
        std::vector<int64_t>& shmemSignalTileShape = input.tileInfo.shape;
        std::vector<int64_t>& shmemSignalTileOffset = input.tileInfo.offset;
        auto shmemSignalTile = shmemSignal->View(function, shmemSignalTileShape, shmemSignalTileOffset);
        auto outTile = outTileFunc(tileIndex);
        auto ubTensor = std::make_shared<LogicalTensor>(function, shmemSignal->Datatype(), Shape{SHMEM_SIGNAL_STRIDE});

        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {predTokenTile, shmemSignalTile},
                                             {outTile, ubTensor});
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
        tileOp.SetAttr(OpAttributeKey::dontTouch, true);
    });
}

void TiledShmemWaitUntil(Function& function, const TileShape& tileShape,
                         const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                         const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 2UL)
        << "TiledShmemWaitUntil iOperand size is not equal to 2";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemWaitUntil oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto shmemSignal = iOperand[1];
    auto out = oOperand[0];

    DummyTileFunc predTokenTileFunc = GetDummyTileFunc(predToken, shmemSignal, tileShape.GetVecTile(), function);
    DummyTileFunc outTileFunc = GetDummyTileFunc(out, shmemSignal, tileShape.GetVecTile(), function);
    ShmemWaitUntilAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    auto vecTile = tileShape.GetVecTile();
    distOpAttr.tileShape = tileShape.GetVecTile().tile;

    DfsTiling(tileShape.GetVecTile(), shmemSignal, [&](uint32_t tileIndex, Input& input) {
        auto predTokenTile = predTokenTileFunc(tileIndex);
        std::vector<int64_t>& shmemSignalTileShape = input.tileInfo.shape;
        std::vector<int64_t>& shmemSignalTileOffset = input.tileInfo.offset;
        auto shmemSignalTile = shmemSignal->View(function, shmemSignalTileShape, shmemSignalTileOffset);
        auto outTile = outTileFunc(tileIndex);

        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {predTokenTile, shmemSignalTile}, {outTile});
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
    });
}

void TiledShmemGet(Function& function, const TileShape& tileShape,
                   const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                   const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 2UL)
        << "TiledShmemGet iOperand size is not equal to 2";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemGet oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto shmemData = iOperand[1];
    auto out = oOperand[0];
    DummyTileFunc predTokenTileFunc = GetDummyTileFunc(predToken, shmemData, tileShape.GetVecTile(), function);
    DfsTiling(tileShape.GetVecTile(), shmemData, [&](uint32_t tileIndex, Input& input) {
        auto predTokenTile = predTokenTileFunc(tileIndex);
        std::vector<int64_t>& shmemDataTileShape = input.tileInfo.shape;
        std::vector<int64_t>& shmemDataTileOffset = input.tileInfo.offset;
        auto shmemDataTile = shmemData->View(function, shmemDataTileShape, shmemDataTileOffset);
        auto [nonShmemDataTileShape, nonShmemDataTileOffset] = GetNonShmemDataTileShapeAndOffset(
            shmemDataTileShape, shmemDataTileOffset, out->shape.size());
        auto outTile = out->View(function, nonShmemDataTileShape, nonShmemDataTileOffset);
        auto copyBufferShape = GetCopyBufferShape(out->Datatype(), shmemDataTile->Datatype(), nonShmemDataTileShape);
        auto ubTensor = CreateAdaptiveUbTensor(function, copyBufferShape, out->Datatype(), shmemDataTile->Datatype());

        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_GET, {predTokenTile, shmemDataTile}, {outTile, ubTensor});

        ShmemGetAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        distOpAttr.copyBufferShape = copyBufferShape;
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
        auto toOffset = OpImmediate::Specified(Offset(nonShmemDataTileOffset.size(), 0));
        tileOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_DEVICE_DDR, toOffset, OpImmediate::Specified(nonShmemDataTileShape),
            OpImmediate::Specified(out->shape), OpImmediate::Specified(shmemDataTile->dynValidShape_)));
        tileOp.SetAttr(OpAttributeKey::isDistCopyOut, true);
    });
}

void TiledShmemLoad(Function& function, const TileShape& tileShape,
                    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 2UL)
        << "TiledShmemLoad iOperand size is not equal to 2";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemLoad oOperand size is not equal to 1";
    auto dummy = iOperand[0];
    auto shmemData = iOperand[1];
    auto outUb = oOperand[0];

    DummyTileFunc dummyTileFunc = GetDummyTileFunc(dummy, shmemData, tileShape.GetVecTile(), function);
    DfsTiling(tileShape.GetVecTile(), shmemData, [&](uint32_t tileIndex, Input& input) {
        auto dummyTile = dummyTileFunc(tileIndex);
        Shape shmemDataTileShape = input.tileInfo.shape;
        Offset shmemDataTileOffset = input.tileInfo.offset;
        auto shmemDataTile = shmemData->View(function, shmemDataTileShape, shmemDataTileOffset);
        auto [nonShmemDataTileShape, nonShmemDataTileOffset] = GetNonShmemDataTileShapeAndOffset(
            shmemDataTileShape, shmemDataTileOffset, outUb->shape.size());
        auto outUbTile = outUb->View(function, nonShmemDataTileShape, nonShmemDataTileOffset);

        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_LOAD, {dummyTile, shmemDataTile}, {outUbTile});

        ShmemGetAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
        tileOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(nonShmemDataTileOffset), MEM_UB, OpImmediate::Specified(shmemDataTile->shape),
            OpImmediate::Specified(outUb->shape), OpImmediate::Specified(shmemDataTile->dynValidShape_)));
        tileOp.SetAttr(OpAttributeKey::isDistCopyOut, false);
    });
}

void TiledShmemSet(Function& function, const TileShape& tileShape,
                   const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                   const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void)op;
    (void)tileShape;

    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, iOperand.size() == 2UL)
        << "TiledShmemSet iOperand size is not equal to 2";
    ASSERT(DistributedErrorCode::INVALID_OPERAND_NUM, oOperand.size() == 1UL)
        << "TiledShmemSet oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto shmemTensor = iOperand[1];
    auto out = oOperand[0];

    ASSERT(DistributedErrorCode::INVALID_ALIGNMENT, UB_BUFFER_BYTE_SIZE % REPEAT_BYTE == 0)
        << "UB_BUFFER_BYTE_SIZE must be a multiple of 256, but got " << UB_BUFFER_BYTE_SIZE;
    ShmemSetAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    uint32_t bufferSize = distOpAttr.isSetData ? UB_BUFFER_BYTE_SIZE : SHMEM_SIZE_ALIGN;
    Shape bufferShape{static_cast<int64_t>(bufferSize / BytesOf(shmemTensor->Datatype()))};
    auto buffer = std::make_shared<LogicalTensor>(function, shmemTensor->Datatype(), bufferShape);
    auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_SET, {predToken, shmemTensor}, {out, buffer});
    distOpAttr.setBufferShape = bufferShape;
    tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    tileOp.SetAttr(OpAttributeKey::ownerRank, distOpAttr.ownerRank);
}

void TiledShmemBindTensor(Function& function, const TileShape& tileShape,
                          const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                          const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void)iOperand;
    (void)tileShape;
    auto& oper = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, oOperand);
    SymbolicScalar bindTensor;
    if (op.HasAttr(OpAttributeKey::bindTensor)) {
        bindTensor = op.GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
        oper.SetAttribute(OpAttributeKey::bindTensor, bindTensor);
    }
}
} // namespace npu::tile_fwk::Distributed
