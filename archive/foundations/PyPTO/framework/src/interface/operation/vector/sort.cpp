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
 * \file sort.cpp
 * \brief
 */

#include <string>
#include <queue>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/operator_tracer.h"
#include "tensor_transformation.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/platform.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";
const std::string TOPK_OFFSET = OP_ATTR_PREFIX + "offset";
const std::string TOPK_VALIDBIT = OP_ATTR_PREFIX + "validBit";
const std::string TOPK_START_INDEX = OP_ATTR_PREFIX + "start_index";
const std::string TOPK_MERGE_SIZE = OP_ATTR_PREFIX + "mergeSize";

const std::string SORT_AXIS = OP_ATTR_PREFIX + "axis";
const std::string SORT_GMSTRIDE = OP_ATTR_PREFIX + "gmstride";
const std::string SORT_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string SORT_ORDER = OP_ATTR_PREFIX + "order";
const std::string SORT_OFFSET = OP_ATTR_PREFIX + "offset";
const std::string SORT_FIRSTSHAPE = OP_ATTR_PREFIX + "firstShape";

constexpr int32_t kBlockSize = 32;
constexpr int32_t sort32Size = 32;
constexpr int32_t kFactorSize = 2;
constexpr int32_t kBlockFpNum = 8;
constexpr int64_t maxNumValue = 8192;

void TiledBitSort(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                  const LogicalTensorPtr& result, TileInfo& resultTileInfo, int axis, int isLargest, int idxStart)
{
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        std::vector<int64_t> tmpShape;
        if (resultTile->shape.size() == 1) {
            tmpShape = {resultTile->shape[resultTile->shape.size() - 1]};
        } else {
            tmpShape = {1, resultTile->shape[resultTile->shape.size() - 1]};
        }
        auto tempTensor = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {resultTile, tempTensor});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        op.SetAttribute(TOPK_OFFSET, static_cast<int>(idxStart));
        return;
    }
    // Jump cur axis
    if (cur == static_cast<size_t>(axis)) {
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest, idxStart);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest, idxStart);
    }
}

void TiledBitSort(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                  const LogicalTensorPtr resOperand, int axis, int isLargest, int idxStart)
{
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledBitSort(function, tileShape, 0, input, resOperand, resultTileInfo, axis, isLargest, idxStart);
}

void TensorBitsortOperation(Function& function, LogicalTensorPtr operand, LogicalTensorPtr resOp, int idxStart,
                            int axis, bool isLargest)
{
    auto& op = function.AddOperation(Opcode::OP_BITSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    op.SetAttribute(TOPK_OFFSET, static_cast<int>(idxStart));
    return;
}

Tensor Sort32(const Tensor& self, int idxStart)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sort32");

    std::unordered_set<DataType> supportedTypes = {DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SORT32");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SORT32");
    CheckTensorShapeSize(self.GetStorage(), "SORT32");
    const auto len = static_cast<int>(self.GetShape().size());
    auto outShape = self.GetShape();
    outShape[len - 1] *= NUM_VALUE_2;
    auto result = Tensor(self.GetStorage()->tensor->datatype, outShape);
    CALL(BitsortOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(),
         idxStart, len - 1, 1);
    return result;
}

void TiledMrgSort(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                  const LogicalTensorPtr& result, TileInfo& resultTileInfo, int axis, int k, int mergeSize)
{
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        std::vector<int64_t> tmpShape;
        if (inputTile->shape.size() == 1) {
            tmpShape = {inputTile->shape[inputTile->shape.size() - 1]};
        } else {
            tmpShape = {1, inputTile->shape[inputTile->shape.size() - 1]};
        }
        auto tempTensor = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_MRGSORT, {inputTile}, {resultTile, tempTensor});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_KVALUE, k);
        op.SetAttribute(TOPK_MERGE_SIZE, mergeSize);
        return;
    }
    // Jump cur axis
    if (static_cast<int>(cur) == axis) {
        TiledMrgSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, k, mergeSize);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledMrgSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, k, mergeSize);
    }
}

void TiledMrgSort(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                  const LogicalTensorPtr resOperand, int axis, int k, int mergeSize)
{
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledMrgSort(function, tileShape, 0, input, resOperand, resultTileInfo, axis, k, mergeSize);
}

void TensorMrgSortOperation(Function& function, LogicalTensorPtr operand, LogicalTensorPtr resOp, int mergeSize,
                            int axis, int k)
{
    auto& op = function.AddOperation(Opcode::OP_MRGSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_MERGE_SIZE, static_cast<int>(mergeSize));
    return;
}

Tensor MrgSort(const Tensor& self, int mergeSize)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "MrgSort");

    std::unordered_set<DataType> supportedTypes = {DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MRGSORT");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "MRGSORT");
    CheckTensorShapeSize(self.GetStorage(), "MRGSORT");
    const auto len = static_cast<int>(self.GetShape().size());
    const auto k = static_cast<int>(self.GetShape()[len - 1]);
    auto outShape = self.GetShape();
    auto result = Tensor(self.GetStorage()->tensor->datatype, outShape);
    CALL(MrgSortOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(),
         mergeSize, len - 1, k);
    return result;
}

void TiledExtract(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                  const LogicalTensorPtr& result, TileInfo& resultTileInfo, int maskMode, int kValue, bool isLargest)
{
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_EXTRACT, {inputTile}, {resultTile});
        op.SetAttribute(EXTRACT_MASKMODE, maskMode);
        op.SetAttribute(TOPK_KVALUE, kValue);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }

    // Jump last axis
    if (cur == input.tensor.GetShape().size() - 1) {
        TiledExtract(function, tileShape, cur + 1, input, result, resultTileInfo, maskMode, kValue, isLargest);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledExtract(function, tileShape, cur + 1, input, result, resultTileInfo, maskMode, kValue, isLargest);
    }
}

void TiledExtract(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                  const LogicalTensorPtr resOperand, int maskMode, int kValue, int isLargest)
{
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledExtract(function, tileShape, 0, input, resOperand, resultTileInfo, maskMode, kValue, isLargest);
}

void TensorExtractOperation(Function& function, LogicalTensorPtr operand, LogicalTensorPtr resOp, int maskMode, int k,
                            bool isLargest)
{
    auto& op = function.AddOperation(Opcode::OP_EXTRACT, {operand}, {resOp});
    op.SetAttribute(EXTRACT_MASKMODE, maskMode);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    return;
}

Tensor TopKExtract(const Tensor& self, int k, bool isIndex)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "TopKExtract");

    std::unordered_set<DataType> supportedTypes = {DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "TOPKEXTRACT");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "TOPKEXTRACT");
    CheckTensorShapeSize(self.GetStorage(), "TOPKEXTRACT");
    DataType dType = isIndex ? DataType::DT_INT32 : self.GetStorage()->tensor->datatype;
    const auto len = static_cast<int>(self.GetShape().size());
    auto outShape = self.GetShape();
    outShape[len - 1] = k;
    auto result = Tensor(dType, outShape);
    CALL(ExtractOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(),
         static_cast<int>(isIndex), k, true);
    return result;
}

void TiledTopK(Function& function, const TileShape& tileShape, size_t cur, Input& input,
               const LogicalTensorPtr& valueResult, const LogicalTensorPtr& indexResult, TileInfo& resultTileInfo,
               int axis, int k, int isLargest)
{
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, k <= vecTile[axis])
        << "The k should less than or equal to" << vecTile[axis];
    if (static_cast<int>(cur) == axis) {
        auto source = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> vecTileAlign = vecTile.tile;
        if (source->shape[axis] > vecTileAlign[axis] * NUM_VALUE_2) {
            int64_t sourceShapeSize = 1;
            for (const auto& num : source->shape) {
                sourceShapeSize *= num;
            }
            int64_t tileShapeSize = sourceShapeSize / source->shape[axis] * vecTileAlign[axis];
            if (sourceShapeSize < maxNumValue) {
                vecTileAlign[axis] = source->shape[axis];
            } else if (tileShapeSize < maxNumValue) {
                vecTileAlign[axis] = std::max(
                    (int64_t)kBlockSize,
                    maxNumValue / (sourceShapeSize / source->shape[axis]) / kBlockSize * kBlockSize);
            }
        }
        vecTileAlign[axis] = (vecTileAlign[axis] + kBlockSize - 1) / kBlockSize * kBlockSize;
        auto axisTileNum = (source->shape[axis] + vecTileAlign[axis] - 1) / vecTileAlign[axis];
        auto axisBlockSizeAlign = vecTileAlign[axis];
        std::vector<int64_t> tileBitsortShape = source->shape;
        std::vector<int64_t> tileMrgsortShape = source->shape;
        std::vector<int64_t> tileSourceShape = source->shape;
        std::vector<int64_t> tileSourceOffset(tileSourceShape.size(), 0);
        std::vector<LogicalTensorPtr> sortList;
        for (int i = 0; i < input.tensor.GetShape()[axis]; i += vecTileAlign[axis]) {
            tileSourceShape[axis] = std::min(vecTileAlign[axis], source->shape[axis] - i);
            tileSourceOffset[axis] = i;
            auto inputTile = source->View(function, tileSourceShape, tileSourceOffset);
            auto tileBitsortRemain = (source->shape[axis] - i + kBlockSize - 1) / kBlockSize * kBlockSize;
            tileBitsortShape[axis] = std::min(axisBlockSizeAlign * kFactorSize, tileBitsortRemain * kFactorSize);
            auto bitsortTile = std::make_shared<LogicalTensor>(function, source->Datatype(), tileBitsortShape);
            std::vector<SymbolicScalar> bitSortDynValidShape(inputTile->GetDynValidShape());
            bitSortDynValidShape[axis] = bitSortDynValidShape[axis] * NUM2;
            bitsortTile->UpdateDynValidShape(bitSortDynValidShape);
            std::vector<int64_t> tmpShape;
            if (bitsortTile->shape.size() == 1) {
                tmpShape = {bitsortTile->shape[bitsortTile->shape.size() - 1]};
            } else {
                tmpShape = {1, bitsortTile->shape[bitsortTile->shape.size() - 1]};
            }
            auto bitsortTempTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), tmpShape);
            auto& bitsortOp = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {bitsortTile, bitsortTempTensor});
            bitsortOp.SetAttribute(TOPK_AXIS, axis);
            bitsortOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
            bitsortOp.SetAttribute(TOPK_OFFSET, static_cast<int>(i));

            int kValue = std::min(k, static_cast<int>(tileSourceShape[axis]));
            tileMrgsortShape[axis] = (kValue + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
            auto mrgsortTile = std::make_shared<LogicalTensor>(function, source->Datatype(), tileMrgsortShape);
            bitSortDynValidShape[axis] = std::min(bitSortDynValidShape[axis],
                                                  (k + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2);
            mrgsortTile->UpdateDynValidShape(bitSortDynValidShape);
            auto mrgsortTempTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), tmpShape);
            auto& mrgsortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitsortTile},
                                                    {mrgsortTile, mrgsortTempTensor});
            mrgsortOp.SetAttribute(TOPK_AXIS, axis);
            mrgsortOp.SetAttribute(TOPK_KVALUE, kValue);
            mrgsortOp.SetAttribute(TOPK_MERGE_SIZE, NUM_VALUE_32);
            sortList.push_back(mrgsortTile);
        }
        tileMrgsortShape[axis] = (k + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
        std::vector<int64_t> mrgsortResultOffset(tileMrgsortShape.size(), 0);
        std::vector<int64_t> tempShape = sortList[0]->shape;
        tempShape[axis] = NUM_VALUE_4 * tempShape[axis];
        for (size_t i = 0; i < tempShape.size() - 1; ++i) {
            tempShape[i] = 1;
        }
        tileMrgsortShape[axis] = tileMrgsortShape[axis] * axisTileNum;
        auto mrgsortBuffer = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tileMrgsortShape);
        std::vector<LogicalTensorPtr> tiledMrgsortList;
        for (int i = 0; i < axisTileNum; i += NUM_VALUE_4) {
            if ((axisTileNum - i) == NUM_VALUE_3) {
                auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tempShape);
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                mrgsortRepeatResult->UpdateDynValidShape(sortList[i]->GetDynValidShape());
                auto& mrgSortMultiQue = function.AddOperation(
                    Opcode::OP_TILEDMRGSORT,
                    {sortList[i], sortList[i + 1], sortList[i + NUM_VALUE_2], sortList[i + NUM_VALUE_2]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_3);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            } else if ((axisTileNum - i) == NUM_VALUE_2) {
                auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tempShape);
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                mrgsortRepeatResult->UpdateDynValidShape(sortList[i]->GetDynValidShape());
                auto& mrgSortMultiQue = function.AddOperation(
                    Opcode::OP_TILEDMRGSORT, {sortList[i], sortList[i + 1], sortList[i + 1], sortList[i + 1]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_2);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            } else if ((axisTileNum - i) == 1) {
                tiledMrgsortList.push_back(sortList[i]);
            } else {
                auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tempShape);
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                mrgsortRepeatResult->UpdateDynValidShape(sortList[i]->GetDynValidShape());
                auto& mrgSortMultiQue = function.AddOperation(
                    Opcode::OP_TILEDMRGSORT,
                    {sortList[i], sortList[i + 1], sortList[i + NUM_VALUE_2], sortList[i + NUM_VALUE_3]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_4);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            }
        }
        int roundNum = 0;
        int width = 1;
        while (width < axisTileNum) {
            width = width << NUM_VALUE_2;
            roundNum++;
        }
        int tileResultIdx = 0;
        for (int i = 1; i < roundNum; ++i) {
            int tileResultNum = tiledMrgsortList.size();
            for (int j = tileResultIdx; j < tileResultNum; j += NUM_VALUE_4) {
                if ((tileResultNum - j) == NUM_VALUE_3) {
                    auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tempShape);
                    mrgsortResultOffset[axis] = (tileResultNum + (j - tileResultIdx) / NUM_VALUE_4) *
                                                sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    mrgsortRepeatResult->UpdateDynValidShape(tiledMrgsortList[j]->GetDynValidShape());
                    auto& mrgSortMultiQue = function.AddOperation(
                        Opcode::OP_TILEDMRGSORT,
                        {tiledMrgsortList[j], tiledMrgsortList[j + 1], tiledMrgsortList[j + NUM_VALUE_2],
                         tiledMrgsortList[j + NUM_VALUE_2]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_3);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                } else if ((tileResultNum - j) == NUM_VALUE_2) {
                    auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tempShape);
                    mrgsortResultOffset[axis] = (tileResultNum + (j - tileResultIdx) / NUM_VALUE_4) *
                                                sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    mrgsortRepeatResult->UpdateDynValidShape(tiledMrgsortList[j]->GetDynValidShape());
                    auto& mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                                                                  {tiledMrgsortList[j], tiledMrgsortList[j + 1],
                                                                   tiledMrgsortList[j + 1], tiledMrgsortList[j + 1]},
                                                                  {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_2);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                } else if ((tileResultNum - j) == 1) {
                    tiledMrgsortList.push_back(tiledMrgsortList[j]);
                } else {
                    auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(), tempShape);
                    mrgsortResultOffset[axis] = (tileResultNum + (j - tileResultIdx) / NUM_VALUE_4) *
                                                sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    mrgsortRepeatResult->UpdateDynValidShape(tiledMrgsortList[j]->GetDynValidShape());
                    auto& mrgSortMultiQue = function.AddOperation(
                        Opcode::OP_TILEDMRGSORT,
                        {tiledMrgsortList[j], tiledMrgsortList[j + 1], tiledMrgsortList[j + NUM_VALUE_2],
                         tiledMrgsortList[j + NUM_VALUE_3]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_4);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                }
            }
            tileResultIdx = tileResultNum;
        }
        auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& valueOp = function.AddOperation(Opcode::OP_EXTRACT, {tiledMrgsortList.back()}, {valueTile});
        valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
        valueOp.SetAttribute(TOPK_KVALUE, k);
        valueOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));

        auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& indexOp = function.AddOperation(Opcode::OP_EXTRACT, {tiledMrgsortList.back()}, {indexTile});
        indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
        indexOp.SetAttribute(TOPK_KVALUE, k);
        indexOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }

    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(valueResult->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledTopK(function, tileShape, cur + 1, input, valueResult, indexResult, resultTileInfo, axis, k, isLargest);
    }
}

void TiledTopKRadixSelect(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                          const LogicalTensorPtr& valueResult, const LogicalTensorPtr& indexResult, int axis, int k,
                          int isLargest, int64_t ubSize)
{
    if (cur == input.tensor.GetShape().size() - 1) {
        auto lastDim = input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = lastDim;
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        input.tileInfo.shape[cur] = valueResult->shape[cur];
        auto valueTile = valueResult->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto indexTile = indexResult->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> tmpShape = {static_cast<int64_t>(AlignUp(lastDim, NUM_VALUE_128) * NUM_VALUE_26)};
        if (cur > 0) {
            tmpShape[0] *= input.tileInfo.shape[cur - 1];
        }
        auto tempTensor = std::make_shared<LogicalTensor>(function, DataType::DT_UINT8, tmpShape);
        auto& newOp = function.AddOperation(Opcode::OP_RADIX_SELECT, {inputTile}, {valueTile, indexTile, tempTensor});
        newOp.SetAttribute(TOPK_AXIS, axis);
        newOp.SetAttribute(TOPK_KVALUE, k);
        newOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTopKRadixSelect(function, tileShape, cur + 1, input, valueResult, indexResult, axis, k, isLargest, ubSize);
    }
}

void TiledTopK(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
               const LogicalTensorPtr valueResult, const LogicalTensorPtr indexResult, int axis, int k, int isLargest,
               TopKAlgo algo)
{
    // Build Init tile info
    TileInfo tileInfo(operand->shape, operand->offset);
    auto input = Input{operand, tileInfo};
    if (algo == TopKAlgo::MERGE_SORT) {
        TileInfo resultTileInfo(valueResult->shape, valueResult->offset);
        TiledTopK(function, tileShape, 0, input, valueResult, indexResult, resultTileInfo, axis, k, isLargest);
    } else if (algo == TopKAlgo::RADIX_SELECT) {
        auto size = operand->shape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape[size - 1] <= tileShape.GetVecTile()[size - 1])
            << "The tile_shape[-1] should greater than or equal to input.shape[-1]. "
            << "tile_shape[-1] = " << tileShape.GetVecTile()[size - 1]
            << ", input.shape[-1] = " << operand->shape[size - 1];
        int64_t ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
        TiledTopKRadixSelect(function, tileShape, 0, input, valueResult, indexResult, axis, k, isLargest, ubSize);
    }
}

void TiledArgSort(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                  const LogicalTensorPtr& resultDices, TileInfo& resultDicesTileInfo, int axis, int isLargest)
{
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultDicesTile = resultDices->View(function, resultDicesTileInfo.shape, resultDicesTileInfo.offset);
        function.AddOperation(Opcode::OP_ARGSORT, {inputTile}, {resultDicesTile});
        return;
    }
    if (cur == static_cast<size_t>(axis)) {
        input.tileInfo.offset[cur] = 0;
        input.tileInfo.shape[cur] = input.tensor.GetShape()[cur];
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultDicesTileInfo.offset[cur] = i;
        resultDicesTileInfo.shape[cur] = std::min(resultDices->shape[cur] - resultDicesTileInfo.offset[cur],
                                                  vecTile[cur]);
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
    }
}

void TiledArgSort(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                  const LogicalTensorPtr resDicesOperand, int axis, int isLargest)
{
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultDicesTileInfo(resDicesOperand->shape.size(), resDicesOperand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledArgSort(function, tileShape, 0, input, resDicesOperand, resultDicesTileInfo, axis, isLargest);
}

void TensorTopK(Function& function, const LogicalTensorPtr& self, LogicalTensorPtr& valueResult,
                LogicalTensorPtr& indexResult, int k, int axis, bool isLargest, TopKAlgo algo)
{
    if (!self->GetDynValidShape().empty()) {
        std::vector<SymbolicScalar> outValidShape;
        for (auto shape : self->GetDynValidShape()) {
            outValidShape.push_back(shape);
        }
        outValidShape[axis] = SymbolicScalar(k);
        valueResult->UpdateDynValidShape(outValidShape);
        indexResult->UpdateDynValidShape(outValidShape);
    }

    auto& op = function.AddOperation(Opcode::OP_TOPK, {self}, {valueResult, indexResult});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    op.SetAttribute(OpAttributeKey::topkAlgo, static_cast<int>(algo));
}

std::tuple<Tensor, Tensor> TopK(const Tensor& self, int k, int axis, bool isLargest, TopKAlgo algo)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "TopK");

    if (algo == TopKAlgo::MERGE_SORT) {
        std::unordered_set<DataType> supportedTypes = {DT_FP32};
        CheckTensorDataType(self.GetStorage(), supportedTypes, "TOPK(Merge Sort)");
    } else if (algo == TopKAlgo::RADIX_SELECT) {
        std::unordered_set<DataType> supportedTypes = {DT_BF16,   DT_FP16,  DT_FP32,   DT_UINT8, DT_INT8,
                                                       DT_UINT16, DT_INT16, DT_UINT32, DT_INT32};
        CheckTensorDataType(self.GetStorage(), supportedTypes, "TOPK(Radix Select)");
    }
    CheckTensorDimRange(self.GetStorage(), 1, 4, "TOPK");
    CheckTensorShapeSize(self.GetStorage(), "TOPK");
    const auto len = static_cast<int>(self.GetShape().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis == len - 1 || axis == -1) << "TopK only support last axis";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          algo != TopKAlgo::RADIX_SELECT || Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510)
        << "When TopK using radix select algo, only DAV_3510 architecture is supported.";
    axis = axis >= 0 ? axis : (axis + len);

    auto topkOutShape = self.GetShape();
    topkOutShape[axis] = k;
    auto indexResult = Tensor(DataType::DT_INT32, topkOutShape);
    auto valueResult = Tensor(self.GetStorage()->tensor->datatype, topkOutShape);
    CALL(TopK, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), valueResult.GetStorage(),
         indexResult.GetStorage(), k, axis, isLargest, algo);
    return std::tie(valueResult, indexResult);
}

bool checkIsExceedUB(const std::vector<int64_t>& tileShape, const std::vector<int64_t>& shape, int axis,
                     int blockSize = 32)
{
    int64_t UBSize = 196608;

    // check shape is out of UB size
    int64_t tileRowShapeSize = 1; // tileShape[0] * tileShape[1] * ... * rawShape[-1]
    for (const auto& num : tileShape) {
        tileRowShapeSize *= num;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileShape[axis] > 0) << "tileShape in axis must greater than 0.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, blockSize > 0) << "blockSize must greater than 0.";
    tileRowShapeSize = tileRowShapeSize / tileShape[axis] * ((shape[axis] + blockSize - 1) / blockSize * blockSize);
    int64_t maxShapeSize = tileRowShapeSize * 2 * 4 * 4; // every element is 8B
    bool isInGM = maxShapeSize >= UBSize ? true : false;
    return isInGM;
}

void TiledSort(Function& function, const TileShape& tileShape, size_t cur, Input& input,
               const LogicalTensorPtr& valueResult, const LogicalTensorPtr& indexResult, TileInfo& resultTileInfo,
               int axis, int descending)
{
    auto& vecTile = tileShape.GetVecTile();
    if (static_cast<int>(cur) == axis) {
        input.tileInfo.offset[axis] = 0;
        auto source = input.tensor.GetStorage()->View(
            function, input.tileInfo.shape, input.tileInfo.offset); // input.tensor是viewTensor, source是tileTensor
        bool isInGM = checkIsExceedUB(vecTile.tile, source->shape, axis, kBlockSize);

        // 行数据可以全部加载到UB上, 直接进行操作, 不进行tile切块
        if (!isInGM) {
            // 每32个元素进行排序
            std::vector<int64_t> bitSortOutputShape = source->shape;
            bitSortOutputShape[axis] = (bitSortOutputShape[axis] + kBlockSize - 1) / kBlockSize * kBlockSize;
            bitSortOutputShape[axis] = bitSortOutputShape[axis] * NUM_VALUE_2;
            auto bitSortOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(),
                                                                       bitSortOutputShape);
            std::vector<int64_t> tmpShape;
            if (bitSortOutputShape.size() == 1) {
                tmpShape = {bitSortOutputShape[axis]};
            } else {
                tmpShape = {1, bitSortOutputShape[axis]};
            }
            auto tempTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), tmpShape);
            auto& bitSortOp = function.AddOperation(Opcode::OP_BITSORT, {source}, {bitSortOutputTensor, tempTensor});
            bitSortOp.SetAttribute(SORT_AXIS, axis);
            bitSortOp.SetAttribute(SORT_ORDER, static_cast<int>(descending));
            bitSortOp.SetAttribute(SORT_OFFSET, static_cast<int>(0));
            std::vector<SymbolicScalar> bitSortDynValidShape(source->GetDynValidShape());
            bitSortDynValidShape[axis] = bitSortDynValidShape[axis] * NUM_VALUE_2;
            bitSortOutputTensor->UpdateDynValidShape(bitSortDynValidShape);
            if (bitSortOutputShape.size() == 1) {
                tempTensor->UpdateDynValidShape({bitSortDynValidShape[axis]});
            } else {
                tempTensor->UpdateDynValidShape({1, bitSortDynValidShape[axis]});
            }

            // 32个元素组成的block之间进行归并
            std::vector<int64_t> mrgSortOutputShape = source->shape;
            mrgSortOutputShape[axis] = (mrgSortOutputShape[axis] + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum *
                                       NUM_VALUE_2;
            auto mrgSortOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(),
                                                                       mrgSortOutputShape);
            auto& mrgSortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitSortOutputTensor},
                                                    {mrgSortOutputTensor, tempTensor});
            mrgSortOp.SetAttribute(TOPK_AXIS, axis);
            mrgSortOp.SetAttribute(TOPK_KVALUE, source->shape[axis]);
            mrgSortOp.SetAttribute(TOPK_MERGE_SIZE, NUM_VALUE_32);
            std::vector<SymbolicScalar> mrgSortDynValidShape(source->GetDynValidShape());
            mrgSortDynValidShape[axis] = source->GetDynValidShape()[axis] * NUM_VALUE_2;
            mrgSortOutputTensor->UpdateDynValidShape(mrgSortDynValidShape);
            if (bitSortOutputShape.size() == 1) {
                tempTensor->UpdateDynValidShape({bitSortDynValidShape[axis]});
            } else {
                tempTensor->UpdateDynValidShape({1, bitSortDynValidShape[axis]});
            }

            // 提取value和index
            auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
            auto& valueOp = function.AddOperation(Opcode::OP_EXTRACT, {mrgSortOutputTensor}, {valueTile});
            valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
            valueOp.SetAttribute(SORT_KVALUE, source->shape[axis]);
            valueOp.SetAttribute(SORT_ORDER, descending);
            valueTile->UpdateDynValidShape(source->GetDynValidShape());

            auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
            auto& indexOp = function.AddOperation(Opcode::OP_EXTRACT, {mrgSortOutputTensor}, {indexTile});
            indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
            indexOp.SetAttribute(SORT_KVALUE, source->shape[axis]);
            indexOp.SetAttribute(SORT_ORDER, descending);
            indexTile->UpdateDynValidShape(source->GetDynValidShape());
            return;
        }

        std::vector<int64_t> vecTileAlign = vecTile.tile; // tile shape after align axis
        vecTileAlign[axis] = (vecTileAlign[axis] + kBlockSize - 1) / kBlockSize * kBlockSize;

        std::vector<int64_t> tileSourceShape = source->shape;
        std::vector<int64_t> tileSourceOffset(tileSourceShape.size(), 0);
        std::vector<int64_t> tileBitSortShape = source->shape;

        // 创建一个2倍source的GM上的空间sortOutputTensor, 用于存储source排序后的结果
        std::vector<int64_t> sortOutputShape = source->shape;
        auto sortOutputValidShape = source->GetDynValidShape();
        // 元素个数k和8对齐，extract中二维的vreduce才能正常转换，因为UB中32B对齐，k*4B和32B对齐，则k与8对齐
        sortOutputShape[axis] = (sortOutputShape[axis] + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
        sortOutputValidShape[axis] = sortOutputValidShape[axis] * NUM_VALUE_2;
        auto sortOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), sortOutputShape,
                                                                sortOutputValidShape);
        std::vector<int64_t> tileOutputShape = sortOutputShape;
        std::vector<int64_t> tileOutputOffset(sortOutputShape.size(), 0);

        for (int64_t i = 0; i < input.tensor.GetShape()[axis]; i += vecTileAlign[axis]) {
            tileSourceShape[axis] = std::min(vecTileAlign[axis], source->shape[axis] - i);
            tileSourceOffset[axis] = i;
            auto inputTile = source->View(function, tileSourceShape, tileSourceOffset);
            tileBitSortShape[axis] = (tileSourceShape[axis] + kBlockSize - 1) / kBlockSize * kBlockSize * NUM_VALUE_2;
            auto bitSortTile = std::make_shared<LogicalTensor>(function, source->Datatype(), tileBitSortShape);
            std::vector<int64_t> tmpShape = {1, tileBitSortShape[axis]};
            if (tileBitSortShape.size() == 1) {
                tmpShape = {tileBitSortShape[axis]};
            } else {
                tmpShape = {1, tileBitSortShape[axis]};
            }
            auto tempTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), tmpShape);
            auto& bitSortOp = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {bitSortTile, tempTensor});
            bitSortOp.SetAttribute(SORT_AXIS, axis);
            bitSortOp.SetAttribute(SORT_ORDER, descending);
            bitSortOp.SetAttribute(SORT_OFFSET, i);
            std::vector<SymbolicScalar> bitSortDynValidShape(inputTile->GetDynValidShape());
            bitSortDynValidShape[axis] = bitSortDynValidShape[axis] * NUM_VALUE_2;
            bitSortTile->UpdateDynValidShape(bitSortDynValidShape);
            if (tileBitSortShape.size() == 1) {
                tempTensor->UpdateDynValidShape({bitSortDynValidShape[axis]});
            } else {
                tempTensor->UpdateDynValidShape({1, bitSortDynValidShape[axis]});
            }

            tileOutputShape[axis] = (tileSourceShape[axis] + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum *
                                    NUM_VALUE_2; // UB 32B对齐，兼顾了DynMrgSort中的k向8对齐
            tileOutputOffset[axis] = i * NUM_VALUE_2;
            auto tmp = std::make_shared<LogicalTensor>(function, source->Datatype(), tileOutputShape);
            auto& mrgSortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitSortTile}, {tmp, tempTensor});
            mrgSortOp.SetAttribute(SORT_AXIS, axis);
            mrgSortOp.SetAttribute(SORT_KVALUE, static_cast<int>(tileSourceShape[axis]));
            mrgSortOp.SetAttribute(TOPK_MERGE_SIZE, NUM_VALUE_32);
            std::vector<SymbolicScalar> mrgSortDynValidShape(inputTile->GetDynValidShape());
            mrgSortDynValidShape[axis] = mrgSortDynValidShape[axis] * NUM_VALUE_2;
            tmp->UpdateDynValidShape(mrgSortDynValidShape);
            if (tileBitSortShape.size() == 1) {
                tempTensor->UpdateDynValidShape({mrgSortDynValidShape[axis]});
            } else {
                tempTensor->UpdateDynValidShape({1, mrgSortDynValidShape[axis]});
            }

            auto& assembleOp = function.AddOperation(config::GetContractOpcode(), {tmp}, {sortOutputTensor});
            assembleOp.iOperand[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            assembleOp.oOperand[0]->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
                MemoryType::MEM_UB, tileOutputOffset,
                std::vector<SymbolicScalar>(tileOutputOffset.begin(), tileOutputOffset.end()),
                tmp->GetDynValidShape()));
        }

        vecTileAlign[axis] = vecTileAlign[axis] * NUM_VALUE_2;
        int64_t tileNum = (sortOutputShape[axis] + vecTileAlign[axis] - 1) / vecTileAlign[axis]; // 计算有多少Tile块

        int64_t roundNum = tileNum;
        std::queue<LogicalTensorPtr> q;
        q.push(sortOutputTensor);
        bool flag = true; // 判断当前是偶数还是奇数阶段
        for (int64_t round = 1; round <= roundNum; round++) {
            auto roundInputTensor = q.front();
            q.pop();
            unsigned firstShape = vecTileAlign[axis];
            auto roundOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), sortOutputShape,
                                                                     sortOutputValidShape);
            std::vector<SymbolicScalar> curValidShape = sortOutputValidShape;
            for (int64_t i = 0; i < sortOutputShape[axis];) {
                tileOutputOffset[axis] = i;
                if (i + vecTileAlign[axis] >= sortOutputShape[axis]) { // 尾块
                    tileOutputShape[axis] = sortOutputShape[axis] - i;
                } else if (!flag && i == 0) { // 奇数阶段的头块
                    tileOutputShape[axis] = vecTileAlign[axis];
                } else { // 两块
                    tileOutputShape[axis] = std::min(NUM_VALUE_2 * vecTileAlign[axis], sortOutputShape[axis] - i);
                }
                i += tileOutputShape[axis];

                auto src = std::make_shared<LogicalTensor>(function, source->Datatype(), tileOutputShape);
                auto& viewOp = function.AddOperation(config::GetSliceOpcode(), {roundInputTensor}, {src});
                curValidShape[axis] = std::max(
                    0, std::min(sortOutputValidShape[axis] - tileOutputOffset[axis], tileOutputShape[axis]));
                viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(
                    tileOutputOffset, MemoryType::MEM_UB,
                    std::vector<SymbolicScalar>(tileOutputOffset.begin(), tileOutputOffset.end()), curValidShape));
                src->UpdateDynValidShape(curValidShape);

                auto outputInUB = std::make_shared<LogicalTensor>(function, src->Datatype(), tileOutputShape);
                auto& twoTileMrgSortOp = function.AddOperation(Opcode::OP_TWOTILEMRGSORT, {src}, {outputInUB});
                twoTileMrgSortOp.SetAttribute(SORT_FIRSTSHAPE, static_cast<int>(firstShape));
                std::vector<SymbolicScalar> tileMrgSortDynValidShape(src->GetDynValidShape());
                outputInUB->UpdateDynValidShape(tileMrgSortDynValidShape);

                auto& assembleOp = function.AddOperation(config::GetContractOpcode(), {outputInUB},
                                                         {roundOutputTensor});
                assembleOp.iOperand[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
                assembleOp.oOperand[0]->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
                    MemoryType::MEM_UB, tileOutputOffset,
                    std::vector<SymbolicScalar>(tileOutputOffset.begin(), tileOutputOffset.end()),
                    outputInUB->GetDynValidShape()));
            }
            q.push(roundOutputTensor);
            flag = !flag;
        }

        auto extractInputTensor = q.front();
        q.pop();
        for (int i = 0; i < sortOutputShape[axis]; i += vecTileAlign[axis]) {
            tileOutputShape[axis] = std::min(vecTileAlign[axis], sortOutputShape[axis] - i);
            tileOutputOffset[axis] = i;
            auto src = extractInputTensor->View(function, tileOutputShape, tileOutputOffset);
            resultTileInfo.shape[axis] = std::min(tileOutputShape[axis] / NUM_VALUE_2,
                                                  source->shape[axis] - i / NUM_VALUE_2);
            resultTileInfo.offset[axis] = i / NUM_VALUE_2;

            auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
            auto& valueOp = function.AddOperation(Opcode::OP_EXTRACT_SINGLE, {src}, {valueTile});
            valueOp.SetAttribute(SORT_ORDER, descending);
            valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
            std::vector<SymbolicScalar> extractDynValidShape(src->GetDynValidShape());
            extractDynValidShape[axis] = extractDynValidShape[axis] / NUM_VALUE_2;
            valueTile->UpdateDynValidShape(extractDynValidShape);

            auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
            auto& indexOp = function.AddOperation(Opcode::OP_EXTRACT_SINGLE, {src}, {indexTile});
            indexOp.SetAttribute(SORT_ORDER, descending);
            indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
            indexTile->UpdateDynValidShape(extractDynValidShape);
        }
        return;
    }

    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(valueResult->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledSort(function, tileShape, cur + 1, input, valueResult, indexResult, resultTileInfo, axis, descending);
    }
}

void TiledSort(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
               const LogicalTensorPtr valueResult, const LogicalTensorPtr indexResult, int axis, int descending)
{
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(valueResult->shape, valueResult->offset);
    auto input = Input{operand, tileInfo};
    TiledSort(function, tileShape, 0, input, valueResult, indexResult, resultTileInfo, axis, descending);
}

void TensorSort(Function& function, const LogicalTensorPtr& self, LogicalTensorPtr& valueResult,
                LogicalTensorPtr& indexResult, int axis, bool descending)
{
    auto validShape = self->GetDynValidShape();
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SORT_UB, {self}, {valueResult, indexResult},
                                           {validShape, validShape});
    op.SetAttribute(SORT_AXIS, static_cast<int>(axis));
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    return;
}

std::tuple<Tensor, Tensor> sort(const Tensor& self, int axis = -1, bool descending = false)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "sort");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SORT");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SORT");
    CheckTensorShapeSize(self.GetStorage(), "SORT");
    auto len = static_cast<int>(self.GetShape().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, len >= 1 && len <= 4) << "Only support 1 dim to 4 dim.\n";

    axis = axis >= 0 ? axis : axis + len;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < len)
        << "Invalid axis value: " << axis << ". Expected range: [-" << len << "," << len - 1 << "]\n";

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, len != 4 || axis != 0) << "Sort not support the 0th axis of 4D input.\n";

    auto validShape = self.GetStorage()->GetDynValidShape();
    auto vecTileShape = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, vecTileShape[axis] % 32 == 0)
        << "The size of the tile shape along axis " << axis << " must be a multiple of 32. Got " << vecTileShape[axis]
        << ".\n";

    if (checkIsExceedUB(vecTileShape.tile, self.GetShape(), axis, 32)) {
        int64_t tileNum = (self.GetShape()[axis] + vecTileShape[axis] - 1) / vecTileShape[axis];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileNum < 128)
            << "For Large Shape in GM, the number of tile on sort axis must less than 128.";
    }

    auto transposeSelf = Transpose(self, {axis, len - 1});
    std::swap(validShape[axis], validShape[len - 1]);
    transposeSelf.GetStorage()->UpdateDynValidShape(validShape);
    std::swap(vecTileShape[axis], vecTileShape[len - 1]);
    TileShape::Current().SetVecTile(vecTileShape);

    auto castSelf = Cast(transposeSelf, DataType::DT_FP32, CastMode::CAST_NONE);
    castSelf.GetStorage()->UpdateDynValidShape(transposeSelf.GetStorage()->GetDynValidShape());

    auto outShape = castSelf.GetShape();
    auto valueResult = Tensor(DataType::DT_FP32, outShape);
    auto indexResult = Tensor(DataType::DT_INT32, outShape);
    CALL(Sort, *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), valueResult.GetStorage(),
         indexResult.GetStorage(), len - 1, descending);

    auto castValueResult = Cast(valueResult, self.GetDataType(), CastMode::CAST_NONE);
    castValueResult.GetStorage()->UpdateDynValidShape(valueResult.GetStorage()->GetDynValidShape());

    TileShape::Current().SetVecTile(vecTileShape);
    auto transposeValueResult = Transpose(castValueResult, {axis, len - 1});
    auto transposeIndexResult = Transpose(indexResult, {axis, len - 1});
    std::swap(validShape[axis], validShape[len - 1]);
    transposeValueResult.GetStorage()->UpdateDynValidShape(validShape);
    transposeIndexResult.GetStorage()->UpdateDynValidShape(validShape);
    std::swap(vecTileShape[axis], vecTileShape[len - 1]);
    TileShape::Current().SetVecTile(vecTileShape);
    return std::tie(transposeValueResult, transposeIndexResult);
}

Tensor ArgSort(const Tensor& self, int axis, bool descending) { return std::get<1>(sort(self, axis, descending)); }

void BitSortOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute(TOPK_AXIS);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    int idxStart = op.GetIntAttribute(TOPK_OFFSET);
    TiledBitSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest, idxStart);
}

void MrgSortOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute(TOPK_AXIS);
    int kValue = op.GetIntAttribute(TOPK_KVALUE);
    int mergeSize = op.GetIntAttribute(TOPK_MERGE_SIZE);
    TiledMrgSort(function, tileShape, iOperand[0], oOperand[0], axis, kValue, mergeSize);
}

void ArgSortOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute("axis");
    int isLargest = op.GetIntAttribute("order");
    TiledArgSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest);
}

void ExtractOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int maskMode = op.GetIntAttribute(EXTRACT_MASKMODE);
    int kValue = op.GetIntAttribute(TOPK_KVALUE);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    TiledExtract(function, tileShape, iOperand[0], oOperand[0], maskMode, kValue, isLargest);
}

void TopkOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute(TOPK_AXIS);
    int kValue = op.GetIntAttribute(TOPK_KVALUE);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    TopKAlgo algo = static_cast<TopKAlgo>(op.GetIntAttribute(OpAttributeKey::topkAlgo));
    TiledTopK(function, tileShape, iOperand[0], oOperand[0], oOperand[1], axis, kValue, isLargest, algo);
}

void SortOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute(SORT_AXIS);
    int descending = op.GetIntAttribute(SORT_ORDER);
    TiledSort(function, tileShape, iOperand[0], oOperand[0], oOperand[1], axis, descending);
}

REGISTER_OPERATION_TILED_FUNC(OP_BITSORT, Opcode::OP_BITSORT, BitSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_MRGSORT, Opcode::OP_MRGSORT, MrgSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ARGSORT, Opcode::OP_ARGSORT, ArgSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXTRACT, Opcode::OP_EXTRACT, ExtractOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TOPK, Opcode::OP_TOPK, TopkOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SORT_UB, Opcode::OP_SORT_UB, SortOperationTileFunc);

} // namespace npu::tile_fwk
