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
 * \file reduction.cpp
 * \brief
 */

#include "unary.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/tilefwk_op.h"
#include "tensor_transformation.h"

namespace npu::tile_fwk {

enum class ReduceType {
    NORMAL,
    EXPAND,
    SINGLE,
};

void TileReduceNew(Function& function, const TileShape& tileShape, const std::string& op, ReduceType reduceType,
                   const LogicalTensorPtr& in, const LogicalTensorPtr& result, int axis = -1)
{
    axis = axis < 0 ? in->shape.size() + axis : axis;
    std::vector<int64_t> tileAshape = in->shape;
    std::vector<int64_t> tileBshape = in->shape;
    std::vector<int64_t> regShape = in->shape;
    std::vector<int64_t> regAshape = in->shape;
    std::vector<int64_t> regBshape = in->shape;
    std::vector<int64_t> regOffset(regShape.size(), 0);
    std::vector<int64_t> tileAoffset(regShape.size(), 0);
    std::vector<int64_t> tileBoffset(regShape.size(), 0);

    std::vector<int64_t> remainderShape = in->shape;
    std::vector<int64_t> remainderOffset(remainderShape.size(), 0);

    auto opNew = op;
    if (opNew == "MAX_COMBINE_AXIS") {
        opNew = "MAX";
    }
    if (opNew == "SUM_COMBINE_AXIS") {
        opNew = "SUM";
    }

    auto source = in;

    auto vecTile = tileShape.GetVecTile();
    int64_t width = (source->shape[axis] + vecTile[axis] - 1) / vecTile[axis] * vecTile[axis]; // 向上对齐
    int padSize = width - source->shape[axis];
    int remainder = 0;

    int p2width = vecTile[axis];
    while (width >= p2width) {
        p2width = p2width << 1;
    }
    p2width = p2width >> 1;

    remainder = width - p2width;
    remainderShape[axis] = remainder;
    remainderOffset[axis] = p2width;

    width = p2width;

    while (width >= NUM2 * vecTile[axis]) // hierarchically pair wise reduce to a
    // single TILE_SHAPE1
    {
        width = width >> 1;

        tileAshape[axis] = width;
        tileBshape[axis] = std::min(width, source->shape[axis] - width); // 带tail的部分
        tileBoffset[axis] = width;

        auto tileA = source->View(function, tileAshape, tileAoffset);
        auto tileB = source->View(function, tileBshape, tileBoffset);

        auto resultA = std::make_shared<LogicalTensor>(
            function, in->Datatype(), reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->shape : source->shape,
            reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->GetDynValidShape() : source->GetDynValidShape());
        for (int j = 0; j < width; j += vecTile[axis]) {
            regAshape[axis] = vecTile[axis];
            regBshape[axis] = std::min(vecTile[axis], tileB->shape[axis] - j); // 带tail的部分
            regOffset[axis] = j;

            auto regA = tileA->View(function, regAshape, regOffset);
            auto regB = tileB->View(function, regBshape, regOffset);
            auto regResult = resultA->View(function, regAshape, regOffset);
            function.AddOperation("TILE_PAIR" + opNew, {regA, regB}, {regResult});
        }

        if (remainder < width) {
            source = resultA;
            continue;
        }

        if ((remainderShape[axis] + remainderOffset[axis] > in->shape[axis])) {
            remainderShape[axis] = remainderShape[axis] - padSize;
        }

        auto tileRemainder = in->View(function, remainderShape, remainderOffset);
        auto resultAnext = std::make_shared<LogicalTensor>(function, in->Datatype(), resultA->shape,
                                                           resultA->GetDynValidShape());
        for (int j = 0; j < width; j += vecTile[axis]) {
            regAshape[axis] = vecTile[axis];
            regBshape[axis] = std::min(vecTile[axis], tileRemainder->shape[axis] - j); // 带tail的部分
            regOffset[axis] = j;

            auto regA = resultA->View(function, regAshape, regOffset);
            auto regB = tileRemainder->View(function, regBshape, regOffset);
            auto regResult = resultAnext->View(function, regAshape, regOffset);
            function.AddOperation("TILE_PAIR" + opNew, {regA, regB}, {regResult});
        }
        remainder -= width;
        remainderOffset[axis] += width;
        remainderShape[axis] -= width;

        source = resultAnext;
    }

    // reduce to a single TILE_SHAPE1
    regShape[axis] = std::min(in->shape[axis], vecTile[axis]);
    regOffset[axis] = 0;

    auto temp = std::make_shared<LogicalTensor>(
        function, in->Datatype(), reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->shape : source->shape,
        reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->GetDynValidShape() : source->GetDynValidShape());
    auto sourceReg = source->View(function, regShape, regOffset);
    switch (reduceType) {
        case npu::tile_fwk::ReduceType::NORMAL: {
            auto resultReg = result->View(function, regShape, regOffset);
            // now the max is in resultReg
            function.AddOperation("TILE_ROW" + op, {sourceReg}, {resultReg});
            break;
        }
        case npu::tile_fwk::ReduceType::EXPAND: {
            auto resultReg = temp->View(function, regShape, regOffset);
            // now the max is in resultReg
            function.AddOperation("TILE_ROWEXP" + op, {sourceReg}, {resultReg});

            auto resultReg1 = temp->View(function, regShape, regOffset);

            for (int j = 0; j < result->shape[1]; j += vecTile[1]) // duplicate to fill result tensor
            {
                regShape[0] = in->shape[0];
                regShape[1] = vecTile[1];

                regOffset[0] = 0;
                regOffset[1] = j;

                resultReg = result->View(function, regShape, regOffset);
                function.AddOperation("TILE_REGISTER_COPY", {resultReg1}, {resultReg});
            }
            break;
        }
        case npu::tile_fwk::ReduceType::SINGLE: {
            std::vector<int64_t> tmpShape = {1, static_cast<int>(BLOCK_SIZE / BytesOf(in->Datatype()))};
            if (op == "SUM" || (static_cast<size_t>(axis) == (in->shape.size() - 1))) {
                if (static_cast<size_t>(axis) == (in->shape.size() - 1)) {
                    tmpShape[0] = sourceReg->shape[std::max(0, axis - 1)];
                    if (static_cast<size_t>(sourceReg->shape[axis]) <= REPEAT_BYTE / BytesOf(in->Datatype())) {
                        tmpShape[0] = 1;
                    } else if (static_cast<size_t>(sourceReg->shape[axis]) <=
                               NUM2 * REPEAT_BYTE / BytesOf(in->Datatype())) {
                        tmpShape[1] = REPEAT_BYTE / BytesOf(in->Datatype());
                    } else {
                        tmpShape[1] = (((sourceReg->shape[axis] * BytesOf(in->Datatype())) / REPEAT_BYTE) / NUM2) *
                                      REPEAT_BYTE / BytesOf(in->Datatype());
                    }
                    if (in->shape.size() == 1) {
                        tmpShape = {tmpShape[1]};
                    }
                    auto tempTensor = std::make_shared<LogicalTensor>(function, in->Datatype(), tmpShape);
                    tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
                    auto& newOp = function.AddOperation("TILE_ROW" + op + "_SINGLE", {sourceReg}, {result, tempTensor});
                    newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
                } else {
                    tmpShape[0] = (sourceReg->shape[axis] + 1) / NUM2;
                    tmpShape[1] = (sourceReg->shape[in->shape.size() - 1] + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM;
                    auto tempTensor = std::make_shared<LogicalTensor>(function, in->Datatype(), tmpShape);
                    tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
                    auto& newOp = function.AddOperation("TILE_ROW" + op + "LINE", {sourceReg}, {result, tempTensor});
                    newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
                }
            } else {
                auto& newOp = function.AddOperation("TILE_ROW" + op + "LINE", {sourceReg}, {result});
                newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
            }
            break;
        }
        default:
            break;
    }
}

void TileArgReduce(Function& function, const TileShape& tileShape, const std::string& op, const LogicalTensorPtr& in,
                   const LogicalTensorPtr& result, int axis = -1)
{
    auto indexDtype = in->Datatype() == DataType::DT_FP32 ? DataType::DT_INT32 : DataType::DT_INT16;
    axis = axis < 0 ? in->shape.size() + axis : axis;
    auto& vecTile = tileShape.GetVecTile();

    std::vector<int64_t> tileValueShape = result->shape;
    std::vector<int64_t> tileSourceShape = in->shape;
    std::vector<int64_t> tileSourceOffset(tileSourceShape.size(), 0);
    std::vector<LogicalTensorPtr> valueList;
    std::vector<LogicalTensorPtr> indexList;
    std::vector<SymbolicScalar> inValidShape = in->GetDynValidShape();
    std::vector<SymbolicScalar> argReudceValidShape;
    if (!inValidShape.empty()) {
        argReudceValidShape = inValidShape;
        argReudceValidShape[axis] = SymbolicScalar(1);
    }
    for (int i = 0; i < in->shape[axis]; i += vecTile[axis]) {
        tileSourceShape[axis] = std::min(vecTile[axis], in->shape[axis] - i);
        tileSourceOffset[axis] = i;
        auto inputTile = in->View(function, tileSourceShape, tileSourceOffset);
        auto valueTile = std::make_shared<LogicalTensor>(function, in->Datatype(), tileValueShape, argReudceValidShape);
        auto indexTile = std::make_shared<LogicalTensor>(function, indexDtype, tileValueShape, argReudceValidShape);
        std::vector<int64_t> tmpShape;
        if (inputTile->shape.size() == 1) {
            tmpShape = {tileSourceShape[axis]};
        } else {
            tmpShape = (static_cast<size_t>(axis) == (inputTile->shape.size() - 1)) ?
                           std::vector<int64_t>{tileSourceShape[axis - 1], tileSourceShape[axis]} :
                           std::vector<int64_t>{1, static_cast<int64_t>(REPEAT_BYTE / BytesOf(in->Datatype()) * NUM3)};
        }
        auto argreduceTempTensor = std::make_shared<LogicalTensor>(function, in->Datatype(), tmpShape,
                                                                   argReudceValidShape);
        if (static_cast<size_t>(axis) == (inputTile->shape.size() - 1)) {
            auto& argreduceOp = function.AddOperation("TILE_ROW" + op + "WITHVALUE_SINGLE", {inputTile},
                                                      {valueTile, indexTile, argreduceTempTensor});
            argreduceOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
        } else {
            auto& argreduceOp = function.AddOperation("TILE_ROW" + op + "WITHVALUE_LINE", {inputTile},
                                                      {valueTile, indexTile, argreduceTempTensor});
            argreduceOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
        }
        auto addsTile = std::make_shared<LogicalTensor>(function, indexDtype, tileValueShape, argReudceValidShape);
        auto& indexUpdateOp = function.AddOperation(Opcode::OP_ADDS, {indexTile}, {addsTile});
        indexUpdateOp.SetAttribute(OpAttributeKey::scalar, Element(indexDtype, i));
        indexUpdateOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", false);
        valueList.push_back(valueTile);
        indexList.push_back(addsTile);
    }

    auto axisTileNum = (in->shape[axis] + vecTile[axis] - 1) / vecTile[axis];
    std::vector<LogicalTensorPtr> workValueList = valueList;
    std::vector<LogicalTensorPtr> workIndexList = indexList;
    int currentSize = axisTileNum;
    while (currentSize > 1) {
        int halfSize = (currentSize + 1) / NUM_VALUE_2;
        std::vector<LogicalTensorPtr> nextValueList;
        std::vector<LogicalTensorPtr> nextIndexList;
        for (int i = 0; i < currentSize; i += NUM_VALUE_2) {
            if ((currentSize - i) == 1) {
                nextValueList.push_back(workValueList[i]);
                nextIndexList.push_back(workIndexList[i]);
            } else {
                auto pairValue = std::make_shared<LogicalTensor>(function, in->Datatype(), workValueList[0]->shape,
                                                                 workValueList[0]->GetDynValidShape());
                auto pairIndex = std::make_shared<LogicalTensor>(function, indexDtype, workIndexList[0]->shape,
                                                                 workIndexList[0]->GetDynValidShape());
                auto pairOpcode = op == "ARGMAX" ? Opcode::OP_PAIRARGMAX : Opcode::OP_PAIRARGMIN;
                function.AddOperation(pairOpcode,
                                      {workValueList[i], workIndexList[i], workValueList[i + 1], workIndexList[i + 1]},
                                      {pairValue, pairIndex});
                nextValueList.push_back(pairValue);
                nextIndexList.push_back(pairIndex);
            }
        }
        workValueList = std::move(nextValueList);
        workIndexList = std::move(nextIndexList);
        currentSize = halfSize;
    }
    if (indexDtype == DataType::DT_INT16 && result->Datatype() == DataType::DT_INT32) {
        auto fp32Index = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, result->shape,
                                                         workIndexList[0]->GetDynValidShape());
        auto& castToFp32 = function.AddOperation(Opcode::OP_CAST, {workIndexList[0]}, {fp32Index});
        castToFp32.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        castToFp32.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(SaturationMode::OFF));
        auto& castOp = function.AddOperation(Opcode::OP_CAST, {fp32Index}, {result});
        castOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        castOp.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(SaturationMode::OFF));
    } else {
        auto& indexUpdateOp = function.AddOperation(Opcode::OP_ADDS, {workIndexList[0]}, {result});
        indexUpdateOp.SetAttribute(OpAttributeKey::scalar, Element(indexDtype, 0));
        indexUpdateOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", false);
    }
    return;
}

void ReduceSingle(size_t cur, const std::string& op, Input& input, const LogicalTensorPtr result,
                  TileInfo& resultTileInfo, int axis, Function& function, const TileShape& tileShape,
                  std::vector<int> order)
{
    if (order[cur] == axis && cur < order.size() - 1) {
        std::swap(order[cur], order[cur + 1]);
    }
    if (order[cur] == axis) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (op == "ARGMAX" || op == "ARGMIN") {
            TileArgReduce(function, tileShape, op, inputTile, resultTile, axis);
        } else {
            TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::SINGLE, inputTile, resultTile, axis);
        }
        return;
    }
    auto vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[order[cur]]; i += vecTile[order[cur]]) {
        resultTileInfo.offset[order[cur]] = i;
        resultTileInfo.shape[order[cur]] = std::min(result->shape[order[cur]] - resultTileInfo.offset[order[cur]],
                                                    vecTile[order[cur]]);
        input.tileInfo.offset[order[cur]] = i % input.tensor.GetShape()[order[cur]];
        input.tileInfo.shape[order[cur]] = std::min(
            input.tensor.GetShape()[order[cur]] - input.tileInfo.offset[order[cur]], vecTile[order[cur]]);
        ReduceSingle(cur + 1, op, input, result, resultTileInfo, axis, function, tileShape, order);
    }
}

void TiledReduceSingle(Function& function, const TileShape& tileShape, const std::string& op,
                       const LogicalTensorPtr& operand, const LogicalTensorPtr& result, int axis)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op == "MAX" || op == "MIN" || op == "SUM" || op == "PROD" ||
                                                  op == "ARGMAX" || op == "ARGMIN" || op == "MAX_COMBINE_AXIS" ||
                                                  op == "SUM_COMBINE_AXIS")
        << "Not support op:" << op;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset should be equal";

    if (axis < 0) {
        axis = operand->shape.size() + axis;
    }

    // for loops before reduce axis
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(result->shape, result->offset);
    auto input = Input{operand, tileInfo};
    std::vector<int> defaultAxisOrder;
    for (size_t i = 0; i < operand->shape.size(); i++) {
        defaultAxisOrder.push_back(i);
    }
    ReduceSingle(0, op, input, result, resultTileInfo, axis, function, tileShape, defaultAxisOrder);
}

[[maybe_unused]] void TensorReduceSingle(Function& function, const std::string& op, const Tensor& operand,
                                         Tensor& result, int axis)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op == "MAX" || op == "MIN" || op == "SUM" || op == "PROD" ||
                                                  op == "ARGMAX" || op == "ARGMIN" || op == "MAX_COMBINE_AXIS" ||
                                                  op == "SUM_COMBINE_AXIS")
        << "Not support op:" << op;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand.GetShape().size() == operand.GetStorage()->offset.size())
        << "The shape size of operand and offset should be equal";
    auto opCode = Opcode::OP_ROWMAX_SINGLE;
    if (op == "MAX") {
        opCode = Opcode::OP_ROWMAX_SINGLE;
    } else if (op == "MIN") {
        opCode = Opcode::OP_ROWMIN_SINGLE;
    } else if (op == "SUM") {
        opCode = Opcode::OP_ROWSUM_SINGLE;
    } else if (op == "PROD") {
        opCode = Opcode::OP_ROWPROD_SINGLE;
    } else if (op == "ARGMAX") {
        opCode = Opcode::OP_ROWARGMAX_SINGLE;
    } else if (op == "ARGMIN") {
        opCode = Opcode::OP_ROWARGMIN_SINGLE;
    }

    if (!operand.GetStorage()->GetDynValidShape().empty()) {
        std::vector<SymbolicScalar> outValidShape;
        for (auto shape : operand.GetStorage()->GetDynValidShape()) {
            outValidShape.push_back(shape);
        }
        outValidShape[axis] = SymbolicScalar(1);
        result.GetStorage()->UpdateDynValidShape(outValidShape);
    }

    auto& newOp = function.AddOperation(opCode, {operand.GetStorage()}, {result.GetStorage()});
    newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int>(axis));
    return;
}

[[maybe_unused]] Tensor ReduceSingle(const std::string& op, const Tensor& operand)
{
    Tensor result(operand.GetStorage()->tensor->datatype, {operand.GetShape()[0], 1});
    Program::GetInstance().AddOperation("REDUCE_" + op + "_SINGLE", {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

static int NormalizeAndValidateReductionAxis(const Tensor& self, int axis)
{
    CheckAxisRange(self, axis);
    auto vecTile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile.valid()) << "TileShape is no set for reduce op";
    return axis;
}

static Tensor ProcessResultShape(const Tensor& result, const Tensor& self, int axis, bool keepDim)
{
    const int lastDim = self.GetShape().size() - 1;
    if (keepDim || lastDim == 0) {
        return result;
    } else {
        std::vector<SymbolicScalar> outValidShape;
        for (auto shape : self.GetStorage()->GetDynValidShape()) {
            outValidShape.push_back(shape);
        }

        auto outShape = result.GetShape();
        outShape.erase(outShape.begin() + axis);
        outValidShape.erase(outValidShape.begin() + axis);

        return Reshape(result, outShape, outValidShape);
    }
}

Tensor Amax(const Tensor& self, int axis, bool keepDim)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Amax");

    std::unordered_set<DataType> supportedTypes;
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8};
    } else {
        supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    }
    CheckTensorDataType(self.GetStorage(), supportedTypes, "AMAX");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "AMAX");
    CheckTensorShapeSize(self.GetStorage(), "AMAX");
    axis = NormalizeAndValidateReductionAxis(self, axis);

    auto resultShape = self.GetShape();
    resultShape[axis] = 1;

    Tensor result(self.GetStorage()->Datatype(), resultShape);
    CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MAX", self, result, axis);

    return ProcessResultShape(result, self, axis, keepDim);
}

Tensor ArgMax(const Tensor& self, int axis, bool keepDim)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ArgMax");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ARGMAX");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ARGMAX");
    CheckTensorShapeSize(self.GetStorage(), "ARGMAX");
    axis = NormalizeAndValidateReductionAxis(self, axis);

    auto resultShape = self.GetShape();
    auto vecTile = TileShape::Current().GetVecTile();
    resultShape[axis] = 1;

    Tensor result(DataType::DT_INT32, resultShape);
    if (self.GetDataType() == DT_FP16) {
        auto castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                             self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "ARGMAX", castSelf, result, axis);
    } else {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "ARGMAX", self, result, axis);
    }

    return ProcessResultShape(result, self, axis, keepDim);
}

Tensor ArgMin(const Tensor& self, int axis, bool keepDim)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ArgMin");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ARGMIN");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ARGMIN");
    CheckTensorShapeSize(self.GetStorage(), "ARGMIN");
    axis = NormalizeAndValidateReductionAxis(self, axis);

    auto resultShape = self.GetShape();
    auto vecTile = TileShape::Current().GetVecTile();
    resultShape[axis] = 1;

    Tensor result(DataType::DT_INT32, resultShape);
    if (self.GetDataType() == DT_FP16) {
        auto castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                             self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "ARGMIN", castSelf, result, axis);
    } else {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "ARGMIN", self, result, axis);
    }

    return ProcessResultShape(result, self, axis, keepDim);
}

Tensor Amin(const Tensor& self, int axis, bool keepDim)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Amin");

    std::unordered_set<DataType> supportedTypes;
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8};
    } else {
        supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    }
    CheckTensorDataType(self.GetStorage(), supportedTypes, "AMIN");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "AMIN");
    CheckTensorShapeSize(self.GetStorage(), "AMIN");
    axis = NormalizeAndValidateReductionAxis(self, axis);

    auto resultShape = self.GetShape();
    resultShape[axis] = 1;

    Tensor result(self.GetStorage()->Datatype(), resultShape);
    CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MIN", self, result, axis);

    return ProcessResultShape(result, self, axis, keepDim);
}

Tensor Sum(const Tensor& self, int axis, bool keepDim)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sum");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SUM");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SUM");
    CheckTensorShapeSize(self.GetStorage(), "SUM");
    axis = NormalizeAndValidateReductionAxis(self, axis);

    auto resultShape = self.GetShape();
    resultShape[axis] = 1;

    Tensor result(self.GetStorage()->Datatype(), resultShape);
    CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "SUM", self, result, axis);

    return ProcessResultShape(result, self, axis, keepDim);
}

Tensor Prod(const Tensor& self, int axis, bool keepDim)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Prod");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "PROD");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "PROD");
    CheckTensorShapeSize(self.GetStorage(), "PROD");

    axis = NormalizeAndValidateReductionAxis(self, axis);

    auto resultShape = self.GetShape();
    resultShape[axis] = 1;

    Tensor result(self.GetStorage()->Datatype(), resultShape);
    CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "PROD", self, result, axis);

    return ProcessResultShape(result, self, axis, keepDim);
}

void TiledReduceExpand(Function& function, const TileShape& tileShape, const std::string& op,
                       const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op == "MAX" || op == "SUM") << "Not support op:" << op;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset should be equal";

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unsupported dimension";
    }

    auto& vecTile = tileShape.GetVecTile();
    TileInfo tileInfo({vecTile[0], operand->shape[1]}, std::vector<int64_t>(operand->offset.size()));

    for (int i = 0; i < operand->shape[0]; i += vecTile[0]) {
        tileInfo.offset[0] = i;
        auto inputTile = operand->View(function, tileInfo.shape, tileInfo.offset);
        auto resultTile = result->View(function, tileInfo.shape, tileInfo.offset);
        TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::EXPAND, inputTile, resultTile);
    }
}

[[maybe_unused]] void TensorReduceExpand(Function& function, const std::string& op, const LogicalTensorPtr& operand,
                                         const LogicalTensorPtr& result)
{
    function.AddOperation(op == "MAX" ? Opcode::OP_ROWEXPMAX : Opcode::OP_ROWEXPSUM, {operand}, {result});
}

void TensorReduceExpand(Function& function, const std::string& op, const Tensor& operand, const Tensor& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op == "MAX" || op == "SUM") << "Not support op:" << op;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand.GetShape().size() == operand.GetStorage()->offset.size())
        << "The shape size of operand and offset must be equal";
    function.AddOperation(op == "MAX" ? Opcode::OP_ROWEXPMAX : Opcode::OP_ROWEXPSUM, {operand.GetStorage()},
                          {result.GetStorage()});
    return;
}

[[maybe_unused]] Tensor ReduceExpand(const std::string& op, const Tensor& operand)
{
    Tensor result(operand.GetStorage()->tensor->datatype, operand.GetShape());
    Program::GetInstance().AddOperation("ROW_" + op + "_EXPAND", {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

void TiledReduceExpandNew(Function& function, const TileShape& tileShape, const std::string& op,
                          const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unsupported dimension";
    }
    auto& vecTile = tileShape.GetVecTile();
    TileInfo tileInfo({vecTile[0], operand->shape[1]}, std::vector<int64_t>(operand->offset.size()));

    for (int i = 0; i < operand->shape[0]; i += vecTile[0]) {
        tileInfo.offset[0] = i;
        auto inputTile = operand->View(function, tileInfo.shape, tileInfo.offset);
        auto resultTile = result->View(function, tileInfo.shape, tileInfo.offset);
        TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::EXPAND, inputTile, resultTile);
    }
}

Tensor RowSumExpand(const Tensor& operand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "RowSumExpand");

    Tensor result(operand.GetStorage()->Datatype(), operand.GetShape());
    CALL(ReduceExpand, *Program::GetInstance().GetCurrentFunction(), "SUM", operand, result);
    return result;
}

Tensor RowMaxExpand(const Tensor& operand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "RowMaxExpand");

    Tensor result(operand.GetStorage()->Datatype(), operand.GetShape());
    CALL(ReduceExpand, *Program::GetInstance().GetCurrentFunction(), "MAX", operand, result);
    return result;
}

void RowMaxSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "MAX", iOperand[0], oOperand[0], axis);
}

void RowMinSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "MIN", iOperand[0], oOperand[0], axis);
}

void RowSumSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "SUM", iOperand[0], oOperand[0], axis);
}

void RowProdSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                    const std::vector<LogicalTensorPtr>& iOperand,
                                    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "PROD", iOperand[0], oOperand[0], axis);
}

void RowExpMaxSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand,
                                      [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    TiledReduceExpand(function, tileShape, "MAX", iOperand[0], oOperand[0]);
}

void RowExpSumSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand,
                                      [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    TiledReduceExpand(function, tileShape, "SUM", iOperand[0], oOperand[0]);
}

void RowArgMaxSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "ARGMAX", iOperand[0], oOperand[0], axis);
}

void RowArgMinSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "ARGMIN", iOperand[0], oOperand[0], axis);
}

REGISTER_OPERATION_TILED_FUNC(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, RowMaxSingleOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROWMIN_SINGLE, Opcode::OP_ROWMIN_SINGLE, RowMinSingleOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, RowSumSingleOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROWPROD_SINGLE, Opcode::OP_ROWPROD_SINGLE, RowProdSingleOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROWARGMAX_SINGLE, Opcode::OP_ROWARGMAX_SINGLE, RowArgMaxSingleOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROWARGMIN_SINGLE, Opcode::OP_ROWARGMIN_SINGLE, RowArgMinSingleOperationTileFunc);

REGISTER_OPERATION_TILED_FUNC(OP_ROWEXPMAX, Opcode::OP_ROWEXPMAX, RowExpMaxSingleOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROWEXPSUM, Opcode::OP_ROWEXPSUM, RowExpSumSingleOperationTileFunc);

} // namespace npu::tile_fwk
