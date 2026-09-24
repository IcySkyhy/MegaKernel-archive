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
 * \file where.cpp
 * \brief
 */

#include "binary.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "passes/pass_utils/graph_utils.h"
#include "tensor_transformation.h"

namespace npu::tile_fwk {

template <typename U, typename W>
void TiledWhereOperation(Function& function, const TileShape& tileShape, size_t cur, Input& condition, U& input,
                         W& other, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == result->shape.size()) {
        auto inputDatatype = DT_FP32;
        if constexpr (std::is_same_v<U, Input>) {
            inputDatatype = input.tensor.GetDataType();
        } else if constexpr (std::is_same_v<U, const Element>) {
            inputDatatype = input.GetDataType();
        }
        DataType selectDtype;
        if (inputDatatype == DT_FP32 || inputDatatype == DT_BF16) {
            selectDtype = DT_FP32;
        } else {
            selectDtype = DT_FP16;
        }
        const size_t ALIGN_SIZE = 32;
        int64_t castConditionTensorSize = 1024;
        int64_t compareConditionTensorSize = 1024;
        int64_t vcmpBitResultTensorSize = 128;
        int64_t startAddrUBTensorSize = 1;
        int64_t inputTempTensorSize = 1024;
        int64_t otherTempTensorSize = 1024;
        int64_t tempByteSize = (castConditionTensorSize + compareConditionTensorSize) * BytesOf(DT_FP16) +
                               vcmpBitResultTensorSize * BytesOf(DT_UINT8) +
                               ((startAddrUBTensorSize * BytesOf(DT_UINT64) + ALIGN_SIZE - 1) / ALIGN_SIZE) *
                                   ALIGN_SIZE +
                               (inputTempTensorSize + otherTempTensorSize) * BytesOf(selectDtype);

        auto conditionTile = condition.tensor.GetStorage()->View(function, condition.tileInfo.shape,
                                                                 condition.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        std::vector<int64_t> tempShape({static_cast<int64_t>(tempByteSize)});
        auto tempTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tempShape);

        int64_t whereBitMode = 0;
        if (condition.tensor.GetDataType() == DT_UINT8) {
            whereBitMode = 1;
        }

        Element convertedInput;
        Element convertedOther;
        if constexpr (std::is_same_v<U, const Element>) {
            if (input.GetDataType() == DT_BF16) {
                double inputValue = input.GetFloatData();
                convertedInput = Element(DataType::DT_FP32, inputValue);
            } else {
                convertedInput = input;
            }
        }
        if constexpr (std::is_same_v<W, const Element>) {
            if (other.GetDataType() == DT_BF16) {
                double otherValue = other.GetFloatData();
                convertedOther = Element(DataType::DT_FP32, otherValue);
            } else {
                convertedOther = other;
            }
        }

        if constexpr (std::is_same_v<U, Input> && std::is_same_v<W, Input>) {
            auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
            auto otherTile = other.tensor.GetStorage()->View(function, other.tileInfo.shape, other.tileInfo.offset);
            auto& op = function.AddOperation(Opcode::OP_WHERE_TT, {conditionTile, inputTile, otherTile},
                                             {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", static_cast<int64_t>(whereBitMode));
        } else if constexpr (std::is_same_v<U, Input> && std::is_same_v<W, const Element>) {
            auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
            auto& op = function.AddOperation(Opcode::OP_WHERE_TS, {conditionTile, inputTile}, {resultTile, tempTensor});
            op.SetAttribute(OpAttributeKey::scalar, convertedOther);
            op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", static_cast<int64_t>(whereBitMode));
        } else if constexpr (std::is_same_v<U, const Element> && std::is_same_v<W, Input>) {
            auto otherTile = other.tensor.GetStorage()->View(function, other.tileInfo.shape, other.tileInfo.offset);
            auto& op = function.AddOperation(Opcode::OP_WHERE_ST, {conditionTile, otherTile}, {resultTile, tempTensor});
            op.SetAttribute(OpAttributeKey::scalar, convertedInput);
            op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", static_cast<int64_t>(whereBitMode));
        } else if constexpr (std::is_same_v<U, const Element> && std::is_same_v<W, const Element>) {
            auto& op = function.AddOperation(Opcode::OP_WHERE_SS, {conditionTile}, {resultTile, tempTensor});
            std::vector<Element> scalars = {convertedInput, convertedOther};
            op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
            op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", static_cast<int64_t>(whereBitMode));
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        // bit模式下condition的尾轴偏移和切割应该都除以8
        // cur == result->shape.size() - 1  特殊处理
        auto conditionDatatype = condition.tensor.GetDataType();
        if (cur == (result->shape.size() - 1) && conditionDatatype == DT_UINT8) {
            condition.tileInfo.offset[cur] = (i / NUM_VALUE_8) % condition.tensor.GetStorage()->shape[cur];
            condition.tileInfo.shape[cur] = std::min(
                condition.tensor.GetStorage()->shape[cur] - condition.tileInfo.offset[cur], vecTile[cur] / NUM_VALUE_8);
        } else {
            condition.tileInfo.offset[cur] = i % condition.tensor.GetStorage()->shape[cur];
            condition.tileInfo.shape[cur] = std::min(
                condition.tensor.GetStorage()->shape[cur] - condition.tileInfo.offset[cur], vecTile[cur]);
        }
        if constexpr (std::is_same_v<U, Input>) {
            input.tileInfo.offset[cur] = i % input.tensor.GetStorage()->shape[cur];
            input.tileInfo.shape[cur] = std::min(input.tensor.GetStorage()->shape[cur] - input.tileInfo.offset[cur],
                                                 vecTile[cur]);
        }
        if constexpr (std::is_same_v<W, Input>) {
            other.tileInfo.offset[cur] = i % other.tensor.GetStorage()->shape[cur];
            other.tileInfo.shape[cur] = std::min(other.tensor.GetStorage()->shape[cur] - other.tileInfo.offset[cur],
                                                 vecTile[cur]);
        }
        TiledWhereOperation(function, tileShape, cur + 1, condition, input, other, result, resultTileInfo);
    }
}

void ExpandTensorLastDimension(const LogicalTensorPtr& TensorPtr)
{
    std::vector<int64_t> inputShape(TensorPtr->shape);
    int lastDim = inputShape.back();
    int bitsNumOfByte = 8;
    int expandLastDim = lastDim * bitsNumOfByte;
    TensorPtr->shape.back() = expandLastDim;
}

void ShrinkTensorLastDimension(const LogicalTensorPtr& TensorPtr)
{
    std::vector<int64_t> inputShape(TensorPtr->shape);
    int lastDim = inputShape.back();
    int bitsNumOfByte = 8;
    int shrinkLastDim = std::max(lastDim / bitsNumOfByte, NUM_VALUE_1);
    TensorPtr->shape.back() = shrinkLastDim;
}

template <typename U, typename W>
void TiledWhereOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& condition,
                         const U& input, const W& other, const LogicalTensorPtr& result)
{
    LogicalTensorPtr conditionPtr = condition;
    LogicalTensorPtr inputPtr = nullptr;
    LogicalTensorPtr otherPtr = nullptr;
    if constexpr (std::is_same_v<U, LogicalTensorPtr>) {
        inputPtr = input;
    }
    if constexpr (std::is_same_v<W, LogicalTensorPtr>) {
        otherPtr = other;
    }
    std::vector<SymbolicScalar> resultValidShape = result->GetDynValidShape();
    std::vector<SymbolicScalar> conditionValidShape = result->GetDynValidShape();
    std::vector<int64_t> conditionExpandShape(result->shape);
    if (condition->Datatype() == DT_UINT8) {
        int bitsNumOfByte = 8;
        CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, tileShape.GetVecTile().tile.back() % bitsNumOfByte == 0)
            << "The tileShape of last axis need to 8 align!";
        conditionValidShape.back() = conditionValidShape.back() / bitsNumOfByte;
        conditionExpandShape.back() = conditionExpandShape.back() / bitsNumOfByte;
    }
    if (condition->shape != conditionExpandShape) {
        auto tmp = std::make_shared<LogicalTensor>(function, condition->Datatype(), conditionExpandShape);
        ExpandWithResultValidShape(function, tileShape, condition, tmp, conditionValidShape);
        conditionPtr = tmp;
    }
    if constexpr (std::is_same_v<U, LogicalTensorPtr>) {
        if (input->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, input->Datatype(), targetShape);
            ExpandWithResultValidShape(function, tileShape, inputPtr, tmp, resultValidShape);
            inputPtr = tmp;
        }
    }
    if constexpr (std::is_same_v<W, LogicalTensorPtr>) {
        if (other->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, other->Datatype(), targetShape);
            ExpandWithResultValidShape(function, tileShape, otherPtr, tmp, resultValidShape);
            otherPtr = tmp;
        }
    }

    TileInfo tileInfoCondition(result->shape.size(), result->offset.size());
    auto inputCondition = Input{conditionPtr, tileInfoCondition};
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    if constexpr (std::is_same_v<U, LogicalTensorPtr> && std::is_same_v<W, LogicalTensorPtr>) {
        TileInfo tileInfoInput(result->shape.size(), result->offset.size());
        TileInfo tileInfoOther(result->shape.size(), result->offset.size());
        auto inputInput = Input{inputPtr, tileInfoInput};
        auto inputOther = Input{otherPtr, tileInfoOther};
        TiledWhereOperation(function, tileShape, 0, inputCondition, inputInput, inputOther, result, resultTileInfo);
    } else if constexpr (std::is_same_v<U, LogicalTensorPtr> && std::is_same_v<W, Element>) {
        TileInfo tileInfoInput(result->shape.size(), result->offset.size());
        auto inputInput = Input{inputPtr, tileInfoInput};
        TiledWhereOperation(function, tileShape, 0, inputCondition, inputInput, other, result, resultTileInfo);
    } else if constexpr (std::is_same_v<U, Element> && std::is_same_v<W, LogicalTensorPtr>) {
        TileInfo tileInfoOther(result->shape.size(), result->offset.size());
        auto inputOther = Input{otherPtr, tileInfoOther};
        TiledWhereOperation(function, tileShape, 0, inputCondition, input, inputOther, result, resultTileInfo);
    } else if constexpr (std::is_same_v<U, Element> && std::is_same_v<W, Element>) {
        TiledWhereOperation(function, tileShape, 0, inputCondition, input, other, result, resultTileInfo);
    }
}

template <typename U, typename W>
std::vector<SymbolicScalar> GetResultValidShape(const LogicalTensorPtr& condition, const U& input, const W& other,
                                                std::vector<int64_t> resultShape)
{
    std::vector<SymbolicScalar> resultValidShape;
    for (size_t i = 0; i < resultShape.size(); ++i) {
        if (!condition->GetDynValidShape().empty() && resultShape[i] == condition->shape[i]) {
            resultValidShape.push_back(condition->GetDynValidShape()[i]);
            continue;
        }
        if constexpr (std::is_same_v<U, LogicalTensorPtr>) {
            if (!input->GetDynValidShape().empty() && resultShape[i] == input->shape[i]) {
                resultValidShape.push_back(input->GetDynValidShape()[i]);
                continue;
            }
        }
        if constexpr (std::is_same_v<W, LogicalTensorPtr>) {
            if (!other->GetDynValidShape().empty() && resultShape[i] == other->shape[i]) {
                resultValidShape.push_back(other->GetDynValidShape()[i]);
                continue;
            }
        }
    }
    if (condition->Datatype() == DT_UINT8 && resultValidShape.size() == (resultShape.size() - 1)) {
        SymbolicScalar temp = condition->GetDynValidShape().back() * NUM_VALUE_8;
        resultValidShape.push_back(temp);
    }
    return resultValidShape;
}

std::vector<int64_t> GetBroadCastShapeReturnInt64_t(LogicalTensorPtr& operand1, LogicalTensorPtr& operand2)
{
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    std::vector<int64_t> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        broadCastShape[i] = std::max(opShape1[i], opShape2[i]);
    }
    return broadCastShape;
}

std::vector<int64_t> GetBroadCastShape(LogicalTensorPtr& operand1, LogicalTensorPtr& operand2,
                                       LogicalTensorPtr& operand3)
{
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
    std::vector<int64_t> opShape3(operand3->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    maxShapeSize = std::max(maxShapeSize, opShape3.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    if (opShape3.size() != maxShapeSize) {
        opShape3.insert(opShape3.begin(), maxShapeSize - opShape3.size(), 1);
    }
    std::vector<int64_t> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        int64_t currentMax = std::max(opShape1[i], opShape2[i]);
        currentMax = std::max(currentMax, opShape3[i]);
        broadCastShape[i] = currentMax;
    }
    return broadCastShape;
}

LogicalTensorPtr BinaryOperationUnsqueeze(const LogicalTensorPtr& operand, const std::vector<int64_t>& broadCastShape)
{
    if (operand->shape.size() < broadCastShape.size()) {
        auto broadCastDims = broadCastShape.size() - operand->shape.size();
        std::vector<int64_t> unsqueezeShape(operand->shape);
        unsqueezeShape.insert(unsqueezeShape.begin(), broadCastDims, 1);
        auto tmpOperand = Reshape(operand, unsqueezeShape).GetStorage();
        return tmpOperand;
    }
    return operand;
}

LogicalTensorPtr TensorWhereOperation(Function& function, const Tensor& condition, const Tensor& input,
                                      const Tensor& other)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, condition.GetShape().size() == condition.GetStorage()->offset.size())
        << "The shape size of condition and offset must be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input.GetShape().size() == input.GetStorage()->offset.size())
        << "The shape size of input and offset must be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, other.GetShape().size() == other.GetStorage()->offset.size())
        << "The shape size of other and offset must be equal";

    if (condition.GetStorage()->Datatype() == DT_UINT8) {
        int bitsNumOfByte = 8;
        int broadcastFlag = 1;
        CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, input.GetStorage()->shape.back() % bitsNumOfByte == 0 ||
                                                         input.GetStorage()->shape.back() == broadcastFlag)
            << "The input shape of last axis need to 8 align or equal to 1";
        CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, other.GetStorage()->shape.back() % bitsNumOfByte == 0 ||
                                                         other.GetStorage()->shape.back() == broadcastFlag)
            << "The other shape of last axis need to 8 align or equal to 1";
    }
    auto conditionT0 = condition.GetStorage();
    auto inputT1 = input.GetStorage();
    auto otherT2 = other.GetStorage();

    std::vector<int64_t> resultShape;
    if (condition.GetStorage()->Datatype() == DT_BOOL) {
        resultShape = GetBroadCastShape(conditionT0, inputT1, otherT2);
        conditionT0 = BinaryOperationUnsqueeze(conditionT0, resultShape);
    } else if (condition.GetStorage()->Datatype() == DT_UINT8) {
        ExpandTensorLastDimension(conditionT0);
        resultShape = GetBroadCastShape(conditionT0, inputT1, otherT2);
        conditionT0 = BinaryOperationUnsqueeze(conditionT0, resultShape);
        ShrinkTensorLastDimension(conditionT0);
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false) << "condition Datatype must be uint8 or bool.";
    }

    inputT1 = BinaryOperationUnsqueeze(inputT1, resultShape);
    otherT2 = BinaryOperationUnsqueeze(otherT2, resultShape);
    std::vector<SymbolicScalar> resultValidShape = GetResultValidShape(conditionT0, inputT1, otherT2, resultShape);
    auto result = std::make_shared<LogicalTensor>(function, input.GetStorage()->Datatype(), resultShape,
                                                  resultValidShape);
    function.AddOperation(Opcode::OP_WHERE_TT, {conditionT0, inputT1, otherT2}, {result});
    result->UpdateDynValidShape(resultValidShape);
    return result;
}

LogicalTensorPtr TensorWhereOperation(Function& function, const Tensor& condition, const Tensor& input,
                                      const Element& other)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, condition.GetShape().size() == condition.GetStorage()->offset.size())
        << "The shape size of condition and offset must be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input.GetShape().size() == input.GetStorage()->offset.size())
        << "The shape size of input and offset must be equal";
    if (condition.GetStorage()->Datatype() == DT_UINT8) {
        int bitsNumOfByte = 8;
        int broadcastFlag = 1;
        CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, input.GetStorage()->shape.back() % bitsNumOfByte == 0 ||
                                                         input.GetStorage()->shape.back() == broadcastFlag)
            << "The input shape of last axis need to 8 align or equal to 1";
    }
    auto conditionT0 = condition.GetStorage();
    auto inputT1 = input.GetStorage();

    std::vector<int64_t> resultShape;
    if (condition.GetStorage()->Datatype() == DT_BOOL) {
        resultShape = GetBroadCastShapeReturnInt64_t(conditionT0, inputT1);
        conditionT0 = BinaryOperationUnsqueeze(conditionT0, resultShape);
    } else if (condition.GetStorage()->Datatype() == DT_UINT8) {
        ExpandTensorLastDimension(conditionT0);
        resultShape = GetBroadCastShapeReturnInt64_t(conditionT0, inputT1);
        conditionT0 = BinaryOperationUnsqueeze(conditionT0, resultShape);
        ShrinkTensorLastDimension(conditionT0);
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false) << "condition Datatype must be uint8 or bool.";
    }
    inputT1 = BinaryOperationUnsqueeze(inputT1, resultShape);
    std::vector<SymbolicScalar> resultValidShape = GetResultValidShape(conditionT0, inputT1, other, resultShape);
    auto result = std::make_shared<LogicalTensor>(function, inputT1->Datatype(), resultShape, resultValidShape);
    auto& op = function.AddOperation(Opcode::OP_WHERE_TS, {conditionT0, inputT1}, {result});
    result->UpdateDynValidShape(resultValidShape);
    op.SetAttribute(OpAttributeKey::scalar, other);
    return result;
}

LogicalTensorPtr TensorWhereOperation(Function& function, const Tensor& condition, const Element& input,
                                      const Tensor& other)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, condition.GetShape().size() == condition.GetStorage()->offset.size())
        << "The shape size of condition and offset must be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, other.GetShape().size() == other.GetStorage()->offset.size())
        << "The shape size of other and offset must be equal";
    if (condition.GetStorage()->Datatype() == DT_UINT8) {
        int bitsNumOfByte = 8;
        int broadcastFlag = 1;
        CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, other.GetStorage()->shape.back() % bitsNumOfByte == 0 ||
                                                         other.GetStorage()->shape.back() == broadcastFlag)
            << "The other shape of last axis need to 8 align or equal to 1";
    }
    auto conditionT0 = condition.GetStorage();
    auto otherT1 = other.GetStorage();
    std::vector<int64_t> resultShape;

    if (condition.GetStorage()->Datatype() == DT_BOOL) {
        resultShape = GetBroadCastShapeReturnInt64_t(conditionT0, otherT1);
        conditionT0 = BinaryOperationUnsqueeze(conditionT0, resultShape);
    } else if (condition.GetStorage()->Datatype() == DT_UINT8) {
        ExpandTensorLastDimension(conditionT0);
        resultShape = GetBroadCastShapeReturnInt64_t(conditionT0, otherT1);
        conditionT0 = BinaryOperationUnsqueeze(conditionT0, resultShape);
        ShrinkTensorLastDimension(conditionT0);
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false) << "condition Datatype must be uint8 or bool.";
    }
    otherT1 = BinaryOperationUnsqueeze(otherT1, resultShape);
    std::vector<SymbolicScalar> resultValidShape = GetResultValidShape(conditionT0, input, otherT1, resultShape);

    auto result = std::make_shared<LogicalTensor>(function, otherT1->Datatype(), resultShape, resultValidShape);
    auto& op = function.AddOperation(Opcode::OP_WHERE_ST, {conditionT0, otherT1}, {result});
    result->UpdateDynValidShape(resultValidShape);
    op.SetAttribute(OpAttributeKey::scalar, input);
    return result;
}

LogicalTensorPtr TensorWhereOperation(Function& function, const Tensor& condition, const Element& input,
                                      const Element& other)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, condition.GetShape().size() == condition.GetStorage()->offset.size())
        << "The shape size of condition and offset must be equal";
    auto conditionT0 = condition.GetStorage();
    std::vector<int64_t> resultShape = {};

    if (condition.GetStorage()->Datatype() == DT_BOOL) {
        resultShape = GetBroadCastShapeReturnInt64_t(conditionT0, conditionT0);
    } else if (condition.GetStorage()->Datatype() == DT_UINT8) {
        ExpandTensorLastDimension(conditionT0);
        resultShape = GetBroadCastShapeReturnInt64_t(conditionT0, conditionT0);
        ShrinkTensorLastDimension(conditionT0);
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false) << "condition Datatype must be uint8 or bool.";
    }
    std::vector<SymbolicScalar> resultValidShape = conditionT0->GetDynValidShape();
    if (conditionT0->Datatype() == DT_UINT8) {
        if (!resultValidShape.empty()) {
            SymbolicScalar temp = conditionT0->GetDynValidShape().back();
            resultValidShape.back() = temp * NUM_VALUE_8;
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, input.GetDataType(), resultShape, resultValidShape);
    auto& op = function.AddOperation(Opcode::OP_WHERE_SS, {conditionT0}, {result});
    result->UpdateDynValidShape(resultValidShape);
    std::vector<Element> scalars = {input, other};
    op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
    return result;
}

Tensor Where(const Tensor& condition, const Tensor& input, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(condition.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");

    static const std::unordered_set<DataType> conditionTypes = {DT_BOOL, DT_UINT8};
    CheckTensorDataType(condition.GetStorage(), conditionTypes, "WHERE");
    static const std::unordered_set<DataType> a2a3InputTypes = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> a5InputTypes = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& inputTypes = GetSupportedDataTypesByArch(a2a3InputTypes, a5InputTypes);
    CheckTensorDataType(input.GetStorage(), inputTypes, "WHERE");
    CheckTensorDimRange(condition.GetStorage(), 1, 4, "WHERE");
    CheckTensorShapeSize(condition.GetStorage(), "WHERE");
    CheckTensorShapeSize(input.GetStorage(), "WHERE");
    CheckTensorShapeSize(other.GetStorage(), "WHERE");
    CheckTensorsDimConsistency({condition.GetStorage(), input.GetStorage(), other.GetStorage()}, "WHERE");
    CheckTensorsDataTypeConsistency(input.GetStorage(), other.GetStorage(), "WHERE");
    CheckTensorsFormatConsistency(condition.GetStorage(), input.GetStorage(), "WHERE");
    CheckTensorsFormatConsistency(condition.GetStorage(), other.GetStorage(), "WHERE");
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, input, other);
}

Tensor Where(const Tensor& condition, const Tensor& input, const Element& otherValue)
{
    DECLARE_TRACER();
    CheckTensorFormat(condition.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");

    static const std::unordered_set<DataType> conditionTypes = {DT_BOOL, DT_UINT8};
    CheckTensorDataType(condition.GetStorage(), conditionTypes, "WHERE");
    static const std::unordered_set<DataType> a2a3InputTypes = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> a5InputTypes = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& inputTypes = GetSupportedDataTypesByArch(a2a3InputTypes, a5InputTypes);
    CheckTensorDataType(input.GetStorage(), inputTypes, "WHERE");
    CheckTensorDimRange(condition.GetStorage(), 1, 4, "WHERE");
    CheckTensorShapeSize(condition.GetStorage(), "WHERE");
    CheckTensorShapeSize(input.GetStorage(), "WHERE");
    CheckTensorsDimConsistency({condition.GetStorage(), input.GetStorage()}, "WHERE");
    CheckTensorsFormatConsistency(condition.GetStorage(), input.GetStorage(), "WHERE");
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, input, otherValue);
}

Tensor Where(const Tensor& condition, const Element& inputValue, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(condition.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");

    static const std::unordered_set<DataType> conditionTypes = {DT_BOOL, DT_UINT8};
    CheckTensorDataType(condition.GetStorage(), conditionTypes, "WHERE");
    static const std::unordered_set<DataType> a2a3InputTypes = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> a5InputTypes = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& inputTypes = GetSupportedDataTypesByArch(a2a3InputTypes, a5InputTypes);
    CheckTensorDataType(other.GetStorage(), inputTypes, "WHERE");
    CheckTensorDimRange(condition.GetStorage(), 1, 4, "WHERE");
    CheckTensorShapeSize(condition.GetStorage(), "WHERE");
    CheckTensorShapeSize(other.GetStorage(), "WHERE");
    CheckTensorsDimConsistency({condition.GetStorage(), other.GetStorage()}, "WHERE");
    CheckTensorsFormatConsistency(condition.GetStorage(), other.GetStorage(), "WHERE");
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, inputValue, other);
}

Tensor Where(const Tensor& condition, const Element& inputValue, const Element& otherValue)
{
    DECLARE_TRACER();
    CheckTensorFormat(condition.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Where");

    static const std::unordered_set<DataType> conditionTypes = {DT_BOOL, DT_UINT8};
    CheckTensorDataType(condition.GetStorage(), conditionTypes, "WHERE");
    CheckTensorDimRange(condition.GetStorage(), 1, 4, "WHERE");
    CheckTensorShapeSize(condition.GetStorage(), "WHERE");
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, inputValue, otherValue);
}

void WhereOperationTileFuncTT(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledWhereOperation(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0]);
}

void WhereOperationTileFuncTS(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledWhereOperation(function, tileShape, iOperand[0], iOperand[1], op.GetElementAttribute(OpAttributeKey::scalar),
                        oOperand[0]);
}

void WhereOperationTileFuncST(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledWhereOperation(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), iOperand[1],
                        oOperand[0]);
}

void WhereOperationTileFuncSS(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledWhereOperation(function, tileShape, iOperand[0], op.GetVectorElementAttribute(OpAttributeKey::vectorScalar)[0],
                        op.GetVectorElementAttribute(OpAttributeKey::vectorScalar)[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_WHERE_TT, Opcode::OP_WHERE_TT, WhereOperationTileFuncTT);
REGISTER_OPERATION_TILED_FUNC(OP_WHERE_TS, Opcode::OP_WHERE_TS, WhereOperationTileFuncTS);
REGISTER_OPERATION_TILED_FUNC(OP_WHERE_ST, Opcode::OP_WHERE_ST, WhereOperationTileFuncST);
REGISTER_OPERATION_TILED_FUNC(OP_WHERE_SS, Opcode::OP_WHERE_SS, WhereOperationTileFuncSS);

} // namespace npu::tile_fwk
