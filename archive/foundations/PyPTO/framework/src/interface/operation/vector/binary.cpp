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
 * \file binary.cpp
 * \brief
 */

#include <limits>
#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
namespace npu::tile_fwk {

std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2)
{
    std::vector<int64_t> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr& operand, const std::vector<int>& broadCastShape)
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

void CheckOperandsValid(const LogicalTensorPtr& operand1, const LogicalTensorPtr& operand2)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand1->shape.size() == operand2->shape.size())
        << "The shape size of the two input tensors must be equal";
}

void CheckBinOpOperandsValid(const LogicalTensorPtr& operand1, const LogicalTensorPtr& operand2)
{
    CheckOperandsValid(operand1, operand2);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        if (operand1->shape[i] != operand2->shape[i] && (operand1->shape[i] != 1 && operand2->shape[i] != 1)) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "shape not support binary operation";
        }
    }
}

void BroadcastOperandTensor(LogicalTensorPtr& operand, LogicalTensorPtr& other, LogicalTensorPtr result,
                            Function& function, const TileShape& tileShape, std::vector<int64_t> dstShape)
{
    if (dstShape.empty()) {
        dstShape = result->shape;
    }
    if (operand->shape == dstShape) {
        return;
    }
    auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
    Expand(function, tileShape, operand, {other}, expanded);
    operand = expanded;
}

void BinaryOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand)
{
    constexpr size_t inOpSize = 2;
    constexpr size_t outOpSize = 1;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == inOpSize) << "iOperand size should be 2";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == outOpSize) << "oOperand size should be 1";
}

// Identify which operand need brc at a specific axis counting from the first
// Return value 0 = NONE, 1 = LEFT_OPERAND, 2 = RIGHT_OPERAND
int BrcAxisBinaryOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2, int64_t axisNum)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand1->shape.size() == operand2->shape.size()) << "Dims not match";
    int64_t shapeSize = operand1->shape.size();
    int operandNum = 0;

    int64_t idx = (axisNum < 0) ? (shapeSize + axisNum) : axisNum;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, idx >= 0 && idx < shapeSize)
        << "axisNum " << axisNum << " out of range for shapeSize " << shapeSize;
    if ((operand1->shape[idx] != 1) && (operand2->shape[idx] == 1)) {
        operandNum = 2;
    } else if ((operand1->shape[idx] == 1) && (operand2->shape[idx] != 1)) {
        operandNum = 1;
    }
    return operandNum;
}

template <BinaryOpType T>
void TiledBinaryOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                          LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                          int64_t precisionType)
{
    size_t shapeSize = input1.tensor->GetShape().size();
    if (cur == shapeSize) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto opName = GetBinaryOpName<T>();
        Operation* op = nullptr;
        if (opName == "BITWISEXOR" || opName == "COPYSIGN" || opName == "POW") {
            std::vector<int64_t> tmpShape(resultTileInfo.shape);
            auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
            auto alignedLast = AlignUp(tmpShape[resultTileInfo.shape.size() - 1], alignSize);
            if (opName == "POW" && result->Datatype() == DT_FP32 &&
                Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201) {
                auto maskColsFloat = AlignUp((alignedLast + NUM_VALUE_8 - 1) / NUM_VALUE_8, BLOCK_SIZE) /
                                     BytesOf(result->Datatype());
                alignedLast = NUM_VALUE_2 * alignedLast + NUM_VALUE_3 * maskColsFloat;
            }
            tmpShape[resultTileInfo.shape.size() - 1] = alignedLast;
            auto tempTensor = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
            op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                        {resultTile, tempTensor});
        } else if (opName == "REM") {
            auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
            std::vector<int64_t> tmpShape;
            tmpShape.push_back(2);
            tmpShape.push_back(AlignUp(resultTileInfo.shape[shapeSize - 1], alignSize));
            auto tmpTensor = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
            op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                        {resultTile, tmpTensor});
        } else if (opName == "ATAN2") {
            std::vector<int64_t> tmpShape(resultTileInfo.shape);
            auto lastDim = tmpShape[resultTileInfo.shape.size() - 1];
            auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
            tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(lastDim, alignSize) * NUM4 * BytesOf(DT_FP32);
            std::vector<SymbolicScalar> tmpValidShape;
            for (auto s : tmpShape) {
                tmpValidShape.emplace_back(s);
            }
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_UINT8, tmpShape, tmpValidShape);
            op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                        {resultTile, workspace});
        } else if (opName == "FLOORDIV") {
            auto tmpShape = resultTileInfo.shape;
            auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
            tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(tmpShape.back(), alignSize) * 6;
            int64_t intermediateBytes = std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL,
                                                        std::multiplies<int64_t>()) *
                                        BytesOf(DT_FP32);
            auto tempTensor = std::make_shared<LogicalTensor>(function, DT_UINT8,
                                                              std::vector<int64_t>{intermediateBytes});
            op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                        {resultTile, tempTensor});
        } else {
            op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
        }

        if (op != nullptr) {
            std::vector<int64_t> brcOperand(shapeSize, 0);
            size_t brcAxesCount = 0;
            for (size_t i = 0; i < shapeSize; i++) {
                brcOperand[i] = BrcAxisBinaryOp(input1.tensor, input2.tensor, i);
                if (brcOperand[i] != 0) {
                    brcAxesCount++;
                }
            }
            if (brcAxesCount > 0) {
                if (brcOperand[shapeSize - 1] != 0 && brcAxesCount >= 2) {
                    op->SetAttribute(OpAttributeKey::excludeBufferReuse, true);
                }
                op->SetAttribute(OpAttributeKey::brcOperand, brcOperand);
            }
        }
        if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW ||
                      T == BinaryOpType::REM) {
            op->SetAttribute(OpAttributeKey::precisionType, precisionType);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledBinaryOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, precisionType);
    }
}

// Determine the target shape for expand before tileop
template <BinaryOpType T>
std::pair<std::vector<int64_t>, std::vector<int64_t>> GetBrcExpandShape(Function& function, LogicalTensorPtr operand1,
                                                                        LogicalTensorPtr operand2,
                                                                        LogicalTensorPtr result)
{
    auto operand1Shape = result->shape;
    auto operand2Shape = result->shape;
    size_t shapeSize = result->shape.size();

    bool isInWhiteList = SUPPORT_BRC_INLINE.count(GetBinaryOpNameCode<T>());
    bool isCombineAxisEnabled = function.paramConfigs_.combineAxis && isInWhiteList;
    if (isInWhiteList) {
        // Outer axis: handled by tileop loop with stride control, keep operand shape.
        if (shapeSize > 2) {
            for (size_t i = 0; i < shapeSize - 2; i++) {
                operand1Shape[i] = operand1->shape[i];
                operand2Shape[i] = operand2->shape[i];
            }
        }
        // The 2nd last axis: skip expand, brcinline
        if (shapeSize > 1) {
            operand1Shape[shapeSize - 2] = operand1->shape[shapeSize - 2];
            operand2Shape[shapeSize - 2] = operand2->shape[shapeSize - 2];
        }
        // The last axis: brcinline when combineAxis is enabled
        if (shapeSize > 0 && isCombineAxisEnabled) {
            operand1Shape[shapeSize - 1] = operand1->shape[shapeSize - 1];
            operand2Shape[shapeSize - 1] = operand2->shape[shapeSize - 1];
        }
    }
    return {operand1Shape, operand2Shape};
}

template <BinaryOpType T>
void TiledBinaryOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                          LogicalTensorPtr operand2, const LogicalTensorPtr& result, int64_t precisionType)
{
    CheckBinOpOperandsValid(operand1, operand2);
    auto [dstShape1, dstShape2] = GetBrcExpandShape<T>(function, operand1, operand2, result);
    BroadcastOperandTensor(operand1, operand2, result, function, tileShape, dstShape1);
    BroadcastOperandTensor(operand2, operand1, result, function, tileShape, dstShape2);

    TileInfo tileInfo1(operand1->shape.size(), operand1->offset.size());
    TileInfo tileInfo2(operand2->shape.size(), operand2->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledBinaryOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo, precisionType);
}

void TiledPReLUOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input, Input& weight,
                         const LogicalTensorPtr& result)
{
    if (cur == 0 && input.tensor.GetShape().size() == 1) {
        // 1D 输入：weight shape [1]，不需要切分，直接初始化 weight tile info
        weight.tileInfo.shape[0] = 1;
        weight.tileInfo.offset[0] = 0;
    }

    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto weightTile = weight.tensor.GetStorage()->View(function, weight.tileInfo.shape, weight.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        int axis = 5 - cur + 1;
        constexpr size_t ALIGN_SIZE = 32;
        constexpr size_t SIZEOFBYTE = 8;
        int64_t tmpSize = ALIGN_SIZE;
        if (axis == 4) {
            tmpSize = (input.tileInfo.shape[cur - 1] + SIZEOFBYTE - 1) / SIZEOFBYTE;
            tmpSize = (tmpSize + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE + ALIGN_SIZE;
        }
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmpShape);
        auto& op = function.AddOperation(Opcode::OP_PRELU, {tile, weightTile}, {resultTile, tmpTensor});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

        size_t dimSize = input.tensor.GetShape().size();
        if (dimSize == 2) {
            std::vector<bool> dimMap({true, false});
            op.SetAttr(OpAttributeKey::rowPad, dimMap);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();

    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        // 1D 输入时，weight 不需要切分
        if (input.tensor.GetShape().size() > 1 && cur == 1) {
            weight.tileInfo.shape[0] = std::min(weight.tensor.GetShape()[0] - i, vecTile[cur]);
            weight.tileInfo.offset[0] = i;
        }
        TiledPReLUOperation(function, tileShape, cur + 1, input, weight, result);
    }
}

void TiledPReLUOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& input,
                         const LogicalTensorPtr& weight, const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input->shape.size() == input->offset.size())
        << "The shape size of input and offset must be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, weight->shape.size() == weight->offset.size())
        << "The shape size of weight and offset must be equal";

    TileInfo inputTileInfo(input->shape.size(), input->offset.size());
    TileInfo weightTileInfo(weight->shape.size(), weight->offset.size());
    auto inputArg = Input{input, inputTileInfo};
    auto weightArg = Input{weight, weightTileInfo};
    TiledPReLUOperation(function, tileShape, 0, inputArg, weightArg, result);
}

void PReLUOperationOperandCheck(const LogicalTensorPtr& selfTensor, const LogicalTensorPtr& weightTensor)
{
    CheckTensorDimRange(selfTensor, 1, 4, "PReLU");
    CheckTensorDimRange(weightTensor, 1, 1, "PReLU");
    CheckTensorShapeSize(selfTensor, "PReLU");
    CheckTensorShapeSize(weightTensor, "PReLU");

    if (selfTensor->shape.size() == 1) {
        // 1D 输入时，weight 必须为 [1]
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, weightTensor->shape[0] == 1)
            << "The weight size should be [1] when input is 1D";
    } else {
        // 2D/3D/4D 输入时，weight 必须与 self 的第二维匹配
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, weightTensor->shape[0] == selfTensor->shape[1])
            << "The weight size should equal to input's second dimension";
    }
}

void PReLUOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledPReLUOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

LogicalTensorPtr TensorPReLUOperation(Function& function, const Tensor& self, const Tensor& weight)
{
    auto selfTensor = self.GetStorage();
    auto weightTensor = weight.GetStorage();

    PReLUOperationOperandCheck(selfTensor, weightTensor);

    auto result = std::make_shared<LogicalTensor>(function, selfTensor->Datatype(), selfTensor->shape,
                                                  selfTensor->GetDynValidShape());
    function.AddOperation(Opcode::OP_PRELU, {selfTensor, weightTensor}, {result});
    return result;
}

Tensor PReLU(const Tensor& self, const Tensor& weight)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "PReLU");
    CheckTensorFormat(weight.GetStorage(), {TileOpFormat::TILEOP_NZ}, "PReLU");

    CheckTensorsDataTypeConsistency(self.GetStorage(), weight.GetStorage(), "PReLU");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "PReLU");

    RETURN_CALL(PReLUOperation, *Program::GetInstance().GetCurrentFunction(), self, weight);
}

Tensor Add(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Add");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Add");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "ADD");

    static const std::unordered_set<DataType> ADD_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> ADD_A5_TYPES = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& supportedTypes = GetSupportedDataTypesByArch(ADD_A2A3_TYPES, ADD_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ADD");
    RETURN_CALL(BinaryOperation<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Sub(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sub");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sub");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "SUB");

    static const std::unordered_set<DataType> SUB_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> SUB_A5_TYPES = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& supportedTypes = GetSupportedDataTypesByArch(SUB_A2A3_TYPES, SUB_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SUB");
    RETURN_CALL(BinaryOperation<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Mul(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Mul");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Mul");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "MUL");

    static const std::unordered_set<DataType> MUL_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> MUL_A5_TYPES = {DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16};
    const auto& supportedTypes = GetSupportedDataTypesByArch(MUL_A2A3_TYPES, MUL_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MUL");
    RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Div(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Div");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Div");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "DIV");

    static const std::unordered_set<DataType> DIV_A2A3_TYPES = {DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32};
    static const std::unordered_set<DataType> DIV_A5_TYPES = {DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32};
    const auto& supportedTypes = GetSupportedDataTypesByArch(DIV_A2A3_TYPES, DIV_A5_TYPES);
    auto isDivSupportedInt = [](DataType dt) { return dt == DT_INT16 || dt == DT_INT32; };
    CheckTensorDataType(self.GetStorage(), supportedTypes, "DIV");

    if (isDivSupportedInt(self.GetDataType())) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Tensor castOther = Cast(other, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [castResult, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(
            *Program::GetInstance().GetCurrentFunction(), castSelf, castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Tensor(castResult);
    }

    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                       self, other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Fmod(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Fmod");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Fmod");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "MOD");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MOD");
    auto selfDtype = self.GetDataType();
    if (selfDtype == DT_FP16) {
        Tensor castSelf = Cast(self, DataType::DT_FP32, CastMode::CAST_NONE);
        Tensor castOther = Cast(other, DataType::DT_FP32, CastMode::CAST_NONE);
        auto [castResult, op] = TensorBinaryOperationWithOp<BinaryOpType::MOD>(
            *Program::GetInstance().GetCurrentFunction(), castSelf, castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Cast(Tensor(castResult), selfDtype, CastMode::CAST_NONE);
    }
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::MOD>(*Program::GetInstance().GetCurrentFunction(),
                                                                       self, other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Remainder(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "REM");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "REM");
    auto selfDtype = self.GetDataType();
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if ((!isA5Architecture && selfDtype == DT_INT16) || selfDtype == DT_FP16) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Tensor castOther = Cast(other, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::REM>(*Program::GetInstance().GetCurrentFunction(),
                                                                           castSelf, castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        Tensor castedResult = Cast(Tensor(result), selfDtype, CastMode::CAST_NONE, SaturationMode::ON);
        return castedResult;
    }
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::REM>(*Program::GetInstance().GetCurrentFunction(),
                                                                       self, other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Maximum(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Maximum");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Maximum");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "MAXIMUM");

    static const std::unordered_set<DataType> MAX_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> MAX_A5_TYPES = {DT_INT32, DT_UINT32, DT_FP32,  DT_INT16, DT_UINT16,
                                                              DT_FP16,  DT_BF16,   DT_UINT8, DT_INT8};
    const auto& supportedTypes = GetSupportedDataTypesByArch(MAX_A2A3_TYPES, MAX_A5_TYPES);
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MAXIMUM");
    RETURN_CALL(BinaryOperation<BinaryOpType::MAXIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1,
                operand2);
}

Tensor Minimum(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Minimum");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Minimum");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "MINIMUM");

    static const std::unordered_set<DataType> MIN_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> MIN_A5_TYPES = {DT_INT32, DT_UINT32, DT_FP32,  DT_INT16, DT_UINT16,
                                                              DT_FP16,  DT_BF16,   DT_UINT8, DT_INT8};
    const auto& supportedTypes = GetSupportedDataTypesByArch(MIN_A2A3_TYPES, MIN_A5_TYPES);
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MINIMUM");
    RETURN_CALL(BinaryOperation<BinaryOpType::MINIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1,
                operand2);
}

Tensor BitwiseAnd(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseAnd");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseAnd");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "BITWISEAND");
    static const std::unordered_set<DataType> BITWISE_AND_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                   DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_AND_TYPES, "BITWISEAND");
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseOr(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseOr");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseOr");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "BITWISEOR");
    static const std::unordered_set<DataType> BITWISE_OR_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                  DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_OR_TYPES, "BITWISEOR");
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEOR>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseXor(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseXor");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseXor");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "BITWISEXOR");
    static const std::unordered_set<DataType> BITWISE_XOR_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                   DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_XOR_TYPES, "BITWISEXOR");
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEXOR>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Gcd(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gcd");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gcd");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "GCD");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "GCD");
    std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_INT16, DT_INT32, DT_UINT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "GCD");
    RETURN_CALL(BinaryOperation<BinaryOpType::GCD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

static std::pair<int64_t, int64_t> GetIntegerScalarRange(DataType dtype)
{
    switch (dtype) {
        case DT_INT8:
            return {std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()};
        case DT_INT16:
            return {std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()};
        case DT_INT32:
            return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
        case DT_UINT8:
            return {0LL, static_cast<int64_t>(std::numeric_limits<uint8_t>::max())};
        default:
            return {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    }
}

Tensor Gcd(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gcd");

    CheckTensorDimRange(self.GetStorage(), 1, 4, "GCD");
    std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_INT16, DT_INT32, DT_UINT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "GCD");
    auto [elemMin, elemMax] = GetIntegerScalarRange(self.GetDataType());
    if (other.GetSignedData() < elemMin || other.GetSignedData() > elemMax) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
            << "The value range for the element is incorrect! expected [" << elemMin << ", " << elemMax << "], got "
            << other.GetSignedData();
    }
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::GCD>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

DataType GetPowRealResultDataType(DataType selfType, DataType otherType)
{
    if (selfType == DT_INT32) {
        return otherType;
    }
    if (otherType == DT_INT32) {
        return selfType;
    }
    if (selfType == DT_BF16) {
        return otherType == DT_FP16 ? DT_FP32 : otherType;
    }
    if (otherType == DT_BF16) {
        return selfType == DT_FP16 ? DT_FP32 : selfType;
    }
    return selfType == DT_FP16 && otherType == DT_FP16 ? DT_FP16 : DT_FP32;
}

DataType GetPowCalcResultDataType(DataType selfType, DataType otherType)
{
    if (selfType == DT_INT32 && otherType == DT_INT32) {
        return DT_INT32;
    }
    return DT_FP32;
}

LogicalTensorPtr CastToResultType(const LogicalTensorPtr& tensor, DataType originType, DataType resultType)
{
    if (originType == resultType) {
        return tensor;
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), tensor, resultType,
                CastMode::CAST_NONE);
}

LogicalTensorPtr GenAllOneTensor(const Shape& shape, std::vector<SymbolicScalar> validShape, const DataType& dataType)
{
    auto result = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
                       SymbolicScalar(), DataType::DT_FP32, shape, validShape);
    if (dataType != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    dataType, CastMode::CAST_NONE);
    }
    return result.GetStorage();
}

void PowCheck(const Tensor& self, const Tensor& other)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "POW");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "POW");
    CheckTensorShapeSize(self.GetStorage(), "POW");
    CheckTensorShapeSize(other.GetStorage(), "POW");
    CheckTensorsDimConsistency({self.GetStorage(), other.GetStorage()}, "POW");
    CheckTensorsShapeConsistencyOrBroadcast({self.GetStorage(), other.GetStorage()}, "POW");
    CheckTensorsFormatConsistency(self.GetStorage(), other.GetStorage(), "POW");
}

Tensor Pow(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");

    PowCheck(self, other);
    DataType selfType = self.GetDataType();
    DataType otherType = other.GetDataType();
    if (IsInteger(selfType) && IsInteger(otherType)) {
        CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "Pow");
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                           self.GetStorage(), other.GetStorage());
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::INTRINSIC));
        return result;
    }
    DataType realResultType = GetPowRealResultDataType(selfType, otherType);
    DataType calcResultType = GetPowCalcResultDataType(selfType, otherType);
    auto selfSt = CastToResultType(self.GetStorage(), selfType, calcResultType);
    auto otherSt = CastToResultType(other.GetStorage(), otherType, calcResultType);
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                       selfSt, otherSt);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (realResultType != calcResultType) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
                    realResultType, CastMode::CAST_NONE);
    }
    return result;
}

LogicalTensorPtr PowSCalc(const LogicalTensorPtr& self, const Element& other, PrecisionType precisionType)
{
    double exponent = other.Cast<double>();
    if (std::abs(exponent - NUM_VALUE_0_5) < NUM_VALUE_EPS) { // sqrt(x)
        RETURN_CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self);
    } else if (std::abs(exponent + NUM_VALUE_0_5) < NUM_VALUE_EPS) { // 1 / sqrt(x)
        auto sqrt = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self);
        auto ones = GenAllOneTensor(self->shape, self->GetDynValidShape(), DT_FP32);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                           ones, sqrt);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::HIGH_PRECISION));
        return result;
    } else if (std::abs(exponent - NUM_VALUE_3) < NUM_VALUE_EPS) { // x * x * x
        auto mul = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
        RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), mul, self);
    } else if (std::abs(exponent - NUM_VALUE_2) < NUM_VALUE_EPS) { // x * x
        RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
    } else if (std::abs(exponent + NUM_VALUE_2) < NUM_VALUE_EPS) { // 1 / (x * x)
        auto mul = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
        auto ones = GenAllOneTensor(self->shape, self->GetDynValidShape(), DT_FP32);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                           ones, mul);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::HIGH_PRECISION));
        return result;
    } else if (std::abs(exponent + NUM_VALUE_1) < NUM_VALUE_EPS) { // 1 / x
        auto ones = GenAllOneTensor(self->shape, self->GetDynValidShape(), DT_FP32);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                           ones, self);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::HIGH_PRECISION));
        return result;
    } else if (self->Datatype() == DT_INT32) {
        auto [res, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::POW>(
            *Program::GetInstance().GetCurrentFunction(), self, Element(DT_INT32, other.Cast<int>()));
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return res;
    } else if (self->Datatype() == DT_FP32) {
        auto otherTensor = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(),
                                Element(DataType::DT_FP32, exponent), SymbolicScalar(), DataType::DT_FP32, self->shape,
                                self->GetDynValidShape());
        auto [res, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                        self, otherTensor);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return res;
    }
    return self;
}

void PowSCheck(const Tensor& self)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "POW");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "POW");
    CheckTensorShapeSize(self.GetStorage(), "POW");
}

Tensor Pow(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");

    PowSCheck(self);
    LogicalTensorPtr castSelf = self.GetStorage();
    if (self.GetDataType() == DT_INT32 && !IsInteger(other.GetDataType())) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    if (IsInteger(castSelf->Datatype())) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, IsInteger(other.GetDataType()))
            << "Scalar dtype incorrect. When self dtype in UINT32/INT16/UINT16/INT8/UINT8, "
            << "Scalar dtype should be int, acture is: " << DataType2String(other.GetDataType());
        auto [res, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::POW>(
            *Program::GetInstance().GetCurrentFunction(), castSelf, Element(DT_INT32, other.Cast<int>()));
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::INTRINSIC));
        return res;
    }
    if (std::abs(other.Cast<double>()) < NUM_VALUE_EPS) {
        return GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    }
    DataType dataType = castSelf->Datatype();
    bool shouldUpToFp32 = dataType == DT_FP16 || dataType == DT_BF16;
    if (shouldUpToFp32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = PowSCalc(castSelf, other, precisionType);
    if (shouldUpToFp32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result, dataType,
                    CastMode::CAST_NONE);
    }
    return result;
}

Tensor FloorDiv(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FLOORDIV");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FLOORDIV");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "FLOORDIV");
    CheckTensorDataType(other.GetStorage(), supportedTypes, "FLOORDIV");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "FLOORDIV");
    RETURN_CALL(BinaryOperation<BinaryOpType::FLOORDIV>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                                bool reverseOperand, int64_t precisionType)
{
    auto opNameCode = GetBinaryOpNameCode<T, true>();
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (opNameCode == Opcode::OP_BITWISEXORS || opNameCode == Opcode::OP_POWS) {
            std::vector<int64_t> tmpShape(resultTileInfo.shape);
            auto alignSize = BLOCK_SIZE / BytesOf(input1.tensor->Datatype());
            tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(tmpShape[resultTileInfo.shape.size() - 1], alignSize);
            auto tempTensor = std::make_shared<LogicalTensor>(function, input1.tensor->Datatype(), tmpShape);
            auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tempTensor});
            tmpOp.SetAttribute(OpAttributeKey::scalar, value);
            tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
            return;
        } else if (opNameCode == Opcode::OP_FLOORDIVS) {
            auto alignSize = BLOCK_SIZE / BytesOf(input1.tensor->Datatype());
            auto tmpShape = resultTileInfo.shape;
            tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(tmpShape.back(), alignSize) * 6;
            int64_t intermediateBytes = std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL,
                                                        std::multiplies<int64_t>()) *
                                        BytesOf(DT_FP32);
            auto tempTensor = std::make_shared<LogicalTensor>(function, DT_UINT8,
                                                              std::vector<int64_t>{intermediateBytes});
            auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tempTensor});
            tmpOp.SetAttribute(OpAttributeKey::scalar, value);
            tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
            return;
        }
        // 确认接口
        auto& op = function.AddOperation(opNameCode, {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW) {
            op.SetAttribute(OpAttributeKey::precisionType, precisionType);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);

        TiledBinaryOperationScalar<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo,
                                      reverseOperand, precisionType);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                Element value, const LogicalTensorPtr& result, bool reverseOperand,
                                int64_t precisionType)
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBinaryOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand,
                                  precisionType);
}

template <BinaryOpType T>
void TiledRemainderSOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                              Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                              bool reverseOperand, int64_t precisionType)
{
    auto opNameCode = GetBinaryOpNameCode<T, true>();
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        int64_t shapeSize = resultTileInfo.shape.size();
        auto alignSize = BLOCK_SIZE / BytesOf(input1.tensor->Datatype());
        std::vector<int64_t> tmpShape{2, 1};
        tmpShape[1] = AlignUp(resultTileInfo.shape[shapeSize - 1], alignSize);
        if (opNameCode == Opcode::OP_REMRS) {
            tmpShape[0] = shapeSize > 1 ? resultTileInfo.shape[shapeSize - 2] + 2 : 3;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, input1.tensor->Datatype(), tmpShape);
        auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tmpTensor});
        tmpOp.SetAttribute(OpAttributeKey::scalar, value);
        tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        tmpOp.SetAttribute(OpAttributeKey::precisionType, precisionType);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledRemainderSOperation<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand,
                                    precisionType);
    }
}

template <BinaryOpType T>
void TiledRemainderSOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1, Element value,
                              const LogicalTensorPtr& result, bool reverseOperand = false,
                              int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC))
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledRemainderSOperation<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand,
                                precisionType);
}

Tensor Add(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Add");

    static const std::unordered_set<DataType> ADD_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> ADD_A5_TYPES = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& supportedTypes = GetSupportedDataTypesByArch(ADD_A2A3_TYPES, ADD_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ADD");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Sub(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sub");

    static const std::unordered_set<DataType> SUB_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> SUB_A5_TYPES = {DT_INT32, DT_FP32,  DT_INT16, DT_FP16,
                                                              DT_BF16,  DT_UINT8, DT_INT8};
    const auto& supportedTypes = GetSupportedDataTypesByArch(SUB_A2A3_TYPES, SUB_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SUB");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Mul(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Mul");

    static const std::unordered_set<DataType> MUL_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
    static const std::unordered_set<DataType> MUL_A5_TYPES = {DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16};
    const auto& supportedTypes = GetSupportedDataTypesByArch(MUL_A2A3_TYPES, MUL_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MUL");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Div(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Div");

    static const std::unordered_set<DataType> DIV_A2A3_TYPES = {DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32};
    static const std::unordered_set<DataType> DIV_A5_TYPES = {DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32};
    const auto& supportedTypes = GetSupportedDataTypesByArch(DIV_A2A3_TYPES, DIV_A5_TYPES);
    auto isDivSupportedInt = [](DataType dt) { return dt == DT_INT16 || dt == DT_INT32; };
    auto isDivSupportedFloat = [](DataType dt) { return dt == DT_FP16 || dt == DT_FP32 || dt == DT_BF16; };
    auto isDivSupportedScalar = [&](DataType dt) { return isDivSupportedFloat(dt) || isDivSupportedInt(dt); };
    CheckTensorDataType(self.GetStorage(), supportedTypes, "DIV");

    if (isDivSupportedInt(self.GetDataType())) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, isDivSupportedInt(other.GetDataType()))
            << "Scalar dtype incorrect. When self dtype is integer (DT_INT16/DT_INT32), "
            << "scalar must be DT_INT16 or DT_INT32, actual is: " << DataType2String(other.GetDataType());
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Element castOther(DT_FP32, other.Cast<float>());
        auto [castResult, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
            *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Tensor(castResult);
    }

    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, isDivSupportedScalar(other.GetDataType()))
        << "Scalar dtype incorrect. When self dtype is float, "
        << "scalar must be float (DT_FP16/DT_FP32/DT_BF16) or integer (DT_INT16/DT_INT32), "
        << "actual is: " << DataType2String(other.GetDataType());

    if (isDivSupportedInt(other.GetDataType())) {
        // 当 self 是 fp16/bf16 且 other 是 int32 时，cast 到 fp32 避免溢出
        bool needPromoteToFp32 = (self.GetDataType() != DT_FP32) && (other.GetDataType() == DT_INT32);
        if (needPromoteToFp32) {
            Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
            Element castOther(DT_FP32, other.Cast<float>());
            auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
                *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), castOther);
            op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
            return Tensor(result);
        }
        // 其他情况：other cast 到 self.dtype
        Element castOther(self.GetDataType(), other.Cast<float>());
        auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
            *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Tensor(result);
    }

    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Fmod(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Fmod");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MOD");
    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::MOD>(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Remainder(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "REM");
    auto selfDtype = self.GetDataType();
    Element castOther = Element(selfDtype, other.Cast<float>());
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if ((!isA5Architecture && selfDtype == DT_INT16) || selfDtype == DT_FP16) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REM>(
            *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        Tensor castedResult = Cast(Tensor(result), selfDtype, CastMode::CAST_NONE, SaturationMode::ON);
        return castedResult;
    }
    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REM>(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), castOther);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Remainder(const Element& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(other.GetStorage(), supportedTypes, "REM");
    auto otherDtype = other.GetDataType();
    Element castSelf = Element(otherDtype, self.Cast<float>());
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if ((!isA5Architecture && otherDtype == DT_INT16) || otherDtype == DT_FP16) {
        Tensor castOther = Cast(other, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REMR>(
            *Program::GetInstance().GetCurrentFunction(), castOther.GetStorage(), castSelf);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        op->SetAttribute(OP_ATTR_PREFIX + "reverseOperand", true);
        Tensor castedResult = Cast(Tensor(result), otherDtype, CastMode::CAST_NONE, SaturationMode::ON);
        return castedResult;
    }
    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REMR>(
        *Program::GetInstance().GetCurrentFunction(), other.GetStorage(), castSelf);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    op->SetAttribute(OP_ATTR_PREFIX + "reverseOperand", true);
    return Tensor(result);
}

Tensor BitwiseAnd(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseAnd");

    static const std::unordered_set<DataType> BITWISE_AND_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                   DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_AND_TYPES, "BITWISEAND");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor BitwiseOr(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseOr");

    static const std::unordered_set<DataType> BITWISE_OR_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                  DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_OR_TYPES, "BITWISEOR");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEOR>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor BitwiseXor(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseXor");

    static const std::unordered_set<DataType> BITWISE_XOR_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                   DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_XOR_TYPES, "BITWISEXOR");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEXOR>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Maximum(const Tensor& operand1, const Element& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Maximum");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MAX");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MAX>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2);
}

Tensor Minimum(const Tensor& operand1, const Element& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Minimum");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MIN");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MIN>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2);
}

Tensor LReLU(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LReLU");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LRELU");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::LRELU>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Atan2(const Tensor& y, const Tensor& x)
{
    DECLARE_TRACER();
    CheckTensorFormat(y.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Atan2");
    CheckTensorFormat(x.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Atan2");

    CheckTensorsDataTypeConsistency(y.GetStorage(), x.GetStorage(), "ATAN2");
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
    CheckTensorDataType(y.GetStorage(), supportedTypes, "ATAN2");
    auto castY = y.GetStorage();
    auto castX = x.GetStorage();
    DataType dataType = y.GetDataType();
    if (dataType != DataType::DT_FP32) {
        castY = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), y.GetStorage(),
                     DataType::DT_FP32, CastMode::CAST_NONE);
        castX = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), x.GetStorage(),
                     DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto res = CALL(BinaryOperation<BinaryOpType::ATAN2>, *Program::GetInstance().GetCurrentFunction(), castY, castX);
    if (dataType != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), res, dataType,
                    CastMode::CAST_NONE);
    }
    return res;
}

Tensor CeilDiv(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CeilDiv");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CeilDiv");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "CEILDIV");
    std::unordered_set<DataType> supportedTypes = {DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CEILDIV");

    Tensor selfFp32 = Cast(self, DataType::DT_FP32);
    Tensor otherFp32 = Cast(other, DataType::DT_FP32);
    Tensor resultFp32 = Div(selfFp32, otherFp32);
    resultFp32 = Ceil(resultFp32);
    Tensor result = Cast(resultFp32, DT_INT32);
    Tensor target = Mul(result, other);
    Tensor diff = Sub(self, target);
    Tensor sgn_diff = Sign(diff);
    Tensor sgn_other = Sign(other);
    Tensor product_sgn = Mul(sgn_diff, sgn_other);
    Tensor inc = Clip(product_sgn, Element(DT_INT32, 0), Element(DT_INT32, 1));
    result = Add(result, inc);
    return result;
}

Tensor CeilDiv(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CeilDiv");

    std::unordered_set<DataType> supportedTypes = {DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CEILDIV");

    Tensor selfFp32 = Cast(self, DataType::DT_FP32);
    Element otherFp32(DT_FP32, other.Cast<float>());
    Tensor resultFp32 = Div(selfFp32, otherFp32);
    resultFp32 = Ceil(resultFp32);
    Tensor result = Cast(resultFp32, DT_INT32);
    Tensor target = Mul(result, other);
    Tensor diff = Sub(self, target);
    Tensor sgn_diff = Sign(diff);
    Element sgn_other;
    if (other.GetSignedData() > 0) {
        sgn_other = Element(DT_INT32, 1);
    } else {
        sgn_other = Element(DT_INT32, -1);
    }
    Tensor product_sgn = Mul(sgn_diff, sgn_other);
    Tensor inc = Clip(product_sgn, Element(DT_INT32, 0), Element(DT_INT32, 1));
    result = Add(result, inc);
    return result;
}

Tensor FloorDiv(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FLOORDIV");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "FLOORDIV");
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, other.GetDataType() == self.GetDataType())
        << "Scalar dtype incorrect. Scalar dtype should be same as self dtype, self dtype is: "
        << DataType2String(self.GetDataType()) << ", actual scalar dtype is: " << DataType2String(other.GetDataType());
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::FLOORDIV>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                   Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                                   bool reverseOperand)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto& op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);

        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo,
                                         reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                   Element value, const LogicalTensorPtr& result, bool reverseOperand)
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

Tensor ScalarAddS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarAddS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_ADD");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarSubS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarSubS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_SUB");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMulS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarMulS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_MUL");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarDivS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarDivS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_DIV");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMaxS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarMaxS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_MAX");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                   LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        function.AddOperation(GetBinaryOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                   LogicalTensorPtr operand2, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(operand1, operand2);

    if (operand1->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
        Expand(function, tileShape, operand1, {operand2}, tmp);
        operand1 = tmp;
    }

    if (operand2->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
        Expand(function, tileShape, operand2, {operand1}, tmp);
        operand2 = tmp;
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

Tensor ScalarAdd(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarAdd");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarAdd");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_ADD");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_ADD");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}
Tensor ScalarSub(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarSub");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarSub");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_SUB");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_SUB");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMul(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarMul");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarMul");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_MUL");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_MUL");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarDiv(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarDiv");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarDiv");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_DIV");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_DIV");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMax(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarMax");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScalarMax");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_MAX");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_MAX");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor CopySign(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CopySign");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CopySign");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "COPYSIGN");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "COPYSIGN");

    DataType selfDType = self.GetDataType();
    DataType otherDType = other.GetDataType();
    Tensor castSelf = self;
    Tensor castOther = other;
    if (selfDType == DT_INT16 || selfDType == DT_INT32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    if (otherDType == DT_INT16 || otherDType == DT_INT32) {
        castOther = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                         other.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    RETURN_CALL(BinaryOperation<BinaryOpType::COPYSIGN>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                castOther);
}

// OP_ADD OP_SUB OP_MUL OP_DIV OP_MAX OP_BITWISEAND OP_BITWISEOR OP_BITWISEXOR
template <BinaryOpType T>
void BinaryOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW ||
                  T == BinaryOpType::REM) {
        if (op.HasAttr(OpAttributeKey::precisionType)) {
            precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
        }
    }
    TiledBinaryOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0], precisionType);
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS OP_BITWISEANDS OP_BITWISEORS OP_BITWISEXORS
template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW) {
        if (op.HasAttr(OpAttributeKey::precisionType)) {
            precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
        }
    }
    TiledBinaryOperationScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                  oOperand[0], false, precisionType);
}

template <BinaryOpType T>
void BinaryOperationScalarResTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand,
                                      [[maybe_unused]] const Operation& op)
{
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD) {
        if (op.HasAttr(OpAttributeKey::precisionType)) {
            precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
        }
    }
    TiledBinaryOperationScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                  oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"), precisionType);
}

template <BinaryOpType T>
void RemainderSTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                        const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    TiledRemainderSOperation<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"), precisionType);
}

// OP_S_ADDS OP_S_SUBS OP_S_MULS OP_S_DIVS OP_S_MAXS
template <BinaryOpType T>
void BinaryOperationAllScalarResTileFunc(Function& function, const TileShape& tileShape,
                                         const std::vector<LogicalTensorPtr>& iOperand,
                                         const std::vector<LogicalTensorPtr>& oOperand,
                                         [[maybe_unused]] const Operation& op)
{
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                     oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
}

// OP_S_ADD OP_S_SUB OP_S_MUL OP_S_DIV OP_S_MAX
template <BinaryOpType T>
void BinaryOperationAllScalarTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand,
                                      [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    CheckTensorsDataTypeConsistency(iOperand[0], iOperand[1], GetBinaryOpName<T>());
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

void TiledAxpyOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& inputSelf,
                        LogicalInput& inputOther, const Element& alpha, const LogicalTensorPtr& result,
                        TileInfo& resultTileInfo)
{
    size_t shapeSize = inputSelf.tensor->GetShape().size();
    if (cur == shapeSize) {
        auto selfTile = inputSelf.tensor->View(function, inputSelf.tileInfo.shape, inputSelf.tileInfo.offset);
        auto otherTile = inputOther.tensor->View(function, inputOther.tileInfo.shape, inputOther.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto& op = function.AddOperation(Opcode::OP_AXPY, {selfTile, otherTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        std::vector<int64_t> brcOperand(shapeSize, 0);
        size_t brcAxesCount = 0;
        for (size_t i = 0; i < shapeSize; i++) {
            int brcResult = BrcAxisBinaryOp(inputSelf.tensor, inputOther.tensor, i);
            brcOperand[i] = (brcResult == 1) ? 0 : brcResult;
            if (brcOperand[i] != 0) {
                brcAxesCount++;
            }
        }
        if (brcOperand[shapeSize - 1] != 0) {
            op.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
            op.SetAttribute(OpAttributeKey::brcbIdx, brcOperand[shapeSize - 1]);
        }
        if (brcAxesCount > 0) {
            op.SetAttribute(OpAttributeKey::brcOperand, brcOperand);
        }
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        inputSelf.tileInfo.offset[cur] = i % inputSelf.tensor->GetShape()[cur];
        inputSelf.tileInfo.shape[cur] = std::min(inputSelf.tensor->GetShape()[cur] - inputSelf.tileInfo.offset[cur],
                                                 vecTile[cur]);
        inputOther.tileInfo.offset[cur] = i % inputOther.tensor->GetShape()[cur];
        inputOther.tileInfo.shape[cur] = std::min(inputOther.tensor->GetShape()[cur] - inputOther.tileInfo.offset[cur],
                                                  vecTile[cur]);
        TiledAxpyOperation(function, tileShape, cur + 1, inputSelf, inputOther, alpha, result, resultTileInfo);
    }
}

void TiledAxpyOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr self, LogicalTensorPtr other,
                        const Element& alpha, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(self, other);
    BroadcastOperandTensor(other, self, result, function, tileShape);

    TileInfo selfTileInfo(self->shape.size(), self->offset.size());
    TileInfo otherTileInfo(other->shape.size(), other->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto inputSelf = LogicalInput{self, selfTileInfo};
    auto inputOther = LogicalInput{other, otherTileInfo};

    TiledAxpyOperation(function, tileShape, 0, inputSelf, inputOther, alpha, result, resultTileInfo);
}

void AxpyOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           const Operation& op)
{
    auto alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledAxpyOperation(function, tileShape, iOperand[0], iOperand[1], alpha, oOperand[0]);
}

LogicalTensorPtr TensorAxpyOperation(Function& function, const Tensor& self, const Tensor& other, float alpha)
{
    auto selfTensor = self.GetStorage();
    auto otherTensor = other.GetStorage();

    if (selfTensor->shape.size() != otherTensor->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(selfTensor, otherTensor);
        selfTensor = BinaryOperationBroadCast(selfTensor, broadCastShape);
        otherTensor = BinaryOperationBroadCast(otherTensor, broadCastShape);
    }

    CheckTensorShapeSize(selfTensor, "AXPY");
    CheckTensorShapeSize(otherTensor, "AXPY");
    CheckBinOpOperandsValid(selfTensor, otherTensor);
    CheckTensorsFormatConsistency(selfTensor, otherTensor, "AXPY");

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(selfTensor, otherTensor);
    size_t shapeSize = resultShape.size();
    if ((!selfTensor->GetDynValidShape().empty()) && (!otherTensor->GetDynValidShape().empty())) {
        for (size_t i = 0; i < shapeSize; ++i) {
            if (resultShape[i] == selfTensor->shape[i]) {
                resultValidShape.push_back(selfTensor->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(otherTensor->GetDynValidShape()[i]);
            }
        }
    }
    // AXPY: y = alpha * x + y, y is in-place updated, cannot broadcast
    // Validate: if any dimension of y is 1 but x is not 1, it's invalid
    for (size_t i = 0; i < shapeSize; i++) {
        if ((selfTensor->shape[i] == 1) && (otherTensor->shape[i] != 1)) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "AXPY: self tensor cannot broadcast, self.shape[" << i << "]=" << selfTensor->shape[i]
                << " but other.shape[" << i << "]=" << otherTensor->shape[i];
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, selfTensor->Datatype(), resultShape, resultValidShape,
                                                  selfTensor->Format());
    auto& op = function.AddOperation(Opcode::OP_AXPY, {selfTensor, otherTensor}, {result});
    op.SetAttribute(OpAttributeKey::scalar, Element(selfTensor->Datatype(), alpha));
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);

    return result;
}

Tensor Axpy(const Tensor& self, const Tensor& other, float alpha)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Axpy");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Axpy");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "AXPY");
    CheckTensorDimRange(other.GetStorage(), 1, 4, "AXPY");

    auto selfDtype = self.GetDataType();
    auto otherDtype = other.GetDataType();
    if (selfDtype == otherDtype) {
        std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16};
        CheckTensorDataType(self.GetStorage(), supportedTypes, "AXPY");
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, selfDtype == DT_FP32 && otherDtype == DT_FP16)
            << "AXPY: when dtype mismatch, only support dst(y)=fp32 with src(x)=fp16.";
    }
    RETURN_CALL(AxpyOperation, *Program::GetInstance().GetCurrentFunction(), self, other, alpha);
}

REGISTER_OPERATION_TILED_FUNC(OP_ADD, Opcode::OP_ADD, BinaryOperationTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUB, Opcode::OP_SUB, BinaryOperationTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MUL, Opcode::OP_MUL, BinaryOperationTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIV, Opcode::OP_DIV, BinaryOperationTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, BinaryOperationTileFunc<BinaryOpType::MAXIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_MINIMUM, Opcode::OP_MINIMUM, BinaryOperationTileFunc<BinaryOpType::MINIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_POW, Opcode::OP_POW, BinaryOperationTileFunc<BinaryOpType::POW>);
REGISTER_OPERATION_TILED_FUNC(OP_MOD, Opcode::OP_MOD, BinaryOperationTileFunc<BinaryOpType::MOD>);
REGISTER_OPERATION_TILED_FUNC(OP_REM, Opcode::OP_REM, BinaryOperationTileFunc<BinaryOpType::REM>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEAND, Opcode::OP_BITWISEAND, BinaryOperationTileFunc<BinaryOpType::BITWISEAND>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEOR, Opcode::OP_BITWISEOR, BinaryOperationTileFunc<BinaryOpType::BITWISEOR>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, BinaryOperationTileFunc<BinaryOpType::BITWISEXOR>);
REGISTER_OPERATION_TILED_FUNC(OP_COPYSIGN, Opcode::OP_COPYSIGN, BinaryOperationTileFunc<BinaryOpType::COPYSIGN>);
REGISTER_OPERATION_TILED_FUNC(OP_GCD, Opcode::OP_GCD, BinaryOperationTileFunc<BinaryOpType::GCD>);
REGISTER_OPERATION_TILED_FUNC(OP_PRELU, Opcode::OP_PRELU, PReLUOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ATAN2, Opcode::OP_ATAN2, BinaryOperationTileFunc<BinaryOpType::ATAN2>);
REGISTER_OPERATION_TILED_FUNC(OP_FLOORDIV, Opcode::OP_FLOORDIV, BinaryOperationTileFunc<BinaryOpType::FLOORDIV>);
REGISTER_OPERATION_TILED_FUNC(OP_AXPY, Opcode::OP_AXPY, AxpyOperationTileFunc);

REGISTER_OPERATION_TILED_FUNC(OP_ADDS, Opcode::OP_ADDS, BinaryOperationScalarTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUBS, Opcode::OP_SUBS, BinaryOperationScalarTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MULS, Opcode::OP_MULS, BinaryOperationScalarTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIVS, Opcode::OP_DIVS, BinaryOperationScalarTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXS, Opcode::OP_MAXS, BinaryOperationScalarTileFunc<BinaryOpType::MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_MINS, Opcode::OP_MINS, BinaryOperationScalarTileFunc<BinaryOpType::MIN>);
REGISTER_OPERATION_TILED_FUNC(OP_POWS, Opcode::OP_POWS, BinaryOperationScalarTileFunc<BinaryOpType::POW>);
REGISTER_OPERATION_TILED_FUNC(OP_LRELU, Opcode::OP_LRELU, BinaryOperationScalarTileFunc<BinaryOpType::LRELU>);
REGISTER_OPERATION_TILED_FUNC(OP_MODS, Opcode::OP_MODS, BinaryOperationScalarTileFunc<BinaryOpType::MOD>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEANDS, Opcode::OP_BITWISEANDS,
                              BinaryOperationScalarTileFunc<BinaryOpType::BITWISEAND>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEORS, Opcode::OP_BITWISEORS,
                              BinaryOperationScalarTileFunc<BinaryOpType::BITWISEOR>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEXORS, Opcode::OP_BITWISEXORS,
                              BinaryOperationScalarTileFunc<BinaryOpType::BITWISEXOR>);
REGISTER_OPERATION_TILED_FUNC(OP_GCDS, Opcode::OP_GCDS, BinaryOperationScalarTileFunc<BinaryOpType::GCD>);
REGISTER_OPERATION_TILED_FUNC(OP_REMS, Opcode::OP_REMS, RemainderSTileFunc<BinaryOpType::REM>);
REGISTER_OPERATION_TILED_FUNC(OP_REMRS, Opcode::OP_REMRS, RemainderSTileFunc<BinaryOpType::REMR>);
REGISTER_OPERATION_TILED_FUNC(OP_FLOORDIVS, Opcode::OP_FLOORDIVS,
                              BinaryOperationScalarResTileFunc<BinaryOpType::FLOORDIV>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MULS, Opcode::OP_S_MULS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MINS, Opcode::OP_S_MINS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MIN>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADD, Opcode::OP_S_ADD, BinaryOperationAllScalarTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUB, Opcode::OP_S_SUB, BinaryOperationAllScalarTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MUL, Opcode::OP_S_MUL, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIV, Opcode::OP_S_DIV, BinaryOperationAllScalarTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAX, Opcode::OP_S_MAX, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MIN, Opcode::OP_S_MIN, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MIN>);

} // namespace npu::tile_fwk
