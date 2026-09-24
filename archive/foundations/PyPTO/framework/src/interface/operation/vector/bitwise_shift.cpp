/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitwise_shift.cpp
 * \brief
 */

#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

enum class BitwiseShiftOpType {
    BITWISERIGHTSHIFT,
    BITWISELEFTSHIFT,
    SBITWISERIGHTSHIFT,
    SBITWISELEFTSHIFT,
};

template <BitwiseShiftOpType T>
std::string GetBitwiseShiftOpName()
{
    switch (T) {
        case BitwiseShiftOpType::BITWISERIGHTSHIFT:
            return "BITWISERIGHTSHIFT";
        case BitwiseShiftOpType::BITWISELEFTSHIFT:
            return "BITWISELEFTSHIFT";
        case BitwiseShiftOpType::SBITWISERIGHTSHIFT:
            return "SBITWISERIGHTSHIFT";
        case BitwiseShiftOpType::SBITWISELEFTSHIFT:
            return "SBITWISELEFTSHIFT";
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown binary op type";
            return "";
    }
}

template <BitwiseShiftOpType T, bool WithElement = false>
Opcode GetBitwiseShiftOpNameCode()
{
    if constexpr (WithElement) {
#define CASE(X)                 \
    case BitwiseShiftOpType::X: \
        return Opcode::OP_##X##S
        switch (T) {
            CASE(BITWISERIGHTSHIFT);
            CASE(BITWISELEFTSHIFT);
            default:
                CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown binary op type";
        }
#undef CASE
    }

#define CASE(X)                 \
    case BitwiseShiftOpType::X: \
        return Opcode::OP_##X
    switch (T) {
        CASE(BITWISERIGHTSHIFT);
        CASE(BITWISELEFTSHIFT);
        CASE(SBITWISERIGHTSHIFT);
        CASE(SBITWISELEFTSHIFT);
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown binary op type";
    }
#undef CASE
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        Shape tmpShape = resultTileInfo.shape;
        auto alignSize = BLOCK_SIZE / static_cast<int64_t>(BytesOf(result->Datatype()));
        tmpShape[tmpShape.size() - 1] = AlignUp(tmpShape[tmpShape.size() - 1], alignSize);
        auto tmpTile = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
        function.AddOperation(GetBitwiseShiftOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile, tmpTile});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledBitwiseShiftOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                LogicalTensorPtr operand2, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(operand1, operand2);
    BroadcastOperandTensor(operand1, operand2, result, function, tileShape);
    BroadcastOperandTensor(operand2, operand1, result, function, tileShape);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    // 如果使能了Combine Axis逻辑，需要将withbrc置为false，避免后续走OP_XX_BRC逻辑
    TiledBitwiseShiftOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

template <BitwiseShiftOpType T>
LogicalTensorPtr TensorBitwiseShiftOperation(Function& function, const Tensor& self, const Tensor& other)
{
    auto operand1 = self.GetStorage();
    auto operand2 = other.GetStorage();
    if (operand1->shape.size() != operand2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operand1, operand2);
        operand1 = BinaryOperationBroadCast(operand1, broadCastShape);
        operand2 = BinaryOperationBroadCast(operand2, broadCastShape);
    }
    auto opName = GetBitwiseShiftOpName<T>();
    CheckBinaryInputTensors(operand1, operand2, opName);
    CheckTensorsDataTypeConsistency(operand1, operand2, opName);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operand1, operand2);
    if ((!operand1->GetDynValidShape().empty()) && (!operand2->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operand1->shape[i]) {
                resultValidShape.push_back(self.GetStorage()->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(other.GetStorage()->GetDynValidShape()[i]);
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, operand1->Datatype(), resultShape, resultValidShape,
                                                  operand1->Format());
    function.AddOperation(GetBitwiseShiftOpNameCode<T>(), {operand1, operand2}, {result});
    return result;
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperationScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& self,
                                      Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                                      bool reverseOperand)
{
    if (cur == self.tensor->GetShape().size()) {
        auto inputTile = self.tensor->View(function, self.tileInfo.shape, self.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(GetBitwiseShiftOpNameCode<T, true>(), {inputTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        self.tileInfo.offset[cur] = i % self.tensor->GetShape()[cur];
        self.tileInfo.shape[cur] = std::min(self.tensor->GetShape()[cur] - self.tileInfo.offset[cur], vecTile[cur]);

        TiledBitwiseShiftOperationScalar<T>(function, tileShape, cur + 1, self, value, result, resultTileInfo,
                                            reverseOperand);
    }
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperationScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand,
                                      Element value, const LogicalTensorPtr& result, bool reverseOperand = false)
{
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto self = LogicalInput{operand, tileInfo};
    TiledBitwiseShiftOperationScalar<T>(function, tileShape, 0, self, value, result, resultTileInfo, reverseOperand);
}

template <BitwiseShiftOpType T>
LogicalTensorPtr TensorBitwiseShiftOperationScalar(Function& function, const LogicalTensorPtr& self,
                                                   const Element& other)
{
    auto opName = GetBitwiseShiftOpName<T>();
    CheckTensorDimRange(self, MIN_TENSOR_DIM, MAX_TENSOR_DIM, opName);
    CheckTensorShapeSize(self, opName);
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->shape, self->GetDynValidShape());
    auto& op = function.AddOperation(GetBitwiseShiftOpNameCode<T, true>(), {self}, {result});
    op.SetAttribute(OpAttributeKey::scalar, other);
    return result;
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperationSelfScalar(Function& function, const TileShape& tileShape, size_t cur, Element& value,
                                          LogicalInput& other, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                                          bool reverseOperand)
{
    if (cur == other.tensor->GetShape().size()) {
        auto inputTile = other.tensor->View(function, other.tileInfo.shape, other.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        Shape tmpShape = resultTileInfo.shape;
        auto alignSize = BLOCK_SIZE / static_cast<int64_t>(BytesOf(result->Datatype()));
        tmpShape[tmpShape.size() - 1] = AlignUp(tmpShape[tmpShape.size() - 1], alignSize);
        auto tmpTile = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);

        auto& op = function.AddOperation(GetBitwiseShiftOpNameCode<T>(), {inputTile}, {resultTile, tmpTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        other.tileInfo.offset[cur] = i % other.tensor->GetShape()[cur];
        other.tileInfo.shape[cur] = std::min(other.tensor->GetShape()[cur] - other.tileInfo.offset[cur], vecTile[cur]);

        TiledBitwiseShiftOperationSelfScalar<T>(function, tileShape, cur + 1, value, other, result, resultTileInfo,
                                                reverseOperand);
    }
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperationSelfScalar(Function& function, const TileShape& tileShape, Element value,
                                          LogicalTensorPtr operand, const LogicalTensorPtr& result,
                                          bool reverseOperand = false)
{
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto other = LogicalInput{operand, tileInfo};
    TiledBitwiseShiftOperationSelfScalar<T>(function, tileShape, 0, value, other, result, resultTileInfo,
                                            reverseOperand);
}

template <BitwiseShiftOpType T>
LogicalTensorPtr TensorBitwiseShiftOperationSelfScalar(Function& function, const Element& self,
                                                       const LogicalTensorPtr& other)
{
    auto opName = GetBitwiseShiftOpName<T>();
    CheckTensorDimRange(other, MIN_TENSOR_DIM, MAX_TENSOR_DIM, opName);
    CheckTensorShapeSize(other, opName);
    auto result = std::make_shared<LogicalTensor>(function, other->Datatype(), other->shape, other->GetDynValidShape());
    auto& op = function.AddOperation(GetBitwiseShiftOpNameCode<T>(), {other}, {result});
    op.SetAttribute(OpAttributeKey::scalar, self);
    return result;
}

namespace {
// TODO i8/u8/i32/u32: pending pto-isa additions on a2a3.
const std::unordered_set<DataType> BITWISESHIFT_A2A3_TYPES = {DT_INT16, DT_UINT16};
const std::unordered_set<DataType> BITWISESHIFT_A5_TYPES = {DT_INT16,  DT_UINT16, DT_INT32,
                                                            DT_UINT32, DT_INT8,   DT_UINT8};
} // namespace

Tensor BitwiseRightShift(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseRightShift");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseRightShift");

    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISESHIFT_A2A3_TYPES, BITWISESHIFT_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BitwiseRightShift");
    RETURN_CALL(BitwiseShiftOperation<BitwiseShiftOpType::BITWISERIGHTSHIFT>,
                *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseRightShift(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseRightShift");

    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISESHIFT_A2A3_TYPES, BITWISESHIFT_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BitwiseRightShift");
    Element newOther = other;
    if (self.GetDataType() != other.GetDataType()) {
        newOther = Element(self.GetDataType(), other.Cast<int32_t>());
    }
    RETURN_CALL(BitwiseShiftOperationScalar<BitwiseShiftOpType::BITWISERIGHTSHIFT>,
                *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), newOther);
}

Tensor BitwiseRightShift(const Element& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseRightShift");

    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISESHIFT_A2A3_TYPES, BITWISESHIFT_A5_TYPES);
    CheckTensorDataType(other.GetStorage(), supportedTypes, "BitwiseRightShift");
    Element newSelf = self;
    if (self.GetDataType() != other.GetDataType()) {
        newSelf = Element(other.GetDataType(), self.Cast<int32_t>());
    }
    RETURN_CALL(BitwiseShiftOperationSelfScalar<BitwiseShiftOpType::SBITWISERIGHTSHIFT>,
                *Program::GetInstance().GetCurrentFunction(), newSelf, other.GetStorage());
}
Tensor BitwiseLeftShift(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseLeftShift");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseLeftShift");

    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISESHIFT_A2A3_TYPES, BITWISESHIFT_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BitwiseLeftShift");
    RETURN_CALL(BitwiseShiftOperation<BitwiseShiftOpType::BITWISELEFTSHIFT>,
                *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseLeftShift(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseLeftShift");

    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISESHIFT_A2A3_TYPES, BITWISESHIFT_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BitwiseLeftShift");
    Element newOther = other;
    if (self.GetDataType() != other.GetDataType()) {
        newOther = Element(self.GetDataType(), other.Cast<int32_t>());
    }
    RETURN_CALL(BitwiseShiftOperationScalar<BitwiseShiftOpType::BITWISELEFTSHIFT>,
                *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), newOther);
}

Tensor BitwiseLeftShift(const Element& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseLeftShift");

    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISESHIFT_A2A3_TYPES, BITWISESHIFT_A5_TYPES);
    CheckTensorDataType(other.GetStorage(), supportedTypes, "BitwiseLeftShift");
    Element newSelf = self;
    if (self.GetDataType() != other.GetDataType()) {
        newSelf = Element(other.GetDataType(), self.Cast<int32_t>());
    }
    RETURN_CALL(BitwiseShiftOperationSelfScalar<BitwiseShiftOpType::SBITWISELEFTSHIFT>,
                *Program::GetInstance().GetCurrentFunction(), newSelf, other.GetStorage());
}

template <BitwiseShiftOpType T>
void BitwiseShiftOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBitwiseShiftOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

template <BitwiseShiftOpType T>
void BitwiseShiftOperationScalarTileFunc(Function& function, const TileShape& tileShape,
                                         const std::vector<LogicalTensorPtr>& iOperand,
                                         const std::vector<LogicalTensorPtr>& oOperand,
                                         [[maybe_unused]] const Operation& op)
{
    TiledBitwiseShiftOperationScalar<T>(function, tileShape, iOperand[0],
                                        op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

template <BitwiseShiftOpType T>
void BitwiseShiftOperationSelfScalarTileFunc(Function& function, const TileShape& tileShape,
                                             const std::vector<LogicalTensorPtr>& iOperand,
                                             const std::vector<LogicalTensorPtr>& oOperand,
                                             [[maybe_unused]] const Operation& op)
{
    TiledBitwiseShiftOperationSelfScalar<T>(function, tileShape, op.GetElementAttribute(OpAttributeKey::scalar),
                                            iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT,
                              BitwiseShiftOperationTileFunc<BitwiseShiftOpType::BITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT,
                              BitwiseShiftOperationTileFunc<BitwiseShiftOpType::BITWISELEFTSHIFT>);

REGISTER_OPERATION_TILED_FUNC(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS,
                              BitwiseShiftOperationScalarTileFunc<BitwiseShiftOpType::BITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS,
                              BitwiseShiftOperationScalarTileFunc<BitwiseShiftOpType::BITWISELEFTSHIFT>);

REGISTER_OPERATION_TILED_FUNC(OP_SBITWISERIGHTSHIFT, Opcode::OP_SBITWISERIGHTSHIFT,
                              BitwiseShiftOperationSelfScalarTileFunc<BitwiseShiftOpType::SBITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_SBITWISELEFTSHIFT, Opcode::OP_SBITWISELEFTSHIFT,
                              BitwiseShiftOperationSelfScalarTileFunc<BitwiseShiftOpType::SBITWISELEFTSHIFT>);
} // namespace npu::tile_fwk
