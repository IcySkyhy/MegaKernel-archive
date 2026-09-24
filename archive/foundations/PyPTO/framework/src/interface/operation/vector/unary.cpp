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
 * \file unary.cpp
 * \brief
 */

#include "binary.h"
#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void UnaryOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                const std::vector<LogicalTensorPtr>& oOperand)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "The input operand size should be 1";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "The output operand size should be 1";
}

template <UnaryOpType T>
void TiledUnaryOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                         const LogicalTensorPtr& result, uint32_t workspaceSize = 0, int64_t precisionType = 0)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_UINT8,
                                                                         std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile, workspace});
        }
        if (T == UnaryOpType::EXP || T == UnaryOpType::SQRT || T == UnaryOpType::LN || T == UnaryOpType::RECIPROCAL) {
            op->SetAttribute(OpAttributeKey::precisionType, precisionType);
        }
        if (T == UnaryOpType::ASIN || T == UnaryOpType::ACOS || T == UnaryOpType::SINH || T == UnaryOpType::ERF ||
            T == UnaryOpType::ASINH || T == UnaryOpType::ATANH || T == UnaryOpType::ISNAN) {
            std::vector<bool> dimMap({true});
            op->SetAttr(OpAttributeKey::rowPad, dimMap);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledUnaryOperation<T>(function, tileShape, cur + 1, input, result, workspaceSize, precisionType);
    }
}

template <UnaryOpType T>
void TiledUnaryOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                         const LogicalTensorPtr& result, int32_t workspaceSize = 0, int64_t precisionType = 0)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result, workspaceSize, precisionType);
}

Tensor Exp(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Exp");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Exp");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::EXP>(*Program::GetInstance().GetCurrentFunction(),
                                                                     self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Ln(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Ln");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "Ln");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::LN>(*Program::GetInstance().GetCurrentFunction(),
                                                                    operand.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor IsFinite(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IsFinite");

    std::unordered_set<DataType> supportedTypes = {DT_FP16,  DT_FP32,   DT_BF16,   DT_INT16, DT_INT4,   DT_INT8,
                                                   DT_INT32, DT_UINT16, DT_UINT32, DT_UINT8, DT_UINT64, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "IsFinite");
    RETURN_CALL(UnaryOperation<UnaryOpType::ISFINITE>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                DT_BOOL);
}

Tensor IsNan(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IsNan");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ISNAN");

    RETURN_CALL(UnaryOperation<UnaryOpType::ISNAN>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                DT_BOOL);
}

Tensor Atan(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Atan");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Atan");
    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto res = CALL(UnaryOperation<UnaryOpType::ATAN>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), res,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return res;
}

Tensor Rsqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Rsqrt");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Rsqrt");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto [sqrtResult, sqrtOp] = TensorUnaryOperationWithOp<UnaryOpType::SQRT>(
        *Program::GetInstance().GetCurrentFunction(), castSelf);
    sqrtOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    auto ones = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
                     SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto [divResult, divOp] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(
        *Program::GetInstance().GetCurrentFunction(), ones, sqrtResult);
    divOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), divResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return divResult;
}

Tensor Sqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sqrt");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sqrt");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::SQRT>(*Program::GetInstance().GetCurrentFunction(),
                                                                      self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Relu(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Relu");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Relu");
    RETURN_CALL(UnaryOperation<UnaryOpType::RELU>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Ceil(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Ceil");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Ceil");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto ceilResult = CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), ceilResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return ceilResult;
}

Tensor Floor(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Floor");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Floor");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto floorResult = CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), floorResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return floorResult;
}

Tensor Trunc(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Trunc");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Trunc");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto truncResult = CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), truncResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return truncResult;
}

Tensor BitwiseNot(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseNot");

    if (self.GetDataType() == DT_BOOL) {
        return LogicalNot(self);
    }
    static const std::unordered_set<DataType> BITWISE_NOT_TYPES = {DT_INT8,   DT_UINT8, DT_INT16,
                                                                   DT_UINT16, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), BITWISE_NOT_TYPES, "BitwiseNot");
    RETURN_CALL(UnaryOperation<UnaryOpType::BITWISENOT>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage());
}

Tensor Reciprocal(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Reciprocal");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "Reciprocal");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::RECIPROCAL>(
        *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Abs(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Abs");

    static const std::unordered_set<DataType> ABS_A2A3_TYPES = {DT_FP16, DT_BF16, DT_FP32};
    static const std::unordered_set<DataType> ABS_A5_TYPES = {DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32};
    const auto& supportedTypes = GetSupportedDataTypesByArch(ABS_A2A3_TYPES, ABS_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Abs");
    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Hub(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Hub");

    RETURN_CALL(UnaryOperation<UnaryOpType::HUB>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Duplicate(const Tensor& operand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Duplicate");

    RETURN_CALL(UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage());
}

Tensor Sinh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sinh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SINH");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SINH");
    CheckTensorShapeSize(self.GetStorage(), "SINH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::SINH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Cosh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Cosh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "COSH");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "COSH");
    CheckTensorShapeSize(self.GetStorage(), "COSH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::COSH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Atanh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Atanh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ATANH");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ATANH");
    CheckTensorShapeSize(self.GetStorage(), "ATANH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ATANH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Erf(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Erf");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Erf");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Erf");
    CheckTensorShapeSize(self.GetStorage(), "Erf");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ERF>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Pack(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pack");

    std::unordered_set<DataType> supportedTypes = {DT_BF16,  DT_FP16,   DT_FP32,  DT_UINT8,  DT_INT8, DT_UINT16,
                                                   DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Pack");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Pack");
    CheckTensorShapeSize(self.GetStorage(), "Pack");
    std::vector<int64_t> tmpShape;
    auto selfShape = self.GetShape();
    int64_t dim = 1;
    for (size_t i = 0; i < selfShape.size(); ++i) {
        dim *= selfShape[i];
    }
    tmpShape.emplace_back(dim);
    auto tmp = Reshape(self, tmpShape).GetStorage();
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int64_t byte = BytesOf(self.GetDataType());
    tmpShape[0] *= byte;
    std::vector<SymbolicScalar> tmpDynValidShape{tmp->GetDynValidShape()[0] * byte};
    auto result = std::make_shared<LogicalTensor>(function, DT_UINT8, tmpShape, tmpDynValidShape, tmp->Format());
    [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_PACK, {tmp}, {result});
    return result;
}

Tensor UnPack(const Tensor& self, DataType dstDataType)
{
    DECLARE_TRACER();

    std::unordered_set<DataType> supportedSrcTypes = {DT_UINT8};
    std::unordered_set<DataType> supportedDstTypes = {DT_FP32,  DT_FP16,  DT_BF16,   DT_INT8,   DT_INT16, DT_INT32,
                                                      DT_INT64, DT_UINT8, DT_UINT64, DT_UINT16, DT_UINT32};
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "UnPack");
    CheckTensorDataType(self.GetStorage(), supportedSrcTypes, "UnPack");
    CheckTensorDimRange(self.GetStorage(), 1, 1, "UnPack");
    CheckTensorShapeSize(self.GetStorage(), "UnPack");
    CheckTensorDataType(dstDataType, supportedDstTypes, "UnPack");
    int64_t dstByte = BytesOf(dstDataType);
    auto srcShape = self.GetShape();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dstByte != 0 && srcShape[0] * BytesOf(DT_UINT8) % dstByte == 0)
        << "Cannot unpack tile to " << DataType2String(dstDataType, true) << ": total bytes not divisible by "
        << dstByte << " bytes";

    std::vector<int64_t> resultShape;
    resultShape.emplace_back(srcShape[0] / dstByte);
    const auto& srcStorage = self.GetStorage();
    std::vector<SymbolicScalar> dstDynValidShape{srcStorage->GetDynValidShape()[0] / dstByte};
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto result = std::make_shared<LogicalTensor>(function, dstDataType, resultShape, dstDynValidShape,
                                                  srcStorage->Format());

    [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_UNPACK, {self.GetStorage()}, {result});
    return result;
}

Tensor Sin(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sin");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sin");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Sin");
    CheckTensorShapeSize(self.GetStorage(), "Sin");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::SIN>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Cos(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Cos");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Cos");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Cos");
    CheckTensorShapeSize(self.GetStorage(), "Cos");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::COS>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Erfc(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Erfc");

    std::unordered_set<DataType> supportedTypes = {DT_BF16, DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Erfc");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Erfc");
    CheckTensorShapeSize(self.GetStorage(), "Erfc");
    if (self.GetDataType() != DataType::DT_FP32) {
        auto castSelf = Cast(self, DataType::DT_FP32);
        auto result = CALL(UnaryOperation<UnaryOpType::ERFC>, *Program::GetInstance().GetCurrentFunction(),
                           castSelf.GetStorage());
        auto castResult = Cast(result, self.GetDataType());
        return castResult;
    }
    auto result = CALL(UnaryOperation<UnaryOpType::ERFC>, *Program::GetInstance().GetCurrentFunction(),
                       self.GetStorage());
    return result;
}

Tensor Asin(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Asin");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ASIN");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ASIN");
    CheckTensorShapeSize(self.GetStorage(), "ASIN");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ASIN>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Acos(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Acos");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ACOS");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ACOS");
    CheckTensorShapeSize(self.GetStorage(), "ACOS");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ACOS>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor ASinh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ASinh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ASinh");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ASinh");
    CheckTensorShapeSize(self.GetStorage(), "ASinh");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ASINH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor ACosh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ACosh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ACosh");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ACosh");
    CheckTensorShapeSize(self.GetStorage(), "ACosh");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ACOSH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

void ExpOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::EXP>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void RsqrtOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReluOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RELU>(function, tileShape, iOperand[0], oOperand[0]);
}

void AtanOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    auto tmpShape = shape.tile;
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[shape.size() - 1] = AlignUp(tmpShape[shape.size() - 1], alignSize) * NUM3;
    auto workspace = BytesOf(DT_FP32) *
                     std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());
    return TiledUnaryOperation<UnaryOpType::ATAN>(function, tileShape, iOperand[0], oOperand[0], workspace);
}

void CeilOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::CEIL>(function, tileShape, iOperand[0], oOperand[0]);
}

void FloorOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::FLOOR>(function, tileShape, iOperand[0], oOperand[0]);
}

void TruncOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::TRUNC>(function, tileShape, iOperand[0], oOperand[0]);
}

void SqrtOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void BitwiseNotOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::BITWISENOT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReciprocalOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::RECIPROCAL>(function, tileShape, iOperand[0], oOperand[0], 0,
                                                        precisionType);
}

void AbsOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
}

void LnOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                         const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::LN>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void IsFiniteOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    // tileShape 对应的中间变量结果，类型为 FP16
    uint32_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP16)) *
                                 std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    uint32_t workspaceSize = intermediateBytes;
    return TiledUnaryOperation<UnaryOpType::ISFINITE>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void IsNanOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto tmpShape = tileShape.GetVecTile().tile;
    int dim = static_cast<int>(tmpShape.size());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP16);
    int64_t tmpW = AlignUp(tmpShape[dim - 1], alignSize);
    int64_t tmpH = (dim >= 2) ? tmpShape[dim - 2] : 1;

    constexpr int64_t kNumBlocks = 3;
    int64_t blockBytes = tmpH * tmpW * BytesOf(DT_FP32);
    uint32_t workspaceSize = kNumBlocks * blockBytes + BLOCK_SIZE;
    return TiledUnaryOperation<UnaryOpType::ISNAN>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void HubOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::HUB>(function, tileShape, iOperand[0], oOperand[0]);
}

template <UnaryOpType T, int64_t TmpBlockNum>
void Fp32AlignedTmpUnaryOperationTileFunc(Function& function, const TileShape& tileShape,
                                          const std::vector<LogicalTensorPtr>& iOperand,
                                          const std::vector<LogicalTensorPtr>& oOperand,
                                          [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    std::vector<int64_t> tmpShape = shape.tile;
    tmpShape[dim - 1] = AlignUp(tmpShape[dim - 1], alignSize) * TmpBlockNum;
    uint64_t intermediateBytes = std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>()) *
                                 BytesOf(DT_FP32);
    return TiledUnaryOperation<T>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void AtanhOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    std::vector<int64_t> tmpShape = shape.tile;
    tmpShape[dim - 1] = AlignUp(tmpShape[dim - 1], alignSize) * NUM_VALUE_5;
    uint64_t intermediateBytes = std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>()) *
                                 BytesOf(DT_FP32);
    return TiledUnaryOperation<UnaryOpType::ATANH>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void ErfOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * 3 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::ERF>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void PackOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto vecTile = tileShape.GetVecTile();
    int64_t byte = BytesOf(iOperand[0]->Datatype());
    int64_t tile = 1;
    for (size_t i = 0; i < vecTile.size(); ++i) {
        tile *= vecTile[i];
    }
    for (int64_t i = 0; i < iOperand[0]->shape[0]; i += tile) {
        std::vector<int64_t> inputShape{std::min(iOperand[0]->shape[0] - i, tile)};
        std::vector<int64_t> outputShape{inputShape[0] * byte};
        std::vector<int64_t> inputOffset{i};
        std::vector<int64_t> outputOffset{i * byte};
        auto inputTile = iOperand[0]->View(function, inputShape, inputOffset);
        auto resultTile = oOperand[0]->View(function, outputShape, outputOffset);
        [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_PACK, {inputTile}, {resultTile});
    }
}

void UnPackOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() != 0) << "UnPack tile shape invalid: vecTile is empty";
    auto dstByte = BytesOf(oOperand[0]->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dstByte != 0 && vecTile[0] % dstByte == 0)
        << "UnPack tile shape invalid: vecTile[0]=" << vecTile[0] << " must be divisible by dstByte=" << dstByte;

    for (int64_t i = 0; i < iOperand[0]->shape[0]; i += vecTile[0]) {
        std::vector<int64_t> inputShape{std::min(iOperand[0]->shape[0] - i, vecTile[0])};
        std::vector<int64_t> offset{i};
        std::vector<int64_t> outputShape{static_cast<int64_t>(inputShape[0] / dstByte)};
        std::vector<int64_t> outputOffset{static_cast<int64_t>(i / dstByte)};
        auto inputTile = iOperand[0]->View(function, inputShape, offset);
        auto resultTile = oOperand[0]->View(function, outputShape, outputOffset);
        [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_UNPACK, {inputTile}, {resultTile});
    }
}

void SinOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * 3 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::SIN>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void CosOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * 3 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::COS>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void ErfcOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * 4 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());
    return TiledUnaryOperation<UnaryOpType::ERFC>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

template <UnaryOpType T>
void AsinAcosOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    int dim = static_cast<int>(shape.size());
    auto repeatElems = REPEAT_BYTE / BytesOf(DT_FP32);
    int64_t tmpW = (shape[dim - 1] + repeatElems - 1) / repeatElems * repeatElems;
    int64_t tmpH = (dim >= 2) ? shape[dim - 2] : 1;
    // 3 H x W tile and 1 mask tile
    uint64_t intermediateBytes = static_cast<uint64_t>(BytesOf(DT_FP32)) * 4 * tmpH * tmpW;
    return TiledUnaryOperation<T>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

REGISTER_OPERATION_TILED_FUNC(OP_EXP, Opcode::OP_EXP, ExpOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RSQRT, Opcode::OP_RSQRT, RsqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RELU, Opcode::OP_RELU, ReluOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ATAN, Opcode::OP_ATAN, AtanOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SQRT, Opcode::OP_SQRT, SqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CEIL, Opcode::OP_CEIL, CeilOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FLOOR, Opcode::OP_FLOOR, FloorOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRUNC, Opcode::OP_TRUNC, TruncOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISENOT, Opcode::OP_BITWISENOT, BitwiseNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ReciprocalOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ABS, Opcode::OP_ABS, AbsOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LN, Opcode::OP_LN, LnOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ISFINITE, Opcode::OP_ISFINITE, IsFiniteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ISNAN, Opcode::OP_ISNAN, IsNanOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_HUB, Opcode::OP_HUB, HubOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SINH, Opcode::OP_SINH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::SINH, NUM_VALUE_4>));
REGISTER_OPERATION_TILED_FUNC(OP_COSH, Opcode::OP_COSH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::COSH, NUM_VALUE_3>));
REGISTER_OPERATION_TILED_FUNC(OP_ATANH, Opcode::OP_ATANH, AtanhOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ERF, Opcode::OP_ERF, ErfOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_PACK, Opcode::OP_PACK, PackOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_UNPACK, Opcode::OP_UNPACK, UnPackOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SIN, Opcode::OP_SIN, SinOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_COS, Opcode::OP_COS, CosOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ERFC, Opcode::OP_ERFC, ErfcOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ASIN, Opcode::OP_ASIN, AsinAcosOperationTileFunc<UnaryOpType::ASIN>);
REGISTER_OPERATION_TILED_FUNC(OP_ACOS, Opcode::OP_ACOS, AsinAcosOperationTileFunc<UnaryOpType::ACOS>);
REGISTER_OPERATION_TILED_FUNC(OP_ASINH, Opcode::OP_ASINH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::ASINH, NUM_VALUE_4>));
REGISTER_OPERATION_TILED_FUNC(OP_ACOSH, Opcode::OP_ACOSH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::ACOSH, NUM_VALUE_3>));
} // namespace npu::tile_fwk
