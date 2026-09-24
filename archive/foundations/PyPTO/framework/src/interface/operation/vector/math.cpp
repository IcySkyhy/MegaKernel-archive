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
 * \file math.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void TiledLogicalNotOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                              const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        constexpr int64_t COUNT_NUM = 2048;
        constexpr int64_t vcmp_bit_size = COUNT_NUM / 8;
        constexpr size_t ALIGN_SIZE = 32;

        DataType select_dtype;
        if (input.tensor.GetDataType() == DT_FP32 || input.tensor.GetDataType() == DT_BF16) {
            select_dtype = DT_FP32;
        } else if (input.tensor.GetDataType() == DT_INT16 || input.tensor.GetDataType() == DT_UINT16) {
            select_dtype = DT_INT16;
        } else if (input.tensor.GetDataType() == DT_INT32 || input.tensor.GetDataType() == DT_UINT32) {
            select_dtype = DT_INT32;
        } else if (input.tensor.GetDataType() == DT_INT64 || input.tensor.GetDataType() == DT_UINT64) {
            select_dtype = DT_INT64;
        } else {
            select_dtype = DT_FP16;
        }

        int64_t total_size;
        if (input.tensor.GetDataType() == DT_INT16 || input.tensor.GetDataType() == DT_UINT16 ||
            input.tensor.GetDataType() == DT_INT32 || input.tensor.GetDataType() == DT_UINT32 ||
            input.tensor.GetDataType() == DT_INT64 || input.tensor.GetDataType() == DT_UINT64) {
            // New integer path: one buffer of 2048 elements at compute width
            // For int64, two-step gather reuses the same buffer
            total_size = COUNT_NUM * BytesOf(select_dtype);
        } else {
            // Existing path: vcmpBitResult + compareCondition + oneCondition + castCondition + startAddrUB
            total_size = COUNT_NUM * 2 + COUNT_NUM * BytesOf(select_dtype) * 2 + vcmp_bit_size + 8;
        }
        total_size = (total_size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        std::vector<int64_t> tmpShape({total_size});

        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT8, tmpShape);
        auto& op = function.AddOperation(Opcode::OP_LOGICALNOT, {tile}, {resultTile, tmpTensor});
        if (input.tensor.GetDataType() == DT_FP32 || input.tensor.GetDataType() == DT_BF16 ||
            input.tensor.GetDataType() == DT_FP16) {
            std::vector<bool> dimMap({true});
            op.SetAttr(OpAttributeKey::rowPad, dimMap);
        }
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLogicalNotOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLogicalNotOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                              const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledLogicalNotOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorLogicalNotOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOGICALNOT, {self}, {result});
    return result;
}

Tensor LogicalNot(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LogicalNot");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16,  DT_UINT8, DT_INT8,   DT_BOOL,
                                                   DT_BF16, DT_INT16, DT_INT32, DT_UINT16, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOGICALNOT");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "LOGICALNOT");
    CheckTensorShapeSize(self.GetStorage(), "LOGICALNOT");
    RETURN_CALL(LogicalNotOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

template <typename T>
int64_t MultiplyLastTwoDims(const std::vector<int64_t>& vec)
{
    constexpr size_t ALIGN_SIZE = 32;
    constexpr size_t ELEMENT_SIZE = sizeof(T);
    constexpr size_t ALIGN_ELEMENTS = ALIGN_SIZE / ELEMENT_SIZE;
    int64_t axis2 = (vec[vec.size() - 1] + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
    return axis2 * vec[vec.size() - 2];
}

void TiledSignOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                        const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        constexpr size_t ALIGN_SIZE = 32;
        int64_t tmpSize = ALIGN_SIZE / BytesOf(DT_FP16);
        if (input.tensor.GetDataType() == DT_INT8) {
            tmpSize = MultiplyLastTwoDims<float16>(input.tileInfo.shape);
        } else if (input.tensor.GetDataType() == DT_FP16 || input.tensor.GetDataType() == DT_FP32) {
            const int64_t typeSize = BytesOf(input.tensor.GetDataType());
            const int64_t alignElements = ALIGN_SIZE / typeSize;
            const auto& shape = input.tileInfo.shape;
            const int64_t tmpH = shape.size() > 1 ? shape[shape.size() - 2] : 1;
            const int64_t tmpW = (shape.back() + alignElements - 1) / alignElements * alignElements;
            // One work tile, one mask tile with the same byte footprint, and one scalar block.
            const int64_t tmpBytes = 2 * tmpH * tmpW * typeSize + ALIGN_SIZE;
            tmpSize = tmpBytes / BytesOf(DT_FP16);
        }

        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP16, tmpShape);
        function.AddOperation(Opcode::OP_SIGN, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledSignOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledSignOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                        const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledSignOperation(function, tileShape, 0, input, result);
}

void TiledSignbitOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        int64_t tmpSize = MultiplyLastTwoDims<float16>(input.tileInfo.shape);
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP16, tmpShape);
        function.AddOperation(Opcode::OP_SIGNBIT, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledSignbitOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledSignbitOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                           const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledSignbitOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorSignOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->tensor->datatype, self->shape,
                                                  self->GetDynValidShape());
    function.AddOperation(Opcode::OP_SIGN, {self}, {result});
    return result;
}

LogicalTensorPtr TensorSignbitOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_BOOL, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_SIGNBIT, {self}, {result});
    return result;
}

Tensor Sign(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sign");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32, DT_INT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SIGN");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SIGN");
    CheckTensorShapeSize(self.GetStorage(), "SIGN");
    RETURN_CALL(SignOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Signbit(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Signbit");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32, DT_INT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SIGNBIT");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SIGNBIT");
    CheckTensorShapeSize(self.GetStorage(), "SIGNBIT");
    RETURN_CALL(SignbitOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

int64_t CmpResAlign(const std::vector<int64_t>& vec)
{
    constexpr size_t ALIGN_SIZE = 32;
    constexpr size_t ALIGN_BIT = 8;
    int64_t axis2 = (vec[vec.size() - 1] + ALIGN_BIT - 1) / ALIGN_BIT * ALIGN_BIT;
    axis2 = (axis2 + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    return axis2 * vec[vec.size() - 2];
}

void TiledTanhOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                        const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        constexpr size_t ALIGN_SIZE = 32;
        int64_t tmpSize = MultiplyLastTwoDims<float>(tileShape.GetVecTile().tile);
        int64_t cmpsize = CmpResAlign(tileShape.GetVecTile().tile);
        if (input.tensor.GetDataType() != DT_FP32) {
            tmpSize = 4 * tmpSize * sizeof(float);
        } else {
            tmpSize = 2 * tmpSize * sizeof(float);
        }
        tmpSize = tmpSize + cmpsize + ALIGN_SIZE;

        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT8, tmpShape);
        function.AddOperation(Opcode::OP_TANH, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTanhOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledTanhOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                        const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledTanhOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorTanhOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->tensor->datatype, self->shape,
                                                  self->GetDynValidShape());
    function.AddOperation(Opcode::OP_TANH, {self}, {result});
    return result;
}

Tensor Tanh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Tanh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "TANH");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "TANH");
    CheckTensorShapeSize(self.GetStorage(), "TANH");

    RETURN_CALL(TanhOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Neg(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Neg");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "NEG");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "NEG");
    CheckTensorShapeSize(self.GetStorage(), "NEG");

    if (IsFloat(self.GetStorage()->Datatype())) {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage(), Element(self.GetStorage()->Datatype(), -1.0));
    } else {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage(), Element(self.GetStorage()->Datatype(), -1));
    }
}

Tensor Log(const Tensor& self, LogBaseType base, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Log");

    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          base == LogBaseType::LOG_E || base == LogBaseType::LOG_2 || base == LogBaseType::LOG_10)
        << "base is incorrect";
    std::unordered_set<DataType> supportedTypes = {DT_BF16, DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOG");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "LOG");
    CheckTensorShapeSize(self.GetStorage(), "LOG");

    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (self.GetStorage()->Datatype() == DataType::DT_FP16 || self.GetStorage()->Datatype() == DataType::DT_BF16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }

    auto resTensor = Tensor(DataType::DT_FP32, self.GetShape());
    resTensor = Ln(operandCast, precisionType);

    auto resTensorBeforeCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (base == LogBaseType::LOG_2) {
        resTensorBeforeCast = CALL(BinaryOperationScalar<BinaryOpType::DIV>,
                                   *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(),
                                   Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_2))));
    } else if (base == LogBaseType::LOG_10) {
        resTensorBeforeCast = CALL(BinaryOperationScalar<BinaryOpType::DIV>,
                                   *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(),
                                   Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_10))));
    } else {
        resTensorBeforeCast = resTensor;
    }

    if (self.GetStorage()->Datatype() == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                    resTensorBeforeCast.GetStorage(), DataType::DT_FP16, CastMode::CAST_NONE);
    } else if (self.GetStorage()->Datatype() == DataType::DT_BF16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                    resTensorBeforeCast.GetStorage(), DataType::DT_BF16, CastMode::CAST_NONE);
    }
    return resTensorBeforeCast;
}

Tensor Log1p(const Tensor& self)
{
    DECLARE_TRACER();
    auto& function = *Program::GetInstance().GetCurrentFunction();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Log1p");

    std::unordered_set<DataType> supportedTypes = {DT_BF16, DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOG1P");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "LOG1P");
    CheckTensorShapeSize(self.GetStorage(), "LOG1P");

    auto operandCast = self.GetStorage();
    if (self.GetStorage()->Datatype() == DataType::DT_FP16 || self.GetStorage()->Datatype() == DataType::DT_BF16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto resTensorBeforeCast = std::make_shared<LogicalTensor>(*Program::GetInstance().GetCurrentFunction(),
                                                               DataType::DT_FP32, self.GetShape(),
                                                               self.GetStorage()->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOG1P, {operandCast}, {resTensorBeforeCast});

    if (self.GetStorage()->Datatype() == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), resTensorBeforeCast,
                    DataType::DT_FP16, CastMode::CAST_NONE);
    } else if (self.GetStorage()->Datatype() == DataType::DT_BF16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), resTensorBeforeCast,
                    DataType::DT_BF16, CastMode::CAST_NONE);
    }
    return Tensor(resTensorBeforeCast);
}

void TiledLog1pOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                         const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        const auto& vecTile = tileShape.GetVecTile().tile;
        auto tileShapeLen = vecTile.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only support 1~4";

        // The tileop keeps four scratch blocks (tmp0/tmp1/tmp2/mask), each the size of the last
        // two tile axes (tileH * tileW). Align the last axis to 32 bytes; treat 1-D tiles as tileH == 1.
        constexpr size_t ALIGN_SIZE = 32;
        auto alignElems = ALIGN_SIZE / BytesOf(DT_FP32);
        int64_t alignedW = AlignUp(vecTile[tileShapeLen - 1], alignElems);
        int64_t tileH = (tileShapeLen >= SHAPE_DIM2) ? vecTile[tileShapeLen - 2] : 1;
        int64_t tmpSize = NUM_VALUE_4 * alignedW * tileH;
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        function.AddOperation(Opcode::OP_LOG1P, {tile}, {resultTile, tmpTensor});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLog1pOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLog1pOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                         const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledLog1pOperation(function, tileShape, 0, input, result);
}

void Log1pOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    TiledLog1pOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void TiledTanOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                       const LogicalTensorPtr& result, DataType srcDtype)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        int64_t tmpSize = 9 * MultiplyLastTwoDims<float>(input.tileInfo.shape);
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        function.AddOperation(Opcode::OP_TAN, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTanOperation(function, tileShape, cur + 1, input, result, srcDtype);
    }
}

void TiledTanOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                       const LogicalTensorPtr& result, DataType srcDtype)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledTanOperation(function, tileShape, 0, input, result, srcDtype);
}

LogicalTensorPtr TensorTanOperation(Function& function, LogicalTensorPtr self)
{
    auto srcDtype = self->tensor->datatype;
    LogicalTensorPtr operandCast = self;
    if (srcDtype == DataType::DT_FP16 || srcDtype == DataType::DT_BF16) {
        operandCast = TensorCastOperation<CastOpType::CAST>(function, self, DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_TAN, {operandCast}, {result});
    if (srcDtype == DataType::DT_FP16 || srcDtype == DataType::DT_BF16) {
        auto resultCast = TensorCastOperation<CastOpType::CAST>(function, result, srcDtype, CastMode::CAST_NONE);
        return resultCast;
    }
    return result;
}

Tensor Tan(const Tensor& operand)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Tan");

    auto dType = operand.GetStorage()->Datatype();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          dType == DataType::DT_FP32 || dType == DataType::DT_FP16 || dType == DataType::DT_BF16)
        << "The datatype is not supported";
    RETURN_CALL(TanOperation, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

LogicalTensorPtr IntegerPow(const Tensor& self, int32_t intExponent)
{
    // 快速幂
    auto result = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    auto current = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, result);

    while (intExponent != NUM_VALUE_0) {
        if (intExponent % NUM_VALUE_2 != NUM_VALUE_0) {
            result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result,
                          current);
        }
        current = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), current,
                       current);
        intExponent /= NUM_VALUE_2;
    }
    return result;
}

LogicalTensorPtr GeneralPow(const Tensor& self, double exponent)
{
    // 如果指数小于0，先计算a^(-b)，最后再取倒数
    bool expLessThanZero = exponent < NUM_VALUE_0;
    exponent = std::abs(exponent);

    LogicalTensorPtr result;
    int32_t intExponent = static_cast<int32_t>(std::floor(exponent));
    if (exponent - intExponent < NUM_VALUE_EPS) {
        result = IntegerPow(self, intExponent);
    } else {
        auto lnSelf = CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage());
        auto exponentLnSelf = CALL(BinaryOperationScalar<BinaryOpType::MUL>,
                                   *Program::GetInstance().GetCurrentFunction(), lnSelf,
                                   Element(DataType::DT_FP32, exponent));
        result = CALL(UnaryOperation<UnaryOpType::EXP>, *Program::GetInstance().GetCurrentFunction(), exponentLnSelf);
    }

    // 指数小于零，结果取倒数
    if (expLessThanZero) {
        auto oneTensor = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
        // 求倒数
        RETURN_CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), oneTensor,
                    result);
    }
    return result;
}

Tensor Pow(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");

    LogicalTensorPtr castSelf = self.GetStorage();
    if ((self.GetDataType() == DT_INT32 || self.GetDataType() == DT_INT16) && other.GetDataType() != DT_INT32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    double exponent = other.Cast<double>();
    // 指数为0，输出全1
    if (std::abs(exponent) < NUM_VALUE_EPS) {
        return GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    }
    DataType dataType = castSelf->Datatype();
    bool shouldUpToFp32 = dataType == DT_FP16 || dataType == DT_BF16;
    if (shouldUpToFp32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = castSelf;
    if (std::abs(exponent - NUM_VALUE_0_5) < NUM_VALUE_EPS) {
        result = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), result);
    } else if (std::abs(exponent - NUM_VALUE_2) < NUM_VALUE_EPS) {
        result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result, result);
    } else if (std::abs(exponent - NUM_VALUE_3) < NUM_VALUE_EPS) {
        auto doubleSelf = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result,
                               result);
        result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), doubleSelf,
                      result);
    } else {
        result = GeneralPow(result, exponent);
    }
    if (shouldUpToFp32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result, dataType,
                    CastMode::CAST_NONE);
    }
    return result;
}

void TiledOneHot(Function& function, const TileShape& tileShape, size_t cur, Input& input, Input& output,
                 int numClasses)
{
    if (cur == output.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto outputTile = output.tensor.GetStorage()->View(function, output.tileInfo.shape, output.tileInfo.offset);
        auto& newOp = function.AddOperation(Opcode::OP_ONEHOT, {inputTile}, {outputTile});
        newOp.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < output.tensor.GetShape()[cur]; i += vecTile[cur]) {
        if (cur < input.tensor.GetShape().size()) {
            input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
            input.tileInfo.offset[cur] = i;
        }
        output.tileInfo.shape[cur] = std::min(output.tensor.GetShape()[cur] - i, vecTile[cur]);
        output.tileInfo.offset[cur] = i;
        TiledOneHot(function, tileShape, cur + 1, input, output, numClasses);
    }
}

void TiledOneHot(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                 const LogicalTensorPtr& result, int numClasses)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, numClasses == tileShape.GetVecTile()[result->shape.size() - 1])
        << "The numClasses and last axis of tileshape should be equal";

    TileInfo inputTileInfo(self->shape.size(), self->offset.size());
    TileInfo outputTileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, inputTileInfo};
    auto output = Input{result, outputTileInfo};
    TiledOneHot(function, tileShape, 0, input, output, numClasses);
}

Tensor TensorOneHot(Function& function, const LogicalTensorPtr& self, int numClasses)
{
    Shape shape(self->shape);
    std::vector<SymbolicScalar> validShape(self->dynValidShape_);
    shape.push_back(static_cast<int64_t>(numClasses));
    validShape.push_back(SymbolicScalar(numClasses));
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT64, shape, validShape);
    auto& op = function.AddOperation(Opcode::OP_ONEHOT, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
    return result;
}

Tensor OneHot(const Tensor& self, int numClasses)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "OneHot");

    std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_INT16, DT_INT32, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ONEHOT");
    CheckTensorDimRange(self.GetStorage(), 1, 3, "ONEHOT");
    CheckTensorShapeSize(self.GetStorage(), "ONEHOT");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, numClasses > 0) << "numClasses must be greater than 0";
    auto res = CALL(OneHot, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), numClasses);
    CheckTensorShapeSize(res.GetStorage(), "ONEHOT");
    return res;
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input0, Input& input1,
                              const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input0.tensor.GetShape().size()) {
        auto tile0 = input0.tensor.GetStorage()->View(function, input0.tileInfo.shape, input0.tileInfo.offset);
        auto tile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        constexpr size_t ALIGN_SIZE = 32;
        const int64_t element_per_chunk = 64;
        int64_t vcmp_bits_size = (element_per_chunk + 7) / 8;
        size_t float_array_size = element_per_chunk * SHAPE_DIM4;
        size_t half_array_size = element_per_chunk * SHAPE_DIM2;
        size_t vcmpBitResult_size = ((vcmp_bits_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t aligned_float_array_size = ((float_array_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t aligned_half_array_size = ((half_array_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t total_bytes = vcmpBitResult_size + 4 * aligned_float_array_size + aligned_half_array_size +
                             ALIGN_SIZE * 2;
        std::vector<int64_t> tmp_shape({static_cast<int64_t>(total_bytes)});
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);

        function.AddOperation(Opcode::OP_LOGICALAND, {tile0, tile1}, {resultTile, tmp_tensor});
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        input0.tileInfo.offset[cur] = i % input0.tensor.GetShape()[cur];
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input0.tileInfo.shape[cur] = std::min(input0.tensor.GetShape()[cur] - input0.tileInfo.offset[cur],
                                              vecTile[cur]);
        input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledLogicalAndOperation(function, tileShape, cur + 1, input0, input1, result, resultTileInfo);
    }
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand0,
                              LogicalTensorPtr operand1, const LogicalTensorPtr& result)
{
    BroadcastOperandTensor(operand0, operand1, result, function, tileShape);
    BroadcastOperandTensor(operand1, operand0, result, function, tileShape);

    TileInfo tileInfo0(result->shape.size(), result->offset.size());
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input0 = Input{operand0, tileInfo0};
    auto input1 = Input{operand1, tileInfo1};
    TiledLogicalAndOperation(function, tileShape, 0, input0, input1, result, resultTileInfo);
}

LogicalTensorPtr TensorLogicalAndOperation(Function& function, const Tensor& self, const Tensor& other)
{
    auto operandT0 = self.GetStorage();
    auto operandT1 = other.GetStorage();
    if (operandT0->shape.size() != operandT1->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT0, operandT1);
        operandT0 = BinaryOperationBroadCast(operandT0, broadCastShape);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
    }

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT0, operandT1);
    if ((!operandT0->GetDynValidShape().empty()) && (!operandT1->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT0->shape[i]) {
                resultValidShape.push_back(operandT0->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            }
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, resultShape, resultValidShape);
    function.AddOperation(Opcode::OP_LOGICALAND, {operandT0, operandT1}, {result});
    return result;
}

Tensor LogicalAnd(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LogicalAnd");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LogicalAnd");

    std::unordered_set<DataType> supportedTypes = {DT_FP32,  DT_FP16, DT_BF16,  DT_INT8,
                                                   DT_UINT8, DT_BOOL, DT_INT16, DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOGICALAND");
    CheckTensorDataType(other.GetStorage(), supportedTypes, "LOGICALAND");
    CheckBinaryInputTensors(self.GetStorage(), other.GetStorage(), "LOGICALAND");
    RETURN_CALL(LogicalAndOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                other.GetStorage());
}

void LogicNotOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledLogicalNotOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void SignOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    TiledSignOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void SignbitOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledSignbitOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void TanhOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    TiledTanhOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void OneHotOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int numClasses = op.GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    TiledOneHot(function, tileShape, iOperand[0], oOperand[0], numClasses);
}

void TanOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledTanOperation(function, tileShape, iOperand[0], oOperand[0], iOperand[0]->tensor->datatype);
}

struct CumOperationTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct CumOperationPara {
    const LogicalTensorPtr& input;
    const LogicalTensorPtr& dstTensor;
    const int axis;
    const bool is_sum;
};

// CumSum/CumProd tiling (rank 1-4):
//   Phase 1 — loop and tile every non-cum dimension (0..rank-1 except axis).
//   Phase 2 — within each fixed non-cum fiber, tile only along axis and propagate
//             carry (last cum-axis element) between consecutive tiles on that axis.
// Carry is scoped per non-cum fiber so higher-rank slices do not clobber each other.

namespace {

std::vector<int> BuildNonCumDimIndices(int rank, int axis)
{
    std::vector<int> dims;
    dims.reserve(rank > 0 ? static_cast<size_t>(rank - 1) : 0);
    for (int d = 0; d < rank; ++d) {
        if (d != axis) {
            dims.push_back(d);
        }
    }
    return dims;
}

void EmitCumAxisTile(Function& function, const CumOperationPara& cumOperationPara, CumOperationTileInfoPara& tileInfo,
                     LogicalTensorPtr& lastCarry)
{
    const LogicalTensorPtr& input = cumOperationPara.input;
    const LogicalTensorPtr& dstTensor = cumOperationPara.dstTensor;
    const int axis = cumOperationPara.axis;
    const bool is_sum = cumOperationPara.is_sum;

    auto dstTile = dstTensor->View(function, tileInfo.dstTileInfo.shape, tileInfo.dstTileInfo.offset);
    auto inputTile = input->View(function, tileInfo.inputTileInfo.shape, tileInfo.inputTileInfo.offset);

    LogicalTensorPtr srcTile = std::make_shared<LogicalTensor>(function, dstTile->Datatype(), dstTile->GetShape(),
                                                               inputTile->GetDynValidShape());
    if (is_sum) {
        auto& op = function.AddOperation(Opcode::OP_CUM_SUM, {inputTile}, {srcTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", is_sum);
    } else {
        auto& op = function.AddOperation(Opcode::OP_CUM_PROD, {inputTile}, {srcTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", is_sum);
    }

    if (tileInfo.dstTileInfo.offset[axis] > 0) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastCarry != nullptr)
            << "carry must be set when cum axis tile offset > 0";
        LogicalTensorPtr lastTile = std::make_shared<LogicalTensor>(function, srcTile->Datatype(), srcTile->GetShape(),
                                                                    srcTile->GetDynValidShape());
        auto& eop = function.AddOperation("TILE_EXPAND", {lastCarry}, {lastTile});
        eop.SetAttribute(OpAttributeKey::expandDims, std::vector<int>{axis});
        if (is_sum) {
            function.AddOperation(Opcode::OP_ADD, {srcTile, lastTile}, {dstTile});
        } else {
            function.AddOperation(Opcode::OP_MUL, {srcTile, lastTile}, {dstTile});
        }
    } else {
        function.AddOperation(Opcode::OP_REGISTER_COPY, {srcTile}, {dstTile});
    }

    std::vector<int64_t> lastShape = tileInfo.dstTileInfo.shape;
    lastShape[axis] = 1;
    std::vector<int64_t> lastViewOffset(lastShape.size(), 0);
    lastViewOffset[axis] = tileInfo.dstTileInfo.shape[axis] - 1;
    lastCarry = dstTile->View(function, lastShape, lastViewOffset);
}

void TiledCumAlongAxis(Function& function, const TileShape& tileShape, const CumOperationPara& cumOperationPara,
                       CumOperationTileInfoPara& tileInfo)
{
    const int axis = cumOperationPara.axis;
    const auto& input = cumOperationPara.input;
    auto& vecTile = tileShape.GetVecTile();
    const int64_t cumTileSize = vecTile[axis];
    const int64_t cumDim = input->GetShape()[axis];

    LogicalTensorPtr lastCarry = nullptr;
    for (int64_t i = 0; i < cumDim; i += cumTileSize) {
        tileInfo.dstTileInfo.offset[axis] = i;
        tileInfo.dstTileInfo.shape[axis] = std::min(cumDim - i, cumTileSize);
        tileInfo.inputTileInfo.offset[axis] = i;
        tileInfo.inputTileInfo.shape[axis] = std::min(cumDim - i, cumTileSize);
        EmitCumAxisTile(function, cumOperationPara, tileInfo, lastCarry);
    }
}

void InnerTiledCumNonAxisDims(size_t idx, const std::vector<int>& nonAxisDims, Function& function,
                              const TileShape& tileShape, const CumOperationPara& cumOperationPara,
                              CumOperationTileInfoPara& tileInfo)
{
    if (idx == nonAxisDims.size()) {
        TiledCumAlongAxis(function, tileShape, cumOperationPara, tileInfo);
        return;
    }

    const int dim = nonAxisDims[idx];
    const auto& input = cumOperationPara.input;
    auto& vecTile = tileShape.GetVecTile();
    const int64_t dimSize = input->GetShape()[dim];
    const int64_t tileSize = vecTile[dim];

    for (int64_t i = 0; i < dimSize; i += tileSize) {
        tileInfo.dstTileInfo.offset[dim] = i;
        tileInfo.dstTileInfo.shape[dim] = std::min(dimSize - i, tileSize);
        tileInfo.inputTileInfo.offset[dim] = i;
        tileInfo.inputTileInfo.shape[dim] = std::min(dimSize - i, tileSize);
        InnerTiledCumNonAxisDims(idx + 1, nonAxisDims, function, tileShape, cumOperationPara, tileInfo);
    }
}

} // namespace

void TiledCumOperation(Function& function, const TileShape& tileShape, const CumOperationPara& cumOperationPara)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          cumOperationPara.input->GetShape().size() == cumOperationPara.input->GetOffset().size())
        << "Shape size and offset size should be equal";

    const int rank = static_cast<int>(cumOperationPara.input->GetShape().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          rank >= static_cast<int>(MIN_TENSOR_DIM) && rank <= static_cast<int>(MAX_TENSOR_DIM))
        << "CumSum/CumProd tiling supports rank 1-4";

    CumOperationTileInfoPara tileInfo{
        TileInfo(cumOperationPara.input->GetShape().size(), cumOperationPara.input->GetOffset().size()),
        TileInfo(cumOperationPara.dstTensor->GetShape().size(), cumOperationPara.dstTensor->GetOffset().size())};

    const std::vector<int> nonAxisDims = BuildNonCumDimIndices(rank, cumOperationPara.axis);
    InnerTiledCumNonAxisDims(0, nonAxisDims, function, tileShape, cumOperationPara, tileInfo);
}

void TensorCumOperation(Function& function, const CumOperationPara& cumOperationPara)
{
    if (cumOperationPara.input->Datatype() == DT_INT16) {
        LogicalTensorPtr inputConvertedFp32 = std::make_shared<LogicalTensor>(
            function, DT_FP32, cumOperationPara.input->GetShape(), cumOperationPara.input->GetDynValidShape());
        Operation& castInputOp = function.AddOperation(Opcode::OP_CAST, {cumOperationPara.input}, {inputConvertedFp32});
        castInputOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr inputConvertedInt32 = std::make_shared<LogicalTensor>(
            function, DT_INT32, cumOperationPara.input->GetShape(), cumOperationPara.input->GetDynValidShape());
        Operation& castInputOp2 = function.AddOperation(Opcode::OP_CAST, {inputConvertedFp32}, {inputConvertedInt32});
        castInputOp2.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_TRUNC);
        LogicalTensorPtr dstConverted = std::make_shared<LogicalTensor>(
            function, DT_INT32, cumOperationPara.dstTensor->GetShape(), inputConvertedInt32->GetDynValidShape());
        auto& op = function.AddOperation(Opcode::OP_CUM_SUM, {inputConvertedInt32}, {dstConverted});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
        cumOperationPara.dstTensor->UpdateDynValidShape(dstConverted->GetDynValidShape());
        Operation& castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConverted}, {cumOperationPara.dstTensor});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        return;
    } else if (cumOperationPara.input->Datatype() == DT_BF16 || cumOperationPara.input->Datatype() == DT_FP16) {
        LogicalTensorPtr inputConverted = std::make_shared<LogicalTensor>(
            function, DT_FP32, cumOperationPara.input->GetShape(), cumOperationPara.input->GetDynValidShape());
        Operation& castInputOp = function.AddOperation(Opcode::OP_CAST, {cumOperationPara.input}, {inputConverted});
        castInputOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr dstConverted = std::make_shared<LogicalTensor>(
            function, DT_FP32, cumOperationPara.dstTensor->GetShape(), inputConverted->GetDynValidShape());
        if (cumOperationPara.is_sum) {
            auto& op = function.AddOperation(Opcode::OP_CUM_SUM, {inputConverted}, {dstConverted});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
            op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
        } else {
            auto& op = function.AddOperation(Opcode::OP_CUM_PROD, {inputConverted}, {dstConverted});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
            op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
        }
        cumOperationPara.dstTensor->UpdateDynValidShape(dstConverted->GetDynValidShape());
        Operation& castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConverted}, {cumOperationPara.dstTensor});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        return;
    }
    if (cumOperationPara.input->Datatype() == DT_INT32) {
        LogicalTensorPtr dstConverted = std::make_shared<LogicalTensor>(
            function, DT_INT32, cumOperationPara.dstTensor->GetShape(), cumOperationPara.input->GetDynValidShape());
        auto& op = function.AddOperation(Opcode::OP_CUM_SUM, {cumOperationPara.input}, {dstConverted});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
        cumOperationPara.dstTensor->UpdateDynValidShape(dstConverted->GetDynValidShape());
        Operation& castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConverted}, {cumOperationPara.dstTensor});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        return;
    } else {
        cumOperationPara.dstTensor->UpdateDynValidShape(cumOperationPara.input->GetDynValidShape());
        if (cumOperationPara.is_sum) {
            auto& op = function.AddOperation(Opcode::OP_CUM_SUM, {cumOperationPara.input},
                                             {cumOperationPara.dstTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
            op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
        } else {
            auto& op = function.AddOperation(Opcode::OP_CUM_PROD, {cumOperationPara.input},
                                             {cumOperationPara.dstTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
            op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
        }
        return;
    }
}

void CheckCumOperation(const Tensor& input, const int& axis, const bool& is_sum)
{
    if (is_sum) {
        std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_INT32, DT_INT16, DT_BF16};
        CheckTensorDataType(input.GetStorage(), supportedTypes, "CUMSUM");
        CheckTensorDimRange(input.GetStorage(), 1, 4, "CUMSUM");
        CheckTensorShapeSize(input.GetStorage(), "CUMSUM");
    } else {
        std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
        CheckTensorDataType(input.GetStorage(), supportedTypes, "CUMPROD");
        CheckTensorDimRange(input.GetStorage(), 1, 4, "CUMPROD");
        CheckTensorShapeSize(input.GetStorage(), "CUMPROD");
    }
    int tmpAxis0 = axis;
    CheckAxisRange(input, tmpAxis0);
    if (input.GetShape().size() == 1) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, tmpAxis0 == 0) << "when input.GetShape().size() is 1, axis must be 0";
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tmpAxis0 == 0 || static_cast<size_t>(tmpAxis0) < input.GetShape().size())
        << "The tmpAxis0 should be 0 and less than shape size";
}

Tensor CumOperation(const Tensor& input, const int& axis, const bool& is_sum)
{
    DECLARE_TRACER();
    CheckCumOperation(input, axis, is_sum);

    auto resultDType = input.GetDataType();
    int shapeSize = input.GetShape().size();
    int tmpAxis0 = axis < 0 ? shapeSize + axis : axis;

    if (resultDType == DataType::DT_INT16 || resultDType == DataType::DT_INT32) {
        resultDType = DataType::DT_INT64;
    }

    const int n_1 = shapeSize - 1;
    const int n_2 = shapeSize - 2;
    if ((resultDType != DataType::DT_INT64) && tmpAxis0 > 0 && tmpAxis0 == n_1) {
        Tensor tmpInput = Transpose(input, {n_2, n_1});
        const int transposedAxis = n_2;

        VecTile oriVectile = TileShape::Current().GetVecTile();
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        int64_t tmp = tmpVectile.tile[n_2];
        tmpVectile.tile[n_2] = tmpVectile.tile[n_1];
        tmpVectile.tile[n_1] = tmp;
        TileShape::Current().SetVecTile(tmpVectile);

        auto tmpValidShape = input.GetStorage()->dynValidShape_;
        SymbolicScalar tmpValid = tmpValidShape[n_2];
        tmpValidShape[n_2] = tmpValidShape[n_1];
        tmpValidShape[n_1] = tmpValid;

        Tensor result(tmpInput.GetDataType(), tmpInput.GetShape());
        CALL(CumOperation, *Program::GetInstance().GetCurrentFunction(),
             {tmpInput.GetStorage(), result.GetStorage(), transposedAxis, is_sum});
        Tensor tmpresult = Transpose(result, {n_2, n_1});
        TileShape::Current().SetVecTile(oriVectile);
        return tmpresult;
    } else {
        Tensor result(resultDType, input.GetShape());
        CALL(CumOperation, *Program::GetInstance().GetCurrentFunction(),
             {input.GetStorage(), result.GetStorage(), tmpAxis0, is_sum});
        return result;
    }
}

Tensor CumSum(const Tensor& input, const int& axis)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CumSum");

    bool is_sum = true;
    Tensor result = CumOperation(input, axis, is_sum);
    return result;
}

Tensor CumProd(const Tensor& input, const int& axis)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CumProd");

    bool is_sum = false;
    Tensor result = CumOperation(input, axis, is_sum);
    return result;
}

void CumSumOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    bool is_sum = op.GetBoolAttribute(OP_ATTR_PREFIX + "flag");
    TiledCumOperation(function, tileShape, {iOperand[0], oOperand[0], axis, is_sum});
}

struct TriULTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct TriULPara {
    const LogicalTensorPtr& input;
    const LogicalTensorPtr& dstTensor;
    const SymbolicScalar diagonal;
    const bool isUpper;
};

void InnerTiledTriUL(size_t cur, Function& function, const TileShape& tileShape, const TriULPara& triULPara,
                     TriULTileInfoPara& triULTileInfo)
{
    const LogicalTensorPtr& input = triULPara.input;
    const LogicalTensorPtr& dstTensor = triULPara.dstTensor;
    SymbolicScalar realDiagonal = triULPara.diagonal;
    const bool isUpper = triULPara.isUpper;
    auto& vecTile = tileShape.GetVecTile();

    if (cur == dstTensor->GetShape().size()) {
        auto dstTile = dstTensor->View(function, triULTileInfo.dstTileInfo.shape, triULTileInfo.dstTileInfo.offset);
        auto inputTile = input->View(function, triULTileInfo.inputTileInfo.shape, triULTileInfo.inputTileInfo.offset);
        realDiagonal = realDiagonal + dstTile->GetOffset()[cur - 2] - dstTile->GetOffset()[cur - 1];
        auto& op = function.AddOperation(Opcode::OP_TRIUL, {inputTile}, {dstTile});
        op.SetAttribute(OpAttributeKey::dynScalar, realDiagonal);
        op.SetAttribute(OpAttributeKey::isUpper, isUpper);
        return;
    }
    int64_t tmpTile = vecTile[cur];

    for (int i = 0; i < input->GetShape()[cur]; i += tmpTile) {
        triULTileInfo.dstTileInfo.offset[cur] = i;
        triULTileInfo.dstTileInfo.shape[cur] = std::min(dstTensor->GetShape()[cur] - i, tmpTile);
        triULTileInfo.inputTileInfo.offset[cur] = i;
        triULTileInfo.inputTileInfo.shape[cur] = std::min(input->GetShape()[cur] - i, tmpTile);
        InnerTiledTriUL(cur + 1, function, tileShape, triULPara, triULTileInfo);
    }
}

void TiledTriUL(Function& function, const TileShape& tileShape, const TriULPara& triULPara)
{
    TriULTileInfoPara triULTileInfo{
        TileInfo(triULPara.input->GetShape().size(), triULPara.input->GetOffset().size()),
        TileInfo(triULPara.dstTensor->GetShape().size(), triULPara.dstTensor->GetOffset().size())};

    InnerTiledTriUL(0, function, tileShape, triULPara, triULTileInfo);
}

void CheckTriULOperationParams(const Tensor& input, const std::string& opName)
{
    static const std::unordered_set<DataType> a2a3Types = {DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_INT8};
    static const std::unordered_set<DataType> a5Types = {DT_FP32,  DT_FP16, DT_BF16,   DT_INT16,
                                                         DT_INT32, DT_INT8, DT_UINT16, DT_UINT32};
    const auto& supportedTypes = GetSupportedDataTypesByArch(a2a3Types, a5Types);
    CheckTensorDataType(input.GetStorage(), supportedTypes, opName);
    CheckTensorDimRange(input.GetStorage(), 2, 5, opName);
    CheckTensorShapeSize(input.GetStorage(), opName);
}

void TensorTriUL(Function& function, const TriULPara& triULPara)
{
    if (triULPara.input->Datatype() == DT_INT8) {
        LogicalTensorPtr inputConverted = std::make_shared<LogicalTensor>(
            function, DT_FP16, triULPara.input->GetShape(), triULPara.input->GetDynValidShape());
        auto& castinputOp = GraphUtils::AddDynOperation(function, Opcode::OP_CAST, {triULPara.input}, {inputConverted});
        castinputOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        castinputOp.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(SaturationMode::ON));
        LogicalTensorPtr dstConverted = std::make_shared<LogicalTensor>(
            function, DT_FP16, triULPara.dstTensor->GetShape(), inputConverted->GetDynValidShape());
        auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_TRIUL, {inputConverted}, {dstConverted});
        op.SetAttribute(OpAttributeKey::dynScalar, triULPara.diagonal);
        op.SetAttribute(OpAttributeKey::isUpper, triULPara.isUpper);
        triULPara.dstTensor->UpdateDynValidShape(dstConverted->GetDynValidShape());
        auto& castDstOp = GraphUtils::AddDynOperation(function, Opcode::OP_CAST, {dstConverted}, {triULPara.dstTensor});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_TRUNC);
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(SaturationMode::ON));
    } else {
        triULPara.dstTensor->UpdateDynValidShape(triULPara.input->GetDynValidShape());
        auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_TRIUL, {triULPara.input}, {triULPara.dstTensor});
        op.SetAttribute(OpAttributeKey::dynScalar, triULPara.diagonal);
        op.SetAttribute(OpAttributeKey::isUpper, triULPara.isUpper);
    }
}

Tensor TriU(const Tensor& input, const SymbolicScalar& diagonal)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "TriU");

    CheckTriULOperationParams(input, "TRIU");
    Tensor result(input.GetDataType(), input.GetShape());
    CALL(TriUL, *Program::GetInstance().GetCurrentFunction(),
         {input.GetStorage(), result.GetStorage(), diagonal, true});
    return result;
}

Tensor TriL(const Tensor& input, const SymbolicScalar& diagonal)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "TriL");

    CheckTriULOperationParams(input, "TRIL");
    Tensor result(input.GetDataType(), input.GetShape());
    CALL(TriUL, *Program::GetInstance().GetCurrentFunction(),
         {input.GetStorage(), result.GetStorage(), diagonal, false});
    return result;
}

void TriULOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    SymbolicScalar diagonal = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    bool isUpper = op.GetBoolAttribute(OpAttributeKey::isUpper);
    TiledTriUL(function, tileShape, {iOperand[0], oOperand[0], diagonal, isUpper});
}

// beginregin: Clip

Tensor Clip(const Tensor& self, const Element& min, const Element& max)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CLIP");
    CheckTensorShapeSize(self.GetStorage(), "CLIP");

    Element min_ = min, max_ = max;

    Tensor result = self;
    if (min_.GetDataType() != DT_BOTTOM) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, min_.GetDataType() == self.GetDataType())
            << "The datatype of inputs should be same";
        result = Maximum(result, min_);
    }
    if (max_.GetDataType() != DT_BOTTOM) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, max_.GetDataType() == self.GetDataType())
            << "The datatype of inputs should be same";
        result = Minimum(result, max_);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}

Tensor Clip(const Tensor& self, const Tensor& min, const Tensor& max)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CLIP");
    CheckTensorShapeSize(self.GetStorage(), "CLIP");

    Tensor result = self;
    if (min.GetStorage() != nullptr) {
        CheckTensorFormat(min.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");
        CheckTensorShapeSize(min.GetStorage(), "CLIP");
        CheckTensorsFormatConsistency(self.GetStorage(), min.GetStorage(), "CLIP");
        std::vector minBroadcastAxes = GetBroadcastAxes(min.GetShape(), self.GetShape());
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, minBroadcastAxes.size() <= 1)
            << "minBroadcastAxes size should be <= 1";
        result = Maximum(result, min);
    }
    if (max.GetStorage() != nullptr) {
        CheckTensorFormat(max.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");
        CheckTensorShapeSize(max.GetStorage(), "CLIP");
        CheckTensorsFormatConsistency(self.GetStorage(), max.GetStorage(), "CLIP");
        std::vector maxBroadcastAxes = GetBroadcastAxes(max.GetShape(), self.GetShape());
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, maxBroadcastAxes.size() <= 1)
            << "maxBroadcastAxes size should be <= 1";
        result = Minimum(result, max);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}
// endregion: Clip

void LogicAndOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledLogicalAndOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

static void VarParamVaildCheck(const Tensor& input, std::vector<int>& dim)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
    CheckTensorDataType(input.GetStorage(), supportedTypes, "VAR");
    CheckTensorDimRange(input.GetStorage(), 1, 4, "VAR");
    CheckTensorShapeSize(input.GetStorage(), "VAR");

    Shape shape = input.GetShape();
    uint64_t shapeSize = shape.size();

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dim.size() <= shapeSize) << "The dim.size() should <= input.size()";
    for (uint64_t i = 0; i < shapeSize; i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape[i] > 0) << "The input shape should > 0";
    }

    if (dim.empty()) {
        for (uint64_t i = 0; i < shapeSize; i++) {
            dim.push_back(static_cast<int>(i));
        }
    }
    std::set<int> dupDimSet(dim.begin(), dim.end());

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dupDimSet.size() == dim.size()) << "There is duplicates elements in dim";
    for (size_t i = 0; i < dim.size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              dim[i] < static_cast<int>(shapeSize) && dim[i] >= -(static_cast<int>(shapeSize)))
            << "The value in dim is out of range";
        if (dim[i] < 0) {
            dim[i] = dim[i] + static_cast<int>(shapeSize);
        }
    }
    std::sort(dim.begin(), dim.end());
}

static Tensor VarResSqueeze(const Tensor& res, const std::vector<int>& dim, const std::vector<int64_t>& oriVecTile,
                            DataType dtype)
{
    std::vector<int64_t> vecTile(oriVecTile.begin(), oriVecTile.end());
    for (auto it = dim.rbegin(); it != dim.rend(); ++it) {
        vecTile.erase(vecTile.begin() + *it);
    }
    int64_t algnedSize = BLOCK_SIZE / BytesOf(dtype);
    if (vecTile.empty()) {
        vecTile.push_back(algnedSize);
    }
    int64_t lastDimSize = vecTile.back();
    if (lastDimSize % algnedSize != 0) {
        vecTile.back() = CeilDiv(lastDimSize, algnedSize) * algnedSize;
    }
    TileShape::Current().SetVecTile(vecTile);
    return Squeeze(res, dim);
}

Tensor Var(const Tensor& input, const std::vector<int>& dim, float correction, bool keepDim)
{
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Var");

    std::vector<int> innerDim(dim.begin(), dim.end());
    VarParamVaildCheck(input, innerDim);

    DataType dtype = input.GetDataType();
    Shape shape = input.GetShape();
    auto castInput = Tensor(DT_FP32, shape);
    if (dtype == DT_FP16 || dtype == DT_BF16) {
        castInput = Cast(input, DT_FP32, CAST_NONE);
    } else {
        castInput = input;
    }

    int calcN = 1;
    auto res = castInput;
    for (size_t i = 0; i < innerDim.size(); i++) {
        calcN *= static_cast<int>(shape[innerDim[i]]);
    }
    res = Div(res, Element(DT_FP32, static_cast<float>(calcN)));
    for (size_t i = 0; i < innerDim.size(); i++) {
        res = Sum(res, innerDim[i], true);
    }

    Shape dstShape = res.GetShape();
    for (size_t i = 0; i < innerDim.size(); i++) {
        dstShape[innerDim[i]] = shape[innerDim[i]];
        res = Expand(res, dstShape);
    }

    res = Sub(castInput, res);
    res = Mul(res, res);
    float count = std::max(0.0f, static_cast<float>(calcN) - correction);
    res = Div(res, Element(DT_FP32, count));
    for (size_t i = 0; i < innerDim.size(); i++) {
        res = Sum(res, innerDim[i], true);
    }
    auto oriVecTile = TileShape::Current().GetVecTile();
    if (!keepDim) {
        res = VarResSqueeze(res, innerDim, oriVecTile.tile, dtype);
    }

    if (dtype == DT_FP16 || dtype == DT_BF16) {
        res = Cast(res, dtype, CAST_NONE);
    }
    if (!keepDim) {
        TileShape::Current().SetVecTile(oriVecTile.tile);
    }

    return res;
}

Tensor TensorExp2(Function& function, const LogicalTensorPtr& self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    if (self->Datatype() == DataType::DT_INT32 || self->Datatype() == DataType::DT_INT16) {
        result = std::make_shared<LogicalTensor>(function, DT_FP32, self->GetShape(), self->GetDynValidShape());
    }
    function.AddOperation(Opcode::OP_EXP2, {self}, {result});
    return result;
}

Tensor Exp2(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Exp2");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "EXP2");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "EXP2");
    CheckTensorShapeSize(self.GetStorage(), "EXP2");

    RETURN_CALL(Exp2, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

void TiledExp2(Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only support 1~4";
        std::vector<int64_t> tmpShape;
        std::vector<int64_t> tmpShape2;
        if (srcTileShape.size() == 1) {
            tmpShape2.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
        } else {
            tmpShape2.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
        }
        auto alignSize2 = BLOCK_SIZE / BytesOf(DT_FP32);
        tmpShape2[tmpShape2.size() - 1] = (tmpShape2[tmpShape2.size() - 1] + alignSize2 - 1) / alignSize2 * alignSize2;
        if (input.tensor.GetDataType() == DT_FP32) {
            tmpShape = {BLOCK_SIZE / sizeof(float)};
        } else {
            if (srcTileShape.size() == 1) {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
            } else {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
            }
            auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
            tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        auto tmpTensorNext = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape2);

        function.AddOperation(Opcode::OP_EXP2, {tile}, {resultTile, tmpTensor, tmpTensorNext});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledExp2(function, tileShape, cur + 1, input, result);
    }
}

void TiledExp2(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
               const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledExp2(function, tileShape, 0, input, result);
}

void Exp2OperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    TiledExp2(function, tileShape, iOperand[0], oOperand[0]);
}

Tensor TensorRound(Function& function, const LogicalTensorPtr& self, const int& decimals = 0)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_ROUND, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "decimals", decimals);
    return result;
}

Tensor Round(const Tensor& self, const int& decimals)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Round");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ROUND");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "ROUND");
    CheckTensorShapeSize(self.GetStorage(), "ROUND");

    RETURN_CALL(Round, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), decimals);
}

void TiledRound(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                const LogicalTensorPtr& result, const int& decimals = 0)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only support 1~4";
        std::vector<int64_t> tmpShape;
        if (result->Datatype() == DT_FP32) {
            tmpShape = {BLOCK_SIZE / sizeof(float)};
        } else {
            if (srcTileShape.size() == 1) {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
            } else {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
            }
            auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
            tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        auto& newOp = function.AddOperation(Opcode::OP_ROUND, {tile}, {resultTile, tmpTensor});
        float powDecimals = std::pow(static_cast<float>(10), static_cast<float>(decimals));
        const int32_t maxFp32Len = 38;
        if (decimals > maxFp32Len) {
            powDecimals = INFINITY;
        }
        newOp.SetAttribute(OP_ATTR_PREFIX + "decimals", decimals);
        newOp.SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, powDecimals));
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledRound(function, tileShape, cur + 1, input, result, decimals);
    }
}

void TiledRound(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                const LogicalTensorPtr& result, const int& decimals = 0)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledRound(function, tileShape, 0, input, result, decimals);
}

void RoundOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int decimals = op.GetIntAttribute(OP_ATTR_PREFIX + "decimals");
    TiledRound(function, tileShape, iOperand[0], oOperand[0], decimals);
}

Tensor TensorExpm1(Function& function, const LogicalTensorPtr& self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    if (self->Datatype() == DataType::DT_INT32 || self->Datatype() == DataType::DT_INT16) {
        result = std::make_shared<LogicalTensor>(function, DT_FP32, self->GetShape(), self->GetDynValidShape());
    }
    function.AddOperation(Opcode::OP_EXPM1, {self}, {result});
    return result;
}

Tensor Expm1(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Expm1");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "EXPM1");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "EXPM1");
    CheckTensorShapeSize(self.GetStorage(), "EXPM1");

    RETURN_CALL(Expm1, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

void TiledExpm1(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only support 1~4";
        std::vector<int64_t> tmpShape;
        if (input.tensor.GetDataType() == DT_FP32) {
            tmpShape = {BLOCK_SIZE / sizeof(float)};
        } else {
            if (srcTileShape.size() == 1) {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
            } else {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
            }
            auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
            tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        function.AddOperation(Opcode::OP_EXPM1, {tile}, {resultTile, tmpTensor});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledExpm1(function, tileShape, cur + 1, input, result);
    }
}

void TiledExpm1(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledExpm1(function, tileShape, 0, input, result);
}

void Expm1OperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledExpm1(function, tileShape, iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, LogicNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ONEHOT, Opcode::OP_ONEHOT, OneHotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXPM1, Opcode::OP_EXPM1, Expm1OperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROUND, Opcode::OP_ROUND, RoundOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXP2, Opcode::OP_EXP2, Exp2OperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOGICALAND, Opcode::OP_LOGICALAND, LogicAndOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CUM_SUM, Opcode::OP_CUM_SUM, CumSumOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CUM_PROD, Opcode::OP_CUM_PROD, CumSumOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRIUL, Opcode::OP_TRIUL, TriULOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SIGN, Opcode::OP_SIGN, SignOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SIGNBIT, Opcode::OP_SIGNBIT, SignbitOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TANH, Opcode::OP_TANH, TanhOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TAN, Opcode::OP_TAN, TanOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOG1P, Opcode::OP_LOG1P, Log1pOperationTileFunc);
} // namespace npu::tile_fwk
