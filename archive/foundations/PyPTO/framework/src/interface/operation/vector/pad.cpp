/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad.cpp
 * \brief
 */

#include <cmath>
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {
void TiledPadImpl(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                  const LogicalTensorPtr& result, TileInfo& resultTileInfo, int64_t padRight, int64_t padBottom,
                  const Element& padValue)
{
    size_t ndim = result->shape.size();
    auto& vecTile = tileShape.GetVecTile();
    if (cur == ndim) {
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (ndim == 1) {
            auto lastInputShape = input.tensor.GetShape()[0];
            auto lastResultShape = result->shape[0];
            auto lastShape = input.tileInfo.shape[0];
            auto lastOffset = input.tileInfo.offset[0];
            if (lastShape <= 0) { // 输入tile大小为0，完全在填充区域
                auto& op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
                op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
            } else if (lastOffset + vecTile[0] > lastInputShape) {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                auto& op = function.AddOperation(Opcode::OP_PAD, {inputTile}, {resultTile});
                auto last = std::min(lastResultShape, lastOffset + vecTile[0]);
                padRight = last > lastInputShape ? last - lastInputShape : 0;
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "pad_right", padRight);
                op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", 0);
            } else {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {resultTile});
            }
        } else {
            auto lastInputShape = input.tensor.GetShape()[ndim - 1];
            auto lastResultShape = result->shape[ndim - 1];
            auto lastShape = input.tileInfo.shape[ndim - 1];
            auto lastOffset = input.tileInfo.offset[ndim - 1];
            auto preInputShape = input.tensor.GetShape()[ndim - 2];
            auto preResultShape = result->shape[ndim - 2];
            auto preShape = input.tileInfo.shape[ndim - 2];
            auto preOffset = input.tileInfo.offset[ndim - 2];
            if (lastShape <= 0 || preShape <= 0) {
                auto& op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
                op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
            } else if (lastOffset + vecTile[ndim - 1] > lastInputShape ||
                       preOffset + vecTile[ndim - 2] > preInputShape) {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                auto& op = function.AddOperation(Opcode::OP_PAD, {inputTile}, {resultTile});
                auto last = std::min(lastResultShape, lastOffset + vecTile[ndim - 1]);
                auto pre = std::min(preResultShape, preOffset + vecTile[ndim - 2]);
                padRight = last > lastInputShape ? last - lastInputShape : 0;
                padBottom = pre > preInputShape ? pre - preInputShape : 0;
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "pad_right", padRight);
                op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", padBottom);
            } else {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {resultTile});
            }
        }
        return;
    }

    for (int64_t i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        TiledPadImpl(function, tileShape, cur + 1, input, result, resultTileInfo, padRight, padBottom, padValue);
    }
}

void TiledPadOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& input,
                       const LogicalTensorPtr& result, int64_t padRight, int64_t padBottom, const Element& padValue)
{
    size_t ndim = result->shape.size();
    TileInfo resultTileInfo(ndim, ndim);
    TileInfo inputTileInfo(ndim, ndim);
    Input padInput{input, inputTileInfo};
    TiledPadImpl(function, tileShape, 0, padInput, result, resultTileInfo, padRight, padBottom, padValue);
}

void TiledFillPadOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& result, const Element& padValue)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_FILLPAD, {tile}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, padValue);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledFillPadOperation(function, tileShape, cur + 1, input, result, padValue);
    }
}

void TiledFillPadOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                           const LogicalTensorPtr& result, const Element& padValue)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledFillPadOperation(function, tileShape, 0, input, result, padValue);
}

LogicalTensorPtr TensorPadOperation(Function& function, const Tensor& self, const std::vector<int64_t>& padding,
                                    const std::string& mode, const Element& value)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, mode == "constant") << "Pad: only 'constant' mode is supported.";
    auto operand = self.GetStorage();
    std::vector<int64_t> outputShape = operand->shape;
    size_t ndim = operand->shape.size();
    int64_t padRight = 0;
    int64_t padBottom = 0;

    if (ndim == 1) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padding.size() == 2)
            << "Pad: 1D tensor only support 2 padding values.";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padding[0] == 0) << "Pad: 1D tensor only support right pad.";
        padRight = padding[1];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padRight >= 0)
            << "Pad: padding values must be non-negative, got pad_right=" << padRight
            << ". Negative padding (for cropping) is not supported.";
        outputShape[0] += padRight;
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padding.size() == 4) << "Pad: only support last 2 axis pad.";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padding[0] == 0 && padding[2] == 0)
            << "Pad: only support bottom and right pad.";
        padRight = padding[1];
        padBottom = padding[3];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padRight >= 0)
            << "Pad: padding values must be non-negative, got pad_right=" << padRight
            << ". Negative padding (for cropping) is not supported.";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, padBottom >= 0)
            << "Pad: padding values must be non-negative, got pad_bottom=" << padBottom
            << ". Negative padding (for cropping) is not supported.";
        outputShape[ndim - 1] += padRight;
        outputShape[ndim - 2] += padBottom;
    }

    std::vector<SymbolicScalar> resultValidShape;
    const auto& inputValidShape = operand->GetDynValidShape();
    if (!inputValidShape.empty()) {
        resultValidShape = inputValidShape;
        if (ndim == 1) {
            resultValidShape[0] = resultValidShape[0] + padRight;
        } else {
            resultValidShape[ndim - 1] = resultValidShape[ndim - 1] + padRight;
            resultValidShape[ndim - 2] = resultValidShape[ndim - 2] + padBottom;
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), outputShape, resultValidShape);
    auto& op = function.AddOperation(Opcode::OP_PAD, {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "pad_right", static_cast<int64_t>(padRight));
    op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", static_cast<int64_t>(padBottom));
    op.SetAttribute(OpAttributeKey::scalar, value);
    return result;
}

Tensor Pad(const Tensor& self, const std::vector<int64_t>& padding, std::string mode, const Element& value)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pad");

    CheckTensorDimRange(self.GetStorage(), 1, 4, "PAD");
    CheckTensorShapeSize(self.GetStorage(), "PAD");
    std::unordered_set<DataType> supportedTypes = {DT_FP32,  DT_FP16,  DT_BF16,   DT_INT8,  DT_INT16,
                                                   DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "PAD");
    RETURN_CALL(PadOperation, *Program::GetInstance().GetCurrentFunction(), self, padding, mode, value);
}

LogicalTensorPtr TensorFillPadOperation(Function& function, const Tensor& self, const std::string& mode,
                                        const Element& value)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, mode == "constant") << "FillPad: only 'constant' mode is supported.";
    auto operand = self.GetStorage();
    std::vector<int64_t> outputShape = operand->shape;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, outputShape.size() == 1 || outputShape.size() == 2)
        << "FillPad: only support 1 dim or 2 dim.";
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), outputShape,
                                                  SymbolicScalar::FromConcrete(outputShape));
    auto& op = function.AddOperation(Opcode::OP_FILLPAD, {operand}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    return result;
}

Tensor FillPad(const Tensor& self, std::string mode, const Element& value)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FillPad");

    CheckTensorDimRange(self.GetStorage(), 1, 2, "FILLPAD");
    CheckTensorShapeSize(self.GetStorage(), "FILLPAD");
    std::unordered_set<DataType> supportedTypes = {DT_FP32,  DT_FP16,  DT_BF16,   DT_INT8,  DT_INT16,
                                                   DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "FILLPAD");
    RETURN_CALL(FillPadOperation, *Program::GetInstance().GetCurrentFunction(), self, mode, value);
}

void PadOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t padRight = op.GetIntAttribute(OP_ATTR_PREFIX + "pad_right");
    int64_t padBottom = op.GetIntAttribute(OP_ATTR_PREFIX + "pad_bottom");
    Element padValue = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledPadOperation(function, tileShape, iOperand[0], oOperand[0], padRight, padBottom, padValue);
}

void FillPadOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    Element padValue = op.GetElementAttribute(OpAttributeKey::scalar);
    return TiledFillPadOperation(function, tileShape, iOperand[0], oOperand[0], padValue);
}

REGISTER_OPERATION_TILED_FUNC(OP_PAD, Opcode::OP_PAD, PadOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FILLPAD, Opcode::OP_FILLPAD, FillPadOperationTileFunc);

} // namespace npu::tile_fwk
