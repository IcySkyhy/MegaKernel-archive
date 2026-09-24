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
 * \file interleave.cpp
 * \brief
 */

#include "binary.h"
#include "interface/utils/common.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

namespace {
const std::vector<NPUArch> INTERLEAVE_SUPPORTED_ARCHITECTURES = {NPUArch::DAV_3510};
} // namespace

void TileInterleaveOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                             LogicalInput& input2, LogicalTensorPtr& result1, LogicalTensorPtr& result2,
                             TileInfo& resultTileInfo)
{
    size_t shapeSize = input1.tensor->GetShape().size();
    auto& vecTile = tileShape.GetVecTile();
    if (cur == shapeSize) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile1 = result1->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto resultTile2 = result2->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto validShape = inputTile1->GetDynValidShape();
        function.AddOperation(Opcode::OP_INTERLEAVE, {inputTile1, inputTile2}, {resultTile1, resultTile2});
        resultTile1->UpdateDynValidShape(validShape);
        resultTile2->UpdateDynValidShape(validShape);
        return;
    }

    for (int i = 0; i < input1.tensor->GetShape()[cur]; i += vecTile[cur]) {
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur],
                                              vecTile[cur]);
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result1->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TileInterleaveOperation(function, tileShape, cur + 1, input1, input2, result1, result2, resultTileInfo);
    }
}

void TileInterleaveOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                             LogicalTensorPtr operand2, LogicalTensorPtr result1, LogicalTensorPtr result2)
{
    TileInfo tileInfo1(operand1->shape.size(), operand1->offset.size());
    TileInfo tileInfo2(operand2->shape.size(), operand2->offset.size());
    TileInfo resultTileInfo(result1->shape.size(), result1->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TileInterleaveOperation(function, tileShape, 0, input1, input2, result1, result2, resultTileInfo);
}

void TensorInterleaveOperation(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& other,
                               LogicalTensorPtr& dst0, LogicalTensorPtr& dst1)
{
    auto validShape = self->GetDynValidShape();
    GraphUtils::AddDynOperation(function, Opcode::OP_INTERLEAVE, {self, other}, {dst0, dst1}, {validShape, validShape});
    return;
}

void TileDeInterleaveOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                               LogicalInput& input2, LogicalTensorPtr& result1, LogicalTensorPtr& result2,
                               TileInfo& resultTileInfo)
{
    size_t shapeSize = result1->GetShape().size();
    auto& vecTile = tileShape.GetVecTile();
    if (cur == shapeSize) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile1 = result1->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto resultTile2 = result2->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto validShape = inputTile1->GetDynValidShape();
        function.AddOperation(Opcode::OP_DEINTERLEAVE, {inputTile1, inputTile2}, {resultTile1, resultTile2});
        resultTile1->UpdateDynValidShape(validShape);
        resultTile2->UpdateDynValidShape(validShape);
        return;
    }

    for (int i = 0; i < result1->GetShape()[cur]; i += vecTile[cur]) {
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur],
                                              vecTile[cur]);
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result1->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TileDeInterleaveOperation(function, tileShape, cur + 1, input1, input2, result1, result2, resultTileInfo);
    }
}

void TileDeInterleaveOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input,
                               LogicalTensorPtr& result1, LogicalTensorPtr& result2, TileInfo& resultTileInfo)
{
    size_t shapeSize = result1->GetShape().size();
    auto& vecTile = tileShape.GetVecTile();
    if (cur == shapeSize) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile1 = result1->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto resultTile2 = result2->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto validShape = inputTile->GetDynValidShape();
        validShape[shapeSize - 1] = validShape[shapeSize - 1] / 2;
        function.AddOperation(Opcode::OP_DEINTERLEAVE_SINGLE, {inputTile}, {resultTile1, resultTile2});
        resultTile1->UpdateDynValidShape(validShape);
        resultTile2->UpdateDynValidShape(validShape);
        return;
    }

    for (int i = 0; i < input.tensor->GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.offset[cur] = i % input.tensor->GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        if (cur == shapeSize - 1) {
            resultTileInfo.offset[cur] = i / 2;
            resultTileInfo.shape[cur] = std::min(result1->shape[cur] - resultTileInfo.offset[cur], vecTile[cur] / 2);
        } else {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result1->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        }
        TileDeInterleaveOperation(function, tileShape, cur + 1, input, result1, result2, resultTileInfo);
    }
}

void TileDeInterleaveOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                               LogicalTensorPtr operand2, LogicalTensorPtr result1, LogicalTensorPtr result2)
{
    TileInfo tileInfo1(operand1->shape.size(), operand1->offset.size());
    TileInfo tileInfo2(operand2->shape.size(), operand2->offset.size());
    TileInfo resultTileInfo(result1->shape.size(), result1->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TileDeInterleaveOperation(function, tileShape, 0, input1, input2, result1, result2, resultTileInfo);
}

void TileDeInterleaveOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand,
                               LogicalTensorPtr result1, LogicalTensorPtr result2)
{
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(result1->shape.size(), result1->offset.size());
    auto input = LogicalInput{operand, tileInfo};
    TileDeInterleaveOperation(function, tileShape, 0, input, result1, result2, resultTileInfo);
}

void TensorDeInterleaveOperation(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& other,
                                 LogicalTensorPtr& dst0, LogicalTensorPtr& dst1)
{
    auto validShape = self->GetDynValidShape();
    dst0->UpdateDynValidShape(validShape);
    dst1->UpdateDynValidShape(validShape);
    GraphUtils::AddDynOperation(function, Opcode::OP_DEINTERLEAVE, {self, other}, {dst0, dst1},
                                {validShape, validShape});
    return;
}

void TensorDeInterleaveOperation(Function& function, const LogicalTensorPtr& self, LogicalTensorPtr& dst0,
                                 LogicalTensorPtr& dst1)
{
    auto validShape = self->GetDynValidShape();
    validShape[validShape.size() - 1] = validShape[validShape.size() - 1] / 2;
    dst0->UpdateDynValidShape(validShape);
    dst1->UpdateDynValidShape(validShape);
    GraphUtils::AddDynOperation(function, Opcode::OP_DEINTERLEAVE_SINGLE, {self}, {dst0, dst1},
                                {validShape, validShape});
    return;
}

static void CheckInterleaveTileShape(const Tensor& self, const char* opName, bool requireFullLastAxis)
{
    auto viewShape = self.GetShape();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, !viewShape.empty()) << opName << " requires non-empty input shape.";
    auto lastAxis = viewShape.size() - 1;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, viewShape[lastAxis] % 2 == 0)
        << opName << " requires input shape last axis to be even, input shape is " << IntVecToStr(viewShape);

    const auto& vecTile = TileShape::Current().GetVecTile();
    if (!vecTile.valid()) {
        return;
    }

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile.size() == viewShape.size())
        << opName << " requires tile_shape rank equal to shape rank, shape is " << IntVecToStr(viewShape)
        << ", tile_shape is " << IntVecToStr(vecTile.tile);
    if (requireFullLastAxis) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile[lastAxis] == viewShape[lastAxis])
            << opName << " requires tile_shape last axis equal to shape last axis, shape last axis is "
            << viewShape[lastAxis] << ", tile_shape last axis is " << vecTile.tile[lastAxis];
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile[lastAxis] % 2 == 0)
        << opName << " requires tile_shape last axis to be even, tile_shape is " << IntVecToStr(vecTile.tile);
}

std::tuple<Tensor, Tensor> Interleave(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Interleave");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Interleave");

    CheckSupportedNPUArch(INTERLEAVE_SUPPORTED_ARCHITECTURES, "Interleave");
    CheckInterleaveTileShape(self, "Interleave", true);
    auto shape = self.GetShape();

    auto dst0 = Tensor(self.GetDataType(), shape);
    auto dst1 = Tensor(self.GetDataType(), shape);
    auto validShape = self.GetStorage()->GetDynValidShape();
    dst0.GetStorage()->UpdateDynValidShape(validShape);
    dst1.GetStorage()->UpdateDynValidShape(validShape);

    CALL(InterleaveOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other.GetStorage(),
         dst0.GetStorage(), dst1.GetStorage());

    return std::tie(dst0, dst1);
}

std::tuple<Tensor, Tensor> DeInterleave(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DeInterleave");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DeInterleave");

    CheckSupportedNPUArch(INTERLEAVE_SUPPORTED_ARCHITECTURES, "DeInterleave");
    CheckInterleaveTileShape(self, "DeInterleave", true);
    auto shape = self.GetShape();
    auto dst0 = Tensor(self.GetDataType(), shape);
    auto dst1 = Tensor(self.GetDataType(), shape);
    auto validShape = self.GetStorage()->GetDynValidShape();
    dst0.GetStorage()->UpdateDynValidShape(validShape);
    dst1.GetStorage()->UpdateDynValidShape(validShape);
    CALL(DeInterleaveOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other.GetStorage(),
         dst0.GetStorage(), dst1.GetStorage());
    return std::tie(dst0, dst1);
}

std::tuple<Tensor, Tensor> DeInterleave(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DeInterleave");

    CheckSupportedNPUArch(INTERLEAVE_SUPPORTED_ARCHITECTURES, "DeInterleave");
    CheckInterleaveTileShape(self, "DeInterleave", false);
    auto shape = self.GetShape();
    shape[shape.size() - 1] = shape[shape.size() - 1] / 2;
    auto dst0 = Tensor(self.GetDataType(), shape);
    auto dst1 = Tensor(self.GetDataType(), shape);
    auto validShape = self.GetStorage()->GetDynValidShape();
    validShape[validShape.size() - 1] = validShape[validShape.size() - 1] / 2;
    dst0.GetStorage()->UpdateDynValidShape(validShape);
    dst1.GetStorage()->UpdateDynValidShape(validShape);
    CALL(DeInterleaveOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dst0.GetStorage(),
         dst1.GetStorage());
    return std::tie(dst0, dst1);
}

void InterleaveOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TileInterleaveOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], oOperand[1]);
}
REGISTER_OPERATION_TILED_FUNC(OP_INTERLEAVE, Opcode::OP_INTERLEAVE, InterleaveOperationTileFunc);

void DeInterleaveOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TileDeInterleaveOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], oOperand[1]);
}
REGISTER_OPERATION_TILED_FUNC(OP_DEINTERLEAVE, Opcode::OP_DEINTERLEAVE, DeInterleaveOperationTileFunc);

void DeInterleaveSingleOperationTileFunc(Function& function, const TileShape& tileShape,
                                         const std::vector<LogicalTensorPtr>& iOperand,
                                         const std::vector<LogicalTensorPtr>& oOperand,
                                         [[maybe_unused]] const Operation& op)
{
    TileDeInterleaveOperation(function, tileShape, iOperand[0], oOperand[0], oOperand[1]);
}
REGISTER_OPERATION_TILED_FUNC(OP_DEINTERLEAVE_SINGLE, Opcode::OP_DEINTERLEAVE_SINGLE,
                              DeInterleaveSingleOperationTileFunc);

} // namespace npu::tile_fwk
