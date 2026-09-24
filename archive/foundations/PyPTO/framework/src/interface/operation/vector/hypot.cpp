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
 * \file hypot.cpp
 * \brief
 */

#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

/**
 * @brief Hypot
 */
void TiledHypotOperationImpl(Function& function, const TileShape& tileShape, size_t cur, Input& input1, Input& input2,
                             const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == result->shape.size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor.GetStorage()->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        size_t element_size = sizeof(float);
        int64_t num_elements = 1;
        if (!resultTileInfo.shape.empty()) {
            num_elements = resultTileInfo.shape.back();
        }
        const size_t ALIGN_SIZE = 32;
        size_t raw_size_bytes = num_elements * element_size;
        size_t aligned_size_bytes = ((raw_size_bytes + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t total_bytes = 2 * aligned_size_bytes;
        std::vector<int64_t> tmp_shape = {static_cast<int64_t>(total_bytes)};
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);
        function.AddOperation(Opcode::OP_HYPOT, {inputTile1, inputTile2}, {resultTile, tmp_tensor});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    int64_t step = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += step) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, step);
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], step);
        input2.tileInfo.offset[cur] = i % input2.tensor.GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], step);
        TiledHypotOperationImpl(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

void TiledHypotOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                         LogicalTensorPtr operand2, const LogicalTensorPtr& result)
{
    auto broadcastOperand = [&](LogicalTensorPtr& operand, LogicalTensorPtr& other) {
        auto dstShape = result->shape;
        if (operand->shape == dstShape) {
            return;
        }
        auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
        Expand(function, tileShape, operand, {other}, expanded);
        operand = expanded;
    };

    CheckBinOpOperandsValid(operand1, operand2);
    broadcastOperand(operand1, operand2);
    broadcastOperand(operand2, operand1);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());

    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};

    TiledHypotOperationImpl(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

LogicalTensorPtr TensorHypotOperation(Function& function, const Tensor& self, const Tensor& other)
{
    auto operandT1 = self.GetStorage();
    auto operandT2 = other.GetStorage();

    if (operandT1->shape.size() != operandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT1, operandT2);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
        operandT2 = BinaryOperationBroadCast(operandT2, broadCastShape);
    }

    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT1, operandT2);

    std::vector<SymbolicScalar> resultValidShape;
    if (!operandT1->GetDynValidShape().empty() && !operandT2->GetDynValidShape().empty()) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT1->shape[i]) {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT2->GetDynValidShape()[i]);
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, operandT1->Datatype(), resultShape, resultValidShape);
    function.AddOperation(Opcode::OP_HYPOT, {operandT1, operandT2}, {result});
    return result;
}

Tensor Hypot(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Hypot");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Hypot");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "HYPOT");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "HYPOT");
    CheckBinaryInputTensors(self.GetStorage(), other.GetStorage(), "HYPOT");
    RETURN_CALL(HypotOperation, *Program::GetInstance().GetCurrentFunction(), self, other);
}

void HypotOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledHypotOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_HYPOT, Opcode::OP_HYPOT, HypotOperationTileFunc);

} // namespace npu::tile_fwk
