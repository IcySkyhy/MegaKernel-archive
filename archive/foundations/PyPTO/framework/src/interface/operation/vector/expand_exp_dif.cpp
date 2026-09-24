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
 * \file expand_exp_dif.cpp
 * \brief
 */

#include "binary.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tensor_transformation.h"

namespace npu::tile_fwk {

void TiledExpandExpDifOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto opName = GetBinaryOpNameCode<BinaryOpType::EXPANDEXPDIF>();
        auto& op = function.AddOperation(opName, {inputTile1, inputTile2}, {resultTile});
        size_t shapeSize = input1.tensor->GetShape().size();
        std::vector<int64_t> brcOperand(shapeSize, 0);
        size_t brcAxesCount = 0;
        for (size_t i = 0; i < shapeSize; i++) {
            brcOperand[i] = BrcAxisBinaryOp(input1.tensor, input2.tensor, i);
            if (brcOperand[i] != 0) {
                brcAxesCount++;
            }
        }
        if (brcAxesCount > 0) {
            op.SetAttribute(OpAttributeKey::brcOperand, brcOperand);
        }
    } else {
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
            TiledExpandExpDifOperation(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
        }
    }
}

void TiledExpandExpDifOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                LogicalTensorPtr operand2, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(operand1, operand2);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledExpandExpDifOperation(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

Tensor ExpandExpDif(const Tensor& input, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ExpandExpDif");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ExpandExpDif");

    CheckTensorsDataTypeConsistency(input.GetStorage(), other.GetStorage(), "EXPANDEXPDIF");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(input.GetStorage(), supportedTypes, "EXPANDEXPDIF");
    CheckBinaryInputTensors(input.GetStorage(), other.GetStorage(), "EXPANDEXPDIF");
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    RETURN_CALL(BinaryOperation<BinaryOpType::EXPANDEXPDIF>, *Program::GetInstance().GetCurrentFunction(), input,
                other);
}

void ExpandExpDifOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledExpandExpDifOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_EXPANDEXPDIF, Opcode::OP_EXPANDEXPDIF, ExpandExpDifOperationTileFunc);

} // namespace npu::tile_fwk
