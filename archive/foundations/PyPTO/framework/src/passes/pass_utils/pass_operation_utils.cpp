/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass_operation_utils.h"

#include <algorithm>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/operation_common.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/tensor_offset.h"

namespace npu::tile_fwk {

Operation& PassOperationUtils::AddOperation(Function& function, Opcode opCode, LogicalTensors iOperands,
                                            const LogicalTensors& oOperands,
                                            std::function<void(Operation&)> beforeInferShapeHandler,
                                            const ir::Span& span, bool inferShape)
{
    auto processedOperands = PreprocessOperationInputs(function, opCode, std::move(iOperands));
    IRBuilder builder;
    auto& op = builder.CreateTensorOpStmt(function, opCode, processedOperands, oOperands, span);

    if (beforeInferShapeHandler) {
        beforeInferShapeHandler(op);
    }

    if (inferShape) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(&op);
    }

    return op;
}

LogicalTensors PassOperationUtils::PreprocessOperationInputs(Function& function, Opcode opCode,
                                                             LogicalTensors iOperands)
{
    CheckTensorDynamicShape(iOperands, opCode);
    for (auto& iOperand : iOperands) {
        FE_ASSERT(FeError::INVALID_VAL, iOperand->shape.size() != 0) << "tensor shape size invalid";
        iOperand = ConnectWithOverlap(function, iOperand);
    }
    return iOperands;
}

std::vector<std::vector<int64_t>> PassOperationUtils::ProcessOffsetAdjustment(LogicalTensors& matches,
                                                                              std::vector<int64_t>& minimumOffsets)
{
    std::vector<std::vector<int64_t>> offsetOfOverlaps;
    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) { return a->offset < b->offset; });

    minimumOffsets = matches.front()->offset;
    for (auto& match : matches) {
        offsetOfOverlaps.emplace_back(match->offset);
        for (size_t idx = 0; idx < minimumOffsets.size(); ++idx) {
            minimumOffsets[idx] = std::min(minimumOffsets[idx], match->offset[idx]);
        }
    }
    for (auto& offsetOfOverlap : offsetOfOverlaps) {
        for (size_t idx = 0; idx < offsetOfOverlap.size(); ++idx) {
            offsetOfOverlap[idx] -= minimumOffsets[idx];
        }
    }
    return offsetOfOverlaps;
}

LogicalTensorPtr PassOperationUtils::HandlePerfectlyMatchWithAll(
    Function& function, LogicalTensorPtr iOperand, const LogicalTensors& matches,
    const std::vector<std::vector<int64_t>>& offsetOfOverlaps)
{
    auto assembleResult = std::make_shared<LogicalTensor>(function, iOperand->Datatype(), iOperand->shape,
                                                          iOperand->GetDynValidShape(), iOperand->Format(),
                                                          "Assemble_" + matches[0]->Symbol());
    FE_ASSERT(FeError::NOT_EXIST, assembleResult->GetProducers().empty()) << "Assemble result should have no producers";
    IRBuilder builder;
    for (size_t idx = 0; idx < matches.size(); ++idx) {
        auto& assembleOp = builder.CreateTensorOpStmt(function, Opcode::OP_ASSEMBLE, {matches[idx]}, {assembleResult});
        assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
            offsetOfOverlaps[idx], SymbolicScalar::FromConcrete(offsetOfOverlaps[idx])));
    }
    return assembleResult;
}

LogicalTensorPtr PassOperationUtils::HandleBeCovered(Function& function, LogicalTensorPtr iOperand,
                                                     const LogicalTensors& matches)
{
    auto viewResult = std::make_shared<LogicalTensor>(function, matches.front()->tensor->datatype, iOperand->shape,
                                                      iOperand->Format(), "View_" + matches.front()->tensor->symbol);
    IRBuilder irBuilder;
    auto& viewOp = irBuilder.CreateTensorOpStmt(function, Opcode::OP_VIEW, {matches.front()}, {viewResult});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(iOperand->GetOffset(), iOperand->GetDynOffset(),
                                                            iOperand->GetDynValidShape()));
    if (!iOperand->GetDynValidShape().empty()) {
        viewResult->UpdateDynValidShape(iOperand->GetDynValidShape());
    }
    return viewResult;
}

LogicalTensorPtr PassOperationUtils::HandleBeCoveredByAll(Function& function, LogicalTensorPtr iOperand,
                                                          const LogicalTensors& matches,
                                                          const std::vector<std::vector<int64_t>>& offsetOfOverlaps)
{
    std::vector<int64_t> minimumOffset;
    std::vector<int64_t> maximumShape;
    CalcShapeAndOffsetOfGroup(matches, minimumOffset, maximumShape);

    auto assembleResult = std::make_shared<LogicalTensor>(function, matches[0]->Datatype(), maximumShape,
                                                          iOperand->Format(), "Assemble_" + matches[0]->Symbol());
    FE_ASSERT(FeError::NOT_EXIST, assembleResult->GetProducers().empty()) << "Assemble result should have no producers";
    IRBuilder builder;
    for (size_t idx = 0; idx < matches.size(); ++idx) {
        auto& assembleOp = builder.CreateTensorOpStmt(function, Opcode::OP_ASSEMBLE, {matches[idx]}, {assembleResult});
        assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
            offsetOfOverlaps[idx], SymbolicScalar::FromConcrete(offsetOfOverlaps[idx])));
    }

    auto viewResult = std::make_shared<LogicalTensor>(function, assembleResult->Datatype(), iOperand->shape,
                                                      iOperand->Format(), "View_" + assembleResult->Symbol());
    auto& viewOp = builder.CreateTensorOpStmt(function, Opcode::OP_VIEW, {assembleResult}, {viewResult});
    std::vector<int64_t> newOffset = TensorOffset::Sub(iOperand->GetOffset(), minimumOffset);
    std::vector<SymbolicScalar> newDynOffset = TensorOffset::Sub(iOperand->GetDynOffset(), minimumOffset);
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffset, newDynOffset, iOperand->GetDynValidShape()));
    return viewResult;
}

LogicalTensorPtr PassOperationUtils::ConnectWithOverlap(Function& function, LogicalTensorPtr iOperand)
{
    auto matches = function.GetTensorMap().Find(iOperand);
    if (matches.empty()) {
        return iOperand;
    }

    auto overlapStatus = CalcOverlap(iOperand, matches);
    FE_ASSERT(!matches.empty()) << "Matches should not be empty";

    std::vector<int64_t> minimumOffsets;
    auto offsetOfOverlaps = ProcessOffsetAdjustment(matches, minimumOffsets);

    switch (overlapStatus) {
        case OverlapStatus::PERFECTLY_MATCH_WITH_ALL:
            return HandlePerfectlyMatchWithAll(function, iOperand, matches, offsetOfOverlaps);
        case OverlapStatus::PERFECTLY_MATCH:
            return matches.front();
        case OverlapStatus::BE_COVERED:
            return HandleBeCovered(function, iOperand, matches);
        case OverlapStatus::BE_COVERED_BY_ALL:
            return HandleBeCoveredByAll(function, iOperand, matches, offsetOfOverlaps);
        default:
            FE_ASSERT(false) << "unexpected behavior";
    }

    FE_ASSERT(false) << "unexpected behavior";
    return nullptr;
}
} // namespace npu::tile_fwk
