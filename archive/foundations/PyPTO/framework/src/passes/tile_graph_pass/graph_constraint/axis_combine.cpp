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
 * \file axis_combine.cpp
 * \brief
 */

#include "axis_combine.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "AxisCombine"

namespace npu {
namespace tile_fwk {

constexpr size_t INPUT_SIZE = 2;

bool InsertCondition(const Opcode& code) { return SUPPORT_BRC_INLINE.count(code) > 0; }

Status AlignedIfNeed(int64_t& currentDim, int64_t padValue)
{
    if (padValue == 0) {
        APASS_LOG_ERROR_F(Elements::Config, "invalid pad base %ld.", static_cast<long>(padValue));
        return FAILED;
    }
    if (currentDim % padValue != 0) {
        currentDim = (currentDim + padValue - 1) / padValue * padValue;
    }
    return SUCCESS;
}

Status GetPaddingValue(const LogicalTensorPtr& tensor, int64_t& padValue)
{
    auto bytes = BytesOf(tensor->Datatype());
    auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
    if (paddingIter == BLOCK_PADDING_DIM.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "tensor %d's datatype is not supported.", tensor->GetMagic());
        return FAILED;
    }
    padValue = paddingIter->second;
    return SUCCESS;
}

inline int GetExpandDim(const std::vector<int64_t>& lhsShape, const std::vector<int64_t>& rhsShape)
{
    for (int i = static_cast<int>(lhsShape.size() - 1); i >= 0; --i) {
        if (lhsShape[i] != rhsShape[i]) {
            return i;
        }
    }
    return -1;
}

LogicalTensorPtr AxisCombine::CreateAlignedTensor(const LogicalTensorPtr& srcTensor,
                                                  const std::vector<int64_t>& alignedShape)
{
    auto alignedTensor = irBuilder_.CreateTensorVar(srcTensor->Datatype(), alignedShape, std::vector<SymbolicScalar>{},
                                                    srcTensor->Format());
    alignedTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    return alignedTensor;
}

static void UpdateOperand(Operation& op, size_t idx, const LogicalTensorPtr& oldTensor,
                          const LogicalTensorPtr& newTensor, std::vector<LogicalTensorPtr>& inputTensor)
{
    oldTensor->RemoveConsumer(op);
    op.ReplaceIOperand(idx, newTensor);
    inputTensor[idx] = newTensor;
}

static std::vector<SymbolicScalar> BuildAlignedValidShape(IRBuilder& irBuilder, LogicalTensors& inputTensor, int idx,
                                                          Shape& shape)
{
    int expandDim = inputTensor[idx]->GetShape().size() - 1;
    auto dynValidShape = CommonUtils::CreateConstIntVector(shape);
    if (!(inputTensor[idx]->GetDynValidShape().empty())) {
        dynValidShape = inputTensor[idx]->GetDynValidShape();
    }
    if (!(inputTensor[idx ^ 1]->GetDynValidShape().empty())) {
        dynValidShape[expandDim] = inputTensor[idx ^ 1]->GetDynValidShape()[expandDim];
    } else {
        dynValidShape[expandDim] = irBuilder.CreateConstInt(inputTensor[idx ^ 1]->GetShape()[expandDim]);
    }
    return dynValidShape;
}

static void UpdateBrcOperandAfterExpand(Operation& op)
{
    if (!op.HasAttr(OpAttributeKey::brcOperand)) {
        return;
    }
    auto brcOperand = op.GetVectorIntAttribute(OpAttributeKey::brcOperand);
    if (!brcOperand.empty()) {
        brcOperand.back() = 0;
    }
    bool anyBrc = std::any_of(brcOperand.begin(), brcOperand.end(), [](int64_t v) { return v != 0; });
    if (anyBrc) {
        op.SetAttribute(OpAttributeKey::brcOperand, brcOperand);
    } else {
        op.RemoveAttr(OpAttributeKey::brcOperand);
    }
}

Status AxisCombine::AlignBroadCastOpInputs([[maybe_unused]] Function& function, Operation& op)
{
    auto inputTensor = op.GetIOperands();
    auto& inTensor0 = inputTensor[0];
    auto& inTensor1 = inputTensor[1];
    const auto& shape0 = inTensor0->GetShape();
    const auto& shape1 = inTensor1->GetShape();
    if (shape0 == shape1 || shape0.back() == shape1.back()) {
        return SUCCESS;
    }
    for (size_t idx = 0; idx < INPUT_SIZE; ++idx) {
        auto srcTensor = inputTensor[idx];
        auto otherTensor = inputTensor[idx ^ 1];
        auto alignedShape = srcTensor->GetShape();
        if (alignedShape.back() != 1) {
            continue;
        }
        int64_t padValue = 0;
        if (GetPaddingValue(srcTensor, padValue) != SUCCESS) {
            return FAILED;
        }
        const bool enableAxisCombine = axisCombineMarker.IsTensorEnableAxisCombine(srcTensor);
        const bool isDAV3510 = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
        if (!enableAxisCombine) {
            padValue = otherTensor->GetShape().back();
            if (AlignedIfNeed(alignedShape.back(), padValue) != SUCCESS) {
                return FAILED;
            }
            auto alignedTensor = CreateAlignedTensor(srcTensor, alignedShape);
            auto& expand = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_EXPAND, {srcTensor}, {alignedTensor});
            auto validShape = BuildAlignedValidShape(irBuilder_, inputTensor, idx, alignedShape);
            expand.SetAttribute(OpAttributeKey::expandDims,
                                std::vector<int>{static_cast<int>(inputTensor[idx]->GetShape().size() - 1)});
            expand.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
            alignedTensor->UpdateDynValidShape(validShape);
            expand.UpdateSubgraphID(op.GetSubgraphID());
            UpdateOperand(op, idx, srcTensor, alignedTensor, inputTensor);
            UpdateBrcOperandAfterExpand(op);
            continue;
        } else if (!isDAV3510 && !IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
            if (AlignedIfNeed(alignedShape.back(), padValue) != SUCCESS) {
                return FAILED;
            }
            auto alignedTensor = CreateAlignedTensor(srcTensor, alignedShape);
            auto& brcb = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_BRCB, {srcTensor}, {alignedTensor});
            auto validShape = BuildAlignedValidShape(irBuilder_, inputTensor, idx, alignedShape);
            alignedTensor->UpdateDynValidShape(validShape);
            brcb.UpdateSubgraphID(op.GetSubgraphID());
            UpdateOperand(op, idx, srcTensor, alignedTensor, inputTensor);
        }
        op.SetAttribute(OpAttributeKey::brcbIdx, static_cast<int64_t>(idx + 1));
    }
    return SUCCESS;
}

Status AxisCombine::Process(Function& function)
{
    for (auto& op : function.Operations()) {
        if (InsertCondition(op.GetOpcode()) && op.GetIOperands().size() == INPUT_SIZE) {
            if (AlignBroadCastOpInputs(function, op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "operation %d's aligned faild. %s", op.GetOpMagic(),
                                  op.GetOpcodeStr().c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status AxisCombine::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start AxisCombine.");
    if (!function.paramConfigs_.combineAxis) {
        APASS_LOG_INFO_F(Elements::Operation, "AxisCombine is skipped.");
        return SUCCESS;
    }
    axisCombineMarker.Run(function);
    if (Process(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AxisCombine process failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End AxisCombine.");
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
