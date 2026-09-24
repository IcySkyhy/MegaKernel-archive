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
 * \file graph_utils.cpp
 * \brief
 */

#include "graph_utils.h"
#include "pass_utils.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/pass_operation_utils.h"

#define MODULE_NAME "GraphUtils"

namespace npu {
namespace tile_fwk {

void GraphUtils::SetDynShape(Operation* newOp, const std::vector<std::vector<SymbolicScalar>>& outDynShape)
{
    if (outDynShape.empty()) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(newOp);
    } else {
        for (size_t i = 0; i < newOp->GetOOperands().size(); ++i) {
            newOp->GetOOperands()[i]->UpdateDynValidShape(outDynShape[i]);
        }
    }
}

Operation& GraphUtils::AddDynOperation(Function& function, const Opcode opCode, LogicalTensors iOperands,
                                       const LogicalTensors& oOperands,
                                       const std::vector<std::vector<SymbolicScalar>>& outDynShape)
{
    auto& newOp = PassOperationUtils::AddOperation(function, opCode, std::move(iOperands), oOperands, nullptr,
                                                   ir::Span::Unknown(), false);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation& GraphUtils::AddAssembleOperation(Function& function, const AssembleOp& assemble,
                                            const std::vector<std::vector<SymbolicScalar>>& outDynShape)
{
    IRBuilder builder;
    if (!IsAssembleLike(assemble.opcode)) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Invalid assemble-family opcode for AddAssembleOperation, fallback to OP_ASSEMBLE.");
    }
    const auto opcode = IsAssembleLike(assemble.opcode) ? assemble.opcode : Opcode::OP_ASSEMBLE;
    auto& newOp = builder.CreateTensorOpStmt(function, opcode, {assemble.input}, {assemble.output});
    if (assemble.originOp != nullptr) {
        newOp.SetScopeInfo(assemble.originOp->GetScopeInfo());
        newOp.CopyAttrFrom(*assemble.originOp, "");
    }
    SetAssembleAttr(newOp, assemble);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation& GraphUtils::AddReshapeOperation(Function& function, const LogicalTensorPtr iOperand,
                                           const LogicalTensorPtr& oOperand, const ReshapeOp& reshapeOp,
                                           const std::vector<SymbolicScalar>& outDynShape)
{
    auto& newOp = PassOperationUtils::AddOperation(function, Opcode::OP_RESHAPE, {iOperand}, {oOperand}, nullptr,
                                                   ir::Span::Unknown(), false);
    if (reshapeOp.originOpPtr != nullptr) {
        newOp.SetScopeInfo(reshapeOp.originOpPtr->GetScopeInfo());
        newOp.CopyAttrFrom(*reshapeOp.originOpPtr, "");
    }
    if (outDynShape.empty()) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(&newOp);
        std::vector<SymbolicScalar> validShape;
        if (!newOp.GetAttr(OP_ATTR_VALID_SHAPE, validShape) || validShape.empty()) {
            newOp.SetAttribute(OP_ATTR_VALID_SHAPE, oOperand->GetDynValidShape());
        }
    } else {
        newOp.SetAttribute(OP_ATTR_VALID_SHAPE, outDynShape);
        oOperand->UpdateDynValidShape(outDynShape);
    }
    return newOp;
}

void GraphUtils::CopyDynStatus(const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& srcTensor)
{
    dstTensor->UpdateDynValidShape(srcTensor->GetDynValidShape());
}

void GraphUtils::UpdateViewAttr(Function& function, Operation& op)
{
    LogicalTensorPtr input = op.GetIOperands().front();
    LogicalTensorPtr output = op.GetIOperands().front();
    auto viewAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    if (function.IsFromInCast(input) || function.IsFromOutCast(output)) {
        if (viewAttribute->GetFromDynOffset().empty()) {
            std::vector<int64_t> fromOffset = viewAttribute->GetFromOffset();
            std::vector<SymbolicScalar> fromDynOffset = CommonUtils::CreateConstIntVector(fromOffset);
            viewAttribute->SetFromOffset(fromOffset, fromDynOffset);
        }
    }
}

void GraphUtils::SetAssembleAttr(Operation& op, const AssembleOp& assemble)
{
    auto assembleOpAttribute = std::make_shared<AssembleOpAttribute>(assemble.from, assemble.toOffset);
    auto fromValidShape = assemble.input->GetDynValidShape();
    assembleOpAttribute->SetFromDynValidShape(fromValidShape);
    op.SetOpAttribute(assembleOpAttribute);
}

bool GraphUtils::IsCVMixPlatform()
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        return true;
    }
    return false;
}

TensorSet GraphUtils::GetTensorsByRawMagic(Function& function, int64_t rawMagic)
{
    TensorSet result;

    for (const auto& tensor : function.inCasts_) {
        if (tensor && tensor->tensor && tensor->tensor->rawmagic == rawMagic) {
            result.insert(tensor);
        }
    }

    for (const auto& tensor : function.outCasts_) {
        if (tensor && tensor->tensor && tensor->tensor->rawmagic == rawMagic) {
            result.insert(tensor);
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetOOperands()) {
            if (tensor && tensor->tensor && tensor->tensor->rawmagic == rawMagic) {
                result.insert(tensor);
            }
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetIOperands()) {
            if (tensor && tensor->tensor && tensor->tensor->rawmagic == rawMagic) {
                result.insert(tensor);
            }
        }
    }
    return result;
}

std::shared_ptr<RawTensor> GraphUtils::GetRawTensorByRawMagic(Function& function, int64_t rawMagic)
{
    auto tensors = GetTensorsByRawMagic(function, rawMagic);
    if (tensors.empty()) {
        return nullptr;
    }
    const auto& firstTensor = *tensors.begin();
    if (firstTensor == nullptr) {
        return nullptr;
    }
    return firstTensor->tensor;
}

TensorSet GraphUtils::GetTensorsByActualRawMagic(Function& function, int64_t actualRawMagic)
{
    TensorSet result;

    for (const auto& tensor : function.inCasts_) {
        if (tensor && tensor->tensor && tensor->tensor->actualRawmagic == actualRawMagic) {
            result.insert(tensor);
        }
    }

    for (const auto& tensor : function.outCasts_) {
        if (tensor && tensor->tensor && tensor->tensor->actualRawmagic == actualRawMagic) {
            result.insert(tensor);
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetOOperands()) {
            if (tensor && tensor->tensor && tensor->tensor->actualRawmagic == actualRawMagic) {
                result.insert(tensor);
            }
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetIOperands()) {
            if (tensor && tensor->tensor && tensor->tensor->actualRawmagic == actualRawMagic) {
                result.insert(tensor);
            }
        }
    }
    return result;
}

std::vector<LogicalTensorPtr> GraphUtils::FindOverlappedTensors(Function& function, const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return {};
    }

    auto candidates = GraphUtils::GetTensorsByRawMagic(function, tensor->tensor->rawmagic);
    if (candidates.empty()) {
        if (!function.HasParent() ||
            function.IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::EXECUTE_GRAPH)) {
            return {};
        }
        return FindOverlappedTensors(function.Parent(), tensor);
    }

    std::vector<LogicalTensorPtr> result;
    for (const auto& candidate : candidates) {
        if (candidate == nullptr || candidate->tensor == nullptr) {
            continue;
        }
        if (candidate->magic == tensor->magic) {
            continue;
        }
        if (CalcOverlap(candidate, tensor) == OverlapStatus::NO_OVER_LAP) {
            continue;
        }
        result.push_back(candidate);
    }
    return result;
}

LogicalTensorPtr GraphUtils::GetTensorByMagic(Function& function, int magic)
{
    for (const auto& tensor : function.inCasts_) {
        if (tensor && tensor->GetMagic() == magic) {
            return tensor;
        }
    }

    for (const auto& tensor : function.outCasts_) {
        if (tensor && tensor->GetMagic() == magic) {
            return tensor;
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetOOperands()) {
            if (tensor && tensor->GetMagic() == magic) {
                return tensor;
            }
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetIOperands()) {
            if (tensor && tensor->GetMagic() == magic) {
                return tensor;
            }
        }
    }
    return nullptr;
}

TensorSet GraphUtils::GetAllTensors(Function& function)
{
    TensorSet result;
    for (const auto& tensor : function.inCasts_) {
        if (tensor) {
            result.insert(tensor);
        }
    }

    for (const auto& tensor : function.outCasts_) {
        if (tensor) {
            result.insert(tensor);
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetOOperands()) {
            if (tensor) {
                result.insert(tensor);
            }
        }
    }

    for (auto& op : function.Operations(false)) {
        for (const auto& tensor : op.GetIOperands()) {
            if (tensor) {
                result.insert(tensor);
            }
        }
    }
    return result;
}

bool GraphUtils::IsCrossCoreMoveOp(Operation* op)
{
    auto convertOpAttr = dynamic_cast<ConvertOpAttribute*>(op->GetOpAttribute().get());
    auto copyOpAttr = dynamic_cast<CopyOpAttribute*>(op->GetOpAttribute().get());
    if (convertOpAttr == nullptr && copyOpAttr == nullptr) {
        return false;
    }
    const std::unordered_set<MemoryType> aicMem{MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                                MemoryType::MEM_L0C};
    const std::unordered_set<MemoryType> aivMem{MemoryType::MEM_UB};
    bool fromAIC = false;
    bool fromAIV = false;
    for (auto& iop : op->GetIOperands()) {
        if (aicMem.count(iop->GetMemoryTypeToBe()) > 0) {
            fromAIC = true;
        } else if (aivMem.count(iop->GetMemoryTypeToBe()) > 0) {
            fromAIV = true;
        }
    }
    for (auto& oop : op->GetOOperands()) {
        if ((fromAIV && aicMem.count(oop->GetMemoryTypeToBe()) > 0) ||
            (fromAIC && aivMem.count(oop->GetMemoryTypeToBe()) > 0)) {
            return true;
        }
    }
    return false;
}
} // namespace tile_fwk
} // namespace npu
