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
 * \file inplace_process.cpp
 * \brief
 */

#include "inplace_process.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "InplaceProcess"

namespace npu {
namespace tile_fwk {
bool InplaceProcess::HasSameConsecutive(Operation& op)
{
    for (const auto& nextOp : op.ConsumerOps()) {
        if (nextOp->GetOpcode() == op.GetOpcode()) {
            return true;
        }
    }
    return false;
}
Status InplaceProcess::PreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for InplaceProcess.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Loopcheck failed before PreGraph; Please check whether there is a loop.");
        return FAILED;
    }
    for (auto& op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] is not partitioned; Please check subGraphIDs. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) &&
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        if (HasSameConsecutive(op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has the same Opcode child op; Plese check child ops. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "unmatched input output memory type for reshape opmagic: %d, input mem type: %s, output mem type: %s; "
                "Please check the input ans output.",
                op.opmagic, MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for InplaceProcess success.");
    return SUCCESS;
}

Status InplaceProcess::InplaceProcessAssemble(Function& function, Operation& op)
{
    if (ValidMeaninglessOp(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Invalid assemble operation; Please check operands size and memory type. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto assembleOut = op.GetOOperands().front();
    // 校验Assemble输出的汇聚后tensor大小是否超过UB上限
    const int UB_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    if (assembleOut->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
        (assembleOut->tensor->GetRawDataSize() > UB_SIZE)) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "Local Buffer Assemble Result Oversized, %d, tensor: %d, size: %ld B; Please check the result size.",
            op.opmagic, assembleOut->magic, assembleOut->tensor->GetRawDataSize());
        return FAILED;
    }
    ProcessAssemble(function, op);
    return SUCCESS;
}

Status InplaceProcess::ProcessOp(Function& function)
{
    auto opList = function.Operations();
    for (auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (ValidMeaninglessOp(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Invalid view operation; Please check operands size and memory type. %s",
                                  GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            ProcessView(function, op);
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW_TYPE) {
            if (ValidMeaninglessOp(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Invalid view operation; Please check operands size and memory type.");
                return FAILED;
            }
            if (ProcessViewType(function, op) == FAILED) {
                return FAILED;
            }
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (std::find(hubRelatedAssembleOpMagics.begin(), hubRelatedAssembleOpMagics.end(), op.GetOpMagic()) !=
                hubRelatedAssembleOpMagics.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "skip processing HUB-related ASSEMBLE[%d]", op.GetOpMagic());
                continue;
            }
            if (InplaceProcessAssemble(function, op) != SUCCESS) {
                return FAILED;
            }
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_HUB) {
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            if (ValidMeaninglessOp(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Invalid reshape operation; Please check operands size and memory type. %s",
                                  GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            ProcessReshape(function, op);
            continue;
        }
        if (ProcessInplaceOp(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Processing inplace op %s[%d] failed. %s", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status InplaceProcess::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start InplaceProcess.");
    auto opList = function.Operations();
    hubRelatedAssembleOpMagics.clear();
    if (ProcessOp(function) != SUCCESS) {
        return FAILED;
    }
    // 将View提取到由inplace最开始的tensor调用，并在原地留下一个NOP保持依赖关系。
    if (RefactorViewConnectForInplace(function) != SUCCESS) {
        return FAILED;
    }
    for (auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_HUB) {
            ProcessHub(function, op);
        }
    }

    APASS_LOG_INFO_F(Elements::Operation, "===> End InplaceProcess.");
    return SUCCESS;
}

void InplaceProcess::ProcessHub(Function& function, Operation& op)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Processing HUB node %d.", op.GetOpMagic());

    // 获取 HUB 的输入和输出 tensor
    auto hubInput = op.GetIOperands()[0];  // HUB 的输入 tensor
    auto hubOutput = op.GetOOperands()[0]; // HUB 的输出 tensor

    // 1. 查找 HUB 输出的所有消费者（应该是 ASSEMBLE 节点）
    auto consumers = hubOutput->GetConsumers();
    for (const auto consumer : consumers) {
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ProcessHubAssembleChain(function, op, *consumer, hubInput, hubOutput);
        }
    }
    auto producers = hubInput->GetProducers();
    for (auto producer : producers) {
        if (OpcodeManager::Inst().IsCopyOut(producer->GetOpcode())) {
            auto copyAttr = dynamic_cast<CopyOpAttribute*>(producer->GetOpAttribute().get());
            if (copyAttr) {
                auto attrOffset = copyAttr->GetToOffset(); // OpImm
                auto tensorOffset = OpImmediate::Specified(hubInput->GetTensorOffset());
                std::vector<OpImmediate> newOffset;
                for (size_t i = 0; i < attrOffset.size(); i++) {
                    newOffset.push_back(attrOffset[i] + tensorOffset[i]);
                }
                copyAttr->SetToOffset(newOffset);
            }
        }
    }
}

void InplaceProcess::ProcessHubAssembleChain(Function& function, Operation& hubOp, Operation& assembleOp,
                                             std::shared_ptr<LogicalTensor> hubInput,
                                             std::shared_ptr<LogicalTensor> hubOutput)
{
    auto assembleInput = assembleOp.GetIOperands()[0];
    auto assembleOutput = assembleOp.GetOOperands()[0];
    if (assembleInput.get() != hubOutput.get()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Assemble input[%d] is not HUB output[%d], chain may be broken",
                         assembleInput->GetMagic(), hubOutput->GetMagic());
        return;
    }
    bool isExactOutcast = false;
    auto outcasts = function.GetOutcast();
    for (const auto& outcast : outcasts) {
        if (outcast.get() == assembleOutput.get()) {
            isExactOutcast = true;
            break;
        }
    }
    if (!isExactOutcast) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Assemble[%d] output is not exact outcast, skip HUB memory reuse processing.",
                         assembleOp.GetOpMagic());
        return;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Found exact HUB-ASSEMBLE-OUTCAST chain: HUB[%d] -> ASSEMBLE[%d] -> OUTCAST[%d]",
                     hubOp.GetOpMagic(), assembleOp.GetOpMagic(), assembleOutput->GetMagic());
    auto hubInputMemType = hubInput->GetMemoryTypeOriginal();
    auto hubOutputMemType = hubOutput->GetMemoryTypeOriginal();
    auto assembleOutputMemType = assembleOutput->GetMemoryTypeOriginal();
    if (hubInputMemType != hubOutputMemType || hubInputMemType != assembleOutputMemType) {
        APASS_LOG_WARN_F(Elements::Tensor, "Memory type mismatch: HUB input=%d, HUB output=%d, ASSEMBLE output=%d",
                         hubInputMemType, hubOutputMemType, assembleOutputMemType);
        return;
    }
    hubInput->tensor = assembleOutput->tensor;
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute*>(assembleOp.GetOpAttribute().get());
    hubInput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    hubOutput->tensor = assembleOutput->tensor;
    hubOutput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    hubRelatedAssembleOpMagics.push_back(assembleOp.GetOpMagic());
    APASS_LOG_INFO_F(Elements::Tensor, "Complete memory reuse established: all tensors share HUB input[%d] memory",
                     hubInput->GetMagic());
}

Status InplaceProcess::ValidMeaninglessOp(const Operation& op) const
{
    // 校验单输入单输出，且输入输出mem类型相同
    if ((op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) || (op.GetIOperands().front() == nullptr) ||
        (op.GetOOperands().front() == nullptr) ||
        (op.GetIOperands().front()->GetMemoryTypeOriginal() != op.GetOOperands().front()->GetMemoryTypeOriginal())) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "InplaceProcess %s[%d] Invalid: IOperands.size is %zu; OOperands.size is %zu; "
            "IOperands.front is nullptr (%d); OOperands.front is nullptr (%d); IOperands.front.MemoryType is %d; "
            "OOperands.front.MemoryType is %d. %s",
            (op.GetOpcodeStr().c_str()), (op.GetOpMagic()), (op.GetIOperands().size()), (op.GetOOperands().size()),
            (op.GetIOperands().front() == nullptr), (op.GetOOperands().front() == nullptr),
            (op.GetIOperands().front()->GetMemoryTypeOriginal()), (op.GetOOperands().front()->GetMemoryTypeOriginal()),
            GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

void InplaceProcess::ProcessView(Function& function, Operation& op) const
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Find Internal View %d.", op.opmagic);
    std::vector<int64_t> inputOffset = op.GetIOperands()[0]->GetOffset();
    for (auto& consumer : op.GetIOperands()[0]->GetConsumers()) {
        if ((consumer->GetOpcode() != Opcode::OP_VIEW) || (consumer->GetOpMagic() != op.GetOpMagic())) {
            continue;
        }
        if (function.IsFromOutCast(consumer->oOperand[0])) {
            APASS_LOG_WARN_F(Elements::Tensor, "InplaceProcess::ProcessView: OP_VIEW oOperand tensor[%d] is outCast.",
                             consumer->oOperand[0]->GetMagic());
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute*>(consumer->GetOpAttribute().get());
        if (viewAttr == nullptr) {
            continue;
        }
        std::vector<int64_t> viewOpOffset = viewAttr->GetFrom();
        auto inputDynOffset = op.GetIOperands()[0]->GetDynOffset();
        if (inputDynOffset.empty()) {
            inputDynOffset = std::vector<SymbolicScalar>(inputOffset.size(), 0);
        }
        auto attrDynOffset = viewAttr->GetFromDynOffset();
        std::vector<SymbolicScalar> outTensorOffset;
        // 增加校验: input --> View --> output 三者的offset size 相同
        for (size_t i = 0; i < inputOffset.size(); i++) {
            viewOpOffset[i] = inputOffset[i] + viewOpOffset[i];
            if (attrDynOffset.size() == inputOffset.size()) {
                attrDynOffset[i] = inputDynOffset[i] + attrDynOffset[i];
            }
        }
        viewAttr->SetFromOffset(viewOpOffset, viewAttr->GetFromDynOffset());
        function.UpdateLinkMap(consumer->oOperand[0], op.GetIOperands()[0]);
        consumer->oOperand[0]->tensor = op.GetIOperands()[0]->tensor;
        TensorOffset newOffset(viewOpOffset, attrDynOffset);
        consumer->oOperand[0]->UpdateOffset(newOffset);
    }
}

Status InplaceProcess::ProcessViewType(Function& function, Operation& op) const
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Find Internal ViewType %d.", op.opmagic);
    auto viewTypeIn = op.GetIOperands()[0];
    auto viewTypeOut = op.GetOOperands()[0];
    if (function.IsFromInCast(viewTypeIn)) {
        viewTypeOut->tensor->actualRawmagic = viewTypeIn->GetRawMagic();
        if (AdjustOffsetAndRawShape(viewTypeIn, viewTypeOut) == FAILED) {
            return FAILED;
        }
        return AlignCopyInConsumer(viewTypeOut);
    }
    if (function.IsFromOutCast(viewTypeOut)) {
        viewTypeIn->tensor->actualRawmagic = viewTypeOut->GetRawMagic();
        if (AdjustOffsetAndRawShape(viewTypeOut, viewTypeIn) == FAILED) {
            return FAILED;
        }
        return AlignCopyOutProducer(viewTypeIn);
    }
    viewTypeOut->tensor->actualRawmagic = viewTypeIn->GetRawMagic();
    return SUCCESS;
}

Status InplaceProcess::AdjustOffsetAndRawShape(LogicalTensorPtr& fromView, LogicalTensorPtr& toView) const
{
    auto fromType = fromView->tensor->datatype;
    auto toType = toView->tensor->datatype;
    auto inEntry = viewTypeTable.find(fromType);
    auto outEntry = viewTypeTable.find(toType);
    if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
        return FAILED;
    }
    int inSize = inEntry->second;
    int outSize = outEntry->second;
    int ratio = inSize > outSize ? inSize / outSize : outSize / inSize;
    bool isExpand = inSize > outSize;
    std::vector<int64_t> fromOffset = fromView->GetOffset();
    std::vector<int64_t> toOffset(fromOffset.size(), 0);
    std::vector<int64_t> inShape = fromView->GetRawTensor()->rawshape;
    std::vector<int64_t> outShape(inShape.size(), 0);
    for (size_t i = 0; i < fromOffset.size(); ++i) {
        if (i != fromOffset.size() - 1) {
            toOffset[i] = fromOffset[i];
            outShape[i] = inShape[i];
            continue;
        }
        if (isExpand) {
            toOffset[i] = fromOffset[i] * ratio;
            outShape[i] = inShape[i] * ratio;
        } else {
            if (fromOffset[i] % ratio != 0 || inShape[i] % ratio != 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "ViewType Offset is not Even.");
                return FAILED;
            }
            toOffset[i] = fromOffset[i] / ratio;
            outShape[i] = inShape[i] / ratio;
        }
    }
    toView->UpdateOffset(toOffset);
    toView->GetRawTensor()->rawshape = outShape;
    return SUCCESS;
}

Status InplaceProcess::AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const
{
    IRBuilder builder;
    APASS_LOG_DEBUG_F(Elements::Tensor, "InplaceProcess::AlignCopyInConsumer tensor[%d].", tensorGm->magic);
    for (auto& consumerOp : tensorGm->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_COPY_IN) {
            std::shared_ptr<CopyOpAttribute> opAttr = std::static_pointer_cast<CopyOpAttribute>(
                consumerOp->GetOpAttribute());
            if (opAttr == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "InplaceProcess::AlignCopyInConsumer OP_COPY_IN %d has no Attribute.",
                                  consumerOp->GetOpMagic());
                return FAILED;
            }
            std::vector<OpImmediate> newFromOffset;
            for (size_t i = 0; i < opAttr->GetFromOffset().size(); i++) {
                newFromOffset.push_back(opAttr->GetFromOffset()[i] +
                                        OpImmediate::Specified(builder.CreateConstInt(tensorGm->offset[i])));
            }
            opAttr->SetFromOffset(newFromOffset);
            opAttr->SetRawShape(OpImmediate::Specified(tensorGm->tensor->GetRawShape()));
        }
    }
    return SUCCESS;
}

Status InplaceProcess::AlignCopyOutProducer(std::shared_ptr<LogicalTensor> tensorGm) const
{
    IRBuilder builder;
    APASS_LOG_DEBUG_F(Elements::Tensor, "InplaceProcess::AlignCopyOutProducer tensor[%d].", tensorGm->magic);
    for (auto& producerOp : tensorGm->GetProducers()) {
        if (producerOp->GetOpcode() == Opcode::OP_COPY_OUT) {
            std::shared_ptr<CopyOpAttribute> opAttr = std::static_pointer_cast<CopyOpAttribute>(
                producerOp->GetOpAttribute());
            if (opAttr == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "InplaceProcess::AlignCopyOutProducer OP_COPY_OUT %d has no Attribute.",
                                  producerOp->GetOpMagic());
                return FAILED;
            }
            std::vector<OpImmediate> newToOffset;
            for (size_t i = 0; i < opAttr->GetToOffset().size(); i++) {
                newToOffset.push_back(opAttr->GetToOffset()[i] +
                                      OpImmediate::Specified(builder.CreateConstInt(tensorGm->offset[i])));
            }
            opAttr->SetToOffset(newToOffset);
            opAttr->SetRawShape(OpImmediate::Specified(tensorGm->tensor->GetRawShape()));
            APASS_LOG_DEBUG_F(Elements::Operation, "InplaceProcess::AlignCopyOutProducer update Attr for %s[%d].",
                              producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
        }
    }
    return SUCCESS;
}

void InplaceProcess::ReplaceRawTensor(Function& function, std::shared_ptr<LogicalTensor> logicalTensor,
                                      const std::shared_ptr<LogicalTensor> targetTensor, const Operation& op)
{
    if (function.IsFromInCast(logicalTensor)) {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "InplaceProcess::ProcessAssemble: OP_ASSEMBLE iOperand tensor[%d] is inCast.",
                         logicalTensor->GetMagic());
        return;
    }
    logicalTensor->tensor = targetTensor->tensor;
    logicalTensor->UpdateOffset(dynamic_cast<AssembleOpAttribute*>(op.GetOpAttribute().get())->GetToTensorOffset());
    /*
        需要将所有和logicalTensor共用一个raw 的所有logical tensor 都刷新
        当前仅往前更新一层
        inplace op1 --> tensor1 --> inplace op2 --> ... --> inplace opN --> tensorN --> Assemble --> T(可能是OCAST)
        后续需要进行优化
     */
    for (const auto& producerOp : logicalTensor->GetProducers()) {
        if (ProcessInplaceOp(function, *producerOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Processing inplace op %s[%d] failed after updating %s[%d]. %s",
                              producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(), op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        }
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "update the offset for Tensor %d.", logicalTensor->magic);
}

void InplaceProcess::ProcessAssemble(Function& function, Operation& op)
{
    auto assembleIn = op.GetIOperands().front();
    auto assembleOut = op.GetOOperands().front();
    bool fromIncast = function.IsFromInCast(assembleIn);
    APASS_LOG_DEBUG_F(Elements::Operation, "assembleIn from Incast: %d.", fromIncast);

    // check each producer of the assem_result
    for (const auto& producer : assembleOut->GetProducers()) {
        if ((producer->GetOpcode() != Opcode::OP_ASSEMBLE) ||
            std::find(visitedAssembleOp.begin(), visitedAssembleOp.end(), producer->GetOpMagic()) !=
                visitedAssembleOp.end()) {
            continue;
        }
        /*
        producer->iOperand[0] 可能来自一个被复用过的op
        raw tensor 与 producer->iOperand[0] 的 raw tensor 相同的所有logical tensor 都应该update
        */
        ReplaceRawTensor(function, producer->iOperand[0], assembleOut, *producer);
        visitedAssembleOp.push_back(producer->GetOpMagic());
    }
}

void InplaceProcess::ProcessReshape(Function& function, Operation& op) const
{
    auto reshapeIn = op.GetIOperands()[0];
    auto reshapeOut = op.GetOOperands()[0];
    APASS_LOG_DEBUG_F(Elements::Operation, " %s[%d] on %s.", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
                      BriefMemoryTypeToString(reshapeIn->GetMemoryTypeOriginal()).c_str());
    if (reshapeOut->tensor->actualRawmagic != -1) {
        return;
    }
    if (function.IsFromOutCast(reshapeOut)) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "InplaceProcess::ProcessReshape: OP_RESHAPE oOperand tensor[%d] is outCast.",
                         reshapeOut->GetMagic());
        return;
    }
    reshapeOut->tensor->actualRawmagic = reshapeIn->GetRawMagic();
    APASS_LOG_DEBUG_F(Elements::Operation, "Update reshape opmagic %d, output's actualRaw: %d.", op.opmagic,
                      reshapeOut->GetRawMagic());
}

Status InplaceProcess::ProcessInplaceOp(Function& function, Operation& op) const
{
    auto opcode = op.GetOpcode();
    std::vector<std::pair<size_t, size_t>> reusePairList;
    if (inplaceOpMap.find(opcode) != inplaceOpMap.end()) {
        reusePairList = inplaceOpMap.at(opcode);
    } else if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
        reusePairList.emplace_back(op.GetIntAttribute(OpAttributeKey::inplaceIdx), 0);
    } else {
        return SUCCESS;
    }
    for (const auto& reusePair : reusePairList) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        if (inputIdx >= op.GetIOperands().size() || outputIdx >= op.GetOOperands().size()) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Invalid inplace op info for %s[%d]. Please check op inputs&outputs, supported inplace info "
                "can be found in inplace_process.h."
                "\n|----detect input size: %zu, recorded inplace input idx: %zu."
                "\n|----detect output size: %zu, recorded inplace output idx: %zu. %s",
                op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands().size(), inputIdx,
                op.GetOOperands().size(), outputIdx, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands()[inputIdx];
        auto tensorOut = op.GetOOperands()[outputIdx];
        if (tensorIn == nullptr || tensorOut == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s[%d] inplace input or output is nullptr.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic());
            return FAILED;
        }
        if (function.IsFromOutCast(tensorOut) && function.IsFromInCast(tensorIn)) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "InplaceProcess::ProcessInplaceOp: inplaceOp iOperand tensor[%d] is inCast and oOperand "
                             "tensor[%d] is outCast.",
                             tensorIn->GetMagic(), tensorOut->GetMagic());
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] output %zu reuses input %zu.", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic(), outputIdx, inputIdx);
        if (function.IsFromOutCast(tensorOut)) {
            if ((tensorIn->tensor->symbol != "") && (tensorOut->tensor->symbol == "")) {
                tensorOut->tensor->symbol = tensorIn->tensor->symbol;
            }
            tensorIn->tensor = tensorOut->tensor;
            tensorIn->UpdateOffset(tensorOut->GetOffset());
            APASS_LOG_DEBUG_F(Elements::Tensor, "Output magic: %d, raw maigc: %d.", tensorOut->magic,
                              tensorOut->tensor->GetRawMagic());
            APASS_LOG_DEBUG_F(Elements::Tensor, "Input magic: %d, raw maigc: %d.", tensorIn->magic,
                              tensorIn->tensor->GetRawMagic());
            continue;
        }
        if ((tensorIn->tensor->symbol == "") && (tensorOut->tensor->symbol != "")) {
            tensorIn->tensor->symbol = tensorOut->tensor->symbol;
        }
        function.UpdateLinkMap(tensorOut, tensorIn);
        tensorOut->tensor = tensorIn->tensor;
        tensorOut->UpdateOffset(tensorIn->GetOffset());
        APASS_LOG_DEBUG_F(Elements::Tensor, "Output magic: %d, raw maigc: %d.", tensorOut->magic,
                          tensorOut->tensor->GetRawMagic());
        APASS_LOG_DEBUG_F(Elements::Tensor, "Input magic: %d, raw maigc: %d.", tensorIn->magic,
                          tensorIn->tensor->GetRawMagic());
    }
    return SUCCESS;
}

LogicalTensorPtr FindInplaceSource(Function& function, Operation& op,
                                   std::unordered_map<Operation*, LogicalTensorPtr>& visited)
{
    if (visited.count(&op) > 0) {
        return visited.at(&op);
    }
    auto inplaceIdx = op.GetIntAttribute(OpAttributeKey::inplaceIdx);
    ASSERT(OperationErr::OP_INVALID_OPERAND_COUNT,
           inplaceIdx >= 0 && inplaceIdx < static_cast<int>(op.GetIOperands().size()))
        << "Invalid inplaceIdx " << inplaceIdx << " for operation " << op.GetOpMagic();
    auto inplaceIOperand = op.GetInputOperand(inplaceIdx);
    LogicalTensorPtr res = nullptr;
    for (const auto producer : inplaceIOperand->GetProducers()) {
        if (!producer->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        auto tmp = FindInplaceSource(function, *producer, visited);
        if (res == nullptr) {
            res = tmp;
        } else {
            ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, res == tmp)
                << "Inconsistent inplace source for operation " << op.GetOpMagic(); // inplace路径应总是交汇于同一起点
        }
    }
    if (res == nullptr) {
        res = inplaceIOperand; // 向前没有inplace了，自己就是起点
    }
    visited.emplace(&op, res);
    return res;
}

static Status ProcessVisitedViewOps(Function& function, const std::unordered_map<Operation*, LogicalTensorPtr>& visited)
{
    for (auto& [op, srcTensor] : visited) {
        if (op->GetOpcode() != Opcode::OP_VIEW)
            continue; // 仅重构View连接
        auto inplaceIdx = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
        if (inplaceIdx != 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Inconsistent inplaceIdx for opsration %d", op->GetOpMagic());
            return FAILED;
        }
        auto iOperand = op->GetInputOperand(inplaceIdx);
        auto oOperand = op->GetOutputOperand(0);
        if (iOperand == srcTensor)
            continue; // 开头的VIEW不需要插入NOP来控制顺序
        bool hasNonInplaceConsumer = false;
        for (auto consumer : oOperand->GetConsumers()) {
            if (consumer->GetOpcode() != Opcode::OP_NOP && !consumer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                hasNonInplaceConsumer = true;
                break;
            }
        }
        if (hasNonInplaceConsumer) {
            continue;
        }
        if (iOperand->GetRawTensor() != srcTensor->GetRawTensor()) {
            APASS_LOG_ERROR_F(Elements::Operation, "RawTensor mismatch for input operand of operation %d",
                              op->GetOpMagic());
            return FAILED;
        }
        if (oOperand->GetRawTensor() != srcTensor->GetRawTensor()) {
            APASS_LOG_ERROR_F(Elements::Operation, "RawTensor mismatch for output operand of operation %d",
                              op->GetOpMagic());
            return FAILED;
        }
        op->ReplaceIOperand(0, srcTensor);
        // 含inplace语义，都为同一个RawTensor
        IRBuilder builder;
        auto nopOutput = builder.CreateTensorVar(srcTensor->GetRawTensor(), Offset(srcTensor->GetOffset().size()),
                                                 srcTensor->GetShape(), std::vector<SymbolicScalar>{});
        nopOutput->SetMemoryTypeBoth(oOperand->GetMemoryTypeOriginal());
        IRBuilder irBuilder;
        auto& nop = irBuilder.CreateTensorOpStmt(function, Opcode::OP_NOP, {iOperand, oOperand}, {nopOutput});
        nop.SetAttribute(OpAttributeKey::inplaceIdx, 0); // 期望上设成任何一个都可以，因为来源一致
        nop.UpdateSubgraphID(op->GetSubgraphID());
        auto consumers = oOperand->GetConsumers(); // deep copy
        for (auto consumer : consumers) {
            if (consumer->GetOpcode() == Opcode::OP_NOP || !consumer->HasAttribute(OpAttributeKey::inplaceIdx))
                continue;
            consumer->ReplaceIOperand(consumer->GetIntAttribute(OpAttributeKey::inplaceIdx), nopOutput);
        }
    }
    return SUCCESS;
}

Status InplaceProcess::RefactorViewConnectForInplace(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start RefactorViewConnectForInplace.");
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW)
            continue;
        if (op.GetInputOperand(0)->GetRawTensor() == op.GetOutputOperand(0)->GetRawTensor()) {
            op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        }
    }

    std::unordered_map<Operation*, LogicalTensorPtr> visited;
    for (Operation& op : function.Operations()) {
        if (!op.HasAttribute(OpAttributeKey::inplaceIdx) || visited.count(&op) > 0)
            continue;
        FindInplaceSource(function, op, visited);
    }

    Status status = ProcessVisitedViewOps(function, visited);
    if (status != SUCCESS) {
        return status; // 如果内部发生错误，直接向外透传 FAILED
    }

    APASS_LOG_INFO_F(Elements::Operation, "===> End RefactorViewConnectForInplace.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
