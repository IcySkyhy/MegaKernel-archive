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
 * \file mix_call_operation_builder.CPP
 * \brief
 */

#include "passes/pass_utils/subfunc_utils.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_call_operation_builder.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
Status MixCallOperationBuilder::CreateCallOps(Function& rootFunc, const std::vector<Operation*>& originalCallOps,
                                              Function* originalMixFunc,
                                              const std::vector<InternalComponentInfo>& components,
                                              const std::vector<uint64_t>& newProgramIDs,
                                              SubgraphToFunction& subgraphToFunction,
                                              std::vector<Function*>& newFunctions)
{
    APASS_LOG_INFO_F(Elements::Operation, "Creating call operations for %zu original call ops and %zu components",
                     originalCallOps.size(), components.size());
    std::vector<CallOpCreationInfo> callOpInfos;
    // 处理所有指向同构originalMixFunc的originalCallOps
    for (auto* originalCallOp : originalCallOps) {
        // 为每个原始callOp分配唯一的wrapId
        uint64_t wrapId = nextWrapId_++;
        APASS_LOG_DEBUG_F(Elements::Operation, "Assigning wrapId=%lu for original callOp %d", wrapId,
                          originalCallOp->GetOpMagic());
        for (size_t i = 0; i < components.size(); i++) {
            CallOpCreationInfo info;
            info.leafFunc = newFunctions[i];
            info.newProgramID = newProgramIDs[i];
            info.componentIndex = i;
            info.originalCallOp = originalCallOp;
            info.wrapId = wrapId;
            auto status = CreateCallOpInRootFunction(rootFunc, *info.leafFunc, info.newProgramID, info.componentIndex,
                                                     info.originalCallOp, originalMixFunc, subgraphToFunction, info);
            if (status != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Failed to create call op for component %zu",
                                  info.componentIndex);
                return FAILED;
            }
            if (!info.createdCallOp) {
                APASS_LOG_ERROR_F(Elements::Operation, "Created call op is null for component %zu",
                                  info.componentIndex);
                return FAILED;
            }
            // 记录这个call op的信息
            callOpInfos.push_back(info);
            APASS_LOG_DEBUG_F(Elements::Operation, "Created call op %d in info for component %zu (wrapId=%lu)",
                              info.createdCallOp->GetOpMagic(), info.componentIndex, wrapId);
        }
    }
    return SUCCESS;
}

Status MixCallOperationBuilder::CreateCallOpInRootFunction(Function& rootFunc, Function& leafFunc,
                                                           uint64_t newProgramID, uint64_t componentIndex,
                                                           Operation* originalCallOp, Function* originalMixFunc,
                                                           SubgraphToFunction& subgraphToFunction,
                                                           CallOpCreationInfo& info)
{
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Creating callOp in root function for leaf: %s, programID=%lu, component=%lu, wrapId=%lu",
                      leafFunc.GetRawName().c_str(), newProgramID, componentIndex, info.wrapId);
    auto originalCallAttr = dynamic_cast<CallOpAttribute*>(originalCallOp->GetOpAttribute().get());
    if (originalCallAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Original callOp %d has no CallOpAttribute",
                          originalCallOp->GetOpMagic());
        return FAILED;
    }
    // 获取原始callOp的operands
    auto originalIOperands = originalCallOp->GetIOperands();
    auto originalOOperands = originalCallOp->GetOOperands();
    APASS_LOG_DEBUG_F(Elements::Operation, "Original callOp %d has %zu iOperands, %zu oOperands",
                      originalCallOp->GetOpMagic(), originalIOperands.size(), originalOOperands.size());
    auto originalIncasts = originalMixFunc->GetIncast();
    auto originalOutcasts = originalMixFunc->GetOutcast();
    APASS_LOG_DEBUG_F(Elements::Function, "Original mix function %s has %zu incasts, %zu outcasts",
                      originalMixFunc->GetRawName().c_str(), originalIncasts.size(), originalOutcasts.size());
    // 从invokeInfo获取incast和outcast参数信息
    const auto& invokeInfo = subgraphToFunction.subFuncInvokeInfos[componentIndex];
    // 构建新的operands列表
    std::vector<LogicalTensorPtr> newIOperands;
    std::vector<LogicalTensorPtr> newOOperands;
    // 用于跟踪已经处理过的tensor
    std::set<LogicalTensorPtr> processedIncasts;
    std::set<LogicalTensorPtr> processedOutcasts;
    // 处理直接外部依赖的IOperands
    APASS_LOG_INFO_F(Elements::Tensor, "===> FindIOperandsAndOOperands start.");
    FindNewIOperandsInOriginalIncast(originalIOperands, originalIncasts, invokeInfo, newIOperands, processedIncasts);
    // 处理直接外部依赖的OOperands
    FindNewOOperandsInOriginalOutcast(originalOOperands, originalOutcasts, invokeInfo, newOOperands, processedOutcasts);
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();
    // 处理传播依赖的IOperands和OOperands
    FindNewIOperandsAndOOperandsInPropagateInOutcast(originalIOperands, originalOOperands, originalIncasts,
                                                     originalOutcasts, actualIncasts, actualOutcasts, newIOperands,
                                                     newOOperands, processedIncasts, processedOutcasts);
    APASS_LOG_INFO_F(Elements::Tensor, "===> FindIOperandsAndOOperands end.");
    auto& callOp = irBuilder_.CreateTensorOpStmt(rootFunc, Opcode::OP_CALL, newIOperands, newOOperands);
    APASS_LOG_INFO_F(Elements::Tensor, "Created operands for new callOp %d: %zu inputs, %zu outputs",
                     callOp.GetOpMagic(), newIOperands.size(), newOOperands.size());
    // 寻找新call op的IOpAttrOffset和OOpAttrOffset
    FindIOpAttrOffsetAndOOpAttrOffset(leafFunc, invokeInfo, info.iOffsets, info.oOffsets, originalMixFunc);
    callOp.SetOperandAttr(info.iOffsets, info.oOffsets);
    SetCallOpAttribute(leafFunc, callOp, originalCallOp, originalCallAttr, newProgramID, componentIndex,
                       subgraphToFunction, info);
    // 将创建的call op记录到info中
    info.createdCallOp = &callOp;
    APASS_LOG_INFO_F(Elements::Operation, "Successfully created callOp %d in root function for programID=%lu, leaf=%s",
                     callOp.GetOpMagic(), newProgramID, leafFunc.GetRawName().c_str());
    return SUCCESS;
}

void MixCallOperationBuilder::FindNewIOperandsInOriginalIncast(
    const std::vector<LogicalTensorPtr>& originalIOperands,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalIncasts, const SubfuncInvokeInfoTy& invokeInfo,
    std::vector<LogicalTensorPtr>& newIOperands, std::set<LogicalTensorPtr>& processedIncasts) const
{
    // 为incast构建新的iOperands
    for (const auto& incastParam : invokeInfo.GetIncastTensorParamList()) {
        int tensorMagic = incastParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedIncasts.insert(incastParam.tensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "  Found: tensor magic=%d -> original iOperand[%d] (tensor magic=%d)",
                              tensorMagic, originalIndex, originalIOperands[originalIndex]->magic);
        }
    }
    // 为global tensor输入构建新的iOperands
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic == -1 || tensorParam.isOutputToGM) {
            continue;
        }
        int tensorMagic = tensorParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedIncasts.insert(tensorParam.tensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "  Found: global input tensor magic=%d -> original iOperand[%d]",
                              tensorMagic, originalIndex);
        }
    }
}

void MixCallOperationBuilder::FindNewOOperandsInOriginalOutcast(
    const std::vector<LogicalTensorPtr>& originalOOperands,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalOutcasts, const SubfuncInvokeInfoTy& invokeInfo,
    std::vector<LogicalTensorPtr>& newOOperands, std::set<LogicalTensorPtr>& processedOutcasts) const
{
    // 为outcast构建新的oOperands
    for (const auto& outcastParam : invokeInfo.GetOutcastTensorParamList()) {
        int tensorMagic = outcastParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOOperands[originalIndex]);
            processedOutcasts.insert(outcastParam.tensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "  Found: tensor magic=%d -> original oOperand[%d] (tensor magic=%d)",
                              tensorMagic, originalIndex, originalOOperands[originalIndex]->magic);
        }
    }
    // 为global tensor输出构建新的oOperands
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic == -1 || tensorParam.tensor == nullptr || !tensorParam.isOutputToGM) {
            continue;
        }
        int tensorMagic = tensorParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOOperands[originalIndex]);
            processedOutcasts.insert(tensorParam.tensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "  Found: global output tensor magic=%d -> original oOperand[%d]",
                              tensorMagic, originalIndex);
        }
    }
}

void MixCallOperationBuilder::FindNewIOperandsAndOOperandsInPropagateInOutcast(
    const std::vector<LogicalTensorPtr>& originalIOperands, const std::vector<LogicalTensorPtr>& originalOOperands,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalIncasts,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalOutcasts,
    const std::vector<std::shared_ptr<LogicalTensor>>& actualIncasts,
    const std::vector<std::shared_ptr<LogicalTensor>>& actualOutcasts, std::vector<LogicalTensorPtr>& newIOperands,
    std::vector<LogicalTensorPtr>& newOOperands, std::set<LogicalTensorPtr>& processedIncasts,
    std::set<LogicalTensorPtr>& processedOutcasts) const
{
    // 处理传播的incast
    for (const auto& incast : actualIncasts) {
        // 检查是否已经在之前的列表中处理过
        if (processedIncasts.count(incast) > 0) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "  Propagated incast tensor magic=%d already processed, skipping",
                              incast->magic);
            continue;
        }
        int tensorMagic = incast->magic;
        APASS_LOG_DEBUG_F(Elements::Tensor, "  Checking propagated incast tensor magic=%d", tensorMagic);
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedIncasts.insert(incast);
            APASS_LOG_DEBUG_F(Elements::Tensor, "    Found: propagated incast tensor magic=%d -> original iOperand[%d]",
                              tensorMagic, originalIndex);
        }
    }
    // 处理传播的outcast
    for (const auto& outcast : actualOutcasts) {
        if (processedOutcasts.count(outcast) > 0) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "  Propagated outcast tensor magic=%d already processed, skipping",
                              outcast->magic);
            continue;
        }
        int tensorMagic = outcast->magic;
        APASS_LOG_DEBUG_F(Elements::Tensor, "  Checking propagated outcast tensor magic=%d", tensorMagic);
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOOperands[originalIndex]);
            processedOutcasts.insert(outcast);
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "    Found: propagated outcast tensor magic=%d -> original oOperand[%d]", tensorMagic,
                              originalIndex);
        }
    }
}

// 辅助函数：在tensor列表中查找指定magic的tensor索引
int MixCallOperationBuilder::FindTensorIndexInList(int tensorMagic,
                                                   const std::vector<LogicalTensorPtr>& tensorList) const
{
    for (size_t i = 0; i < tensorList.size(); ++i) {
        if (tensorList[i] != nullptr && tensorList[i]->magic == tensorMagic) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MixCallOperationBuilder::FindIOpAttrOffsetAndOOpAttrOffset(Function& leafFunc,
                                                                const SubfuncInvokeInfoTy& invokeInfo,
                                                                std::vector<OperandAttribute>& iOffsets,
                                                                std::vector<OperandAttribute>& oOffsets,
                                                                Function* originalMixFunc) const
{
    // 清空offset向量
    iOffsets.clear();
    oOffsets.clear();
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();

    APASS_LOG_INFO_F(Elements::Function, "===> FindIOpAttrOffsetAndOOpAttrOffset start.");
    APASS_LOG_DEBUG_F(Elements::Function,
                      "Leaf function %s has %zu actual incasts, %zu actual outcasts after dependency propagation",
                      leafFunc.GetRawName().c_str(), actualIncasts.size(), actualOutcasts.size());
    std::set<LogicalTensorPtr> processedIncasts;
    std::set<LogicalTensorPtr> processedOutcasts;
    // 处理直接参数（在原始invokeInfo中能找到的）
    ExtractInfo extractInfo{iOffsets, oOffsets, processedIncasts, processedOutcasts};
    // 使用invokeInfo中预先构造的incast信息
    if (!FindIOpAttrOffsetFromIncast(invokeInfo, leafFunc, extractInfo)) {
        return;
    }
    // 使用invokeInfo中的outcast信息
    if (!FindOOpAttrOffsetFromOutcast(invokeInfo, leafFunc, extractInfo)) {
        return;
    }
    // 使用invokeInfo中的global tensor信息
    if (!FindIOOpAttrOffsetGlobalTensor(invokeInfo, leafFunc, extractInfo)) {
        return;
    }
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    if (!FindIOpAttrOffsetFromActualIncasts(actualIncasts, extractInfo, originalMixFunc)) {
        return;
    }
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    if (!FindOOpAttrOffsetFromActualOutcasts(actualOutcasts, extractInfo, originalMixFunc)) {
        return;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> FindIOpAttrOffsetAndOOpAttrOffset end.");
}

bool MixCallOperationBuilder::FindIOpAttrOffsetFromIncast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc,
                                                          ExtractInfo& extractInfo) const
{
    // 使用invokeInfo中预先构造的incast信息
    for (const auto& in : invokeInfo.GetIncastTensorParamList()) {
        if (in.opMagic == -1) { // 跳过无效的incast
            continue;
        }
        auto attr = GetOperandAttr(in.opMagic, in.operandIdx, leafFunc, false);
        if (attr.offset == -1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to get offset for incast (op=%d, idx=%d)!", in.opMagic,
                              in.operandIdx);
            continue;
        }
        extractInfo.iOffsets.push_back(attr);
        extractInfo.processedIncasts.insert(in.tensor);
        APASS_LOG_DEBUG_F(Elements::Operation, "Incast (op=%d, idx=%d)", in.opMagic, in.operandIdx);
    }
    return true;
}

bool MixCallOperationBuilder::FindOOpAttrOffsetFromOutcast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc,
                                                           ExtractInfo& extractInfo) const
{
    // 使用invokeInfo中预先构造的outcast信息
    for (const auto& out : invokeInfo.GetOutcastTensorParamList()) {
        if (out.opMagic == -1) {
            continue;
        }
        auto attr = GetOperandAttr(out.opMagic, out.operandIdx, leafFunc, true);
        if (attr.offset == -1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to get offset for outcast (op=%d, idx=%d)!", out.opMagic,
                              out.operandIdx);
            continue;
        }
        extractInfo.oOffsets.push_back(attr);
        extractInfo.processedOutcasts.insert(out.tensor);
        APASS_LOG_DEBUG_F(Elements::Operation, "Outcast (op=%d, idx=%d)", out.opMagic, out.operandIdx);
    }
    return true;
}

// 统一的offset获取函数
OperandAttribute MixCallOperationBuilder::GetOperandAttr(int opMagic, int operandIdx, Function& leafFunc,
                                                         bool isOutput) const
{
    auto operations = leafFunc.Operations(false);
    for (auto& op : operations) {
        if (op.GetOpMagic() != opMagic) {
            continue;
        }
        if (isOutput) {
            if (operandIdx >= 0 && static_cast<size_t>(operandIdx) < op.GetOOperands().size()) {
                return op.GetOOpAttr(operandIdx);
            }
        } else {
            if (operandIdx >= 0 && static_cast<size_t>(operandIdx) < op.GetIOperands().size()) {
                return op.GetIOpAttr(operandIdx);
            }
        }
    }

    APASS_LOG_WARN_F(Elements::Operation, "Could not find offset for op %d idx %d (isOutput=%d)", opMagic, operandIdx,
                     isOutput);
    return -1;
}

bool MixCallOperationBuilder::FindIOOpAttrOffsetGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc,
                                                             ExtractInfo& extractInfo) const
{
    for (const auto& tensor : invokeInfo.GetTensorParamList()) {
        if (tensor.opMagic == -1) {
            continue;
        }
        OperandAttribute attr = GetOperandAttr(tensor.opMagic, tensor.operandIdx, leafFunc, tensor.isOutputToGM);
        if (attr.offset == -1) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to get offset for global tensor (op=%d, idx=%d, isOutput=%d)!",
                              tensor.opMagic, tensor.operandIdx, tensor.isOutputToGM);
            continue;
        }
        if (tensor.isOutputToGM) {
            extractInfo.oOffsets.push_back(attr);
            extractInfo.processedOutcasts.insert(tensor.tensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "Global tensor -> Outcast: opmagic=%d, idx=%d -> oOpAttrOffset=%d",
                              tensor.opMagic, tensor.operandIdx, attr.offset);
        } else {
            extractInfo.iOffsets.push_back(attr);
            extractInfo.processedIncasts.insert(tensor.tensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "Global tensor -> Incast: opmagic=%d, idx=%d -> iOpAttrOffset=%d",
                              tensor.opMagic, tensor.operandIdx, attr.offset);
        }
    }
    return true;
}

bool MixCallOperationBuilder::FindIOpAttrOffsetFromActualIncasts(
    const std::vector<std::shared_ptr<LogicalTensor>>& actualIncasts, ExtractInfo& extractInfo,
    Function* originalMixFunc) const
{
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    for (const auto& incast : actualIncasts) {
        if (incast == nullptr || extractInfo.processedIncasts.count(incast) > 0) {
            continue;
        }

        // 这是传播依赖添加的参数，需要特殊处理
        // 在原始Mix function中查找这个tensor的offset
        auto attr = FindOriginalAttrInMixFunction(incast, originalMixFunc);
        if (attr.offset == -1) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to find offset for propagated incast tensor %d!",
                              incast->GetRawMagic());
            return false; // 直接报错返回
        }
        extractInfo.iOffsets.push_back(attr);
        extractInfo.processedIncasts.insert(incast);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Extracted propagated incast: tensor rawmagic = %d, attr.offset = %d",
                          incast->GetRawMagic(), attr.offset);
    }
    return true;
}

bool MixCallOperationBuilder::FindOOpAttrOffsetFromActualOutcasts(
    const std::vector<std::shared_ptr<LogicalTensor>>& actualOutcasts, ExtractInfo& extractInfo,
    Function* originalMixFunc) const
{
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    for (const auto& outcast : actualOutcasts) {
        if (outcast == nullptr || extractInfo.processedOutcasts.count(outcast) > 0) {
            continue;
        }
        auto shape = outcast->GetShape();
        if (shape.empty()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Propagated outcast tensor %d has empty shape!",
                              outcast->GetRawMagic());
            return false;
        }
        auto attr = FindOriginalAttrInMixFunction(outcast, originalMixFunc);
        if (attr.offset == -1) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to find offset for propagated outcast tensor %d!",
                              outcast->GetRawMagic());
            return false; // 直接报错返回
        }
        extractInfo.oOffsets.push_back(attr);
        extractInfo.processedOutcasts.insert(outcast);
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Extracted propagated outcast: tensor rawmagic = %d -> original attr.offset = %d",
                          outcast->GetRawMagic(), attr.offset);
    }
    return true;
}

OperandAttribute MixCallOperationBuilder::FindOriginalAttrInMixFunctionByRawMagic(LogicalTensorPtr tensor,
                                                                                  Function* originalMixFunc) const
{
    auto operations = originalMixFunc->Operations(false);
    int rawMagic = tensor->GetRawMagic();
    for (auto& op : operations) {
        if (op.IsNOP())
            continue;
        auto iOperands = op.GetIOperands();
        for (size_t i = 0; i < iOperands.size(); i++) {
            auto inputTensor = iOperands[i];
            if (inputTensor->GetRawMagic() == rawMagic) {
                auto attr = op.GetIOpAttr(i);
                if (attr.offset != -1) {
                    return attr;
                }
            }
        }
        auto oOperands = op.GetOOperands();
        for (size_t k = 0; k < oOperands.size(); k++) {
            auto outputTensor = oOperands[k];
            if (outputTensor->GetRawMagic() == rawMagic) {
                auto attr = op.GetOOpAttr(k);
                if (attr.offset != -1) {
                    return attr;
                }
            }
        }
    }
    return -1;
}

OperandAttribute MixCallOperationBuilder::FindOriginalAttrInMixFunction(LogicalTensorPtr tensor,
                                                                        Function* originalMixFunc) const
{
    if (tensor == nullptr || originalMixFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor or function is nullptr in FindOriginalOffsetInMixFunction");
        return -1;
    }
    int rawMagic = tensor->GetRawMagic();
    APASS_LOG_DEBUG_F(Elements::Tensor, "Finding original offset for tensor raw magic=%d in function %s", rawMagic,
                      originalMixFunc->GetRawName().c_str());
    auto operations = originalMixFunc->Operations(false);
    for (auto& op : operations) {
        if (op.IsNOP())
            continue;
        auto iOperands = op.GetIOperands();
        for (size_t i = 0; i < iOperands.size(); i++) {
            auto inputTensor = iOperands[i];
            if (inputTensor.get() == tensor.get()) {
                auto attr = op.GetIOpAttr(i);
                if (attr.offset != -1) {
                    return attr;
                }
            }
        }
        auto oOperands = op.GetOOperands();
        for (size_t i = 0; i < oOperands.size(); i++) {
            auto outputTensor = oOperands[i];
            if (outputTensor.get() == tensor.get()) {
                auto attr = op.GetOOpAttr(i);
                if (attr.offset != -1) {
                    return attr;
                }
            }
        }
    }
    OperandAttribute attrByMagic = FindOriginalAttrInMixFunctionByRawMagic(tensor, originalMixFunc);
    if (attrByMagic.offset != -1) {
        return attrByMagic;
    }
    APASS_LOG_ERROR_F(Elements::Tensor, "Tensor raw magic=%d not found in function %s operations", rawMagic,
                      originalMixFunc->GetRawName().c_str());
    return -1;
}

// 设置新创建的callOp的属性
void MixCallOperationBuilder::SetCallOpAttribute(Function& leafFunc, Operation& callOp, Operation* originalCallOp,
                                                 CallOpAttribute* originalCallAttr, uint64_t newProgramID,
                                                 uint64_t componentIndex, SubgraphToFunction& subgraphToFunction,
                                                 CallOpCreationInfo& info)
{
    if (originalCallAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Original callOp %d has no CallOpAttribute",
                          originalCallOp->GetOpMagic());
        return;
    }
    // 使用原mix callOp的argList
    auto argList = originalCallAttr->GetArgList();
    std::map<int, SymbolicScalar> outIndexToExpr;
    leafFunc.GetOutcastSymbolicExpr(outIndexToExpr);
    // 创建CallOpAttribute（使用从原始CallOp提取的argList）
    auto callAttr = leafFunc.CreateCallOpAttribute(argList, outIndexToExpr);
    auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(callAttr);
    if (callOpAttr != nullptr) {
        callOpAttr->wrapId = info.wrapId;
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Set wrapId=%lu to callOp attribute for programID=%lu (from original callOp %d)", info.wrapId,
                          newProgramID, originalCallOp->GetOpMagic());
    }
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(newProgramID);
    if (componentIndex < subgraphToFunction.subFuncInvokeInfos.size()) {
        callOp.SetSubFuncInvokeInfo(subgraphToFunction.subFuncInvokeInfos[componentIndex]);
    }
    if (callOpAttr != nullptr && callOpAttr->invokeInfo_ != nullptr) {
        callOpAttr->invokeInfo_->UpdateProgramSubgraphId(newProgramID);
    }
    subgraphToFunction.SetSemanticLabel(leafFunc.GetProgramOp(), callOp);
    APASS_LOG_DEBUG_F(
        Elements::Operation,
        "Created callOp %d: %zu arg blocks (from original callOp %d), %zu input offsets, %zu output offsets",
        callOp.GetOpMagic(), argList.size(), originalCallOp->GetOpMagic(), info.iOffsets.size(), info.oOffsets.size());
}
} // namespace tile_fwk
} // namespace npu
