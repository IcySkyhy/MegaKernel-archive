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
 * \file infer_discontinuous_input_utils.cpp
 * \brief utils of infer discontinuous input
 */

#include "passes/pass_utils/infer_discontinuous_input_utils.h"
#include "interface/configs/config_manager_ng.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "InferDiscontinuousInput"

namespace npu::tile_fwk {
inline bool ShapeToSize(Shape& shapes, int64_t& out)
{
    auto [result, overflow] = CommonUtils::SafeMultiplyShape(shapes);
    if (overflow) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Shape multiply overflow.");
        return false;
    }
    out = result;
    return true;
}

inline bool PerfectOffsetOverlap(std::vector<int>& rawTensorIds, std::vector<Shape>& rawShapes,
                                 std::vector<Shape>& shapes, std::vector<Offset>& offsets,
                                 std::vector<Offset>& offsetTos)
{
    std::unordered_map<int, Offset> rawIdToRawOffset;
    std::unordered_map<int, int64_t> rawEmptySize;
    std::unordered_map<int, int64_t> rawValueSize;
    for (size_t i = 0; i < rawTensorIds.size(); i++) {
        int rawId = rawTensorIds[i];
        Offset rawOffset(rawShapes[i].size(), 0);
        for (size_t dim = 0; dim < rawShapes[i].size(); dim++) {
            rawOffset[dim] = offsetTos[i][dim] - offsets[i][dim];
        }
        if (rawIdToRawOffset.count(rawId) == 0) {
            rawIdToRawOffset[rawId] = rawOffset;
        } else {
            if (rawIdToRawOffset[rawId] != rawOffset) {
                return false;
            }
        }
        int64_t rawSize = 0;
        if (rawEmptySize.count(rawId) == 0) {
            if (!ShapeToSize(rawShapes[i], rawSize)) {
                return false;
            }
            rawEmptySize[rawId] = rawSize;
        }
        int64_t sliceSize = 0;
        if (!ShapeToSize(shapes[i], sliceSize)) {
            return false;
        }
        rawValueSize[rawId] += sliceSize;
    }
    for (const auto& rawPr : rawEmptySize) {
        if (rawPr.second != rawValueSize[rawPr.first]) {
            return false;
        }
    }
    return true;
}

inline bool IsTraceableView(Operation* cur)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(cur->GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        return false;
    }
    for (auto& fromOffset : viewOpAttribute->GetFromDynOffset()) {
        if (fromOffset.IsSymbol()) {
            return false;
        }
        if (fromOffset.IsExpression()) {
            return false;
        }
    }
    for (auto& dynShape : viewOpAttribute->GetToDynValidShape()) {
        if (dynShape.IsSymbol()) {
            return false;
        }
        if (dynShape.IsExpression()) {
            return false;
        }
    }
    return true;
}

inline bool NoViewConflict(Function& function,
                           const std::vector<std::pair<LogicalTensorPtr, Operation*>>& inplaceTensors)
{
    std::vector<Operation*> viewOps(inplaceTensors.size(), nullptr);
    for (size_t i = 0; i < inplaceTensors.size(); i++) {
        auto tensor = inplaceTensors[i].first;
        for (auto& producer : tensor->GetProducers()) {
            if (IsViewLike(producer->GetOpcode())) {
                viewOps[i] = producer;
            }
        }
    }
    for (size_t i = 0; i < inplaceTensors.size(); i++) {
        if (viewOps[i] == nullptr) {
            continue;
        }
        // dynamic view check
        if (!IsTraceableView(viewOps[i])) {
            return false;
        }
        // incast outcast Check
        for (auto& producerTensor : viewOps[i]->GetIOperands()) {
            if (FunctionUtils::GetNodeType(*producerTensor, function) != NodeType::LOCAL) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::pair<LogicalTensorPtr, Operation*>> InferDiscontinuousInputUtils::GetInplacedTileTensors(
    LogicalTensorPtr targetTensor)
{
    static const std::unordered_set<Opcode> inplaceNodes{Opcode::OP_VIEW,     Opcode::OP_SLICE,
                                                         Opcode::OP_ASSEMBLE, Opcode::OP_CONTRACT,
                                                         Opcode::OP_RESHAPE,  Opcode::OP_INDEX_OUTCAST};
    std::vector<std::pair<LogicalTensorPtr, Operation*>> inplacedTensor;
    for (auto& producer : targetTensor->GetProducers()) {
        if (inplaceNodes.count(producer->GetOpcode()) == 0) {
            continue;
        }
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            // consumers 可能为空（如 RemoveRedundantOp 删除消费者后），此时 *begin() 会解引用 end 迭代器
            const auto& consumers = targetTensor->GetConsumers();
            if (consumers.empty()) {
                continue;
            }
            auto consumerOp = *consumers.begin();
            if (consumerOp == nullptr || IsAssembleLike(consumerOp->GetOpcode())) {
                continue;
            }
            constexpr int kIndexOutcastInplaceInputIdx = 2;
            inplacedTensor.emplace_back(
                std::make_pair(producer->GetInputOperand(kIndexOutcastInplaceInputIdx), producer));
            continue;
        }
        for (auto& inputTensor : producer->GetIOperands()) {
            inplacedTensor.emplace_back(std::make_pair(inputTensor, producer));
        }
    }
    return inplacedTensor;
}

std::vector<size_t> InferDiscontinuousInputUtils::GetInputTileConflict(
    Function& function, const std::vector<std::pair<LogicalTensorPtr, Operation*>>& inplaceTensors,
    bool checkViewConflict)
{
    std::vector<int> rawTensorMagics;
    std::vector<Shape> rawShapes;
    std::vector<Shape> shapes;
    std::vector<Offset> offsets;
    std::vector<Offset> offsetTos;

    bool assembleCheck = true;
    for (const auto& pr : inplaceTensors) {
        if (pr.first->GetMemoryTypeOriginal() != pr.second->GetOOperands()[0]->GetMemoryTypeOriginal()) {
            assembleCheck = false;
            break;
        }
        std::shared_ptr<AssembleOpAttribute> attr = std::dynamic_pointer_cast<AssembleOpAttribute>(
            pr.second->GetOpAttribute());
        if (attr == nullptr) {
            assembleCheck = false;
            break;
        }
        offsetTos.push_back(attr->GetToOffset());
        rawTensorMagics.push_back(pr.first->GetRawMagic());
        rawShapes.push_back(pr.first->GetRawTensor()->GetRawShape());
        shapes.push_back(pr.first->GetShape());
        offsets.push_back(pr.first->GetOffset());
    }
    std::vector<size_t> copyIdx;
    if (!assembleCheck) {
        return {};
    }
    // RemoveRedundantOp 末尾调用时跳过 NoViewConflict：REGISTER_COPY 删除后 ASSEMBLE 输入可能
    // 穿透到上游 VIEW(inCast)，但 ASSEMBLE 输出位置不重叠即无冲突，无需因 VIEW 输入非 LOCAL 而插 copy
    bool isOffsetOverlap = PerfectOffsetOverlap(rawTensorMagics, rawShapes, shapes, offsets, offsetTos);
    bool noViewConflict = checkViewConflict ? NoViewConflict(function, inplaceTensors) : true;
    if (!(isOffsetOverlap && noViewConflict) && inplaceTensors.size() > 1) {
        for (size_t i = 0; i < inplaceTensors.size(); i++) {
            copyIdx.push_back(i);
        }
    }

    return copyIdx;
}

void InferDiscontinuousInputUtils::DDRTensorAssignUB(
    Function& function, std::unordered_map<LogicalTensorPtr, std::unordered_set<Operation*>>& insertedNodes)
{
    auto opList = function.Operations(true, SortOperationsMode::LIGHTWEIGHT).DuplicatedOpList();
    insertedNodes.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); ++i) {
        Operation* currOp = opList[i];
        if (!IsAssembleLike(currOp->GetOpcode())) {
            continue;
        }
        for (LogicalTensorPtr ioperand : currOp->GetIOperands()) {
            if (ioperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            auto& producers = ioperand->GetProducers();
            if (producers.size() != 1 || !IsViewLike((*producers.begin())->GetOpcode())) {
                continue;
            }
            auto& inOp = *producers.begin();
            auto viewOut = inOp->GetOOperands().front();
            auto outShape = viewOut->GetShape();
            bool isDynAxis = false;
            for (size_t dim = 0; dim < outShape.size(); dim++) {
                if (outShape[dim] < 0) {
                    insertedNodes[ioperand].insert(currOp);
                    isDynAxis = true;
                    break;
                }
            }
            if (isDynAxis) {
                continue;
            }
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(inOp->GetOpAttribute());
            viewOpAttribute->SetToType(MemoryType::MEM_UB);
            ioperand->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            ioperand->SetMemoryTypeToBe(MemoryType::MEM_UB);
            insertedNodes[ioperand].insert(currOp);
        }
    }
}

Status InferDiscontinuousInputUtils::Process(Function& function, bool checkViewConflict)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferDiscontinuousInput for function [%s].",
                     function.GetRawName().c_str());
    Init(function);
    if (InferFromIncast(function, checkViewConflict) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Infer INCAST and OUTCAST address failed.");
        return FAILED;
    }
    if (InsertTensorCopy(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Insert copy op failed.");
        return FAILED;
    }
    if (!newOps_.empty()) {
        if (InferShapeUtils::InferShape(function, newOps_) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
            return FAILED;
        }
    }

    APASS_LOG_INFO_F(Elements::Function, "===> End InferDiscontinuousInput for function [%s].",
                     function.GetRawName().c_str());
    return SUCCESS;
}

std::vector<std::pair<LogicalTensorPtr, Operation*>> InferDiscontinuousInputUtils::FilterCopyScenes(
    Function& function, const std::vector<std::pair<LogicalTensorPtr, Operation*>>& inplaceTensors,
    bool checkViewConflict)
{
    std::vector<std::pair<LogicalTensorPtr, Operation*>> needInsertCopys;
    if (inplaceTensors.empty()) {
        return needInsertCopys;
    }
    auto copyIdx = GetInputTileConflict(function, inplaceTensors, checkViewConflict);
    for (auto idx : copyIdx) {
        needInsertCopys.push_back(inplaceTensors[idx]);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Input tensor [%d] conflict.", inplaceTensors[idx].first->GetMagic());
    }
    return needInsertCopys;
}

void InferDiscontinuousInputUtils::Init(Function& function)
{
    auto opList = function.Operations(true, SortOperationsMode::LIGHTWEIGHT).DuplicatedOpList();
    tensorProducers_.reserve(opList.size());
    for (auto currOp : opList) {
        for (auto outTensor : currOp->GetOOperands()) {
            tensorProducers_[outTensor] = outTensor->GetProducers().size();
        }
    }
}

// 从INCAST出发，按前向推导
Status InferDiscontinuousInputUtils::InferFromIncast(Function& function, bool checkViewConflict)
{
    auto opList = function.Operations(true, SortOperationsMode::LIGHTWEIGHT).DuplicatedOpList();
    insertCopys_.reserve(opList.size());
    for (auto currOp : opList) {
        for (auto& outputTensor : currOp->GetOOperands()) {
            auto& producerCnt = tensorProducers_[outputTensor];
            producerCnt--;
            if (producerCnt != 0) {
                continue;
            }
            auto inplacedTensor = GetInplacedTileTensors(outputTensor);
            auto filteredTensor = FilterCopyScenes(function, inplacedTensor, checkViewConflict);
            insertCopys_.emplace(outputTensor, std::move(filteredTensor));
        }
    }
    return SUCCESS;
}

void InferDiscontinuousInputUtils::InsertViewOp(Function& function, LogicalTensorPtr iOperand,
                                                LogicalTensorPtr oOperand)
{
    auto& insertViewOp = irBuilder_.CreateTensorOpStmt(function, config::GetSliceOpcode(), {iOperand}, {oOperand});
    newOps_.push_back(&insertViewOp);
    insertViewOp.SetOpAttribute(
        std::make_shared<ViewOpAttribute>(iOperand->GetOffset(), oOperand->GetMemoryTypeOriginal(),
                                          iOperand->GetDynOffset(), iOperand->GetDynValidShape()));
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert %s op [%d].", insertViewOp.GetOpcodeStr().c_str(),
                      insertViewOp.GetOpMagic());
}

void InferDiscontinuousInputUtils::InsertAssembleOp(Function& function, LogicalTensorPtr iOperand,
                                                    LogicalTensorPtr oOperand)
{
    auto& insertAssembleOp = irBuilder_.CreateTensorOpStmt(function, config::GetContractOpcode(), {iOperand},
                                                           {oOperand});
    newOps_.push_back(&insertAssembleOp);
    insertAssembleOp.SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(iOperand->GetMemoryTypeOriginal(), oOperand->GetOffset(),
                                              oOperand->GetDynOffset(), oOperand->GetDynValidShape()));
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert %s op [%d].", insertAssembleOp.GetOpcodeStr().c_str(),
                      insertAssembleOp.GetOpMagic());
}

void InferDiscontinuousInputUtils::InsertCopyOp(Function& function, LogicalTensorPtr iOperand,
                                                LogicalTensorPtr oOperand)
{
    if ((iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) &&
        (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(iOperand->Datatype(),
                                                                              iOperand->GetShape(), iOperand->Format());
        Offset newOffset(iOperand->GetShape().size(), 0);
        LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, newOffset, iOperand->GetShape(),
                                                                iOperand->GetDynValidShape());
        newTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        newTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
        InsertViewOp(function, iOperand, newTensor);
        InsertAssembleOp(function, newTensor, oOperand);
        return;
    }
    if ((iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) &&
        (oOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR)) {
        InsertViewOp(function, iOperand, oOperand);
        return;
    }
    if ((iOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) &&
        (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
        InsertAssembleOp(function, iOperand, oOperand);
        return;
    }
    std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(iOperand->Datatype(), iOperand->GetShape(),
                                                                          iOperand->Format());
    Offset newOffset(iOperand->GetShape().size(), 0);
    LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, newOffset, iOperand->GetShape(),
                                                            iOperand->GetDynValidShape());
    newTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    newTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    InsertAssembleOp(function, iOperand, newTensor);
    InsertViewOp(function, newTensor, oOperand);
}

Status InferDiscontinuousInputUtils::InsertTensorCopy(Function& function)
{
    std::unordered_map<LogicalTensorPtr, std::unordered_set<Operation*>> insertedNodes;
    DDRTensorAssignUB(function, insertedNodes);
    // 按 tensor magic 排序后遍历，保证插入顺序确定性
    std::vector<std::pair<LogicalTensorPtr, std::vector<std::pair<LogicalTensorPtr, Operation*>>>> sortedCopys(
        insertCopys_.begin(), insertCopys_.end());
    std::sort(sortedCopys.begin(), sortedCopys.end(),
              [](const auto& a, const auto& b) { return a.first->GetMagic() < b.first->GetMagic(); });
    for (auto& copyInserts : sortedCopys) {
        auto& inplaceNodes = copyInserts.second;
        for (auto& inplaceNode : inplaceNodes) {
            auto& inputTensor = inplaceNode.first;
            auto& nodeSet = insertedNodes[inputTensor];
            if (nodeSet.count(inplaceNode.second) != 0U) {
                continue;
            }
            nodeSet.insert(inplaceNode.second);
            std::shared_ptr<RawTensor> newRawTensor = irBuilder_.CreateRawTensor(
                inputTensor->Datatype(), inputTensor->GetShape(), inputTensor->Format());
            Offset newOffset(inputTensor->GetShape().size(), 0);
            LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, newOffset, inputTensor->GetShape(),
                                                                    inputTensor->GetDynValidShape());
            LogicalTensorPtr customTensor = inplaceNode.second->GetOOperands()[0];
            newTensor->SetMemoryTypeOriginal(customTensor->GetMemoryTypeOriginal(), true);
            newTensor->SetMemoryTypeToBe(newTensor->GetMemoryTypeOriginal());
            InsertCopyOp(function, inputTensor, newTensor);
            inputTensor->RemoveConsumer(inplaceNode.second);
            inplaceNode.second->ReplaceInput(newTensor, inputTensor);
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
