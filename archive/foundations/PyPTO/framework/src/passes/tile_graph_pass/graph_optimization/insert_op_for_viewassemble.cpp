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
 * \file insert_op_for_viewassemble.cpp
 * \brief
 */
#include "insert_op_for_viewassemble.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InsertOpForViewAssemble"

namespace npu {
namespace tile_fwk {
void InsertOpForViewAssemble::InsertViewAssemble(Function& function, Operation* viewOp, Operation* assembleOp)
{
    auto& moveOutTensorPtr = viewOp->GetOOperands()[0];
    auto shape = moveOutTensorPtr->GetShape();
    LogicalTensorPtr ddrTensorPtr = irBuilder_.CreateTensorVar(moveOutTensorPtr->Datatype(), shape,
                                                               std::vector<SymbolicScalar>{});
    ddrTensorPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    ddrTensorPtr->UpdateDynValidShape(moveOutTensorPtr->GetDynValidShape());
    LogicalTensorPtr moveInTensorPtr = irBuilder_.CreateTensorVar(moveOutTensorPtr->Datatype(), shape,
                                                                  std::vector<SymbolicScalar>{});
    moveInTensorPtr->SetMemoryTypeBoth(moveOutTensorPtr->GetMemoryTypeOriginal(), true);
    moveInTensorPtr->UpdateDynValidShape(moveOutTensorPtr->GetDynValidShape());
    std::vector<int64_t> offset(moveOutTensorPtr->GetShape().size(), 0);
    std::vector<SymbolicScalar> dynOffset(moveOutTensorPtr->GetShape().size(), 0);
    Operation& assemble = irBuilder_.CreateTensorOpStmt(function, config::GetContractOpcode(), {moveOutTensorPtr},
                                                        {ddrTensorPtr});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(moveOutTensorPtr->GetMemoryTypeOriginal(), offset,
                                                                  dynOffset, moveOutTensorPtr->GetDynValidShape()));
    Operation& view = irBuilder_.CreateTensorOpStmt(function, config::GetSliceOpcode(), {ddrTensorPtr},
                                                    {moveInTensorPtr});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, moveOutTensorPtr->GetMemoryTypeOriginal(), dynOffset,
                                                          moveOutTensorPtr->GetDynValidShape()));
    assembleOp->ReplaceInput(moveInTensorPtr, moveOutTensorPtr);
}

Status InsertOpForViewAssemble::InsertCopy(Function& function, Operation* viewOp, Operation* assOp)
{
    if (assOp->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        if (viewOp->GetOpcode() != config::GetSliceOpcode() || assOp->GetOpcode() != config::GetContractOpcode()) {
            return SUCCESS;
        }
        assOp->GetIOperands()[0]->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(viewOp->GetOpAttribute());
        if (viewOpAttr != nullptr) {
            viewOpAttr->SetToType(MemoryType::MEM_UB);
        }
        auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assOp->GetOpAttribute());
        if (assembleOpAttr != nullptr) {
            assembleOpAttr->SetFromType(MemoryType::MEM_UB);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Set Assemble Op[%d] iOperand MEM_DDR.", assOp->GetOpMagic());
    } else if (assOp->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
               assOp->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        APASS_LOG_INFO_F(Elements::Operation, "Insert Copy For View[%d], Assemble[%d].", viewOp->GetOpMagic(),
                         assOp->GetOpMagic());
        InsertViewAssemble(function, viewOp, assOp);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Assemble inTensor %d memory type is unexpected, InsertCopy failed.",
                          assOp->GetIOperands()[0]->GetMagic());
        return FAILED;
    }
    return SUCCESS;
}

bool InsertOpForViewAssemble::NeedInsertCopy(LogicalTensorPtr& assembleOut)
{
    bool isNeedInsert = false;
    for (auto& assOp : assembleOut->GetProducers()) {
        auto& prodOp = *assOp->GetIOperands()[0]->GetProducers().begin();
        if (!IsViewLike(prodOp->GetOpcode())) {
            isNeedInsert = true;
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] producerOp %s[%d] is not viewOp.", assOp->GetOpMagic(),
                             prodOp->GetOpcodeStr().c_str(), prodOp->GetOpMagic());
            continue;
        }
        recordOpPair_.push_back(std::make_pair(prodOp, assOp));
        if (isNeedInsert)
            continue;
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(assOp->GetOpAttribute());
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(prodOp->GetOpAttribute());
        if (assembleAttr == nullptr || viewAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "View or Assemble attribute is nullptr, NeedInsertCopy Failed.");
            return false;
        }
        if (assembleAttr->GetToOffset() != viewAttr->GetFromOffset()) {
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetToOffset is not equal to viewOp[%d] GetFromOffset.",
                             assOp->GetOpMagic(), prodOp->GetOpMagic());
            isNeedInsert = true;
            continue;
        }
        if (assembleAttr->GetToDynOffset().size() != viewAttr->GetFromDynOffset().size()) {
            APASS_LOG_INFO_F(Elements::Operation,
                             "assOp[%d] GetToDynOffset size is not equal to viewOp[%d] GetFromDynOffset size.",
                             assOp->GetOpMagic(), prodOp->GetOpMagic());
            isNeedInsert = true;
            continue;
        }
        for (size_t i = 0; i < assembleAttr->GetToDynOffset().size(); i++) {
            if (assembleAttr->GetToDynOffset()[i].Dump() != viewAttr->GetFromDynOffset()[i].Dump()) {
                APASS_LOG_INFO_F(Elements::Operation,
                                 "assOp[%d] GetToDynOffset value is not equal to viewOp[%d] GetFromDynOffset value.",
                                 assOp->GetOpMagic(), prodOp->GetOpMagic());
                isNeedInsert = true;
                break;
            }
        }
        auto viewIn = prodOp->GetIOperands()[0];
        auto inShape = viewIn->GetShape();
        auto outShape = assembleOut->GetShape();
        if (inShape.size() != outShape.size()) {
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetShape size is not equal to viewOp[%d] GetShape size.",
                             assOp->GetOpMagic(), prodOp->GetOpMagic());
            isNeedInsert = true;
            continue;
        }
        for (size_t i = 0; i < inShape.size(); i++) {
            if (inShape[i] != outShape[i]) {
                APASS_LOG_INFO_F(Elements::Operation,
                                 "assOp[%d] GetShape value is not equal to viewOp[%d] GetShape value.",
                                 assOp->GetOpMagic(), prodOp->GetOpMagic());
                isNeedInsert = true;
                break;
            }
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "isNeedInsert value is %d.", isNeedInsert);
    return isNeedInsert;
}

Status InsertOpForViewAssemble::JudgedViewAssemble(Function& function)
{
    for (auto& op : function.Operations()) {
        if (!IsAssembleLike(op.GetOpcode())) {
            continue;
        }
        auto& prodOp = *op.GetIOperands()[0]->GetProducers().begin();
        if (op.GetOOperands()[0]->GetMemoryTypeOriginal() != op.GetIOperands()[0]->GetMemoryTypeOriginal()) {
            if (notProcessOut_.find(op.GetOOperands()[0]) == notProcessOut_.end()) {
                notProcessOut_.insert(op.GetOOperands()[0]);
            }
            continue;
        }
        if (IsViewLike(prodOp->GetOpcode()) && assembleOutSet_.find(op.GetOOperands()[0]) == assembleOutSet_.end() &&
            prodOp->GetIOperands()[0]->GetMemoryTypeOriginal() == prodOp->GetOOperands()[0]->GetMemoryTypeOriginal() &&
            notProcessOut_.find(op.GetOOperands()[0]) == notProcessOut_.end()) {
            assembleOutSet_.insert(op.GetOOperands()[0]);
            APASS_LOG_INFO_F(Elements::Operation, "assembleOutSet_ insert oOperand %d",
                             op.GetOOperands()[0]->GetMagic());
        }
    }
    for (auto assembleOut : assembleOutSet_) {
        recordOpPair_.clear();
        if (NeedInsertCopy(assembleOut)) {
            for (auto& opPair : recordOpPair_) {
                auto viewOp = opPair.first;
                auto assOp = opPair.second;
                if (InsertCopy(function, viewOp, assOp) == FAILED) {
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status InsertOpForViewAssemble::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start InsertOpForViewAssemble");
    if (JudgedViewAssemble(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "JudgedViewAssemble Failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End InsertOpForViewAssemble");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
