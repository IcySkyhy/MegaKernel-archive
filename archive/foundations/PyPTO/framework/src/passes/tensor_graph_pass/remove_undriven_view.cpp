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
 * \file remove_undriven_view.cpp
 * \brief
 */

#include "remove_undriven_view.h"
#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveUndrivenView"

using namespace npu::tile_fwk;

Status Process(Function& function)
{
    bool hasDeleted = false;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE_SSA) {
            continue;
        }
        if (!op.HasAttribute(OpAttributeKey::inplaceIdx)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Missing required attribute 'INPLACE_IDX' for ASSEMBLE_SSA [%d]. %s",
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto inplaceIdx = op.GetIntAttribute(OpAttributeKey::inplaceIdx);
        auto iOperand = op.GetInputOperand(inplaceIdx);
        if (iOperand->GetProducers().size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Invalid producer count for ASSEMBLE_SSA [%d]. Expected 1,  got %zu. %s", op.GetOpMagic(),
                              iOperand->GetProducers().size(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto& producerOp = **iOperand->GetProducers().begin();
        if (producerOp.GetOpcode() != Opcode::OP_VIEW) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Invalid producer type for ASSEMBLE_SSA [%d]. Expected OP_VIEW, got %s. %s",
                              op.GetOpMagic(), producerOp.GetOpcodeStr().c_str(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (!producerOp.GetInputOperand(0)->GetProducers().empty()) {
            continue;
        }
        // 降级为普通的Assemble
        op.SetOpCode(Opcode::OP_ASSEMBLE);
        op.RemoveAttr(OpAttributeKey::inplaceIdx);
        iOperand->RemoveConsumer(op);
        op.EraseInput(iOperand);
        // 删除undriven的View
        producerOp.SetAsDeleted();
        hasDeleted = true;
    }
    if (hasDeleted) {
        function.EraseOperations();
    }
    return SUCCESS;
}

Status RemoveUndrivenView::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveUndrivenView.");
    if (Process(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "===> Process failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveUndrivenView.");
    return SUCCESS;
}
