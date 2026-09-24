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
 * \file set_copy_attr.cpp
 * \brief
 */

#include "set_copy_attr.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
void SetCopyAttr::ProcessSpecialMTEOperation(Operation& op) const
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Process Special MTE Operation %d.", op.opmagic);
    auto inputTensor = op.iOperand.front();
    auto outputTensor = op.oOperand.front();
    if ((inputTensor == nullptr) || (outputTensor == nullptr)) {
        return;
    }
    if (op.GetOpcode() == Opcode::OP_INDEX_PUT || op.GetOpcode() == Opcode::OP_INDEX_ADD) {
        outputTensor = inputTensor;
    }
    /* transpose datamove 输入和输出的shape不相同 */
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        inputTensor->GetMemoryTypeOriginal(), OpImmediate::Specified(outputTensor->GetTensorOffset()),
        OpImmediate::Specified(outputTensor->GetShape()),
        OpImmediate::Specified(outputTensor->tensor->GetDynRawShape())));
}

void SetCopyAttr::ProcessMoveInOperation(Operation& op) const
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Process MoveIn Operation %d.", op.opmagic);
    auto inputTensor = op.iOperand.front();
    if (inputTensor == nullptr) {
        return;
    }
    TensorOffset offset = inputTensor->GetTensorOffset();
    auto producers = inputTensor->GetProducers();
    auto outputTensor = op.oOperand.front();
    if (!producers.empty()) {
        auto pre = *(producers.begin());
        if (pre != nullptr && pre->GetOpcode() == Opcode::OP_VIEW) {
            auto attr = dynamic_cast<ViewOpAttribute*>(pre->GetOpAttribute().get());
            if (attr != nullptr) {
                op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(TensorOffset(attr->GetFromOffset(), attr->GetFromDynOffset())),
                    outputTensor->GetMemoryTypeOriginal(), OpImmediate::Specified(inputTensor->GetShape()),
                    OpImmediate::Specified(inputTensor->tensor->GetDynRawShape()),
                    OpImmediate::Specified(inputTensor->GetDynValidShape())));
                return;
            }
        }
    }
    if (op.GetOpcode() != Opcode::OP_L1_COPY_IN_A_SCALE && op.GetOpcode() != Opcode::OP_L1_COPY_IN_B_SCALE) {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset), MemoryType::MEM_UB, OpImmediate::Specified(inputTensor->GetShape()),
            OpImmediate::Specified(inputTensor->tensor->GetDynRawShape()),
            OpImmediate::Specified(inputTensor->GetDynValidShape())));
    }
}
} // namespace npu::tile_fwk
