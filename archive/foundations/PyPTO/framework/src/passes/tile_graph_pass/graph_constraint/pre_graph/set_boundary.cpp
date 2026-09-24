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
 * \file set_boundary.cpp
 * \brief
 */

#include "set_boundary.h"
#include "interface/tensor/irbuilder.h"

namespace npu::tile_fwk {
void SetBoundary::InsertTemporaryCopyIn(Function& function, Operation& op) const
{
    if (!OpcodeManager::Inst().HasStaticAttribute(op.GetOpcode(), OpAttributeKey::requiresBoundaryCopy)) {
        return;
    }
    for (auto& input : op.GetIOperands()) {
        if (input->GetProducers().size() == 0 && input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            // insert Copy_In before the op
            LogicalTensors operandGm;
            IRBuilder builder;
            LogicalTensorPtr tensorGM = builder.CreateTensorVar(input->Datatype(), input->shape,
                                                                std::vector<SymbolicScalar>{}, input->Format());
            GraphUtils::CopyDynStatus(tensorGM, input);
            tensorGM->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            tensorGM->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
            operandGm.push_back(tensorGM);

            LogicalTensors operandUb;
            operandUb.push_back(input);

            // add UB_Alloc && UB_COPY_IN
            auto& ubCopyIn = builder.CreateTensorOpStmt(function, Opcode::OP_COPY_IN, operandGm, operandUb);
            ubCopyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(input->GetTensorOffset()), MemoryType::MEM_UB,
                OpImmediate::Specified(input->GetShape()), OpImmediate::Specified(input->tensor->GetDynRawShape()),
                OpImmediate::Specified(input->GetDynValidShape())));
            ubCopyIn.SetAttribute(OpAttributeKey::isCube, false);
            ubCopyIn.UpdateSubgraphID(op.GetSubgraphID());
        }
    }
}

bool IsDiffSubgraphId(int& oriSubgraphId, Operation& op)
{
    int opSubgraphId = op.GetSubgraphID();
    if (oriSubgraphId == -1) {
        oriSubgraphId = opSubgraphId;
    }
    if (oriSubgraphId != opSubgraphId) {
        return true;
    }
    return false;
}

bool IsTensorSubgraphBoundary(LogicalTensorPtr t)
{
    int subgraphId = -1;
    for (const auto& op : t->GetProducers()) {
        if (IsDiffSubgraphId(subgraphId, *op) == true) {
            return true;
        }
    }
    for (const auto& op : t->GetConsumers()) {
        if (IsDiffSubgraphId(subgraphId, *op) == true) {
            return true;
        }
    }

    return false;
}

void SetBoundary::SetTensorBoundary(Function& function) const
{
    for (auto& op : function.Operations()) {
        /* memory map size > 1 代表该tensor被多个子图使用，那么标记为boundary*/
        for (auto& input : op.GetIOperands()) {
            if (input->GetProducers().size() == 0) {
                InsertTemporaryCopyIn(function, op);
            }
        }
    }
}
} // namespace npu::tile_fwk
