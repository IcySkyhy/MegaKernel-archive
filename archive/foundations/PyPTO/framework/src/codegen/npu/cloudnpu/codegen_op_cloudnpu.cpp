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
 * \file codegen_op_cloudnpu.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"

namespace npu::tile_fwk {

CodeGenOpCloudNPU::CodeGenOpCloudNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx)
{
    InitOpsGenMap();
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    GetGmParamIdx(ctx.operation);
    UpdateTileTensorInfo();
    UpdateLoopInfo();
}

void CodeGenOpCloudNPU::GetGmParamIdx(const Operation& oper)
{
    if (oper.HasAttribute(OpAttributeKey::gmTensorParamIdxInCall)) {
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute(OpAttributeKey::gmTensorParamIdxInCall);
    }

    for (size_t i = 0; i < oper.GetOOperands().size(); ++i) {
        if (oper.GetOOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
            paramLocation[i] = oper.GetOOpAttrOffset(i);
        }
    }

    size_t iOffset = oper.GetOOperands().size() == 0 ? 1 : oper.GetOOperands().size();
    for (size_t i = 0; i < oper.GetIOperands().size(); ++i) {
        if (oper.GetIOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
            paramLocation[i + iOffset] = oper.GetIOpAttrOffset(i);
        }
    }
}

} // namespace npu::tile_fwk
