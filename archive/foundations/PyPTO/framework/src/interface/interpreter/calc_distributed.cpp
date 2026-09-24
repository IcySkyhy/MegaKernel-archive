/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file calc_distributed.cpp
 * \brief
 */

#include <memory>
#include <iostream>
#include "interface/interpreter/operation.h"
#include "tensor/symbolic_scalar.h"
#include "tilefwk/error.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/comm_group_recorder.h"
#include "calc.h"
#include "communication.h"
#include "interface/operation/distributed/distributed_common.h"

namespace npu::tile_fwk {
void ExecuteOpBindTensor(ExecuteOperationContext* ctx) { (void)ctx; }
REGISTER_CALC_OP(OP_BIND_TENSOR, Opcode::OP_BIND_TENSOR, ExecuteOpBindTensor);

void ExecuteOpShmemSet(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0x2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1 || ctx->ooperandInplaceDataViewList->size() == 0x2);
    auto& shm = ctx->ioperandDataViewList->at(1);

    Distributed::ShmemSetAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);

    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    size_t slotSize = shm->GetSize() * BytesOf(shm->GetDataType());
    if (!attr.isSetData) {
        context->Signal(context->GetRank(), 0, slotSize, shm->GetShmStorageOffset());
    } else {
        context->Set(context->GetRank(), 0, slotSize, shm->GetShmStorageOffset());
    }
}
REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);

void ExecuteOpShmemPut(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0x3);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1 || ctx->ooperandInplaceDataViewList->size() == 0x2);
    auto& in = ctx->ioperandDataViewList->at(1);
    auto& shm = ctx->ioperandDataViewList->at(0x2);

    Distributed::ShmemPutAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);

    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    int atomicType = 0;
    if (attr.atomicType == Distributed::AtomicType::ADD) {
        atomicType = 1;
    }
    if (attr.atomicType == Distributed::AtomicType::SET) {
        atomicType = 0;
    }

    if (shm->GetDataType() != in->GetDataType()) {
        auto castedIn = LogicalTensorData::CreateEmpty(shm->GetDataType(), shm->GetShape(), shm->GetValidShape(),
                                                       shm->GetShape());
        calc::Cast(castedIn, in);
        context->Put(castedIn, dstRank, shm->GetShmStorageOffset(), atomicType);
    } else {
        context->Put(in, dstRank, shm->GetShmStorageOffset(), atomicType);
    }
}
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);

void ExecuteOpShmemSignal(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0x2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 0x2 || ctx->ooperandInplaceDataViewList->size() == 1);
    auto& shm = ctx->ioperandDataViewList->at(1);

    Distributed::ShmemSignalAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);

    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    int atomicType = 0;
    if (attr.atomicType == Distributed::AtomicType::SET) {
        atomicType = 0;
    }
    if (attr.atomicType == Distributed::AtomicType::ADD) {
        atomicType = 1;
    }
    int value = attr.signalValue;
    bool notifyAll = attr.notifyAll;
    size_t slotSize = shm->GetSize() * BytesOf(shm->GetDataType());
    context->Signal(dstRank, value, slotSize, shm->GetShmStorageOffset(), atomicType, notifyAll);
}
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);

void ExecuteOpShmemWaitUntil(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0x2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    auto& shm = ctx->ioperandDataViewList->at(1);

    Distributed::ShmemWaitUntilAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);

    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int srcRank = context->GetRank();
    int expect = attr.expectedSum;
    bool reset = attr.resetSignal;
    size_t slotSize = shm->GetSize() * BytesOf(shm->GetDataType());

    context->Wait(srcRank, expect, slotSize, shm->GetShmStorageOffset(), reset);
}
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);

void ExecuteOpShmemGet(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0x2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 0x2 || ctx->ooperandInplaceDataViewList->size() == 1);
    auto& shm = ctx->ioperandDataViewList->at(1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);

    Distributed::ShmemGetAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);

    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int srcRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    LogicalTensorDataPtr tmp = context->Get(srcRank, out->GetDataType(), out->GetShape(), shm->GetShmStorageOffset());
    calc::Copy(out, tmp);
}
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);
REGISTER_CALC_OP(OP_SHMEM_LOAD, Opcode::OP_SHMEM_LOAD, ExecuteOpShmemGet);
} // namespace npu::tile_fwk
