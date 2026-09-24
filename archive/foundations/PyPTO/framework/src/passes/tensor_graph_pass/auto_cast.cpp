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
 * \file auto_cast.cpp
 * \brief
 */

#include "auto_cast.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_check/auto_cast_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "AutoCast"

namespace npu {
namespace tile_fwk {

LogicalTensorPtr AutoCast::CreateFp32TensorLike(const LogicalTensorPtr& tensor)
{
    return irBuilder_.CreateTensorVar(DataType::DT_FP32, tensor->GetShape(), tensor->GetDynValidShape(),
                                      tensor->Format());
}

Status AutoCast::GetInOutConnectedTensor(Function& function)
{
    inCastConnectedTensors_.clear();
    outCastConnectedTensors_.clear();

    std::vector<std::shared_ptr<LogicalTensor>> inCastConnected(function.inCasts_);
    while (inCastConnected.size() > 0) {
        std::shared_ptr<LogicalTensor> currTensor = inCastConnected.back();
        inCastConnected.pop_back();
        if (inCastConnectedTensors_.count(currTensor->GetMagic()) > 0) {
            continue;
        }
        inCastConnectedTensors_.insert(currTensor->GetMagic());
        for (auto& consumer : currTensor->GetConsumers()) {
            if (consumer->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            for (auto& tensor : consumer->GetOOperands()) {
                inCastConnected.push_back(tensor);
            }
        }
    }

    std::vector<std::shared_ptr<LogicalTensor>> outCastConnected(function.outCasts_);
    while (outCastConnected.size() > 0) {
        std::shared_ptr<LogicalTensor> currTensor = outCastConnected.back();
        outCastConnected.pop_back();
        if (outCastConnectedTensors_.count(currTensor->GetMagic()) > 0) {
            continue;
        }
        outCastConnectedTensors_.insert(currTensor->GetMagic());
        for (auto& producer : currTensor->GetProducers()) {
            if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            for (auto& tensor : producer->GetIOperands()) {
                outCastConnected.push_back(tensor);
            }
        }
    }
    return SUCCESS;
}

Status AutoCast::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start AutoCast for function [%s].", function.GetRawName().c_str());
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        legalCastPair.insert({DataType::DT_INT32, DataType::DT_FP16});
    }
    if (GetInOutConnectedTensor(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to get InOutCast-connected tensor.");
        return FAILED;
    }
    if (InsertBF16Cast(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to insert CAST for BF16 unsupported Operations.");
        return FAILED;
    }
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) && InsertFP16Cast(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to insert CAST for FP16 unsupported Operations.");
        return FAILED;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && InsertInt32Fp16Cast(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to insert fp32 between int32 to fp16 cast.");
        return FAILED;
    }
    if (RemoveRedundantCastChain(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to remove redundant CAST.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End AutoCast for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status AutoCast::InsertInt32Fp16Cast(Function& function)
{
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation* op = opList[opIdx];
        if (op->GetOpcode() != Opcode::OP_CAST) {
            continue;
        }
        auto iOperands = op->GetIOperands();
        auto oOperands = op->GetOOperands();
        if (iOperands.empty() || oOperands.empty()) {
            continue;
        }
        LogicalTensorPtr srcTensor = iOperands[0];
        LogicalTensorPtr tgtTensor = oOperands[0];

        if (srcTensor->Datatype() != DataType::DT_INT32 || tgtTensor->Datatype() != DataType::DT_FP16) {
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "Cast[%d] is cast between int32 and fp16.", op->GetOpMagic());
        auto fp32Tensor = CreateFp32TensorLike(tgtTensor);
        InsertCastOp(function, srcTensor, fp32Tensor, op->GetTileShape(), op->GetScopeInfo());
        op->ReplaceInput(fp32Tensor, srcTensor);
    }
    return SUCCESS;
}

bool AutoCast::SupportBF16(Operation* op)
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (UNSUPPORT_BF16_ARCH35_OPS.count(op->GetOpcode()) > 0)
            return false;
    } else {
        if (UNSUPPORT_BF16_OPS.count(op->GetOpcode()) > 0) {
            APASS_LOG_INFO_F(Elements::Operation, "Op[%d] can find in UNSUPPORT_BF16_OPS.", op->GetOpMagic());
            return false;
        }
    }
    return true;
}

bool AutoCast::SupportFP16(Operation* op)
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (UNSUPPORT_FP16_ARCH35_OPS.count(op->GetOpcode()) > 0)
            return false;
    } else {
        if (UNSUPPORT_FP16_OPS.count(op->GetOpcode()) > 0) {
            APASS_LOG_INFO_F(Elements::Operation, "Op[%d] can find in UNSUPPORT_FP16_OPS.", op->GetOpMagic());
            return false;
        }
    }
    return true;
}

void AutoCast::InsertCastOp(Function& function, LogicalTensorPtr src, LogicalTensorPtr tgt, const TileShape& tileShape,
                            const Operation::ScopeInfo& scopeInfo)
{
    Operation& newCast = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_CAST, {src}, {tgt});
    newCast.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
    auto newTileShape = tileShape;
    auto vecTile = newTileShape.GetVecTile();
    int activeStart = std::max(0, static_cast<int>(vecTile.tile.size()) - static_cast<int>(tgt->GetShape().size()));
    vecTile.tile = std::vector<int64_t>(vecTile.tile.begin() + activeStart, vecTile.tile.end());
    newTileShape.SetVecTile(vecTile);
    newCast.UpdateTileShape(newTileShape);
    newCast.SetScopeInfo(scopeInfo);
    addedCast_.insert(&newCast);
}

Status AutoCast::InsertBF16Cast(Function& function)
{
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> oldMagic2Input;
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation* op = opList[opIdx];
        if (SupportBF16(op)) {
            continue;
        }
        auto iOperands = op->GetIOperands();
        std::unordered_set<int> visitedIOp;
        for (auto& iop : iOperands) {
            if (visitedIOp.count(iop->GetMagic()) > 0 || iop->Datatype() != DataType::DT_BF16) {
                continue;
            }
            visitedIOp.insert(iop->GetMagic());
            if (oldMagic2Input.count(iop->GetMagic()) > 0) {
                auto newInput = oldMagic2Input[iop->GetMagic()];
                op->ReplaceInput(newInput, iop);
                continue;
            }
            auto newInput = CreateFp32TensorLike(iop);
            InsertCastOp(function, iop, newInput, op->GetTileShape(), op->GetScopeInfo());
            op->ReplaceInput(newInput, iop);
            oldMagic2Input[iop->GetMagic()] = newInput;
            if (inCastConnectedTensors_.count(iop->GetMagic()) > 0) {
                inCastConnectedTensors_.insert(newInput->GetMagic());
            }
        }
        auto oOperands = op->GetOOperands();
        std::unordered_set<int> visitedOOp;
        for (auto& oop : oOperands) {
            if (visitedOOp.count(oop->GetMagic()) > 0 || oop->Datatype() != DataType::DT_BF16) {
                continue;
            }
            visitedOOp.insert(oop->GetMagic());
            if (oop->Datatype() == DataType::DT_BF16) {
                auto newOutput = CreateFp32TensorLike(oop);
                op->ReplaceOutput(newOutput, oop);
                InsertCastOp(function, newOutput, oop, op->GetTileShape(), op->GetScopeInfo());
                oldMagic2Input[oop->GetMagic()] = newOutput;
                if (outCastConnectedTensors_.count(oop->GetMagic()) > 0) {
                    outCastConnectedTensors_.insert(newOutput->GetMagic());
                }
            }
        }
    }
    return SUCCESS;
}

Status AutoCast::InsertFP16Cast(Function& function)
{
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> oldMagic2Input;
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation* op = opList[opIdx];
        if (SupportFP16(op)) {
            continue;
        }
        auto iOperands = op->GetIOperands();
        std::unordered_set<int> visitedIOp;
        for (auto& iop : iOperands) {
            if (visitedIOp.count(iop->GetMagic()) > 0 || iop->Datatype() != DataType::DT_FP16) {
                continue;
            }
            visitedIOp.insert(iop->GetMagic());
            if (oldMagic2Input.count(iop->GetMagic()) > 0) {
                auto newInput = oldMagic2Input[iop->GetMagic()];
                op->ReplaceInput(newInput, iop);
                continue;
            }
            auto newInput = CreateFp32TensorLike(iop);
            InsertCastOp(function, iop, newInput, op->GetTileShape(), op->GetScopeInfo());
            op->ReplaceInput(newInput, iop);
            oldMagic2Input[iop->GetMagic()] = newInput;
            if (inCastConnectedTensors_.count(iop->GetMagic()) > 0) {
                inCastConnectedTensors_.insert(newInput->GetMagic());
            }
        }
        auto oOperands = op->GetOOperands();
        std::unordered_set<int> visitedOOp;
        for (auto& oop : oOperands) {
            if (visitedOOp.count(oop->GetMagic()) > 0 || oop->Datatype() != DataType::DT_FP16) {
                continue;
            }
            visitedOOp.insert(oop->GetMagic());
            auto newOutput = CreateFp32TensorLike(oop);
            op->ReplaceOutput(newOutput, oop);
            InsertCastOp(function, newOutput, oop, op->GetTileShape(), op->GetScopeInfo());
            oldMagic2Input[oop->GetMagic()] = newOutput;
            if (outCastConnectedTensors_.count(oop->GetMagic()) > 0) {
                outCastConnectedTensors_.insert(newOutput->GetMagic());
            }
        }
    }
    return SUCCESS;
}

bool AutoCast::IsLegalCast(DataType ds, DataType dt)
{
    if (legalCastPair.count(std::make_pair(ds, dt)) > 0) {
        return true;
    }
    return false;
}

std::vector<Operation*> AutoCast::GetCastChain(Operation* tailOp)
{
    std::vector<Operation*> tailToHeadChain;
    bool isFront = false;
    Operation* currOp = tailOp;
    while (!isFront) {
        if (currOp->ProducerOps().size() != 1 || (*currOp->ProducerOps().begin())->GetOpcode() != Opcode::OP_CAST ||
            addedCast_.count(*currOp->ProducerOps().begin()) == 0) {
            isFront = true;
            tailToHeadChain.push_back(currOp);
            break;
        }
        tailToHeadChain.push_back(currOp);
        currOp = *(currOp->ProducerOps().begin());
    }
    return tailToHeadChain;
}

Status AutoCast::ShortenChain(Function& function, const std::vector<Operation*>& castChain, Operation* tailOp)
{
    std::shared_ptr<LogicalTensor> tgtTensor = *(tailOp->GetOOperands().begin());
    DataType tgtType = tgtTensor->Datatype();
    bool isTgtOut = (FunctionUtils::GetNodeType(*tgtTensor, function) == NodeType::OUTCAST);
    bool isTgtOutConnected = (outCastConnectedTensors_.count(tgtTensor->GetMagic()) > 0);
    for (int i = static_cast<int>(castChain.size()) - 1; i >= 0; i--) {
        std::shared_ptr<LogicalTensor> srcTensor = *(castChain[i]->GetIOperands().begin());
        DataType srcType = srcTensor->Datatype();
        bool isSrcIn = (FunctionUtils::GetNodeType(*srcTensor, function) == NodeType::INCAST);
        bool isSrcOut = (FunctionUtils::GetNodeType(*srcTensor, function) == NodeType::OUTCAST);
        bool isSrcInConnected = (inCastConnectedTensors_.count(srcTensor->GetMagic()) > 0);
        if (srcType == tgtType && !(isTgtOutConnected && isSrcInConnected)) {
            if (!isTgtOut) {
                auto consumers = tgtTensor->GetConsumers();
                for (auto& consumerOp : consumers) {
                    consumerOp->ReplaceInput(srcTensor, tgtTensor);
                }
                break;
            }
            if (!isSrcIn && !isSrcOut) {
                auto srcProducers = srcTensor->GetProducers();
                auto srcConsumers = srcTensor->GetConsumers();
                auto tgtProducers = tgtTensor->GetProducers();
                for (auto& tgtProducerOp : tgtProducers) {
                    tgtProducerOp->ReplaceOutput(srcTensor, tgtTensor);
                }
                for (auto& srcProducerOp : srcProducers) {
                    srcProducerOp->ReplaceOutput(tgtTensor, srcTensor);
                }
                for (auto& srcConsumerOp : srcConsumers) {
                    srcConsumerOp->ReplaceInput(tgtTensor, srcTensor);
                }
                break;
            }
        }
        if (i != 0 && IsLegalCast(srcType, tgtType)) {
            tgtTensor->RemoveProducer(tailOp);
            auto origTileShape = (*srcTensor->GetConsumers().begin())->GetTileShape();
            auto origScopeInfo = (*srcTensor->GetConsumers().begin())->GetScopeInfo();
            InsertCastOp(function, srcTensor, tgtTensor, origTileShape, origScopeInfo);
            break;
        }
    }
    return SUCCESS;
}

Status AutoCast::RemoveRedundantCastChain(Function& function)
{
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation* op = opList[opIdx];
        if (op->GetOpcode() != Opcode::OP_CAST || addedCast_.count(op) == 0) {
            continue;
        }
        bool allCast = true;
        for (auto& nextOp : op->ConsumerOps()) {
            if (nextOp->GetOpcode() != Opcode::OP_CAST) {
                allCast = false;
                break;
            }
        }
        if (allCast && op->ConsumerOps().size() > 0) {
            continue;
        }
        std::vector<Operation*> castChain = GetCastChain(op);
        ShortenChain(function, castChain, op);
    }
    return SUCCESS;
}

Status AutoCast::DefaultEnabledPreCheck(Function& function)
{
    AutoCastChecker checker;
    return checker.DoDefaultEnabledPreCheck(function);
}

Status AutoCast::PostCheck(Function& function)
{
    AutoCastChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace tile_fwk
} // namespace npu
