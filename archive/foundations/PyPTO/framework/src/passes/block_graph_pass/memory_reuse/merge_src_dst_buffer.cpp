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
 * \file merge_src_dst_buffer.cpp
 * \brief
 */

#include "merge_src_dst_buffer.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "SrcDstBufferMerge"

namespace npu::tile_fwk {

void SrcDstBufferMergeImpl::InitializeTensorMemorymap(Operation& op) const
{
    for (auto& input : op.GetIOperands()) {
        TileRange range;
        range.memId = input->tensor->GetRawMagic();
        input->memoryrange = range;
    }
    for (auto& output : op.GetOOperands()) {
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->memoryrange = range;
    }
}

void SrcDstBufferMergeImpl::InitTensorMaxSize(const LogicalTensorPtr& output)
{
    for (auto& consumer : output->GetConsumers()) {
        tensorConsumers_[output->memoryrange.memId].insert(consumer->GetOpMagic());
        if (tensorMaxSize_.find(output->memoryrange.memId) == tensorMaxSize_.end()) {
            tensorMaxSize_[output->memoryrange.memId] = output->tensor->GetRawDataSize();
            continue;
        }
        tensorMaxSize_[output->memoryrange.memId] = std::max(tensorMaxSize_[output->memoryrange.memId],
                                                             output->tensor->GetRawDataSize());
    }
}

Status SrcDstBufferMergeImpl::CheckOpValid(const Operation* op, int opId)
{
    if (op == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op:%d is null.%s", opId, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

void SrcDstBufferMergeImpl::InitOpOutput(const Operation& op)
{
    int outId = 0;
    for (auto& output : op.GetOOperands()) {
        if (output == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Op:%s, magic:%d, output:%d is null.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), outId);
            ++outId;
            continue;
        }
        if (output->memoryrange.memId == -1) {
            output->memoryrange.memId = output->GetMagic();
        }
        InitTensorMaxSize(output);
        ++outId;
    }
}

Status SrcDstBufferMergeImpl::Init(const std::vector<Operation*>& opList)
{
    if (opList.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "OpList empty.");
        return FAILED;
    }
    if (opList.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "First op is null.");
        return FAILED;
    }

    int opId = 0;
    for (auto& op : opList) {
        if (CheckOpValid(op, opId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CheckOpValid failed.");
            return FAILED;
        }
        InitializeTensorMemorymap(*op);
        InitOpOutput(*op);
        ++opId;
    }

    return SUCCESS;
}

bool SrcDstBufferMergeImpl::CheckIgnoreScene(const Operation& oriOps)
{
    /* use opcode is unfavorable for reading and modification, maybe use opcalctype */
    const std::set<Opcode> ignoreOps = {Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT,
                                        Opcode::OP_UB_COPY_ND2NZ};
    if (ignoreOps.count(oriOps.GetOpcode()) != 0) {
        return true;
    }

    if (OpcodeManager::Inst().HasStaticAttribute(oriOps.GetOpcode(), OpAttributeKey::excludeBufferReuse)) {
        return true;
    }

    if (oriOps.HasAttr(OpAttributeKey::excludeBufferReuse)) {
        return true;
    }

    for (auto& output : oriOps.GetOOperands()) {
        if (output == nullptr) {
            return true;
        }
    }
    return false;
}

Status SrcDstBufferMergeImpl::CheckHasInplaced(const Operation& oriOps, const Operation& ops,
                                               std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
                                               bool& hasInplaced)
{
    if (oriOps.HasAttr(OpAttributeKey::inplaceInfo)) {
        std::map<int, int> inplaceInfo;
        if (!oriOps.GetAttr(OpAttributeKey::inplaceInfo, inplaceInfo)) {
            APASS_LOG_ERROR_F(Elements::Tensor, "OriOps:%s[%d] get inplaceInfo error.%s", oriOps.GetOpcodeStr().c_str(),
                              oriOps.GetOpMagic(), GetFormatBacktrace(oriOps).c_str());
            return FAILED;
        }
        for (auto& [iIdx, oIdx] : inplaceInfo) {
            if (ops.GetIOperands().size() < static_cast<size_t>(iIdx) ||
                ops.GetOOperands().size() < static_cast<size_t>(oIdx)) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "The number of inputs or outputs for op:%s[%d] does not match inplaceInfo, inputs size: %zu, iIdx: "
                    "%d, outputs size: %zu, oIdx: %d.",
                    oriOps.GetOpcodeStr().c_str(), oriOps.GetOpMagic(), ops.GetIOperands().size(), iIdx,
                    ops.GetOOperands().size(), oIdx);
                return FAILED;
            }
            auto in = ops.GetIOperands()[iIdx];
            auto out = ops.GetOOperands()[oIdx];
            out->memoryrange.memId = in->memoryrange.memId;
            tensorConsumers_[in->memoryrange.memId].insert(tensorConsumers_[out->memoryrange.memId].begin(),
                                                           tensorConsumers_[out->memoryrange.memId].end());
            replacedTensors[out->memoryrange.memId] = in;
        }
        hasInplaced = true;
        return SUCCESS;
    }
    return SUCCESS;
}

Status SrcDstBufferMergeImpl::FindReplaced(const Operation& oriOps, const Operation& ops,
                                           std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
                                           bool& hasFound)
{
    if (OpcodeManager::Inst().GetCoreType(ops.GetOpcode()) == OpCoreType::AIC) {
        return ProcessL0MemoryReuse(ops, replacedTensors, hasFound);
    } else {
        return ProcessInplaceReuse(oriOps, ops, replacedTensors, hasFound);
    }
}

void SrcDstBufferMergeImpl::NotFindReplacedProcess(
    const Operation& ops, const std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors)
{
    for (auto& out : ops.GetOOperands()) {
        auto outTensorMemId = out->memoryrange.memId;
        auto it = replacedTensors.find(outTensorMemId);
        if (it != replacedTensors.end()) {
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Find memId: %d replaced by memId: %d, tensor magic: %d, continue replacing",
                              outTensorMemId, it->second->memoryrange.memId, out->GetMagic());
            out->memoryrange.memId = it->second->memoryrange.memId;
        }
    }
}

Status SrcDstBufferMergeImpl::Run(Function& func)
{
    if (func.rootFunc_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "RootFunc is null.");
        return FAILED;
    }
    for (auto& subProgram : func.rootFunc_->programs_) {
        APASS_LOG_INFO_F(Elements::Operation, "Merge src dst for program id : [%lu]", subProgram.first);
        auto opList = subProgram.second->Operations(false).DuplicatedOpList();
        if (Init(opList) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Init failed; Please check the Init method.");
            return FAILED;
        }
        auto oriOps(opList);
        std::unordered_map<int, std::shared_ptr<LogicalTensor>> replacedTensors;
        for (size_t i = 0; i < oriOps.size(); i++) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Attempt memory reuse for op:%s[%d]",
                              oriOps[i]->GetOpcodeStr().c_str(), oriOps[i]->GetOpMagic());
            if (CheckIgnoreScene(*oriOps[i])) {
                continue;
            }
            bool hasInplaced = false;
            if (CheckHasInplaced(*oriOps[i], *opList[i], replacedTensors, hasInplaced) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "CheckHasInplaced failed; Please check the CheckHasInplaced method.");
                return FAILED;
            }
            if (hasInplaced) {
                continue;
            }
            bool hasFound = false;
            if (FindReplaced(*oriOps[i], *opList[i], replacedTensors, hasFound) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "FindReplaced failed; Please check the FindReplaced method.");
                return FAILED;
            }
            if (!hasFound) {
                NotFindReplacedProcess(*opList[i], replacedTensors);
            }
        }
    }
    return SUCCESS;
}

bool SrcDstBufferMergeImpl::CheckAssembleReuse(const LogicalTensorPtr& outOperand)
{
    for (auto consumer : outOperand->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (auto assembleOutTensor : consumer->GetOOperands()) {
            if (assembleOutTensor->memoryrange.memId == outOperand->memoryrange.memId) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Assemble cannot be reused.");
                return false;
            }
        }
    }
    return true;
}

bool SrcDstBufferMergeImpl::CanSrcDstReuse(const Operation& ops, std::shared_ptr<LogicalTensor> iOperand,
                                           std::shared_ptr<LogicalTensor> oOperand)
{
    if (std::find(SCATTER_ELEMENT_OPS.begin(), SCATTER_ELEMENT_OPS.end(), ops.GetOpcode()) !=
        SCATTER_ELEMENT_OPS.end()) {
        if (iOperand == ops.GetIOperands()[0]) {
            return true;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Try to reuse memory, iOperand %d -> oOperand %d", iOperand->GetMagic(),
                      oOperand->GetMagic());
    if (oOperand->GetMemoryTypeOriginal() != iOperand->GetMemoryTypeOriginal()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "iOperand memtype %s is not same as oOperand memtype %s",
                          MemoryTypeToString(iOperand->GetMemoryTypeOriginal()).c_str(),
                          MemoryTypeToString(iOperand->GetMemoryTypeOriginal()).c_str());
        return false;
    }
    if (tensorMaxSize_[oOperand->memoryrange.memId] != tensorMaxSize_[iOperand->memoryrange.memId]) {
        APASS_LOG_DEBUG_F(
            Elements::Tensor, "Output tensor (memId=%d, size=%ld) != input tensor (memId=%d, size=%ld), op:%s[%d]",
            oOperand->memoryrange.memId, tensorMaxSize_[oOperand->memoryrange.memId], iOperand->memoryrange.memId,
            tensorMaxSize_[iOperand->memoryrange.memId], ops.GetOpcodeStr().c_str(), ops.GetOpMagic());
        return false;
    }
    if (BytesOf(oOperand->Datatype()) != BytesOf(iOperand->Datatype())) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Bytes of output datatype[%zu] != Bytes of output datatype[%zu], op:%s[%d]",
                          BytesOf(oOperand->Datatype()), BytesOf(iOperand->Datatype()), ops.GetOpcodeStr().c_str(),
                          ops.GetOpMagic());
        return false;
    }
    if (!CheckAssembleReuse(oOperand)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Check Assemble op which cannot be reused, current op:%s[%d]",
                          ops.GetOpcodeStr().c_str(), ops.GetOpMagic());
        return false;
    }
    // 确保复用UB buffer后不会被覆写
    auto iter = tensorConsumers_.find(iOperand->memoryrange.memId);
    if (iter != tensorConsumers_.end() && iter->second.size() > 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] has more than 1 output.", ops.GetOpcodeStr().c_str(),
                          ops.GetOpMagic());
        return false;
    }
    APASS_LOG_DEBUG_F(
        Elements::Tensor,
        "Reusable, iOperand magic: %d, memId: %d ,datatype: %s, oOperand magic: %d, memId: %d datatype: %s, op:%s[%d]",
        iOperand->GetMagic(), iOperand->memoryrange.memId, DataType2String(iOperand->Datatype()), oOperand->GetMagic(),
        oOperand->memoryrange.memId, DataType2String(oOperand->Datatype()), ops.GetOpcodeStr().c_str(),
        ops.GetOpMagic());
    return true;
}

Status SrcDstBufferMergeImpl::ProcessInplaceReuse(
    const Operation& oriOps, const Operation& ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors, bool& hasFound)
{
    if (ops.GetOOperands().size() == 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Operation %s[%d] has no outOperands", ops.GetOpcodeStr().c_str(),
                          ops.GetOpMagic());
        return FAILED;
    }
    auto out = ops.GetOOperands()[0];
    auto outTensorMagic = out->memoryrange.memId;
    for (auto in : oriOps.GetIOperands()) {
        if (in != nullptr && CanSrcDstReuse(oriOps, in, out)) {
            // 当前输出复用输入
            auto inTensorMagic = in->memoryrange.memId;
            if (inTensorMagic == outTensorMagic) {
                continue;
            }
            APASS_LOG_DEBUG_F(Elements::Tensor, "Set out tensor %d reuse src tensor %d", out->GetMagic(),
                              in->GetMagic());
            out->memoryrange.memId = in->memoryrange.memId;
            if (tensorConsumers_[outTensorMagic].size() > tensorConsumers_[inTensorMagic].size()) {
                tensorConsumers_[inTensorMagic] = tensorConsumers_[outTensorMagic];
            }
            replacedTensors[outTensorMagic] = in;
            hasFound = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}

bool SrcDstBufferMergeImpl::IsL1ToL0Transfer(const Operation& op)
{
    for (auto& inputTensor : op.GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L1) {
            return false;
        }
    }
    for (auto& outputTensor : op.GetOOperands()) {
        if (outputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L0A &&
            outputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L0B) {
            return false;
        }
    }
    ASSERT(OperationErr::OP_INVALID_OPERAND_COUNT, op.GetIOperands().size() == 1 && op.GetOOperands().size() == 1)
        << "The L1-to-L0 copy op can have only one input and one output tensor";
    return true;
}

bool SrcDstBufferMergeImpl::IsL0CToL1Transfer(const Operation& op)
{
    for (auto& inputTensor : op.GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
            return false;
        }
    }
    for (auto& outputTensor : op.GetOOperands()) {
        if (outputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L1) {
            return false;
        }
    }
    ASSERT(OperationErr::OP_INVALID_OPERAND_COUNT, op.GetIOperands().size() == 1 && op.GetOOperands().size() == 1)
        << "The L0C-to-L1 copy op can have only one input and one output tensor";
    return true;
}

Status SrcDstBufferMergeImpl::ProcessL0MemoryReuse(
    const Operation& op, std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors, bool& hasFound)
{
    if (!IsL1ToL0Transfer(op)) {
        return SUCCESS;
    }
    auto inputTensor = op.GetIOperands().front();
    auto outputTensor = op.GetOOperands().front();
    if (inputTensor == nullptr || outputTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] failed to obtain the input/output tensor",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    if (tensorConsumers_[inputTensor->memoryrange.memId].size() > 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d], memId[%d], memType[%s] has more than 1 consumer.",
                          inputTensor->GetMagic(), inputTensor->memoryrange.memId,
                          MemoryTypeToString(inputTensor->GetMemoryTypeOriginal()).c_str());
        return SUCCESS;
    }
    for (auto& producerOp : inputTensor->GetProducers()) {
        if (!IsL0CToL1Transfer(*producerOp)) {
            return SUCCESS;
        }
        auto l0cTensor = producerOp->GetIOperands().front();
        if (tensorConsumers_[l0cTensor->memoryrange.memId].size() > 1) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d], memId[%d], memType[%s] has more than 1 consumer.",
                              l0cTensor->GetMagic(), l0cTensor->memoryrange.memId,
                              MemoryTypeToString(l0cTensor->GetMemoryTypeOriginal()).c_str());
            return SUCCESS;
        }
        auto checkOp = *l0cTensor->GetProducers().begin();
        if (FindReuseableL0Tensor(*checkOp, replacedTensors, outputTensor, hasFound) != SUCCESS) {
            return FAILED;
        }
        if (hasFound) {
            break;
        }
    }
    return SUCCESS;
}

Status SrcDstBufferMergeImpl::FindReuseableL0Tensor(
    const Operation& op, std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
    LogicalTensorPtr needReplacedTensor, bool& hasFound)
{
    if (npu::tile_fwk::OpcodeManager::Inst().GetOpCalcType(op.GetOpcode()) != npu::tile_fwk::OpCalcType::MATMUL) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] is not a matmul operation.", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic());
        return SUCCESS;
    }
    for (const auto& inputTensor : op.GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() != needReplacedTensor->GetMemoryTypeOriginal()) {
            continue;
        }
        if (tensorMaxSize_[inputTensor->memoryrange.memId] != tensorMaxSize_[needReplacedTensor->memoryrange.memId]) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor,
                "Matmul input tensor (memId=%d, size=%ld) != needReplaced tensor (memId=%d, size=%ld), op:%s[%d].",
                inputTensor->memoryrange.memId, tensorMaxSize_[inputTensor->memoryrange.memId],
                needReplacedTensor->memoryrange.memId, tensorMaxSize_[needReplacedTensor->memoryrange.memId],
                op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return SUCCESS;
        }
        if (tensorConsumers_[inputTensor->memoryrange.memId].size() > 1) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d], memId[%d] has more than 1 consumer.",
                              inputTensor->GetMagic(), inputTensor->memoryrange.memId);
            return SUCCESS;
        }
        if (hasReusedL0Tensors_.count(inputTensor->GetMagic())) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d], memId[%d] has been reused", inputTensor->GetMagic(),
                              inputTensor->memoryrange.memId);
            return SUCCESS;
        }
        if (tensorConsumers_[needReplacedTensor->memoryrange.memId].size() >
            tensorConsumers_[inputTensor->memoryrange.memId].size()) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Needreplaced tensor[%d] consumers > matmul input tensor[%d] consumers, perform refresh.",
                              needReplacedTensor->GetMagic(), inputTensor->GetMagic());
            tensorConsumers_[inputTensor->memoryrange.memId] = tensorConsumers_[needReplacedTensor->GetMagic()];
        }
        replacedTensors[needReplacedTensor->memoryrange.memId] = inputTensor;
        APASS_LOG_INFO_F(
            Elements::Operation,
            "Successfully performed L0 memory reuse, Needreplaced tensor[%d] memId[%d] , input tensor[%d] memId[%d]",
            needReplacedTensor->GetMagic(), needReplacedTensor->memoryrange.memId, inputTensor->GetMagic(),
            inputTensor->memoryrange.memId);
        needReplacedTensor->memoryrange.memId = inputTensor->memoryrange.memId;
        hasReusedL0Tensors_.insert(inputTensor->GetMagic());
        hasFound = true;
        return SUCCESS;
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
