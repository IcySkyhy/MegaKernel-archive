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
 * \file schedule_ooo_check.cpp
 * \brief
 */

#include "schedule_ooo_checker.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOSchedule"
#endif

namespace npu {
namespace tile_fwk {
bool OoOScheduleChecker::PreCheckTensorInfo(const LogicalTensorPtr tensor)
{
    // memorytypeOriginal和Tobe要一致
    if (tensor->GetMemoryTypeOriginal() != tensor->GetMemoryTypeToBe()) {
        APASS_LOG_ERROR_C(TensorErr::TENSOR_INVALID_MEMORY_TYPE, Elements::Tensor,
                          "Tensor[%d] memorytypeOriginal is not equal to memorytypeTobe, OoOSchedule Precheck failed!",
                          tensor->GetMagic());
        return false;
    }
    // 子图边界上的tensor不检查
    if (SubgraphUtils::IsBoundary(tensor)) {
        return true;
    }
    // memoryrange对应的memoryid不为-1
    if (tensor->memoryrange.memId == -1) {
        APASS_LOG_ERROR_C(TensorErr::TENSOR_MEMORY_ALLOCATION, Elements::Tensor,
                          "Tensor[%d] memId does not exist, OoOSchedule Precheck failed!", tensor->GetMagic());
        return false;
    }
    return true;
}

bool OoOScheduleChecker::PreCheckOpInfo(const Operation* op)
{
    // 检查op的输入tensors
    for (auto inTensor : op->GetIOperands()) {
        if (!PreCheckTensorInfo(inTensor)) {
            return false;
        }
    }
    // 检查op的输出tensors
    for (auto outTensor : op->GetOOperands()) {
        if (!PreCheckTensorInfo(outTensor)) {
            return false;
        }
    }
    // 检查op不可以是call op
    if (op->GetOpcode() == Opcode::OP_CALL) {
        APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                          "Block graph has call op, OoOSchedule Precheck failed!");
        return false;
    }
    // 开始检查ASSEMBLE/RESHAPE/VIEW op
    if (op->GetOpcode() != Opcode::OP_ASSEMBLE && op->GetOpcode() != Opcode::OP_RESHAPE &&
        op->GetOpcode() != Opcode::OP_VIEW && op->GetOpcode() != Opcode::OP_NOP) {
        return true;
    }
    // 检查ASSEMBLE/RESHAPE/VIEW op的latency
    if (op->GetLatency() != 1) {
        APASS_LOG_WARN_F(Elements::Operation, "%s[%d] Op latency is not 1, OoOSchedule Precheck warning!",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic());
    }
    // 检查输出不在DDR上的Op
    if (op->GetOOperands()[0]->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
        // 输入tensor的memid要与输出tensor的memid保持一致
        int memId = op->GetOOperands()[0]->memoryrange.memId;
        for (auto inTensor : op->GetIOperands()) {
            if (inTensor->memoryrange.memId != memId && inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_INVALID_MEMORY_TYPE, Elements::Operation,
                                  "%s[%d] input output tensors memId does not match, OoOSchedule Precheck failed!",
                                  op->GetOpcodeStr().c_str(), op->GetOpMagic());
                return false;
            }
        }
    }
    return true;
}

Status OoOScheduleChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start OoOSchedule Precheck.");
    int programIdx = 0;
    int programSize = function.rootFunc_->programs_.size();
    tensorListBeforePass_.resize(programSize);
    for (auto& program : function.rootFunc_->programs_) { // 对每个子图分别进行precheck
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OoOSchedule Precheck begin.", program.first);
        auto opList = program.second->Operations().DuplicatedOpList();
        if (opList.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "Operation List is empty!");
            APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OoOSchedule Precheck end.", program.first);
            continue;
        }
        for (auto& op : opList) {
            if (op == nullptr) {
                APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                                  "Operation is nullptr, OoOSchedule Precheck failed!");
                return FAILED;
            }
        }
        // 检查单ASSEMBLE/RESHAPE/VIEW op在UB上没有alloc
        if ((opList.size() == 1) &&
            (opList.front()->GetOpcode() == Opcode::OP_ASSEMBLE || opList.front()->GetOpcode() == Opcode::OP_RESHAPE ||
             opList.front()->GetOpcode() == Opcode::OP_VIEW) &&
            opList.front()->GetOOperands()[0]->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                              "Single Op: localBuffer does not have alloc, OoOSchedule Precheck failed!");
            return FAILED;
        }
        std::unordered_set<LogicalTensorPtr> tensorList;
        for (auto& op : opList) {
            if (!PreCheckOpInfo(op)) {
                APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                                  "PreCheckOpInfo failed; Please check the PreCheckOpInfo method.");
                return FAILED;
            }
            APASS_LOG_INFO_F(Elements::Operation, "Before OoOSchedule op: %s, %d.", op->GetOpcodeStr().c_str(),
                             op->GetOpMagic()); // topo order
            // 记录所有op的输入tensor和输出tensor
            auto ioperands = op->GetIOperands();
            auto ooperands = op->GetOOperands();
            std::copy(ioperands.begin(), ioperands.end(), std::inserter(tensorList, tensorList.end()));
            std::copy(ooperands.begin(), ooperands.end(), std::inserter(tensorList, tensorList.end()));
        }
        tensorListBeforePass_[programIdx] = tensorList;
        programIdx++;
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OoOSchedule Precheck end.", program.first);
    }
    APASS_LOG_INFO_F(Elements::Operation, "OoOSchedule Precheck completed successfully!");
    return SUCCESS;
}

bool OoOScheduleChecker::PostCheckOpMagic(std::set<int> opSet, const Operation* op, const int programIdx)
{
    if (!opSet.insert(op->GetOpMagic()).second) {
        APASS_LOG_ERROR_F(Elements::Operation, "Program %d: %d opmagic is not unique, OoOSchedule Postcheck failed!",
                          programIdx, op->GetOpMagic());
        return false;
    }
    return true;
}

bool OoOScheduleChecker::PostCheckNewOpConnection(const std::vector<Operation*> opListBeforePass,
                                                  const std::vector<int> opMagicListBeforePass, const Operation* op,
                                                  const int programIdx)
{
    auto it = std::find(opMagicListBeforePass.begin(), opMagicListBeforePass.end(), op->GetOpMagic());
    if (it == opMagicListBeforePass.end()) {
        return true;
    }
    int index = std::distance(opMagicListBeforePass.begin(), it);
    auto& opBefore = opListBeforePass[index];
    auto inTensorsBefore = opBefore->GetIOperands();                            // 老op的ioperands
    std::vector<std::set<Operation*, LogicalTensor::CompareOp>> opBeforeIncast; // 老op的ioperands的生产者
    for (auto& inTensorBefore : inTensorsBefore) {
        opBeforeIncast.emplace_back(inTensorBefore->GetProducers());
    }
    auto inTensorsAfter = op->GetIOperands(); // 新op的ioperands
    int shape = inTensorsAfter.size();
    for (int i = 0; i < shape; i++) {
        auto opAfterIncast = inTensorsAfter[i]->GetProducers(); // 新op的ioperands的生产者
        std::set<Operation*, LogicalTensor::CompareOp> beforeHasAfterNot;
        std::set<Operation*, LogicalTensor::CompareOp> beforeNotAfterHas;
        std::set_difference(opBeforeIncast[i].begin(), opBeforeIncast[i].end(), opAfterIncast.begin(),
                            opAfterIncast.end(), std::inserter(beforeHasAfterNot, beforeHasAfterNot.begin()));
        std::set_difference(opAfterIncast.begin(), opAfterIncast.end(), opBeforeIncast[i].begin(),
                            opBeforeIncast[i].end(), std::inserter(beforeNotAfterHas, beforeNotAfterHas.begin()));
        if (beforeHasAfterNot.empty() && beforeNotAfterHas.empty()) {
            continue;
        }
        std::vector<Operation*> copyins;
        for (auto& opNew : beforeNotAfterHas) {
            if (opNew->GetOpcode() != Opcode::OP_COPY_IN) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Program %d: %d op's successors include unexpected op %s, OoOSchedule Postcheck failed!",
                    programIdx, op->GetOpMagic(), opNew->GetOpcodeStr().c_str());
                return false;
            }
            copyins.emplace_back(opNew);
        }
        std::vector<Operation*> copyouts;
        for (auto& copyin : copyins) {
            auto opPtr = *(copyin->GetIOperands()[0]->GetProducers().begin());
            if (opPtr->GetOpcode() != Opcode::OP_COPY_OUT) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Program %d: %d op's successors include unexpected op %s, OoOSchedule Postcheck failed!",
                    programIdx, copyin->GetOpMagic(), opPtr->GetOpcodeStr().c_str());
                return false;
            }
        }
        std::set<Operation*, LogicalTensor::CompareOp> mainres;
        for (auto& copyout : copyouts) {
            mainres.insert(*(copyout->GetIOperands()[0]->GetProducers()).begin());
        }
        if (mainres != beforeHasAfterNot) {
            std::set<Operation*, LogicalTensor::CompareOp> difference;
            std::set_difference(beforeHasAfterNot.begin(), beforeHasAfterNot.end(), mainres.begin(), mainres.end(),
                                std::inserter(difference, difference.begin()));
            for (auto& dif : difference) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Program %d: %d op is not found after OoOSchedule, OoOSchedule Postcheck failed!",
                                  programIdx, dif->GetOpMagic());
                return false;
            }
        }
    }
    return true;
}

bool OoOScheduleChecker::PostCheckSpecialOp(const Operation* op)
{
    if (op->GetOpcode() == Opcode::OP_ASSEMBLE || op->GetOpcode() == Opcode::OP_RESHAPE ||
        op->GetOpcode() == Opcode::OP_VIEW) {
        // 检查输出不在DDR的op: alloc标签不允许打在ASSEMBLE/RESHAPE/VIEW的输出tensor上
        if (op->GetOOperands()[0]->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            bool needAlloc = false;
            if (op->GetOOperands()[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc) && needAlloc) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "ASSEMBLE/RESHAPE/VIEW op output tensor has alloc attribute, OoOSchedule Postcheck failed!");
                return false;
            }
        }
    }
    return true;
}

bool OoOScheduleChecker::PostCheckTensorMagic(std::set<int> tensorSet, const LogicalTensorPtr tensor,
                                              const int programIdx)
{
    if (!tensorSet.insert(tensor->GetMagic()).second) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Program %d: %d tensormagic is not unique, OoOSchedule Postcheck failed!", programIdx,
                          tensor->GetMagic());
        return false;
    }
    return true;
}

bool OoOScheduleChecker::PostCheckLocalTensor(const LogicalTensorPtr tensor, const int programIdx)
{
    MemoryType memType = tensor->GetMemoryTypeOriginal();
    if (memType == MemoryType::MEM_UB || memType == MemoryType::MEM_L1 || memType == MemoryType::MEM_L0A ||
        memType == MemoryType::MEM_L0B || memType == MemoryType::MEM_L0C) {
        int memoryRange = tensor->memoryrange.end - tensor->memoryrange.start;
        if (memoryRange == 0) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Program %d: %d tensor memory range is 0, OoOSchedule Postcheck failed!", programIdx,
                              tensor->GetMagic());
            return false;
        }
        int tensorshape = 1;
        for (auto num : tensor->GetShape()) {
            tensorshape *= num;
        }
        int tensorsize = tensorshape * BytesOf(tensor->Datatype());
        if (memoryRange < tensorsize) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Program %d: %d tensor memory range < tensor size, OoOSchedule Postcheck failed!",
                              programIdx, tensor->GetMagic());
            return false;
        }
    }
    return true;
}

bool OoOScheduleChecker::PostCheckGlobalTensor(const LogicalTensorPtr tensor, const int programIdx)
{
    MemoryType memType = tensor->GetMemoryTypeOriginal();
    if (memType >= MemoryType::MEM_DEVICE_DDR && !(SubgraphUtils::IsBoundary(tensor))) {
        if (tensor->memoryrange.memId == -1) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Program %d: %d global tensor memid is -1, OoOSchedule Postcheck failed!", programIdx,
                              tensor->GetMagic());
            return false;
        }
    }
    return true;
}

bool OoOScheduleChecker::PostCheckDynValidShape(const LogicalTensorPtr tensor, const int programIdx)
{
    if (tensor->dynValidShape_.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Program %d: %d Dyn validshape is empty, OoOSchedule Postcheck failed!",
                          programIdx, tensor->GetMagic());
        return false;
    }
    return true;
}

bool OoOScheduleChecker::PostCheckNewTensor(std::pair<const int, Function*> program, const int programIdx)
{
    std::vector<LogicalTensorPtr> newTensors;
    std::unordered_set<int> tensorMagicBeforePass;
    std::unordered_set<int> tensorMagicAfterPass;
    for (auto& tensor : tensorListBeforePass_[programIdx]) {
        tensorMagicBeforePass.insert(tensor->GetMagic());
    }
    for (auto& tensor : tensorListAfterPass_[programIdx]) {
        tensorMagicAfterPass.insert(tensor->GetMagic());
    }
    for (auto& tensor : tensorListAfterPass_[programIdx]) {
        int magic = tensor->GetMagic();
        if (tensorMagicBeforePass.find(magic) == tensorMagicBeforePass.end()) {
            newTensors.emplace_back(tensor);
        }
    }
    for (auto& newtensor : newTensors) {
        auto matchTensors = GraphUtils::FindOverlappedTensors(*(program.second), newtensor);
        bool existFlag = GraphUtils::GetTensorByMagic(*(program.second), newtensor->GetMagic()) == newtensor;
        if (!existFlag && matchTensors.size() != 0) {
            existFlag = true;
        }
        if (existFlag == false) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Program %d: %d new tensor does not exist in tensormap, OoOSchedule Postcheck failed!",
                              programIdx, newtensor->GetMagic());
            return false;
        }
        if ((newtensor->shape.size() == 0) && (SubgraphUtils::IsBoundary(newtensor))) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Program %d: %d new tensor shape is null, OoOSchedule Postcheck failed!", programIdx,
                              newtensor->GetMagic());
            return false;
        }
    }
    return true;
}

Status OoOScheduleChecker::PostCheckTensor(const LogicalTensorPtr& tensor, const std::set<int>& tensorSet,
                                           int programIdx)
{
    if (!PostCheckTensorMagic(tensorSet, tensor, programIdx)) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "PostCheckTensorMagic failed; Please check the PostCheckTensorMagic method.");
        return FAILED; // tensor magic不能重复
    };
    if (!PostCheckLocalTensor(tensor, programIdx)) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "PostCheckLocalTensor failed; Please check the PostCheckLocalTensor method.");
        return FAILED; // 0 < memoryrange < shape*dtype
    };
    if (!PostCheckGlobalTensor(tensor, programIdx)) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "PostCheckGlobalTensor failed; Please check the PostCheckGlobalTensor method.");
        return FAILED; // global tensor memid不为-1
    };
    if (!PostCheckDynValidShape(tensor, programIdx)) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "PostCheckDynValidShape failed; Please check the PostCheckDynValidShape method.");
        return FAILED; // valid shape存在
    };
    return SUCCESS;
}

Status OoOScheduleChecker::PostCheckSubGraph(const std::pair<uint64_t, Function*>& program, int programIdx)
{
    auto opList = program.second->Operations().DuplicatedOpList();
    if (opList.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "Operation List is empty! \nSubgraph[%zu] OoOSchedule Precheck end.",
                         program.first);
        return SUCCESS;
    }
    for (auto& op : opList) {
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation is nullptr, OoOSchedule Postcheck failed!");
            return FAILED;
        }
    }
    std::unordered_set<LogicalTensorPtr> tensorList;
    std::set<int> opSet;
    auto opListBeforePass = oriFunctions_[programIdx]->Operations().DuplicatedOpList();
    std::vector<int> opMagicListBeforePass;
    for (auto& op : opListBeforePass) {
        opMagicListBeforePass.emplace_back(op->GetOpMagic());
    }
    for (auto& op : opList) {
        if (!PostCheckOpMagic(opSet, op, programIdx)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "PostCheckOpMagic failed; Please check the PostCheckOpMagic method.");
            return FAILED; // opmagic 不能重复
        };
        if (!PostCheckNewOpConnection(opListBeforePass, opMagicListBeforePass, op, programIdx)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "PostCheckNewOpConnection failed; Please check the PostCheckNewOpConnection method.");
            return FAILED; // 图连接关系找全
        };
        if (!PostCheckSpecialOp(op)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "PostCheckSpecialOp failed; Please check the PostCheckSpecialOp method.");
            return FAILED; // 特别检查ASSEMBLE/RESHAPE/VIEW op
        };
        // 记录所有op的输入tensor和输出tensor
        auto ioperands = op->GetIOperands();
        auto ooperands = op->GetOOperands();
        std::copy(ioperands.begin(), ioperands.end(), std::inserter(tensorList, tensorList.end()));
        std::copy(ooperands.begin(), ooperands.end(), std::inserter(tensorList, tensorList.end()));
    }
    tensorListAfterPass_[programIdx] = tensorList;
    std::set<int> tensorSet;
    for (auto& tensor : tensorList) {
        if (PostCheckTensor(tensor, tensorSet, programIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "PostCheckTensor failed; Please check the PostCheckTensor method.");
            return FAILED;
        }
    }
    // 新增tensor要出现在tensormap中; 新增tensor的orishape不为0
    if (!PostCheckNewTensor(program, programIdx)) {
        APASS_LOG_ERROR_F(Elements::Tensor, "PostCheckNewTensor failed; Please check the PostCheckNewTensor method.");
        return FAILED;
    };
    return SUCCESS;
}

Status OoOScheduleChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start OoOSchedule Postcheck.");
    int programSize = function.rootFunc_->programs_.size();
    tensorListAfterPass_.resize(programSize);
    int programIdx = 0;
    for (auto& program : function.rootFunc_->programs_) { // 对每个子图分别进行postcheck
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OoOSchedule Postcheck begin.", program.first);
        if (PostCheckSubGraph(program, programIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Subgraph[%d] OoOSchedule Postcheck failed!", programIdx);
            return FAILED;
        }
        programIdx++;
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OoOSchedule Postcheck end.", program.first);
    }
    APASS_LOG_INFO_F(Elements::Operation, "OoOSchedule Postcheck completed successfully!");
    return SUCCESS;
}

void OoOScheduleChecker::SetOriFunctions(const std::vector<Function*>& oriFunctions) { oriFunctions_ = oriFunctions; }
} // namespace tile_fwk
} // namespace npu
