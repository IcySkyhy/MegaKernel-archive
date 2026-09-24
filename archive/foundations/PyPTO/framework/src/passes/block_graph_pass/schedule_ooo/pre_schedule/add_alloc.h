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
 * \file add_alloc.h
 * \brief
 */

#ifndef PASS_ADD_ALLOC_H
#define PASS_ADD_ALLOC_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#undef MODULE_NAME
#define MODULE_NAME "AddAlloc"

namespace npu::tile_fwk {
struct TensorAllocMsg {
    std::vector<std::reference_wrapper<Operation>> producer;
    bool isAllocated{false};
    MemoryType memType;
    int memId;
    size_t firstOperandIdx{0};
};

class AddAlloc : public Pass {
public:
    AddAlloc() : Pass("AddAlloc") {}
    ~AddAlloc() override = default;

private:
    Status RunOnFunction(Function& function) override
    {
        APASS_LOG_INFO_F(Elements::Function, "===> Start AddAlloc.");
        for (auto& program : function.rootFunc_->programs_) {
            if (AddAndCheckAlloc(*program.second) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "AddAndCheckAlloc failed.");
                return FAILED;
            }
        }
        APASS_LOG_INFO_F(Elements::Function, "===> End AddAlloc.");
        return SUCCESS;
    }
    // 按color去判断是否需要插入alloc
    Status GenAllocNode(Function& function);
    Status AddAndCheckAlloc(Function& function);
    Status UpdateTensorAllocMsg(Operation& op, size_t i,
                                std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const;
    Status FindTensorAllocMsg(Operation& op, std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const;
    Status CreateAllocNode(const TensorAllocMsg& tensorAllocMsg, Function& function);
    Status GenAllocOpcode(const Opcode& allocOpcode, const TensorAllocMsg& tensorAllocMsg, Function& function);
    Status GenTensorAllocMsgMap(Function& function, std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const;
    Status SetTensorAllocMsg(Operation& op, std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const;

    TensorAllocMsg ConstructTensorAllocMsg(Operation& op, size_t i, int memId) const;
    const std::unordered_map<MemoryType, Opcode> allocOpcodeMap = {
        {MemoryType::MEM_L0A, Opcode::OP_L0A_ALLOC},
        {MemoryType::MEM_UB, Opcode::OP_UB_ALLOC},
        {MemoryType::MEM_VECTOR_REG, Opcode::OP_REG_ALLOC},
        {MemoryType::MEM_L1, Opcode::OP_L1_ALLOC},
        {MemoryType::MEM_L0B, Opcode::OP_L0B_ALLOC},
        {MemoryType::MEM_L0C, Opcode::OP_L0C_ALLOC},
        {MemoryType::MEM_BT, Opcode::OP_BT_ALLOC},
        {MemoryType::MEM_FIX, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_QUANT_PRE, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_RELU_PRE, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_RELU_POST, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_QUANT_POST, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_ELT_ANTIQ, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_MTE2_ANTIQ, Opcode::OP_FIX_ALLOC}};
};

} // namespace npu::tile_fwk
#endif // PASS_ADD_ALLOC_H
