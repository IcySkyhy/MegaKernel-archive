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
 * \file codegen_scalar.cpp
 * \brief
 */

#include "interface/tensor/logical_tensor.h"
#include "codegen_op_npu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
std::string CodeGenOpNPU::GenBarrier() const
{
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    std::ostringstream oss;
    oss << "pipe_barrier(" << pipeId1 << ")" << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintSyncInSingleKernel(bool isWait) const
{
    std::string syncOp = isWait ? "wait_flag" : "set_flag";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    std::ostringstream oss;
    std::vector<std::string> tileOpParams = {pipeId1, pipeId2, "EVENT_ID" + std::to_string(syncQueue.eventId_)};
    oss << syncOp << WrapParamByParentheses(tileOpParams) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenSyncSetOp() const { return PrintSyncInSingleKernel(); }

std::string CodeGenOpNPU::GenSyncWaitOp() const { return PrintSyncInSingleKernel(true); }

void InsertSetSysCnt(std::ostringstream& oss)
{
    oss << "#ifdef OPEN_MIX_PERF\n";
    oss << "{\n";
    oss << "    __gm__ uint64_t* setEventBase = reinterpret_cast<__gm__ uint64_t*>(taskStat->perfDataBaseAddr + "
           "taskStat->setEventAddr);\n";
    oss << "    setEventBase[taskStat->setEventNum++] = get_sys_cnt();\n";
    oss << "}\n";
    oss << "#endif\n";
}

std::string CodeGenOpNPU::GenCVSyncSetOp() const
{
    auto pipeId = GetPipeId(syncQueue.pipeId_);
    std::ostringstream oss;
    InsertSetSysCnt(oss);
    oss << "set_intra_block(" << pipeId << ", " << std::to_string(syncQueue.eventId_) << ");\n";
    return oss.str();
}

void InsertWaitSysCnt(std::ostringstream& oss)
{
    oss << "#ifdef OPEN_MIX_PERF\n";
    oss << "{\n";
    oss << "    __gm__ uint64_t* waitEventBase = reinterpret_cast<__gm__ uint64_t*>(taskStat->perfDataBaseAddr + "
           "taskStat->waitEventAddr);\n";
    oss << "    waitEventBase[taskStat->waitEventNum++] = get_sys_cnt();\n";
    oss << "}\n";
    oss << "#endif\n";
}

std::string CodeGenOpNPU::GenCVSyncWaitOp() const
{
    auto pipeId = GetPipeId(syncQueue.trigPipeId_);
    std::ostringstream oss;
    InsertWaitSysCnt(oss);
    oss << "wait_intra_block(" << pipeId << ", " << std::to_string(syncQueue.eventId_) << ");\n";
    return oss.str();
}

static const std::unordered_map<int, std::string> aicpuCallNumDict = {
    {AICPU_CALL_NUM_COPYOUT_RESOLVE, "AICPU_CALL_NUM_COPYOUT_RESOLVE"},
};

std::string CodeGenOpNPU::GenAicpuCallOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::aicpuCall))
        << "OpAttributeKey::aicpuCall not found";
    uint32_t call = static_cast<uint32_t>(AnyCast<int64_t>(opAttrs.find(OpAttributeKey::aicpuCall)->second));
    uint16_t callNum = call >> AICPU_CALL_ARG_BIT;
    uint16_t callArg = call & ((1 << AICPU_CALL_ARG_BIT) - 1);

    std::ostringstream oss;
    std::string callNumName = std::to_string(callNum);
    if (aicpuCallNumDict.count(callNum)) {
        callNumName = aicpuCallNumDict.find(callNum)->second;
    }
    oss << tileOpName << "<" << callNumName << "," << callArg << ">(GET_CURRENT_TASKID());\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenFFTSCrossCoreSyncOp() const
{
    auto pipeId = GetPipeId(syncQueue.pipeId_);
    std::ostringstream oss;
    InsertSetSysCnt(oss);
    oss << "ffts_cross_core_sync(" << pipeId << ", getFFTSMsg(0x2, " << std::to_string(syncQueue.eventId_) << "))"
        << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenWaitFlagDevOp() const
{
    auto pipeId = GetPipeId(syncQueue.pipeId_);
    std::ostringstream oss;
    InsertWaitSysCnt(oss);
    oss << "wait_flag_dev(" << pipeId << ", " << std::to_string(syncQueue.eventId_) << ")" << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
