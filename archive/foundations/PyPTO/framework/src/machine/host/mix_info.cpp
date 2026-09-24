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
 * \file mix_info.cpp
 * \brief
 */

#include <fstream>
#include <nlohmann/json.hpp>
#include "mix_info.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
using json = nlohmann::json;
namespace npu {
namespace tile_fwk {
namespace mix_info {
constexpr uint32_t AVG_EVENTS_PER_TASK = 16;
constexpr int MAX_DFX_TASK_NUM_PER_CORE = 10000;
constexpr uint32_t MAX_TOTAL_EVENT_NUMS = AVG_EVENTS_PER_TASK * MAX_DFX_TASK_NUM_PER_CORE;

int GetMixInfoEventId(const Operation& op)
{
    switch (op.GetOpcode()) {
        case Opcode::OP_FFTS_CROSS_CORE_SYNC:
        case Opcode::OP_WAIT_FLAG_DEV:
            return MIX_INFO_FORCE_SYNC_EVENT_ID;
        default:
            return op.GetSyncQueue().eventId_;
    }
}

void GetExecuteFunc(Function* func, std::map<int, std::set<Function*>>& leafFunctions)
{
    auto funcType = func->GetGraphType();
    if (func->IsFunctionTypeAndGraphType(
            {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH},
            GraphType::TENSOR_GRAPH)) {
        for (auto callop : func->GetCallopList()) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            auto callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
            if (callFunc == nullptr) {
                continue;
            }
            GetExecuteFunc(callFunc, leafFunctions);
        }
        return;
    } else if (funcType == GraphType::EXECUTE_GRAPH) {
        for (auto callop : func->GetCallopList()) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            auto wrapId = callopAttr->wrapId;
            if (wrapId == -1) {
                continue;
            }
            auto callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
            if (callFunc == nullptr) {
                continue;
            }
            leafFunctions[wrapId].insert(callFunc);
        }
        return;
    } else if (funcType == GraphType::TILE_GRAPH) {
        GetExecuteFunc(func->GetRootFunction(), leafFunctions);
    }
    return;
}

void DumpMixInfoToJson(const std::map<uint64_t, std::map<int, WrapInfo>>& wrapInfos)
{
    std::vector<MixInfo> wrapinfoList;
    for (auto& [mixId, rootWrapinfo] : wrapInfos) {
        MixInfo mixInfo;
        mixInfo.mixId = mixId;
        for (auto& [wrapId, wrapInfo] : rootWrapinfo) {
            (void)wrapId;
            mixInfo.wrapInfos.push_back(wrapInfo);
        }
        wrapinfoList.push_back(mixInfo);
    }
    json j = wrapinfoList;
    std::string path = npu::tile_fwk::config::GetAbsoluteTopFolder() + "/mix_event_info.json";
    std::ofstream of(path);
    if (of.is_open()) {
        of << j.dump(4);
        of.close();
    }
}

int DumpMixInfo(Function* topFunc)
{
    if (topFunc == nullptr) {
        return 0;
    }
    std::map<int, std::set<Function*>> leafFunctions;
    GetExecuteFunc(topFunc, leafFunctions);
    std::map<uint64_t, std::map<int, WrapInfo>> wrapInfos;
    uint32_t totalSetEventCount = 0;
    uint32_t totalWaitEventCount = 0;
    for (auto& [wrapID, leafFuncs] : leafFunctions) {
        for (auto& leafFunc : leafFuncs) {
            auto leafAttr = leafFunc->GetLeafFuncAttribute();
            if (leafAttr == nullptr) {
                continue;
            }
            auto mixId = leafAttr->mixId;
            if (wrapInfos.find(mixId) == wrapInfos.end() || wrapInfos[mixId].find(wrapID) == wrapInfos[mixId].end()) {
                WrapInfo info;
                info.wrapID = wrapID;
                wrapInfos[mixId][wrapID] = info;
            }
            CoreTask leafFuncSyncInfo;
            leafFuncSyncInfo.hashValue = leafFunc->GetFunctionHash().GetHash();
            for (auto& op : leafFunc->Operations(false).DuplicatedOpList()) {
                SyncInfo syncInfo;
                if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC || op->GetOpcode() == Opcode::OP_FFTS_CROSS_CORE_SYNC) {
                    syncInfo.isSet = true;
                    totalSetEventCount++;
                } else if (op->GetOpcode() == Opcode::OP_CV_SYNC_DST || op->GetOpcode() == Opcode::OP_WAIT_FLAG_DEV) {
                    syncInfo.isSet = false;
                    totalWaitEventCount++;
                } else {
                    continue;
                }
                syncInfo.eventID = GetMixInfoEventId(*op);
                leafFuncSyncInfo.syncMsg.push_back(syncInfo);
            }
            wrapInfos[mixId][wrapID].coreTask.push_back(leafFuncSyncInfo);
        }
    }
    DumpMixInfoToJson(wrapInfos);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, totalSetEventCount <= MAX_TOTAL_EVENT_NUMS)
        << "TotalSetEventCount (" << totalSetEventCount << ") is larger than MAX_TOTAL_EVENT_NUMS("
        << MAX_TOTAL_EVENT_NUMS << ")";
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, totalWaitEventCount <= MAX_TOTAL_EVENT_NUMS)
        << "TotalWaitEventCount (" << totalWaitEventCount << ") is larger than MAX_TOTAL_EVENT_NUMS("
        << MAX_TOTAL_EVENT_NUMS << ")";
    return 0;
}
} // namespace mix_info
} // namespace tile_fwk
} // namespace npu
