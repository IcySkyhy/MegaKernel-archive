/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tune_sync_for_vf.cpp
 * \brief
 */

#include "passes/block_graph_pass/tune_sync_for_vf.h"
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/pass_log/pass_log.h"
#include "interface/configs/config_manager_ng.h"

#define MODULE_NAME "TuneSyncForVF"

namespace npu {
namespace tile_fwk {
bool TuneSyncForVF::NeedAdjustSetFlag(Function* subGraphFunc, Operation* vecTileOp0, Operation* vecTileOp1,
                                      Operation* setFlag)
{
    PipeType pipeX = setFlag->syncQueue_.trigPipeId_;
    float tv = static_cast<float>(subGraphFunc->pipeEndTime[PipeType::PIPE_V]);
    float tx = static_cast<float>(subGraphFunc->pipeEndTime[pipeX]);
    float t0 = static_cast<float>(vecTileOp0->cycleStart);
    float t1 = static_cast<float>(vecTileOp0->cycleEnd);
    float t2 = static_cast<float>(vecTileOp1->cycleEnd);
    float ty = t0 + vfPrarm * vecTileOp0->GetLatency() + vfPrarm * vecTileOp1->GetLatency();
    Operation* tileOpZ = subGraphFunc->setOpMap[setFlag];
    float tb = static_cast<float>(tileOpZ->cycleStart);
    if (std::max(tv - t2 + ty, tx + std::max(static_cast<float>(0), (ty - std::max(t1, tb)))) < tv) {
        return true;
    }
    return false;
}

bool TuneSyncForVF::NeedAdjustWaitFlag(Function* subGraphFunc, Operation* vecTileOp0, Operation* vecTileOp1,
                                       Operation* waitFlag)
{
    PipeType pipeX = waitFlag->syncQueue_.pipeId_;
    float tv = static_cast<float>(subGraphFunc->pipeEndTime[PipeType::PIPE_V]);
    float tx = static_cast<float>(subGraphFunc->pipeEndTime[pipeX]);
    float t0 = static_cast<float>(vecTileOp0->cycleStart);
    float t2 = static_cast<float>(vecTileOp1->cycleEnd);
    Operation* tileOpZ = subGraphFunc->waitOpMap[waitFlag];
    float tb = static_cast<float>(tileOpZ->cycleEnd);
    float ty = std::max(t0, tb) + vfPrarm * vecTileOp0->GetLatency() + vfPrarm * vecTileOp1->GetLatency();
    if (std::max(tv - t2 + ty, tx) < tv) {
        return true;
    }
    return false;
}

void TuneSyncForVF::GenPipeOpMap(Function* subGraphFunc)
{
    PipeSync ps;
    pipeOpMap.clear();
    std::vector<Operation*> oriOpList = subGraphFunc->oriOpList;
    for (auto& op : oriOpList) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
        ps.AdjustOpCfg(opcfg, *op);
        if (opcfg.coreType_ != CoreType::AIV) {
            continue;
        }
        auto& startVec = pipeOpMap[opcfg.pipeIdStart_];
        if (std::find(startVec.begin(), startVec.end(), op) == startVec.end()) {
            startVec.emplace_back(op);
        }
        auto& endVec = pipeOpMap[opcfg.pipeIdEnd_];
        if (std::find(endVec.begin(), endVec.end(), op) == endVec.end()) {
            endVec.emplace_back(op);
        }
    }
}

size_t TuneSyncForVF::MoveOpsForMerge(size_t vecTileOp0Idx, size_t vecTileOp1Idx, int groupNum,
                                      std::vector<Operation*>& setFlagList, std::vector<Operation*>& waitFlagList)
{
    // 将setwaitflag删掉
    std::vector<size_t> setWaitIdx;
    for (size_t k = vecTileOp0Idx + 1; k < vecTileOp1Idx; k++) {
        setWaitIdx.emplace_back(k);
    }
    for (auto it = setWaitIdx.rbegin(); it != setWaitIdx.rend(); it++) {
        opList_.erase(opList_.begin() + *it);
    }
    // 删掉这些op后，vecTileOp1Idx = vecTileOp0Idx + 1, 在vecTileOp1Idx右侧将setflag插入
    auto insertPos = opList_.begin() + vecTileOp0Idx + NUM_VALUE_2;
    opList_.insert(insertPos, setFlagList.begin(), setFlagList.end());
    // 在vecTileOp0Idx集合的左侧将waitflag插入
    size_t mergedSize = mergedOps[groupNum].size();
    auto insertPos2 = opList_.begin() + vecTileOp0Idx - mergedSize + NUM_VALUE_2;
    opList_.insert(insertPos2, waitFlagList.begin(), waitFlagList.end());
    return mergedSize;
}

Status TuneSyncForVF::UpdatePipeVTime(Operation* vecTileOp1, int groupNum, size_t mergedSize, int& curVFStartTime,
                                      int& curVecTileOp1EndTime)
{
    curVFStartTime = mergedOps[groupNum][0]->cycleStart; // 当前vf融合op开始时间
    int prevEndTime = curVFStartTime;
    int preVectileOp1EndTime = mergedOps[groupNum][mergedOps[groupNum].size() - 1]->cycleEnd;
    for (size_t i = 0; i < mergedSize; i++) {
        mergedOps[groupNum][i]->cycleStart = prevEndTime;
        auto oriLatency = mergedOps[groupNum][i]->GetLatency();
        mergedOps[groupNum][i]->UpdateLatency(static_cast<int>(oriLatency * vfPrarm));
        auto newLatency = mergedOps[groupNum][i]->GetLatency();
        mergedOps[groupNum][i]->cycleEnd = mergedOps[groupNum][i]->cycleStart + newLatency;
        prevEndTime = mergedOps[groupNum][i]->cycleEnd;
    }
    curVecTileOp1EndTime = mergedOps[groupNum][mergedOps[groupNum].size() - 1]->cycleEnd; // 当前vf融合op结束时间
    int moveFrontTime = preVectileOp1EndTime - curVecTileOp1EndTime;
    // 找到vecTileOp1在pipe_v中的位置，然后将后面的op的开始终止时间全部提前moveFrontTime
    auto& pipeVops = pipeOpMap[PipeType::PIPE_V];
    bool findFlag = false;
    for (size_t k = 0; k < pipeVops.size(); k++) {
        if (pipeVops[k]->GetOpMagic() == vecTileOp1->GetOpMagic()) {
            findFlag = true;
            for (size_t j = k + 1; j < pipeVops.size(); j++) {
                pipeVops[j]->cycleStart -= moveFrontTime;
                pipeVops[j]->cycleEnd -= moveFrontTime;
            }
            break;
        }
    }
    if (!findFlag) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot find %d %s in %s oplist, UpdatePipeVTime failed.",
                          vecTileOp1->GetOpMagic(), vecTileOp1->GetOpcodeStr().c_str(),
                          GetPipeTypeDict().Find(PipeType::PIPE_V).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status TuneSyncForVF::UpdateSetPipeTime(Function* subGraphFunc, std::vector<Operation*>& setFlagList,
                                        const int& curVecTileOp1EndTime)
{
    for (auto& setFlag : setFlagList) {
        bool findFlag = false;
        auto pipeX = setFlag->syncQueue_.trigPipeId_;
        auto& tileOpZ = subGraphFunc->setOpMap[setFlag];
        // 在pipeX的队列中找到tileopZ
        auto& pipeXops = pipeOpMap[pipeX];
        for (size_t k = 0; k < pipeXops.size(); k++) {
            if (pipeXops[k]->GetOpMagic() == tileOpZ->GetOpMagic()) {
                findFlag = true;
                int preOpEndTime;
                if (k == 0) {
                    preOpEndTime = pipeXops[0]->cycleStart;
                } else {
                    preOpEndTime = pipeXops[k - 1]->cycleEnd;
                }
                auto tileOpZNewStartTime = std::max(preOpEndTime, curVecTileOp1EndTime);
                int moveDist = tileOpZ->cycleStart - tileOpZNewStartTime;
                for (size_t j = k; j < pipeXops.size(); j++) {
                    pipeXops[j]->cycleStart -= moveDist;
                    pipeXops[j]->cycleEnd -= moveDist;
                }
                break;
            }
        }
        if (!findFlag) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find %d %s in %s oplist, UpdateSetPipeTime failed.",
                              tileOpZ->GetOpMagic(), tileOpZ->GetOpcodeStr().c_str(),
                              GetPipeTypeDict().Find(pipeX).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status TuneSyncForVF::UpdateWaitPipeTime(Function* subGraphFunc, std::vector<Operation*>& waitFlagList,
                                         const int& curVFStartTime, int& maxMoveBackDist)
{
    for (auto& waitFlag : waitFlagList) {
        bool findFlag = false;
        auto pipeX = waitFlag->syncQueue_.pipeId_;
        auto& tileOpZ = subGraphFunc->waitOpMap[waitFlag];
        // 在pipeX的队列中找到tileopZ
        auto& pipeXops = pipeOpMap[pipeX];
        for (size_t k = 0; k < pipeXops.size(); k++) {
            if (pipeXops[k]->GetOpMagic() == tileOpZ->GetOpMagic()) {
                findFlag = true;
                maxMoveBackDist = std::max(maxMoveBackDist, tileOpZ->cycleEnd - curVFStartTime);
                break;
            }
        }
        if (!findFlag) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find %d %s in %s oplist, UpdateWaitPipeTime failed.",
                              tileOpZ->GetOpMagic(), tileOpZ->GetOpcodeStr().c_str(),
                              GetPipeTypeDict().Find(pipeX).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status TuneSyncForVF::MoveBackPipeVOps(int groupNum, const int& maxMoveBackDist)
{
    auto& firstOp = mergedOps[groupNum][0];
    bool findFlag = false;
    auto& pipeVops = pipeOpMap[PipeType::PIPE_V];
    for (size_t k = 0; k < pipeVops.size(); k++) {
        if (pipeVops[k]->GetOpMagic() == firstOp->GetOpMagic()) {
            findFlag = true;
            for (size_t j = k; j < pipeVops.size(); j++) {
                pipeVops[j]->cycleStart += maxMoveBackDist;
                pipeVops[j]->cycleEnd += maxMoveBackDist;
            }
            break;
        }
    }
    if (!findFlag) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot find %d %s in %s oplist, MoveBackPipeVOps failed.",
                          firstOp->GetOpMagic(), firstOp->GetOpcodeStr().c_str(),
                          GetPipeTypeDict().Find(PipeType::PIPE_V).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status TuneSyncForVF::AdjustSetWaitFlag(Function* subGraphFunc, std::vector<Operation*>& setFlagList,
                                        std::vector<Operation*>& waitFlagList, size_t vecTileOp0Idx,
                                        size_t vecTileOp1Idx, int groupNum)
{
    auto vecTileOp1 = opList_[vecTileOp1Idx];
    // 改变opList执行顺序
    size_t mergedSize = MoveOpsForMerge(vecTileOp0Idx, vecTileOp1Idx, groupNum, setFlagList, waitFlagList);

    // 更新各pipe上op的时间戳
    // pipe_v
    int curVFStartTime;       // 当前vf融合op开始时间
    int curVecTileOp1EndTime; // 当前vf融合op结束时间
    if (UpdatePipeVTime(vecTileOp1, groupNum, mergedSize, curVFStartTime, curVecTileOp1EndTime) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustSetWaitFlag failed at UpdatePipeVTime.");
        return FAILED;
    }

    // setflag对应的各pipe
    if (UpdateSetPipeTime(subGraphFunc, setFlagList, curVecTileOp1EndTime) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustSetWaitFlag failed at UpdateSetPipeTime.");
        return FAILED;
    }

    // waitflag对应的各pipe  需要让pipev的op后移来满足依赖关系
    int maxMoveBackDist{0};
    if (UpdateWaitPipeTime(subGraphFunc, waitFlagList, curVFStartTime, maxMoveBackDist) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustSetWaitFlag failed at UpdateWaitPipeTime.");
        return FAILED;
    }

    // 后移vf融合op及其之后的pipe_v Op
    if (MoveBackPipeVOps(groupNum, maxMoveBackDist) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustSetWaitFlag failed at MoveBackPipeVOps.");
        return FAILED;
    }

    return SUCCESS;
}

void TuneSyncForVF::FindPipeVIdx(std::vector<size_t>& pipeVIdx, AIVCore coreType)
{
    PipeSync ps;
    for (size_t i = 0; i < opList_.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList_[i]->GetOpcode());
        ps.AdjustOpCfg(opcfg, *opList_[i]);
        if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList_[i]->GetAIVCore() == coreType) {
            pipeVIdx.emplace_back(i);
        }
    }
}

bool TuneSyncForVF::IsMergeable(size_t left, size_t right, std::vector<Operation*>& setFlagList,
                                std::vector<Operation*>& waitFlagList)
{
    // 判断两个pipeV op间是否有SYNC_SRC或者SYNC_DST或者既不是SYNC_SRC也不是SYNC_DST
    bool hasNonSetWaitOp = false;
    for (size_t k = left + 1; k < right; k++) {
        if (opList_[k]->GetOpcode() == Opcode::OP_SYNC_SRC) {
            setFlagList.emplace_back(opList_[k]);
        } else if (opList_[k]->GetOpcode() == Opcode::OP_SYNC_DST) {
            waitFlagList.emplace_back(opList_[k]);
        } else {
            hasNonSetWaitOp = true;
            break;
        }
    }
    // 两个pipeV的op间的op如果有一个既不是SYNC_SRC也不是SYNC_DST,则说明这两个pipeV op不能合并
    if (hasNonSetWaitOp) {
        return false;
    }
    return true;
}

bool TuneSyncForVF::NeedAdjustOpSeq(Function* subGraphFunc, const std::vector<Operation*>& setFlagList,
                                    const std::vector<Operation*>& waitFlagList, size_t left, size_t right)
{
    if (setFlagList.empty() && waitFlagList.empty()) {
        return true;
    }
    for (auto& setFlag : setFlagList) {
        if (NeedAdjustSetFlag(subGraphFunc, opList_[left], opList_[right], setFlag)) {
            return true;
        }
    }
    for (auto& waitFlag : waitFlagList) {
        if (NeedAdjustWaitFlag(subGraphFunc, opList_[left], opList_[right], waitFlag)) {
            return true;
        }
    }
    return false;
}

void TuneSyncForVF::AddVecTileopsToGroup(int& groupNum, size_t left, size_t right)
{
    for (size_t i = 0; i < mergedOps.size(); i++) {
        for (size_t j = 0; j < mergedOps[i].size(); j++) {
            if (mergedOps[i][j] == opList_[left]) {
                groupNum = i;
                break;
            }
        }
    }
    if (groupNum == -1) {
        // 将vecTileop0和vecTileop1添加到mergedOps中
        std::vector<Operation*> newOp = {opList_[left], opList_[right]};
        mergedOps.emplace_back(newOp);
        groupNum = mergedOps.size() - 1;
    } else {
        mergedOps[groupNum].emplace_back(opList_[right]);
    }
}

Status TuneSyncForVF::ChangeOpSeq(Function* subGraphFunc, bool isAIV1)
{
    AIVCore coreType;
    if (!isAIV1) {
        coreType = AIVCore::AIV0;
    } else {
        coreType = AIVCore::AIV1;
    }
    std::vector<size_t> pipeVIdx;
    FindPipeVIdx(pipeVIdx, coreType);
    if (pipeVIdx.size() <= 1) {
        return SUCCESS;
    }

    mergedOps.clear();
    for (size_t idx = 0; idx + 1 < pipeVIdx.size(); idx++) {
        size_t left = pipeVIdx[idx];
        size_t right = pipeVIdx[idx + 1];
        APASS_LOG_DEBUG_F(Elements::Operation, "Try to merge %d %s and %d %s", opList_[left]->GetOpMagic(),
                          opList_[left]->GetOpcodeStr().c_str(), opList_[right]->GetOpMagic(),
                          opList_[right]->GetOpcodeStr().c_str());

        // 判断是否可以进行调整
        std::vector<Operation*> setFlagList;
        std::vector<Operation*> waitFlagList;
        if (!IsMergeable(left, right, setFlagList, waitFlagList)) {
            continue;
        }

        // 判断是否需要进行调整（所有的SYNC_SRC和SYNC_DST中，只要有一个是有收益的，就进行融合）
        int64_t tunevfMode = config::GetPassOption<int64_t>(SG_SET_TUNEVF_MODE);
        if (tunevfMode != NUM2) {
            if (!NeedAdjustOpSeq(subGraphFunc, setFlagList, waitFlagList, left, right)) {
                continue;
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Need merge.");

        // vecTileop1和vecTileop2需要融合，将其加入mergedOps中
        int groupNum = -1;
        AddVecTileopsToGroup(groupNum, left, right);

        // 进行调整
        if (AdjustSetWaitFlag(subGraphFunc, setFlagList, waitFlagList, left, right, groupNum) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ChangeOpSeq failed at function AdjustSetWaitFlag.");
            return FAILED;
        }
        // 由于移动，pipeVop的idx会发生变化，需要重新更新pipeVIdx
        pipeVIdx.clear();
        FindPipeVIdx(pipeVIdx, coreType);
    }
    return SUCCESS;
}

void TuneSyncForVF::LogOpList(const std::string& label)
{
    for (const auto& op : opList_) {
        if (op->GetOpcodeStr().find("SYNC_SRC") != std::string::npos ||
            op->GetOpcodeStr().find("SYNC_DST") != std::string::npos || op->GetOpcode() == Opcode::OP_BAR_V ||
            op->GetOpcode() == Opcode::OP_BAR_M) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "%s operation %d: %s, setpipe type: %s, setcore type: %s, waitpipe type: %s, waitcore type: %s, "
                "eventid: %d",
                label.c_str(), op->GetOpMagic(), op->GetOpcodeStr().c_str(),
                GetPipeTypeDict().Find(op->syncQueue_.pipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.coreType_).c_str(),
                GetPipeTypeDict().Find(op->syncQueue_.trigPipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.trigCoreType_).c_str(), op->syncQueue_.eventId_);
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s operation %d: %s", label.c_str(), op->GetOpMagic(),
                          op->GetOpcodeStr().c_str());
    }
}

Status TuneSyncForVF::RunOnFunction(Function& function)
{
    if (!config::GetPassGlobalConfig(KEY_ENABLE_VF, false)) {
        APASS_LOG_DEBUG_F(Elements::Function, "TuneSyncForVF is skipped for ENABLE_VF is false.");
        return SUCCESS;
    }
    int64_t tunevfMode = config::GetPassOption<int64_t>(SG_SET_TUNEVF_MODE);
    if (tunevfMode == 1) {
        APASS_LOG_DEBUG_F(Elements::Function, "TuneSyncForVF is bypassed for set_tunevf_mode=1.");
        return SUCCESS;
    }
    size_t funcId = 0;
    for (auto& program : function.rootFunc_->programs_) {
        std::vector<Operation*> opList(program.second->Operations(false).DuplicatedOpList());
        opList_ = opList;
        APASS_LOG_DEBUG_F(Elements::Function, "=======================function %zu ======================", funcId);
        LogOpList("Input");
        GenPipeOpMap(program.second);
        if (ChangeOpSeq(program.second, false) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "RunOnFunction failed at function ChangeOpSeq.");
            return FAILED;
        }
        if (ChangeOpSeq(program.second, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "RunOnFunction failed at function ChangeOpSeq.");
            return FAILED;
        }
        program.second->ScheduleBy(opList_, true);
        APASS_LOG_DEBUG_F(Elements::Function, "---------------------------------------------------");
        LogOpList("Output");
        funcId++;
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
