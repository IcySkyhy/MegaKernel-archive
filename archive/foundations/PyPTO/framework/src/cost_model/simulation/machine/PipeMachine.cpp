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
 * \file PipeMachine.cpp
 * \brief
 */

#include "cost_model/simulation/machine/PipeMachine.h"
#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/cache/CacheMachine.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/common/ISA.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {
PipeMachine::PipeMachine() { machineType = MachineType::PIPE; }

PipeMachine::PipeMachine(CostModel::MachineType mType, CostModel::CorePipeType pType, int pId) : PipeMachine()
{
    machineType = mType;
    pipeType = pType;
    pipeId = pId;
}

void PipeMachine::Step()
{
    if (needTerminate) {
        return;
    }
    RunAtBegin();
    ReceivePacket();
    ReceiveL2Packet();
    ProcessTileOp();
    RunAtEnd();
}

void PipeMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    stats = std::make_shared<CoreStats>(GetSim()->GetReporter());
    auto pipeQuery = MACHINE_PIPE_SET.at(parentMachine->machineType).find(pipeType);
    if (pipeQuery != MACHINE_PIPE_SET.at(parentMachine->machineType).end()) {
        uint64_t initCycle = GetSim()->GetCycles();
        parentMachine->LoggerRecordTileOp("Pipe Init", pipeId, initCycle, initCycle + 1);
        GetSim()->AddCycles();
    }
    InitQueueDelay();
}

void PipeMachine::Reset() {}

void PipeMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();
    needTerminate = IsTerminate();

    // Updat next simulation cycles
    nextCycles = INT_MAX;
    nextCycles = std::min(nextCycles, GetQueueNextCycles());
    if (retireCycle > 0) {
        nextCycles = std::min(nextCycles, retireCycle);
    }
    GetSim()->UpdateNextCycles(nextCycles);
}

std::shared_ptr<SimSys> PipeMachine::GetSim() { return sim; }

void PipeMachine::Report() {}

void PipeMachine::InitQueueDelay()
{
    submissionQueue.SetWriteDelay(0);
    submissionQueue.SetReadDelay(0);
    completionQueue.SetWriteDelay(0);
    completionQueue.SetReadDelay(0);
    outcastReferenceQueue.SetWriteDelay(0);
    outcastReferenceQueue.SetReadDelay(0);
    incastReferenceQueue.SetWriteDelay(0);
    incastReferenceQueue.SetReadDelay(0);
    releaseQueue.SetWriteDelay(0);
    releaseQueue.SetReadDelay(0);
    cacheRespQueue.SetWriteDelay(0);
    cacheRespQueue.SetReadDelay(0);
}

void PipeMachine::StepQueue()
{
    uint64_t intervalCycles = GetSim()->GetCycles() - lastCycles;
    submissionQueue.UpdateIntervalCycles(intervalCycles);
    completionQueue.UpdateIntervalCycles(intervalCycles);
    cacheRespQueue.UpdateIntervalCycles(intervalCycles);
    submissionQueue.Step();
    completionQueue.Step();
    cacheRespQueue.Step();
}

uint64_t PipeMachine::GetQueueNextCycles()
{
    uint64_t res = INT_MAX;
    uint64_t gCycles = GetSim()->GetCycles();
    res = std::min(res, gCycles + submissionQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + completionQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + cacheRespQueue.GetMinWaitCycles());
    return res;
}

bool PipeMachine::IsTerminate()
{
    return (!executingTask && submissionQueue.IsTerminate() && completionQueue.IsTerminate() &&
            outcastReferenceQueue.IsTerminate() && incastReferenceQueue.IsTerminate() && releaseQueue.IsTerminate() &&
            cacheRespQueue.IsTerminate());
}

void PipeMachine::RunAtBegin() { nextCycles = INT_MAX; }

void PipeMachine::RunAtEnd()
{
    if (!executingTask) {
        return;
    }
    auto corePtr = std::dynamic_pointer_cast<CoreMachine>(parentMachine);
    if (retireCycle <= GetSim()->GetCycles()) {
        PushCompletion(executingTaskId, magic);
        std::string info;
        uint64_t sCycle = 0;
        uint64_t eCycle = 0;
        magic = -1;
        nextCycles = GetSim()->GetCycles() + 1;
        GetSim()->UpdateNextCycles(GetSim()->GetCycles() + 1);
        retireCycle = 0;
        if (tileOp != nullptr) {
            tileOp->exeInfo.cycleInfo.executeEndCycle = GetSim()->GetCycles();
            info += tileOp->Dump(true);
            info += " SUBGRAPH[" + std::to_string(tileOp->subgraphId) + "] TASK[" + std::to_string(executingTaskId) +
                    "] ";
            info += "(" + DecimalTo26(executingTaskId) + ")";
            sCycle = tileOp->exeInfo.cycleInfo.executeStartCycle;
            eCycle = tileOp->exeInfo.cycleInfo.executeEndCycle;
        } else if (tile != nullptr) {
            tile->exeInfo.cycleInfo.executeEndCycle = GetSim()->GetCycles();
            info += tile->Dump();
            info += " SUBGRAPH[" + std::to_string(tile->subgraphId) + "] TASK[" + std::to_string(executingTaskId) + "]";
            sCycle = tile->exeInfo.cycleInfo.executeStartCycle;
            eCycle = tile->exeInfo.cycleInfo.executeEndCycle;
        }
        parentMachine->LoggerRecordTileOp(info, pipeId, sCycle, eCycle);
        corePtr->LoggerRecordTileOpFlow(tileOp);
        corePtr->stats->totalPipeUseCycles[int(pipeType)] += (eCycle - sCycle);
        tileOp = nullptr;
        tile = nullptr;
        SetMachineExecuting(false);
        LoggerRecordPipeWL(pipeId, CounterType::QUEUE_POP);
    } else {
        nextCycles = retireCycle;
        GetSim()->UpdateNextCycles(retireCycle);
    }
}

void PipeMachine::ReceivePacket()
{
    if (submissionQueue.Empty() || executingTask) {
        return;
    }
    TaskPack packet;
    submissionQueue.Dequeue(packet);
    tile = packet.tileopTask.tile;
    tileOp = packet.tileopTask.tileOp;
    executingTaskId = packet.taskId;
    SIMULATION_LOGI("[Cycle: %lu][PipeMachine: %zu][ReceivePacket] get task %lu magic %d", GetSim()->GetCycles(),
                    machineId, packet.taskId, packet.tileopTask.magic);
    SetMachineExecuting(true);
    LoggerRecordPipeWL(pipeId, CounterType::QUEUE_PUSH);
}

void PipeMachine::ReceiveL2Packet()
{
    if (cacheRespQueue.Empty() || !waitL2CacheResponse) {
        return;
    }
    CachePacket packet;
    cacheRespQueue.Dequeue(packet);
    // Process Packet
    SIMULATION_LOGI("[Cycle: %lu][PipeMachine: %zu][ReceiveL2Resp] %lu", GetSim()->GetCycles(), machineId, packet.pid);

    waitL2CacheResponse = false;
    nextCycles = GetSim()->GetCycles() + 1;
    GetSim()->UpdateNextCycles(GetSim()->GetCycles() + 1);
    retireCycle = GetSim()->GetCycles();
}

void PipeMachine::SendCachePacket(bool read)
{
    CachePacket packet;
    packet.pid = machineId;
    packet.type = CachePacketType::REQUEST;
    packet.requestType = read ? CacheRequestType::DATA_READ_REQ : CacheRequestType::DATA_WRITE_REQ;
    packet.addr = tileOp->GetAddress();
    packet.size = tileOp->GetSize();
    packet.cycleInfo.pktSendCycle = GetSim()->GetCycles();
    auto l2Cache = std::dynamic_pointer_cast<CacheMachine>(l2cacheMachine);
    l2Cache->RequestData(packet);
    waitL2CacheResponse = true;
}

void PipeMachine::ProcessTileOp()
{
    if (retireCycle > 0 || (tileOp == nullptr && tile == nullptr)) {
        return;
    }
    // Simulate MTE access L2Cache. retireCycle = L2Packet received cycle;
    // Calculate latency based on shape and config->handle_threshold;
    uint64_t latency = 0;
    if (tileOp != nullptr) {
        latency = pipeImpl->PostSimulate(tileOp);
        SIMULATION_LOGI("[task: %lu][op: %s] latency: %lu", tileOp->taskId, tileOp->opcode.c_str(), latency);
        if (sim->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
            // For calendar schuedule fluctuate
            float basePercent = 100.0;
            latency = uint64_t(float(latency) * (1 + float(GetSim()->config.pipeBoardVibration) / basePercent));
        }
        if (tileOp->specialOp) {
            latency = 1;
        }
        if (l2cacheMachine) {
            if (pipeType == CostModel::CorePipeType::PIPE_MTE_IN) {
                latency = INT_MAX;
                SendCachePacket(true);
            } else if (pipeType == CostModel::CorePipeType::PIPE_MTE_OUT) {
                latency = INT_MAX;
                SendCachePacket(false);
            }
        }
        tileOp->exeInfo.latency = latency;
        tileOp->exeInfo.cycleInfo.executeStartCycle = GetSim()->GetCycles();
        magic = tileOp->magic;
    } else if (tile != nullptr) {
        latency = 1;
        tile->exeInfo.latency = 1;
        tile->exeInfo.cycleInfo.executeStartCycle = GetSim()->GetCycles();
        magic = tile->magic;
    }
    SIMULATION_LOGI("[Cycle: %lu][PipeMachine: %zu] latency: %lu", GetSim()->GetCycles(), machineId, latency);
    nextCycles = GetSim()->GetCycles() + latency;
    GetSim()->UpdateNextCycles(GetSim()->GetCycles() + latency);
    retireCycle = GetSim()->GetCycles() + latency;
}

void PipeMachine::PushCompletion(int taskId, int curMagic)
{
    SIMULATION_LOGI("[Cycle: %lu][PipeMachine: %zu][PushCompletion] push task %d magic %d  in completion queue.",
                    GetSim()->GetCycles(), machineId, taskId, magic);
    CompletedPacket packet;
    packet.taskId = taskId;
    packet.currentType = machineType;
    packet.pipeMsg.magic = curMagic;
    completionQueue.Enqueue(packet);
}

void PipeMachine::LoggerRecordPipeWL(size_t pId, CounterType type)
{
    GetSim()->GetLogger()->AddCounterEvent(parentMachine->machineId, pId + parentMachine->reversedTidNum, type);
}
} // namespace CostModel
