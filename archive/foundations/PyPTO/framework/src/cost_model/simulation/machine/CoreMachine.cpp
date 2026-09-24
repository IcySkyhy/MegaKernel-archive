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
 * \file CoreMachine.cpp
 * \brief
 */

#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/machine/DeviceMachine.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/arch/TileAllocPipeImpl.h"
#include "cost_model/simulation/value/TileCalculator.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {

CoreMachine::CoreMachine()
{
    machineType = MachineType::AIV;
    pipeMachineIndex.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
    readyQueues.clear();
    for (int i = 0; i < static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE); i++) {
        readyQueues.emplace_back(static_cast<CorePipeType>(i), i);
    }
}

CoreMachine::CoreMachine(CostModel::MachineType type) : CoreMachine() { machineType = type; }

void CoreMachine::Step()
{
    if (needTerminate) {
        return;
    }
    RunAtBegin();
    ReceivePacket();
    RetireTileOp();
    IssueTileOp();
    RunAtEnd();
    SetCalendar();
}

void CoreMachine::SetCalendar()
{
    if ((GetSim()->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) && needSet) {
        for (auto wait : sim->taskWaitBeforeSetMap[executingTaskId]) {
            int counterId = wait.first;
            int expectValue = wait.second;
            if (sim->calendarCounter[counterId] < expectValue) {
                SIMULATION_LOGI(
                    "[Cycle: %lu][CoreMachine][SetCalendar] task id %lu counter %d expectValue %d current value %d",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(executingTaskId),
                    counterId, expectValue, sim->calendarCounter[counterId]);

                needWaitCounter = true;
                return;
            }
        }
        needSet = false;
        needWaitCounter = false;

        auto it = sim->taskSetMap.find(executingTaskId);
        if (it != sim->taskSetMap.end()) {
            counterSetQueue.Enqueue(calendarSecondSet);
        }

        SIMULATION_LOGI("[Cycle: %lu][CoreMachine][SetCalendar] task id %lu SetCalendar pass!!!",
                        static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(executingTaskId));
    }
}

void CoreMachine::InitBufferSize()
{
    bufferSize.clear();
    aliveBuffer.clear();
    bufferSize[CorePipeType::PIPE_VECTOR_BMU] = 0;
    bufferSize[CorePipeType::PIPE_CUBE_BMU_L0A] = 0;
    bufferSize[CorePipeType::PIPE_CUBE_BMU_L0B] = 0;
    bufferSize[CorePipeType::PIPE_CUBE_BMU_L0C] = 0;
    bufferSize[CorePipeType::PIPE_CUBE_BMU_L1] = 0;
}

void CoreMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    stats = std::make_shared<CoreStats>(GetSim()->GetReporter());
    LoggerRecordTaskStart("Core Machine Init");
    GetSim()->AddCycles();
    LoggerRecordTaskEnd();
    GetSim()->AddCycles();
    InitQueueDelay();
}

void CoreMachine::Reset() {}

void CoreMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();
    needTerminate = IsTerminate();

    // Updat next simulation cycles
    nextCycles = INT_MAX;
    nextCycles = std::min(nextCycles, GetQueueNextCycles());
    if (coreNextNeedStep) {
        nextCycles = std::min(nextCycles, GetSim()->GetCycles() + 1);
    }
    if (exectingFixLatencyTask) {
        nextCycles = std::min(nextCycles, fixedLatencyTaskEndCycle);
    }
    GetSim()->UpdateNextCycles(nextCycles);
}

std::shared_ptr<SimSys> CoreMachine::GetSim() { return sim; }

void CoreMachine::Report()
{
    int machineSeq = GetMachineSeq(machineId);
    std::string name = MachineName(machineType);
    name += "_" + std::to_string(machineSeq);
    stats->Report(name);
}

void CoreMachine::InitQueueDelay()
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
    counterSetQueue.SetWriteDelay(config.calendarSetQueueWDelay);
    counterSetQueue.SetReadDelay(0);
}

void CoreMachine::StepQueue()
{
    uint64_t intervalCycles = GetSim()->GetCycles() - lastCycles;
    submissionQueue.UpdateIntervalCycles(intervalCycles);
    completionQueue.UpdateIntervalCycles(intervalCycles);
    outcastReferenceQueue.UpdateIntervalCycles(intervalCycles);
    incastReferenceQueue.UpdateIntervalCycles(intervalCycles);
    releaseQueue.UpdateIntervalCycles(intervalCycles);
    cacheRespQueue.UpdateIntervalCycles(intervalCycles);
    counterSetQueue.UpdateIntervalCycles(intervalCycles);
    submissionQueue.Step();
    completionQueue.Step();
    outcastReferenceQueue.Step();
    incastReferenceQueue.Step();
    releaseQueue.Step();
    cacheRespQueue.Step();
    counterSetQueue.Step();
    while (!counterSetQueue.Empty()) {
        counterSetQueue.CalendarPopFront();
        sim->calendarCounter[sim->taskSetMap[executingTaskId]]++;

        SIMULATION_LOGI(
            "[Cycle: %lu][CoreMachine][StepQueue] task id %lu set counter %d value before %d value after %d",
            static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(executingTaskId),
            sim->taskSetMap[executingTaskId], sim->calendarCounter[sim->taskSetMap[executingTaskId]] - 1,
            sim->calendarCounter[sim->taskSetMap[executingTaskId]]);
    }
}

void CoreMachine::SetTileState(std::shared_ptr<TileState>& state) { tileState = state; }

uint64_t CoreMachine::GetQueueNextCycles()
{
    uint64_t res = INT_MAX;
    uint64_t gCycles = GetSim()->GetCycles();
    if (!executingTask) {
        res = std::min(res, gCycles + submissionQueue.GetMinWaitCycles());
    }
    res = std::min(res, gCycles + completionQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + outcastReferenceQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + incastReferenceQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + releaseQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + cacheRespQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + counterSetQueue.GetMinWaitCycles());
    return res;
}

bool CoreMachine::IsTerminate()
{
    // wait counter to the expect value before set in calendar mode
    if (needWaitCounter) {
        return false;
    }

    return (!executingTask && submissionQueue.IsTerminate() && completionQueue.IsTerminate() &&
            outcastReferenceQueue.IsTerminate() && incastReferenceQueue.IsTerminate() && releaseQueue.IsTerminate() &&
            cacheRespQueue.IsTerminate() && counterSetQueue.IsTerminate());
}

void CoreMachine::PushCompletion(uint64_t taskId)
{
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine%lu][PushCompletion] push task %lu in completion queue",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId),
                    static_cast<unsigned long>(taskId));

    stats->completedTaskNum++;
    // not push packet in calendar mode
    if (sim->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
        return;
    }
    RecordLeafPipeExecuteTime();
    CompletedPacket packet;
    packet.taskId = taskId;
    packet.currentType = machineType;
    packet.cycleInfo.taskExecuteStartCycle = executionStartCycle;
    packet.cycleInfo.taskExecuteEndCycle = GetSim()->GetCycles();
    auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(GetSim()->machineGroup[int(MachineType::DEVICE)][0]);
    if (devicePtr != nullptr) {
        auto it = devicePtr->savedTaskMap.find(taskId);
        if (it != devicePtr->savedTaskMap.end() && it->second != nullptr) {
            it->second->taskEndCycle = GetSim()->GetCycles();
        }
    }
    completionQueue.Enqueue(packet);
    GetSim()->taskCompleteSeq[taskId] = GetSim()->taskCompleteSeqIndex++;
    GetSim()->GetCalendarGenerator()->LogTaskComplete(taskId, machineId, executionStartCycle, GetSim()->GetCycles());
}

void CoreMachine::ReceivePacket()
{
    if (submissionQueue.Empty() || executingTask) {
        return;
    }
    TaskPack packetHead;
    submissionQueue.Front(packetHead);
    if (GetSim()->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE) &&
        !CalendarCountReady(packetHead)) {
        return;
    }

    TaskPack packet;
    submissionQueue.Dequeue(packet);
    ProcessDeviceTaskPacket(packet);
}

bool CoreMachine::CalendarCountReady(const TaskPack& packetHead)
{
    // Calendar schedule wait counter
    if (needWaitCounter || !counterSetQueue.IsTerminate()) {
        return false;
    }
    for (auto wait : sim->taskWaitMap[packetHead.taskId]) {
        int counterId = wait.first;
        int expectValue = wait.second;
        if (sim->calendarCounter[counterId] < expectValue) {
            SIMULATION_LOGI(
                "[Cycle: %lu][CoreMachine][ReceivePacket] task id%lu counter %d expectValue %d current value %d",
                static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(packetHead.taskId),
                counterId, expectValue, sim->calendarCounter[counterId]);

            return false;
        }
    }
    if (GetSim()->config.calendarMode == static_cast<uint64_t>(CalendarMode::GLOBAL_COUNTER)) {
        counterSetQueue.Enqueue(calendarFirstSet);
    }
    return true;
}

void CoreMachine::ProcessDeviceTaskPacket(const TaskPack& packet)
{
    auto functionHash = packet.task.functionHash;

    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][ReceivePacket] get task %lu from parent machine",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(packet.taskId));

    bool hit = sim->functionCache.Lookup(functionHash, machineId, functionCacheTid);
    if (!hit) {
        SIMULATION_LOGI("[Cycle: %lu][CoreMachine][ReceivePacket] Function Cache Miss",
                        static_cast<unsigned long>(GetSim()->GetCycles()));
    }
    FunctionPtr function = sim->functionCache.GetFunction(functionHash);
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][ReceivePacket] CoreMachine: %lu Receive Function:%s",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId),
                    function->funcName.c_str());

    std::string logLabel = packet.task.taskPtr->GetTaskName();
    logLabel += (" (" + packet.task.taskPtr->GetColorLabel(config.logLabelMode) + ")");
    std::string logInfo = packet.task.taskPtr->GetTaskFullName();
    LoggerRecordTaskStart(logLabel, logInfo);
    if (packet.task.taskPtr != nullptr && packet.task.taskPtr->fixedLatency) {
        exectingFixLatencyTask = true;
        fixedLatencyTaskEndCycle = GetSim()->GetCycles() + packet.task.taskPtr->fixedLatencyVal;
        PrintRelativeCycleInfo(function, packet.task.taskPtr);
    } else if (function->hasRecordInfo) {
        exectingFixLatencyTask = true;
        fixedLatencyTaskEndCycle = GetSim()->GetCycles() + function->totalCycles;
        PrintRelativeCycleInfo(function, packet.task.taskPtr);
    }
    InitCore();
    GenDependence(function);
    SortTileAndTileOp(function);
    Dispatch();
    SetMachineExecuting(true);
    InitBufferSize();
    executingFunctionHash = functionHash;
    executingTaskId = packet.taskId;
    executingFunctionName = function->funcName;
    executingFunctionPtr = function;
    executionStartCycle = GetSim()->GetCycles();
    if (packet.task.taskPtr != nullptr) {
        packet.task.taskPtr->taskStartCycle = executionStartCycle;
    }
    GetSim()->taskToCounter[packet.taskId].push_back(GetSim()->globalCounter++);
}

void CoreMachine::InitCore()
{
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][InitCore] ******Init Cost Model ******",
                    static_cast<unsigned long>(GetSim()->GetCycles()));

    scheduler.sim = GetSim();
    totalOperations = 0;
    commitOperations = 0;
    tiles.clear();
    tileOps.clear();
    for (auto& queue : readyQueues) {
        queue.Reset();
    }
    tileAllocSequence.clear();
    tileAllocSequence.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
    local = std::make_shared<TileState>();
    ResetLeafPipeExecuteTime();
}

void CoreMachine::GenDependence(std::shared_ptr<CostModel::Function> func)
{
    if (exectingFixLatencyTask) {
        return;
    }
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][GenDependence] ***** Generate Dependence ******",
                    static_cast<unsigned long>(GetSim()->GetCycles()));
    // The function can be called multiple times.
    // Therefore, we need to build new TilePtr and TileOpPtr.
    for (auto tensor : func->tiles) {
        tensor->exeInfo.Reset();
        TilePtr newTile = std::make_shared<CostModel::Tile>(*tensor);
        tiles.insert(std::make_pair(tensor->magic, newTile));
    }

    for (auto& in : func->incastMagic) {
        tiles[in]->exeInfo.isIncast = true;
        if (tiles[in]->producers.empty()) {
            tiles[in]->exeInfo.isWritten = true;
        } else {
            tiles[in]->exeInfo.isWritten = false;
        }
    }

    for (auto& out : func->outcastMagic) {
        tiles[out]->exeInfo.isOutcast = true;
    }

    int totalOperationNum = func->tileOps.size();
    for (int i = 0; i < totalOperationNum; i++) {
        auto tileop = func->tileOps[i];
        tileop->exeInfo.Reset();
        TileOpPtr newTileop = std::make_shared<TileOp>(*tileop);
        newTileop->taskId = executingTaskId;
        tileOps.insert(std::make_pair(tileop->magic, newTileop));

        bool srcTileHasProducesor = false;
        for (auto& src : newTileop->iOperand) {
            auto it = tiles.find(src->magic);
            if (it == tiles.end()) {
                // Tile is in iOperand but not in the incast, process as an incast
                src->exeInfo.Reset();
                TilePtr newSrc = std::make_shared<CostModel::Tile>(*src);
                src = newSrc;
                tiles.insert(std::make_pair(src->magic, newSrc));
                tiles[src->magic]->exeInfo.isWritten = true;
            }
            auto srcTile = tiles[src->magic];
            src = srcTile;
            newTileop->exeInfo.domCount += srcTile->exeInfo.domCount;
            if (!srcTile->producers.empty()) {
                srcTileHasProducesor = true;
            }
        }
        newTileop->exeInfo.noSrcWakeup = ((newTileop->iOperand.size() == 0) || !srcTileHasProducesor);

        bool allDstTileMemKnown = true;
        for (auto& dst : newTileop->oOperand) {
            auto it = tiles.find(dst->magic);
            if (it == tiles.end()) {
                dst->exeInfo.Reset();
                TilePtr newDst = std::make_shared<CostModel::Tile>(*dst);
                dst = newDst;
                tiles.insert(std::make_pair(dst->magic, newDst));
            }
            auto dstTile = tiles[dst->magic];
            dst = dstTile; // Replace oOperand with TilePtr in CoreMachine local tiles.
            dstTile->exeInfo.domCount += newTileop->exeInfo.domCount;
            if (dstTile->bufType == BUF_UNKNOWN || dstTile->bufType == BUF_DDR) {
                allDstTileMemKnown = false;
            }
        }
        newTileop->exeInfo.noDstWakeup = ((newTileop->oOperand.size() == 0) || !allDstTileMemKnown);
    }
    for (auto funcTile : func->tiles) {
        auto& tile = tiles[funcTile->magic];
        if (tile->producer != nullptr) {
            tile->producer = tileOps[tile->producer->magic];
        }
        for (auto produce : tile->producers) {
            produce = tileOps[produce->magic];
        }
        for (auto cons : tile->consumers) {
            cons = tileOps[cons->magic];
        }
    }
    totalOperations = tileOps.size();

    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][GenDependence] Tile Number:",
                    static_cast<unsigned long>(GetSim()->GetCycles()));
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][GenDependence] Operation Number:%zu",
                    static_cast<unsigned long>(GetSim()->GetCycles()), tileOps.size());
}

void CoreMachine::SortTileAndTileOp(FunctionPtr func)
{
    if (exectingFixLatencyTask) {
        return;
    }
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][SortTileAndTileOp] ****** Sort Tile Alloc ******",
                    static_cast<unsigned long>(GetSim()->GetCycles()));

    if (func->hasSchedule) {
        tileAllocSequence = func->tileAllocSequence;
    } else {
        scheduler.SortTile(tiles, tileOps, tileAllocSequence);
        func->hasSchedule = true;
        func->tileAllocSequence = tileAllocSequence;
    }
}

// process tile that buffer type is UNKNOW and DDR.
void CoreMachine::MarkTileAlloc(std::vector<int>& sequence)
{
    for (auto& tileIndex : sequence) {
        auto tile = tiles[tileIndex];
        tile->exeInfo.isAllocated = true;
        if (tile->producers.empty()) {
            tile->exeInfo.isWritten = true;
        } else {
            tile->exeInfo.isWritten = false;
        }
    }
}

void CoreMachine::Dispatch()
{
    if (exectingFixLatencyTask) {
        return;
    }
    // Put all tile alloc into ready queue
    for (size_t i = 0; i < tileAllocSequence.size(); i++) {
        auto& sequence = tileAllocSequence[i];
        if (IsTileAlloc(static_cast<CorePipeType>(i))) {
            if (static_cast<CorePipeType>(i) == CorePipeType::PIPE_TILE_ALLOC) {
                MarkTileAlloc(sequence);
            } else {
                for (auto& tileIndex : sequence) {
                    auto tile = tiles[tileIndex];
                    readyQueues[static_cast<int>(tile->pipeType)].Insert(tileIndex);
                }
            }
        } else {
            for (auto& tileopIndex : sequence) {
                auto tileop = tileOps[tileopIndex];
                tileop->exeInfo.issued = true;
                readyQueues[static_cast<int>(tileop->pipeType)].Insert(tileopIndex);
            }
        }
    }
}

void CoreMachine::SelectPipeToIssue(int qId, int& pipeSelect, int& pipeIndexSelect, int& freeNum)
{
    for (size_t k = 0; k < pipeMachineIndex[qId].size(); k++) {
        if (numTileopSentToPipe[qId][k] < config.tileopSentToPipeThreshold) {
            if (pipeSelect < 0) {
                pipeSelect = pipeMachineIndex[qId][k];
                pipeIndexSelect = k;
            } else {
                freeNum++;
            }
        }
    }
}

void CoreMachine::IssueTileOp()
{
    if (exectingFixLatencyTask) {
        return;
    }
    for (size_t qid = 0; qid < readyQueues.size(); qid++) {
        // pick one tile operation if:
        // 1. ready tile operation in queue
        if (readyQueues[qid].Empty()) {
            continue;
        }
        // Add BMU back pressure issue tile alloc.
        if (config.bufferBackPressure && IsTileAlloc(static_cast<CorePipeType>(qid))) {
            int tileMagic = readyQueues[qid].Front();
            auto tile = tiles[tileMagic];
            auto buffer = tile->pipeType;
            if ((bufferSize[buffer] + tile->SizeinBytes()) > GetSim()->GetBufferThreshold(buffer)) {
                continue;
            }
        }
        allEmpty = false;
        noExecution = false;

        // 2. pipeline ready to take new tile operation
        // one pipe could have multiple pipe machine, default one
        int pipeSelect = -1;
        int pipeIndexSelect = -1;
        int freePipeNum = 0;
        SelectPipeToIssue(qid, pipeSelect, pipeIndexSelect, freePipeNum);

        if (pipeSelect == -1) {
            continue;
        }
        noIssue = false;
        auto selectedMachine = std::dynamic_pointer_cast<PipeMachine>(subMachines[pipeSelect]);
        int magic = readyQueues[qid].Front();
        std::string logInfo;
        readyQueues[qid].Pop();
        TaskPack packet;
        packet.taskId = executingTaskId;
        packet.tileopTask.magic = magic;
        if (IsTileAlloc(static_cast<CorePipeType>(qid))) {
            auto tile = tiles[magic];
            packet.tileopTask.tile = tile;
            tile->exeInfo.cycleInfo.issueCycle = GetSim()->GetCycles();
            tile->exeInfo.exePipeId = pipeSelect;
            // add buffer size
            auto buffer = tile->pipeType;
            bufferSize[buffer] += tile->SizeinBytes();
            aliveBuffer[buffer].insert(magic);
            if (bufferSize[buffer] > GetSim()->GetBufferThreshold(buffer)) {
                SIMULATION_LOGI("[Cycle: %lu][CoreMachine][IssueTileOp] MachineId:%lu Buffer %s exceeds limit!!!",
                                static_cast<unsigned long>(GetSim()->GetCycles()),
                                static_cast<unsigned long>(machineId), CorePipeName(buffer).c_str());
            }
            logInfo = tile->Dump();
        } else {
            auto tileop = tileOps[magic];
            packet.tileopTask.tileOp = tileop;
            tileop->exeInfo.cycleInfo.issueCycle = GetSim()->GetCycles();
            tileop->exeInfo.exePipeId = pipeSelect;
            if (GetSim()->enableExpectValue) {
                TileCalculator::Self().Calculate(tileop, tileop->funcPtr->invoke[executingTaskId], local, tileState);
            }
            logInfo = tileop->Dump();
        }
        numTileopSentToPipe[qid][pipeIndexSelect]++;
        selectedMachine->SubmitTask(packet);
        noIssue = false;
        SIMULATION_LOGI("[Cycle: %lu][CoreMachine][IssueTileOp] MachineId:%lu ISSUE %s",
                        static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId),
                        logInfo.c_str());

        if (freePipeNum > 0 && !readyQueues[qid].Empty()) {
            coreNextNeedStep = true;
        }
    }
}

void CoreMachine::RetirePipeCompletion(std::shared_ptr<PipeMachine> pipeMachine, int magic)
{
    std::string logInfo;
    if (IsTileAlloc(pipeMachine->pipeType)) {
        auto tile = tiles[magic];
        tile->exeInfo.isAllocated = true;
        WakeupTileProducer(magic);
        WakeupTileConsumers(tile->magic);
        tile->exeInfo.cycleInfo.retireCycle = GetSim()->GetCycles();
        stats->retiredTileAllocNum++;
        leafPipeExecuteTime[pipeMachine->pipeType] += tile->exeInfo.latency;
        logInfo = tile->Dump();
    } else {
        commitOperations++;
        retiredOperations.emplace_back(magic);
        auto tileop = tileOps[magic];
        tileop->exeInfo.retired = true;
        for (const auto& srcTile : tileop->iOperand) {
            srcTile->exeInfo.readReference++;
            CheckReleaseSrcTile(srcTile->magic);
        }
        for (const auto& dstTile : tileop->oOperand) {
            dstTile->exeInfo.writeReference++;
            if (dstTile->exeInfo.writeReference == dstTile->producers.size()) {
                dstTile->exeInfo.isWritten = true;
                WakeupTileConsumers(dstTile->magic);
            }
        }
        tileop->exeInfo.cycleInfo.retireCycle = GetSim()->GetCycles();
        stats->retiredTileOpNum++;
        leafPipeExecuteTime[pipeMachine->pipeType] += tileop->exeInfo.latency;
        logInfo = tileop->Dump();
    }
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][RetireTileOp] MachineId:%lu retire: %s",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId),
                    logInfo.c_str());
}

void CoreMachine::RetireTileOp()
{
    if (exectingFixLatencyTask) {
        return;
    }
    // Retire Stage
    for (size_t qid = 0; qid < pipeMachineIndex.size(); qid++) {
        for (size_t k = 0; k < pipeMachineIndex[qid].size(); k++) {
            auto& pipeIndex = pipeMachineIndex[qid][k];
            auto pipeMachine = std::dynamic_pointer_cast<PipeMachine>(subMachines[pipeIndex]);
            CompletedPacket packet;
            bool getPacket = pipeMachine->completionQueue.Front(packet);
            bool taskIdEqual = (packet.taskId == static_cast<uint64_t>(executingTaskId));
            if (getPacket && taskIdEqual) {
                pipeMachine->completionQueue.PopFront();
                numTileopSentToPipe[qid][k]--;
                noExecution = false;
                allEmpty = false;
                int magic = packet.pipeMsg.magic;
                RetirePipeCompletion(pipeMachine, magic);
            } else if (pipeMachine->executingTask || !pipeMachine->completionQueue.IsTerminate() ||
                       (getPacket && !taskIdEqual) || !pipeMachine->submissionQueue.IsTerminate()) {
                noExecution = false;
                allEmpty = false;
            }
        }
    }
}

void CoreMachine::RunAtBegin()
{
    nextCycles = INT_MAX;
    allEmpty = true;
    noRetired = true;
    noIssue = true;
    noExecution = true;
    coreNextNeedStep = false;
}

void CoreMachine::RunAtEnd()
{
    if (executingTask && !exectingFixLatencyTask) {
        if (noIssue) {
            if (allEmpty) {
                CheckDeadlock();
            }
            if (noExecution) {
                CheckDeadlock();
            }
        }
        if (commitOperations >= totalOperations) {
            CheckDeadlock();
        }
    } else if (executingTask && exectingFixLatencyTask) {
        if (fixedLatencyTaskEndCycle <= GetSim()->GetCycles()) {
            SIMULATION_LOGI("[Cycle: %lu][CoreMachine:%lu][CheckDeadlock] Completed!!!!!",
                            static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId));
            LoggerRecordTaskEnd();
            exectingFixLatencyTask = false;
            SetMachineExecuting(false);
            PushCompletion(executingTaskId);
            GetSim()->taskToCounter[executingTaskId].push_back(GetSim()->globalCounter++);
            needSet = true;
        }
    }
}

void CoreMachine::AnalysisDeadlock(std::set<int>& unissuedTileMagics)
{
    // Analysis deadlock source.
    // Tileop
    std::set<int> deadLockSrcOpMagic;
    for (auto& opmagic : unissuedTileMagics) {
        auto& op = tileOps[opmagic];
        SIMULATION_LOGW("[AnalysisDeadlock] Uissued Tileop: %s", op->Dump().c_str());
        bool srcReady = true;
        for (auto& src : op->iOperand) {
            for (auto& ptr : src->producers) {
                auto& pro = tileOps[ptr->magic];
                if (!pro->exeInfo.issued || !pro->exeInfo.retired) {
                    srcReady = false;
                    break;
                }
            }
        }
        if (srcReady) {
            deadLockSrcOpMagic.insert(op->magic);
        }
    }
    // Tile
    for (const auto& tile : tiles) {
        if (!tile.second->exeInfo.isWritten || !tile.second->exeInfo.isAllocated) {
            SIMULATION_LOGW("[AnalysisDeadlock] unissued tile: %s", tile.second->Dump().c_str());
        }
    }
    for (auto& alive : aliveBuffer) {
        SIMULATION_LOGW("[AnalysisDeadlock] Alive Buffer [%s]", CorePipeName(alive.first).c_str());
        for (auto& magic : alive.second) {
            SIMULATION_LOGW("[AnalysisDeadlock] Alive Tile: %s", tiles[magic]->Dump().c_str());
        }
    }
    for (auto& magic : deadLockSrcOpMagic) {
        SIMULATION_LOGW("[AnalysisDeadlock] DeadLock Source Tileop: %s", tileOps[magic]->Dump().c_str());
    }
    for (auto& readyQ : readyQueues) {
        if (!readyQ.Empty()) {
            int front = readyQ.Front();
            SIMULATION_LOGW("[AnalysisDeadlock] ReadyQ[%s] size: %zu, front: %s", CorePipeName(readyQ.iqType).c_str(),
                            readyQ.readyQueue.size(), tiles[front]->Dump().c_str());
        }
    }
    SIMULATION_LOGW("[Cycle: %lu][CoreMachine][AnalysisDeadlock] ERROR: DEADLOCK!!! [MachineID: %lu]",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId));
    SIMULATION_LOGW("[Cycle: %lu][CoreMachine][AnalysisDeadlock] DeadLock Task: %lu",
                    static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(executingTaskId));

    GetSim()->DebugDrawFunc(sim->functionCache.GetFunction(executingFunctionHash), tiles, tileOps);
}

void CoreMachine::CheckDeadlock()
{
    uint64_t unissuedTileOps = 0;
    uint64_t retiredTileOps = 0;
    std::set<int> unissuedTileMagics;
    for (auto& tileop : tileOps) {
        if (tileop.second->exeInfo.retired) {
            retiredTileOps++;
        }
        if (!tileop.second->exeInfo.issued || !tileop.second->exeInfo.retired) {
            unissuedTileOps++;
            unissuedTileMagics.insert(tileop.first);
        }
    }
    if (unissuedTileOps > 0) {
        SIMULATION_LOGW("[Cycle: %lu][CoreMachine][CheckDeadlock] Total Tile Operations %zu",
                        static_cast<unsigned long>(GetSim()->GetCycles()), tileOps.size());

        SIMULATION_LOGW("[Cycle: %lu][CoreMachine][CheckDeadlock] Retired Tile Operations %lu",
                        static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(retiredTileOps));

        SIMULATION_LOGW("[Cycle: %lu][CoreMachine][CheckDeadlock] Unissued Tile Operations %lu",
                        static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(unissuedTileOps));

        AnalysisDeadlock(unissuedTileMagics);
        GetSim()->ReportDeadlock(machineId);
        return;
    }
    SIMULATION_LOGI("[Cycle: %lu][CoreMachine][CheckDeadlock] Completed!!!!!!!!!!!!!!!",
                    static_cast<unsigned long>(GetSim()->GetCycles()));
    if (executingTask && GetSim() && GetSim()->GetLogger()) {
        LoggerRecordTaskEnd();
    }
    GetSim()->taskToCounter[executingTaskId].push_back(GetSim()->globalCounter++);
    SetMachineExecuting(false);
    PushCompletion(executingTaskId);
    needSet = true;
}

void CoreMachine::CheckOperationReady(int magic)
{
    auto tileop = tileOps[magic];
    bool ready = true;
    // Check src tile has been written
    for (const auto& srcTile : tileop->iOperand) {
        if (!srcTile->exeInfo.isWritten || !srcTile->exeInfo.isAllocated) {
            ready = false;
            break;
        }
    }
    // Check dst tile has been allocated
    for (const auto& dstTile : tileop->oOperand) {
        if (!dstTile->exeInfo.isAllocated) {
            ready = false;
            break;
        }
    }

    if (ready && !tileop->exeInfo.issued) {
        // The two srcs of an operation may be the same.
        // In this scenario, the issue occurs twice.
        tileop->exeInfo.issued = true;
        readyQueues[static_cast<int>(tileop->pipeType)].Insert(magic);
        SIMULATION_LOGI("[Cycle: %lu][CoreMachine][CheckOperationReady] MachineId: %lu",
                        static_cast<unsigned long>(GetSim()->GetCycles()), static_cast<unsigned long>(machineId));
    }
}

void CoreMachine::WakeupTileProducer(int tileMagic)
{
    auto tile = tiles[tileMagic];
    if (tile->producers.empty()) {
        tile->exeInfo.isWritten = true;
        return;
    }
    for (const auto& producer : tile->producers) {
        CheckOperationReady(producer->magic);
    }
}

void CoreMachine::WakeupTileConsumers(int tileMagic)
{
    auto tile = tiles[tileMagic];
    if (tile->consumers.empty() && IsTileBufferAlloc(tile->pipeType)) {
        CheckReleaseSrcTile(tileMagic);
        return;
    }
    for (auto& consumer : tile->consumers) {
        CheckOperationReady(consumer->magic);
    }
}

void CoreMachine::CheckReleaseSrcTile(int magic)
{
    auto& srcTile = tiles[magic];
    if (srcTile->exeInfo.readReference == srcTile->consumers.size()) {
        // decrease buffer size, Release Tile Buffer
        bufferSize[srcTile->pipeType] -= srcTile->SizeinBytes();
        aliveBuffer[srcTile->pipeType].erase(magic); // remove tile from aliveBuffer
    }
}

uint64_t CoreMachine::GetPipeNum(CostModel::CorePipeType type) const
{
    switch (type) {
        case CostModel::CorePipeType::PIPE_TILE_ALLOC:
            return config.pipeTileAllocNum;
        case CostModel::CorePipeType::PIPE_VECTOR_BMU:
            return config.pipeVectorBmuNum;
        case CostModel::CorePipeType::PIPE_CUBE_BMU_L1:
            return config.pipeCubeBmuL1NUM;
        case CostModel::CorePipeType::PIPE_CUBE_BMU_L0A:
            return config.pipeCubeBmuL0ANUM;
        case CostModel::CorePipeType::PIPE_CUBE_BMU_L0B:
            return config.pipeCubeBmuL0BNUM;
        case CostModel::CorePipeType::PIPE_CUBE_BMU_L0C:
            return config.pipeCubeBmuL0CNUM;
        case CostModel::CorePipeType::PIPE_MTE_IN:
            return config.pipeMteInNum;
        case CostModel::CorePipeType::PIPE_MTE1:
            return config.pipeMte1Num;
        case CostModel::CorePipeType::PIPE_VECTOR_ALU:
            return config.pipeVectorAluNum;
        case CostModel::CorePipeType::PIPE_CUBE:
            return config.pipeCubeNum;
        case CostModel::CorePipeType::PIPE_MTE_OUT:
            return config.pipeMteOutNum;
        default:
            return 1;
    }
}

void CoreMachine::ResetLeafPipeExecuteTime()
{
    leafPipeExecuteTime.clear();
    leafPipeExecuteTime[CorePipeType::PIPE_VECTOR_BMU] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L1] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L0A] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L0B] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L0C] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_MTE_IN] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_MTE1] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_VECTOR_ALU] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_CUBE] = 0;
    leafPipeExecuteTime[CorePipeType::PIPE_MTE_OUT] = 0;
}

void CoreMachine::RecordLeafPipeExecuteTime()
{
    if (executingFunctionPtr->hasRecordInfo) {
        return;
    }
    executingFunctionPtr->hasRecordInfo = true;
    executingFunctionPtr->pipeExecuteTime = leafPipeExecuteTime;
    executingFunctionPtr->startCycles = executionStartCycle;
    executingFunctionPtr->totalCycles = GetSim()->GetCycles() - executionStartCycle;
    for (auto& tileOp : tileOps) {
        executingFunctionPtr->tileOpMap[tileOp.first]->exeInfo = tileOp.second->exeInfo;
    }
    for (auto& tile : tiles) {
        executingFunctionPtr->tileMap[tile.first]->exeInfo = tile.second->exeInfo;
    }
}

void CoreMachine::LoggerRecordTileOpFlow(TileOpPtr tileOp)
{
    if (!config.enableTileOpFlow || tileOp == nullptr) {
        return;
    }
    auto logger = GetSim()->GetLogger();
    auto curMagic = tileOp->magic;
    for (auto& in : tileOp->iOperand) {
        for (auto& prodOp : in->producers) {
            logger->AddTileOpFlow(machineId, prodOp->magic, curMagic);
        }
    }
    for (auto& out : tileOp->oOperand) {
        if (out->exeInfo.exePipeId < 0) {
            continue;
        }
        logger->AddTileOpFlow(machineId, out->magic, curMagic);
    }
}

void CoreMachine::PrintRelativeCycleInfo(FunctionPtr func, std::shared_ptr<Task> task)
{
    // Calculate Relative cycle
    func->CalculateRelativeCycle(GetSim()->GetCycles(), task->proportion);

    // Log TileOp trace
    for (auto& opMagic : func->opMagicSequence) {
        auto& tileOp = func->tileOpMap[opMagic];
        std::string info = tileOp->Dump(true);
        info += (" Task[" + std::to_string(task->taskId) + "]-r");
        LoggerRecordTileOp(info, tileOp->exeInfo.exePipeId, tileOp->exeInfo.cycleInfo.relativeStartCycle,
                           tileOp->exeInfo.cycleInfo.relativeEndCycle);
        LoggerRecordTileOpFlow(tileOp);
    }

    // Add stat
    for (auto& pipe : func->pipeExecuteTime) {
        uint64_t pipeStat = pipe.second;
        if (IsMTEPipe(pipe.first)) {
            pipeStat = uint64_t(double(pipeStat) * task->proportion);
        }
        stats->totalPipeUseCycles[int(pipe.first)] += pipeStat;
    }
}

void ReadyQueue::Insert(int idx) { readyQueue.push_back(idx); }

bool ReadyQueue::Empty() const { return readyQueue.empty(); }

int ReadyQueue::Front()
{
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, !readyQueue.empty()) << "[SIMULATION]: "
                                                                               << "readyQueue is empty";
    int idx = readyQueue.front();
    return idx;
}

int ReadyQueue::Pop()
{
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, !readyQueue.empty()) << "[SIMULATION]: "
                                                                               << "readyQueue is empty";
    int idx = readyQueue.front();
    readyQueue.pop_front();
    return idx;
}

void ReadyQueue::Reset() { readyQueue.clear(); }
} // namespace CostModel
