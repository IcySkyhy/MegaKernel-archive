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
 * \file CoreMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <deque>
#include <vector>
#include <unordered_map>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/machine/PipeMachine.h"
#include "cost_model/simulation/machine/Scheduler.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/cache/FunctionCache.h"
#include "cost_model/simulation/config/CoreConfig.h"
#include "cost_model/simulation/statistics/CoreStats.h"
#include "cost_model/simulation/value/TileState.h"

namespace CostModel {

class ReadyQueue {
public:
    int iqId = -1;
    CorePipeType iqType = CorePipeType::TOTAL_CORE_PIPE_TYPE;
    std::deque<int> readyQueue;

    ReadyQueue(CorePipeType type, int id) : iqId(id), iqType(type) { readyQueue.clear(); };

    void Insert(int idx);
    bool Empty() const;
    int Front();
    int Pop();
    void Reset();
};

class CoreMachine : public Machine {
public:
    // CoreMachine Components
    CoreConfig config;
    std::shared_ptr<CoreStats> stats = nullptr;
    Scheduler scheduler;
    std::vector<std::vector<int>> pipeMachineIndex;
    std::unordered_map<int, CorePipeType> pipeTypeMap;
    std::vector<std::vector<uint64_t>> numTileopSentToPipe;
    std::vector<ReadyQueue> readyQueues;

    // For Calendar schedule
    bool needWaitCounter = false;
    SimQueue<int> counterSetQueue;
    bool needSet = false;
    void SetCalendar();

    // Status Check Deadlock
    bool allEmpty = true;
    bool noRetired = true;
    bool noIssue = true;
    bool noExecution = true;
    int calendarFirstSet = 1;
    int calendarSecondSet = 2;
    // Execute Info
    std::unordered_map<int, TilePtr> tiles;
    std::unordered_map<int, TileOpPtr> tileOps;
    std::vector<std::vector<int>> tileAllocSequence;
    bool coreNextNeedStep = false;

    // local stat
    std::unordered_map<CostModel::CorePipeType, uint64_t> leafPipeExecuteTime;

    uint64_t totalOperations = 0;
    uint64_t commitOperations = 0;
    uint64_t executionStartCycle = 0;
    uint64_t executingTaskId = 0;
    bool exectingFixLatencyTask = false;
    uint64_t fixedLatencyTaskEndCycle = 0;
    FunctionPtr executingFunctionPtr = nullptr;
    uint64_t executingFunctionHash = -1;
    std::string executingFunctionName = "";
    std::vector<int> retiredOperations;
    std::unordered_map<CorePipeType, uint64_t> bufferSize;
    std::unordered_map<CorePipeType, std::set<int>> aliveBuffer;
    std::shared_ptr<TileState> tileState;
    std::shared_ptr<TileState> local;

    CoreMachine();
    explicit CoreMachine(MachineType type);
    void RunAtBegin();
    void RunAtEnd();
    bool CalendarCountReady(const TaskPack& packetHead);
    void ProcessDeviceTaskPacket(const TaskPack& packet);
    void ReceivePacket();
    void InitCore();
    void GenDependence(FunctionPtr func);
    void SortTileAndTileOp(FunctionPtr func);
    void MarkTileAlloc(std::vector<int>& sequence); // process tile that buffer type is UNKNOW and DDR.
    void Dispatch();
    void SelectPipeToIssue(int qId, int& pipeSelect, int& pipeIndexSelect, int& freeNum);
    void IssueTileOp();
    void RetirePipeCompletion(std::shared_ptr<PipeMachine> pipeMachine, int magic);
    void RetireTileOp();
    void AnalysisDeadlock(std::set<int>& unissuedTileMagics);
    void CheckDeadlock();
    void PushCompletion(uint64_t taskId);
    void InitBufferSize();

    void CheckOperationReady(int magic);
    void WakeupTileProducer(int tileMagic);
    void WakeupTileConsumers(int tileMagic);
    void CheckReleaseSrcTile(int magic);

    uint64_t GetPipeNum(CorePipeType type) const;
    void ResetLeafPipeExecuteTime();
    void RecordLeafPipeExecuteTime();

    void PrintRelativeCycleInfo(FunctionPtr func, std::shared_ptr<Task> task);
    void LoggerRecordTileOpFlow(TileOpPtr tileOp);

    void Step() override;
    void Xfer() override;
    void Build() override;
    void Reset() override;
    std::shared_ptr<SimSys> GetSim() override;
    void Report() override;
    void InitQueueDelay() override;
    void StepQueue() override;
    bool IsTerminate() override;
    uint64_t GetQueueNextCycles();
    void SetTileState(std::shared_ptr<TileState>& state);
};
} // namespace CostModel
