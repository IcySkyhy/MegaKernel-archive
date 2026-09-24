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
 * \file DeviceMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <random>
#include <deque>
#include <mutex>
#include <unordered_set>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/config/ModelConfig.h"
#include "cost_model/simulation/config/DeviceConfig.h"
#include "cost_model/simulation/statistics/DeviceStats.h"
#include "cost_model/simulation/statistics/TraceLogger.h"
#include "cost_model/simulation/value/TileState.h"

namespace CostModel {
class DeviceMachine : public Machine {
public:
    // base info
    bool cubeVecMix = false;
    std::size_t readyQueuePid;
    std::size_t readyQueueTotalTid;
    std::map<MachineType, std::size_t> readyQueueTid;

    // status info
    bool taskBuilded = false;
    bool replayPreExecute = false;
    bool hasPreExecute = false;
    uint64_t replayPreStartTime = 0;
    uint64_t lastHeartModulo = 0;
    uint64_t currentHeartModulo = 0;
    std::map<MachineType, std::deque<uint64_t>> readyQueues;
    std::set<uint64_t> readySet; // For replay mode

    TaskMap taskMap;
    TaskMap savedTaskMap;
    std::deque<TaskMap> taskMapQueue;
    std::map<uint64_t, uint64_t> executingTaskMap;

    DeviceConfig config;
    std::shared_ptr<DeviceStats> stats;
    std::shared_ptr<TileState> tileStateGolden;
    std::shared_ptr<TileState> tileState;

    // For Replay Mode
    uint64_t currentSeq = 0;
    // key: machineId, value: tasks queue
    std::unordered_map<uint64_t, std::deque<ReplayTaskEntry>> replayTasksInfoMap;

    void RunAtBegin();
    void RunAtEnd();
    void SubmitDeviceTask();
    void BuildDeviceTask();
    TaskMap BuildATaskMap();
    void InitFunctions();
    void BuildLeafFunctionTasks();
    void BuildSubtasksFromRootFuncTopo();
    void BuildSubTasksFromTopoJson();
    void BuildSingleFuncTask();
    void PushReadyQueue(MachineType mType, uint64_t taskId);
    uint64_t PopReadyQueue(MachineType mType);
    bool EraseReadyQueue(MachineType mType, uint64_t taskId);

    // For replay mode
    void BuildReplayInfo();
    bool IsReady(uint64_t taskId);
    void InsertReadySet(uint64_t taskId);
    void CheckHUBTaskReplayInfo(uint64_t taskId);
    void EraseReadySet(uint64_t taskId);
    void SetReplayPreStart();
    void SetReplayPreEnd();
    void EnableScaleTaskExecuteTime();
    void ScaleTaskExecuteTime(ReplayTaskEntry& replayInfo);

    void Step() override;
    void Xfer() override;
    void Build() override;
    void Reset() override;
    std::shared_ptr<SimSys> GetSim() override;
    void Report() override;
    void InitQueueDelay() override;
    void StepQueue() override;
    bool IsTerminate() override;

private:
    void CalculateTileGolden();
    void CalculateFunctionArgTile(FunctionPtr func, std::shared_ptr<TileState> state);
    void CalculateFunctionTileGolden(FunctionPtr func, std::shared_ptr<TileState> local,
                                     std::shared_ptr<TileState> global, int esgId);
    void PrintFunctionOutputTile(FunctionPtr func, std::shared_ptr<TileState> state);
    void PrintTopo();
    void PrintFunctionTopo(FunctionPtr func);
};

} // namespace CostModel
