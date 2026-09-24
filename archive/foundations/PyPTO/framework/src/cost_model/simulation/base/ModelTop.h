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
 * \file ModelTop.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <climits>
#include <memory>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/cache/CacheMachine.h"
#include "cost_model/simulation/cache/FunctionCache.h"
#include "cost_model/simulation/config/ModelConfig.h"
#include "cost_model/simulation/statistics/ModelStats.h"
#include "cost_model/simulation/base/Reporter.h"
#include "cost_model/simulation/statistics/TraceLogger.h"
#include "cost_model/simulation/machine/DeviceMachine.h"
#include "cost_model/simulation/machine/AICPUMachine.h"
#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/machine/PipeMachine.h"
#include "cost_model/simulation/arch/GenCalendar/GenCalendar.h"
#include "cost_model/simulation/pv/PvModel.h"

namespace CostModel {

// Op types for performance metrics (longest-path per-op-type computation amount)
constexpr int PERF_OP_TYPE_COUNT = 11;
enum PerfOpType {
    PERF_MATMUL = 0,
    PERF_ELEM_BINARY = 1,
    PERF_ELEM_UNARY = 2,
    PERF_ELEM_RELU = 3,
    PERF_ELEM_REDUCE = 4,
    PERF_ELEM_SORT = 5,
    PERF_ELEM_SELECT = 6,
    PERF_UB_COPY_IN = 7,
    PERF_UB_COPY_OUT = 8,
    PERF_L1_COPY_IN = 9,
    PERF_L0C_COPY_OUT = 10,
};
inline const char* PERF_OP_TYPE_NAMES[] = {"matmul",      "elem-binary", "elem-unary",  "elem-relu",
                                           "elem-reduce", "elem-sort",   "elem-select", "ub-copyin",
                                           "ub-copyout",  "l1-copyin",   "l0c-copyout"};

using MachinePtr = std::shared_ptr<Machine>;
using DevicePtr = std::shared_ptr<DeviceMachine>;
using AICPUPtr = std::shared_ptr<AICPUMachine>;

class SimSys : public std::enable_shared_from_this<SimSys> {
public:
    uint64_t globalCycles = 0;
    uint64_t nextSimulationCycles = INT_MAX;
    uint64_t lastSimulationCycles = INT_MAX;
    std::size_t topMachineViewPid = 1000;
    uint64_t queuePidPtr = 2000;
    std::map<std::string, uint64_t> queuePidMap;
    bool terminate = false;
    bool deadlock = false;

    std::vector<std::shared_ptr<Machine>> machines;
    std::vector<std::vector<std::shared_ptr<Machine>>> machineGroup;
    std::unordered_map<Pid, std::shared_ptr<Machine>> pidToMachineMp;
    std::shared_ptr<TraceLogger> totalTraceLogger = nullptr;
    std::shared_ptr<GenCalendar> calendarGenerator = nullptr;

    ModelConfig config;
    std::shared_ptr<ModelStats> stats = nullptr;
    CostModel::Reporter reporter;

    std::shared_ptr<CacheMachine> l2Cache = nullptr;
    FunctionCache functionCache;
    std::string startFuncName = "";
    uint64_t startFuncHash = 0;
    bool testSingleFunc = false;
    uint64_t singleFuncHash = 0;

    // For CoreMachine load balance in SMT mode
    AICoreWorkLoadStatus wlStatus;
    size_t subMachineCount;

    std::unordered_map<MachineType, int> machineTypeSeq;

    // For Add CoreMachine backend PipeMachine
    MachineType lastCoreMachineType = MachineType::UNKNOWN;
    std::vector<std::shared_ptr<Machine>> pipeMachines;
    std::vector<std::vector<int>> pipeMachineIndex;
    std::vector<std::vector<uint64_t>> numTileopSentToPipe;
    std::unordered_map<CorePipeType, UnifiedPipeMachinePtr> pipeImplMap;
    std::unordered_map<CorePipeType, uint64_t> bufferSizeThreshold;

    // Dynamic workflow
    bool dynamicWorkflow = false;
    std::map<uint64_t, uint64_t> leafFunctionTime;

    std::map<int, std::vector<std::pair<int, int>>> taskWaitMap;
    std::map<int, int> taskSetMap;
    std::map<int, int> taskFirstSetMap;
    std::map<int, std::vector<std::pair<int, int>>> taskWaitBeforeSetMap;
    std::vector<int> calendarCounter;
    std::map<int, std::vector<std::pair<int, uint64_t>>> corePacketMap;
    TaskMap calendarTaskMap; // For calendar
    std::map<int, int> taskSetExpectMap;
    std::map<int, std::vector<int>> taskToCounter;
    std::map<int, uint64_t> taskToHash;
    uint64_t globalCounter = 0;

    std::map<int, int> taskCompleteSeq;
    int taskCompleteSeqIndex = 0;
    // Config Parameters
    SimMode mode = SimMode::NORMAL;
    std::string jsonPath = ""; // Input json file to load
    int logLevel = 3;          // 1: DEBUG; 2: INFO; 3: WARN; 4: ERROR, 5: FATAL
    int accLevel = 1;
    bool enableExpectValue = false;
    uint64_t executeCycleThreshold = -1;
    std::string outdir = "";
    std::string graphsOutdir = "";
    bool drawGraph = true;
    std::vector<std::string> cfgs;
    std::string topoOutFile = "";
    std::chrono::system_clock::time_point simStartTime;

    SimSys();
    std::shared_ptr<SimSys> GetShared();
    uint64_t GetCycles() const;
    void UpdateNextCycles(uint64_t nextCycle);
    void ResetCycles(uint64_t cycles);
    void AddCycles(uint64_t overTime = 1);
    std::shared_ptr<TraceLogger> GetLogger();
    std::shared_ptr<GenCalendar> GetCalendarGenerator();
    CostModel::Reporter* GetReporter();
    bool IsMachine(CostModel::Pid pid, CostModel::Tid tid);
    bool IsQueue(CostModel::Tid tid);
    bool IsWorkPipe(CostModel::Pid pid, CostModel::Tid tid, std::string& name);

    // Build Device-AICPU-AICORE-PIPE System
    void AddExtraConfigs();
    void LogRegisterMachine(MachinePtr machine, size_t id, int coreIdx);
    void InitMachineStartSeq();
    void BuildCaches();

    void BuildHUBCore();
    MachinePtr GetHUBCore();
    UnifiedPipeMachinePtr GetPipeImpl(CorePipeType pType);
    void BuildPipes(uint64_t index, std::shared_ptr<CoreMachine> coreMachine);
    void BuildCore(DevicePtr device, AICPUPtr cpu, uint64_t idInCPU, MachineType type);
    void BuildAICPU(DevicePtr device, uint64_t idInDevice);
    void BuildDevice();
    void BuildSystemStat();
    void BuildSystem();

    void InitCoreTask(); // Init CoreMachine Task
    void Reset();
    void AddMachine(std::shared_ptr<Machine> m);

    void Step();

    bool CheckAllEmpty();
    bool IsTerminate() const;
    void ReportDeadlock(size_t machineId);
    bool IsDeadlock() const;
    void InitBufferThreshold(PipeConfig& pipeConfig);
    uint64_t GetBufferThreshold(CorePipeType pType);

    void CalendarDispatchTasksToCore(int key, std::shared_ptr<CoreMachine> coreMachine);
    void InitCalendarMode();
    void LoggerRecordCoreStart(std::string name, size_t machineId, std::string hint = "");
    void LoggerRecordCoreCompleted(size_t machineId);
    void OutputTrace(std::string prefix = "");
    void OutputPerfettoTrace(std::string prefix = "");
    void DumpFunctionExecuteTime(std::string prefix = "");
    void OutputConfig(std::string prefix = "");
    void OutputLogForSwimLane(std::string prefix = "");
    void OutputLogForPipeSwimLane(std::string prefix = "");
    void OutputCalendarScheduleCpp(std::string prefix = "");
    void OutputPerfMetrics(std::string prefix = "");
    void ProcessTaskMap(TaskMap& taskMap, std::string prefix = "");
    void DrawTasks(const TaskMap& taskMap, std::string prefix = "");
    void DebugDrawFunc(FunctionPtr func, std::unordered_map<int, TilePtr>& tiles,
                       std::unordered_map<int, TileOpPtr>& tileOps);
    void DumpTasksTopo(const TaskMap& taskMap, std::string prefix = "");
    void ResetStat(bool start);
    void PrintCoreStat();
    void PrintStat();
    uint64_t RegisterQueuePid(std::string key);
    void GetDeviceReadyQueueInfo(size_t& devicePid, std::set<uint64_t>& readyQueueTidSet);

    static std::string GetFileName(const std::string& file);
    static std::string GetFileName(const std::string& dir, const std::string& inputFile, const std::string& preFix,
                                   const std::string& suffix);

private:
    std::map<uint64_t, std::vector<uint64_t>> ComputeTaskAmounts(const TaskMap& taskMap);
    struct LongestPathResult {
        std::map<uint64_t, std::vector<uint64_t>> longestPath;
        std::map<uint64_t, std::vector<uint64_t>> chosenPred;
    };
    LongestPathResult ComputeLongestPaths(const TaskMap& taskMap,
                                          const std::map<uint64_t, std::vector<uint64_t>>& computeAmountMap);
    Json FormatPerfMetricsJson(const std::map<uint64_t, std::vector<uint64_t>>& longestPath,
                               const std::map<uint64_t, std::vector<uint64_t>>& chosenPred);
};
} // namespace CostModel
