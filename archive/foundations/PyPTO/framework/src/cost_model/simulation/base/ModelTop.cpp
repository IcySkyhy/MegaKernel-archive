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
 * \file ModelTop.cpp
 * \brief
 */

#include <cstdlib>
#include <iostream>
#include <deque>
#include <set>
#include "cost_model/simulation/base/ModelTop.h"

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/common/Packet.h"
#include "cost_model/simulation/tools/visualizer.h"
#include "cost_model/simulation/arch/PipeFactory.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "tilefwk/error_code.h"
#include "utils/file_utils.h"
#include "interface/utils/common.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"

namespace {

using namespace CostModel;

static const std::unordered_map<std::string, PerfOpType> VEC_OPCODE_MAP = {
    // elem-binary: arithmetic
    {"ADD", PERF_ELEM_BINARY},
    {"SUB", PERF_ELEM_BINARY},
    {"MUL", PERF_ELEM_BINARY},
    {"DIV", PERF_ELEM_BINARY},
    {"ADD_BRC", PERF_ELEM_BINARY},
    {"SUB_BRC", PERF_ELEM_BINARY},
    {"MUL_BRC", PERF_ELEM_BINARY},
    {"DIV_BRC", PERF_ELEM_BINARY},
    {"MAX_BRC", PERF_ELEM_BINARY},
    {"MIN_BRC", PERF_ELEM_BINARY},
    {"S_ADD", PERF_ELEM_BINARY},
    {"S_SUB", PERF_ELEM_BINARY},
    {"S_MUL", PERF_ELEM_BINARY},
    {"S_DIV", PERF_ELEM_BINARY},
    {"S_MAX", PERF_ELEM_BINARY},
    {"S_MIN", PERF_ELEM_BINARY},
    {"ADDS", PERF_ELEM_BINARY},
    {"SUBS", PERF_ELEM_BINARY},
    {"MULS", PERF_ELEM_BINARY},
    {"DIVS", PERF_ELEM_BINARY},
    {"MAXS", PERF_ELEM_BINARY},
    {"MINS", PERF_ELEM_BINARY},
    {"S_ADDS", PERF_ELEM_BINARY},
    {"S_SUBS", PERF_ELEM_BINARY},
    {"S_MULS", PERF_ELEM_BINARY},
    {"S_DIVS", PERF_ELEM_BINARY},
    {"S_MAXS", PERF_ELEM_BINARY},
    {"S_MINS", PERF_ELEM_BINARY},
    {"MAXIMUM", PERF_ELEM_BINARY},
    {"MINIMUM", PERF_ELEM_BINARY},
    {"POW", PERF_ELEM_BINARY},
    // elem-unary: math / transform
    {"EXP", PERF_ELEM_UNARY},
    {"SQRT", PERF_ELEM_UNARY},
    {"RSQRT", PERF_ELEM_UNARY},
    {"RECIPROCAL", PERF_ELEM_UNARY},
    {"NEG", PERF_ELEM_UNARY},
    {"LN", PERF_ELEM_UNARY},
    {"EXPM1", PERF_ELEM_UNARY},
    {"ABS", PERF_ELEM_UNARY},
    {"FLOOR", PERF_ELEM_UNARY},
    {"CEIL", PERF_ELEM_UNARY},
    {"ROUND", PERF_ELEM_UNARY},
    {"TRUNC", PERF_ELEM_UNARY},
    {"SIGN", PERF_ELEM_UNARY},
    {"SIGNBIT", PERF_ELEM_UNARY},
    {"CAST", PERF_ELEM_UNARY},
    {"CONVERT", PERF_ELEM_UNARY},
    {"BITWISENOT", PERF_ELEM_UNARY},
    {"LOGICALNOT", PERF_ELEM_UNARY},
    {"ISFINITE", PERF_ELEM_UNARY},
    {"LOGICALAND", PERF_ELEM_UNARY},
    {"TRANSPOSE_VNCHWCONV", PERF_ELEM_UNARY},
    {"BRCB", PERF_ELEM_UNARY},
    // elem-relu: activation
    {"RELU", PERF_ELEM_RELU},
    {"HUB", PERF_ELEM_RELU},
    // elem-reduce: dimension reduction
    {"ROWMAX", PERF_ELEM_REDUCE},
    {"ROWSUM", PERF_ELEM_REDUCE},
    {"ROWEXPMAX", PERF_ELEM_REDUCE},
    {"ROWEXPSUM", PERF_ELEM_REDUCE},
    {"ROWSUMLINE", PERF_ELEM_REDUCE},
    {"ROWMAXLINE", PERF_ELEM_REDUCE},
    {"ROWMINLINE", PERF_ELEM_REDUCE},
    {"ROWMAX_SINGLE", PERF_ELEM_REDUCE},
    {"ROWMIN_SINGLE", PERF_ELEM_REDUCE},
    {"ROWSUM_SINGLE", PERF_ELEM_REDUCE},
    {"REDUCE_ACC", PERF_ELEM_REDUCE},
    {"CUM_SUM", PERF_ELEM_REDUCE},
    {"PAIRMAX", PERF_ELEM_REDUCE},
    {"PAIRMIN", PERF_ELEM_REDUCE},
    {"PAIRSUM", PERF_ELEM_REDUCE},
    // elem-sort: sort / topk
    {"TOPK", PERF_ELEM_SORT},
    {"TOPK_SORT", PERF_ELEM_SORT},
    {"TOPK_MERGE", PERF_ELEM_SORT},
    {"TOPK_EXTRACT", PERF_ELEM_SORT},
    {"TILEDMRGSORT", PERF_ELEM_SORT},
    {"BITSORT", PERF_ELEM_SORT},
    {"MRGSORT", PERF_ELEM_SORT},
    {"ARGSORT", PERF_ELEM_SORT},
    {"SORT", PERF_ELEM_SORT},
    {"EXTRACT", PERF_ELEM_SORT},
    {"COMP_SWAP", PERF_ELEM_SORT},
    {"MERGE", PERF_ELEM_SORT},
    // elem-select: compare / gather / scatter / concat / reshape-like-in-vec
    {"CMP", PERF_ELEM_SELECT},
    {"CMPS", PERF_ELEM_SELECT},
    {"WHERE_TT", PERF_ELEM_SELECT},
    {"WHERE_TS", PERF_ELEM_SELECT},
    {"WHERE_ST", PERF_ELEM_SELECT},
    {"WHERE_SS", PERF_ELEM_SELECT},
    {"GATHER", PERF_ELEM_SELECT},
    {"GATHER_FROM_UB", PERF_ELEM_SELECT},
    {"GATHER_ELEMENT", PERF_ELEM_SELECT},
    {"SCATTER", PERF_ELEM_SELECT},
    {"SCATTER_ELEMENT", PERF_ELEM_SELECT},
    {"SCATTER_UPDATE", PERF_ELEM_SELECT},
    {"SCATTER_SCALAR", PERF_ELEM_SELECT},
    {"INDEX_PUT", PERF_ELEM_SELECT},
    {"INDEX_ADD", PERF_ELEM_SELECT},
    {"INDEX_OUTCAST", PERF_ELEM_SELECT},
    {"CONCAT", PERF_ELEM_SELECT},
    {"DUPLICATE", PERF_ELEM_SELECT},
    {"EXPAND", PERF_ELEM_SELECT},
    {"ONEHOT", PERF_ELEM_SELECT},
    {"COMPACT", PERF_ELEM_SELECT},
    {"REGISTER_COPY", PERF_ELEM_SELECT},
    {"VEC_DUP", PERF_ELEM_SELECT},
    {"UB_TO_UB", PERF_ELEM_SELECT},
    {"UB_COPY_ND2NZ", PERF_ELEM_SELECT},
    {"FILLPAD", PERF_ELEM_SELECT},
    {"MAX_POOL", PERF_ELEM_SELECT},
    {"RANGE", PERF_ELEM_SELECT},
    {"FUSED_OP", PERF_ELEM_SELECT},
    {"CALL_NOT_EXPAND", PERF_ELEM_SELECT},
};

int GetOpTypeIndex(const TileOpPtr& tileOp)
{
    const auto& opcode = tileOp->opcode;
    auto pipeType = tileOp->pipeType;

    switch (pipeType) {
        case CorePipeType::PIPE_CUBE:
            return PERF_MATMUL;

        case CorePipeType::PIPE_VECTOR_ALU: {
            auto it = VEC_OPCODE_MAP.find(opcode);
            if (it != VEC_OPCODE_MAP.end())
                return it->second;
        }
            return -1;

        case CorePipeType::PIPE_MTE_IN:
            if (opcode.find("L1_COPY_IN") != std::string::npos || opcode == "GATHER_IN_L1")
                return PERF_L1_COPY_IN;
            return PERF_UB_COPY_IN;

        case CorePipeType::PIPE_MTE_OUT:
            if (opcode == "L0C_COPY_OUT")
                return PERF_L0C_COPY_OUT;
            return PERF_UB_COPY_OUT;

        default:
            return -1;
    }
}

uint64_t TileShapeElements(const TileOpPtr& tileOp)
{
    if (!tileOp || tileOp->iOperand.empty())
        return 0;
    auto& tile = tileOp->iOperand[0];
    if (!tile)
        return 0;
    return static_cast<uint64_t>(tile->SizeinBytes());
}

std::vector<uint64_t> ComputeComputeAmount(const FunctionPtr& func)
{
    std::vector<uint64_t> result(PERF_OP_TYPE_COUNT, 0);
    for (const auto& kv : func->tileOpMap) {
        if (!kv.second)
            continue;
        int typeIdx = GetOpTypeIndex(kv.second);
        if (typeIdx < 0)
            continue;
        result[typeIdx] += TileShapeElements(kv.second);
    }
    return result;
}

} // anonymous namespace

namespace CostModel {

constexpr uint64_t L0C_THRESHOLD_MULTIPLIER = 2;

SimSys::SimSys()
{
    globalCycles = 0;
    simStartTime = std::chrono::system_clock::now();
    calendarGenerator = std::make_shared<GenCalendar>();
    totalTraceLogger = std::make_shared<TraceLogger>();
    totalTraceLogger->topMachineViewPid = topMachineViewPid;
    stats = std::make_shared<ModelStats>(GetReporter());
    stats->Reset();
}
std::shared_ptr<SimSys> SimSys::GetShared() { return shared_from_this(); }

bool SimSys::IsMachine(CostModel::Pid pid, CostModel::Tid tid)
{
    if (pid == topMachineViewPid) {
        return false;
    }
    if (tid <= machines[0]->coreTid || tid >= machines[0]->reversedTidNum) {
        return false;
    }
    return true;
}

bool SimSys::IsQueue(CostModel::Tid tid)
{
    if (tid <= machines[0]->coreTid || tid >= machines[0]->reversedTidNum) {
        return false;
    }
    return true;
}

bool SimSys::IsWorkPipe(CostModel::Pid pid, CostModel::Tid tid, std::string& name)
{
    if (!IsCoreMachine(GetMachineType(pid))) {
        return false;
    }
    auto core = std::dynamic_pointer_cast<CoreMachine>(pidToMachineMp[pid]);
    int pipeId = tid - core->reversedTidNum;
    auto pipeType = core->pipeTypeMap[pipeId];
    if (!IsTileAlloc(pipeType)) {
        name = CorePipeName(pipeType);
        return true;
    }
    return false;
}

void SimSys::InitMachineStartSeq()
{
    machineTypeSeq[MachineType::AIC] = 0;
    machineTypeSeq[MachineType::AIV] = config.deviceMachineNumber * config.aicpuMachineNumber *
                                       config.cubeMachineNumberPerAICPU;
    machineTypeSeq[MachineType::MIXAICORE] = 0;
    machineTypeSeq[MachineType::DEVICE] = config.deviceMachineNumber * config.aicpuMachineNumber *
                                          config.coreMachineNumberPerAICPU;
    machineTypeSeq[MachineType::CPU] = machineTypeSeq[MachineType::DEVICE] + config.deviceMachineNumber;
    machineTypeSeq[MachineType::HUB] = machineTypeSeq[MachineType::CPU] + config.aicpuMachineNumber;
    machineTypeSeq[MachineType::PIPE] = 0;
}

void SimSys::BuildCaches()
{
    functionCache.SetSim(GetShared());
    functionCache.SetMaxCacheSize(config.functionCacheSize);

    // Create the L2 cache.
    auto l2BankId = 0;
    l2Cache = std::make_shared<CacheMachine>(CacheType::L2CACHE, config.deviceArch);
    l2Cache->machineType = MachineType::CACHE;
    l2Cache->machineId = GetProcessID(MachineType::CACHE, l2BankId);
    l2Cache->sim = GetShared();
    AddMachine(l2Cache);
    l2Cache->Build();
}

UnifiedPipeMachinePtr SimSys::GetPipeImpl(CorePipeType pType)
{
    if (IsTileAlloc(pType)) {
        return nullptr;
    }
    if (pipeImplMap[pType] == nullptr) {
        pipeImplMap[pType] = PipeFactory::Create(pType, config.deviceArch);
    }
    return pipeImplMap[pType];
}

void SimSys::BuildPipes(uint64_t index, std::shared_ptr<CoreMachine> coreMachine)
{
    // SMT: CoreMachine Share the PipeMachine backend
    if (coreMachine->machineType == lastCoreMachineType && index % config.coreMachineSmtNum != 0) {
        return;
    }
    lastCoreMachineType = coreMachine->machineType;
    stats->pipeGroupNum++;
    pipeMachines.clear();
    pipeMachineIndex.clear();
    numTileopSentToPipe.clear();
    pipeMachineIndex.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
    numTileopSentToPipe.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
    wlStatus.AddMachineGroup();
    int pipeId = 0;
    for (int pipeType = 0; pipeType < static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE); pipeType++) {
        uint64_t pipeNum = coreMachine->GetPipeNum(static_cast<CorePipeType>(pipeType));
        for (uint64_t num = 0; num < pipeNum; num++) {
            int pipeSeq = machineTypeSeq[MachineType::PIPE]++;
            auto pipeMachine = std::make_shared<PipeMachine>(MachineType::PIPE, static_cast<CorePipeType>(pipeType),
                                                             pipeId);
            pipeMachine->machineType = MachineType::PIPE;
            pipeMachine->machineId = GetProcessID(MachineType::PIPE, pipeSeq);
            pipeMachine->pipeImpl = GetPipeImpl(static_cast<CorePipeType>(pipeType));
            pipeMachine->sim = GetShared();
            AddMachine(pipeMachine);

            std::string pipeName = "Pipe_" + std::to_string(pipeSeq);
            totalTraceLogger->SetProcessName(pipeName, pipeMachine->machineId, GetMachineSeq(coreMachine->machineId));
            totalTraceLogger->SetThreadName("Core_Machine_View", pipeMachine->machineId, pipeMachine->coreTid);
            pipeMachine->SetQueueCounter();
            pipeMachine->parentMachine = coreMachine;
            pipeMachine->Build();
            pipeMachine->l2cacheMachine = config.mteUseL2Cache ? l2Cache : nullptr;

            pipeMachineIndex[pipeType].emplace_back(pipeId);
            numTileopSentToPipe[pipeType].emplace_back(0);
            pipeMachines.emplace_back(pipeMachine);
            coreMachine->LoggerRecordPipe(CorePipeName(static_cast<CorePipeType>(pipeType)), pipeId);
            coreMachine->pipeTypeMap[pipeId] = static_cast<CorePipeType>(pipeType);
            pipeId++;
        }
    }
}

void SimSys::CalendarDispatchTasksToCore(int key, std::shared_ptr<CoreMachine> coreMachine)
{
    if (config.calendarMode == static_cast<uint64_t>(CalendarMode::DEVICE)) {
        return;
    }
    for (const auto& task : corePacketMap[key]) {
        TaskPack packet;
        packet.taskId = task.first;
        packet.task.taskPtr = calendarTaskMap[task.first];
        ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, packet.task.taskPtr != nullptr)
            << "[SIMULATION]: "
            << "task does not exist. taskId=" << packet.taskId;
        packet.task.functionHash = task.second;
        coreMachine->SubmitTask(packet);
    }
}

void SimSys::InitCalendarMode()
{
    if (machineGroup[int(MachineType::DEVICE)].empty()) {
        return;
    }

    auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(machineGroup[int(MachineType::DEVICE)][0]);
    devicePtr->InitFunctions();
    calendarTaskMap = devicePtr->taskMapQueue.front();
    devicePtr->taskMapQueue.pop_front();
    for (int mType = 0; mType < int(MachineType::TOTAL_MACHINE_TYPE); mType++) {
        if (!IsCoreMachine(mType)) {
            continue;
        }
        for (auto& aiCore : machineGroup[mType]) {
            auto coreIdx = GetMachineSeq(aiCore->machineId);
            auto corePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
            CalendarDispatchTasksToCore(coreIdx, corePtr);
        }
    }
}

void SimSys::BuildHUBCore()
{
    auto type = MachineType::HUB;
    int coreIdx = machineTypeSeq[type]++;
    auto coreMachine = std::make_shared<CoreMachine>(type);
    coreMachine->machineId = GetProcessID(coreMachine->machineType, coreIdx);
    coreMachine->sim = GetShared();
    coreMachine->Build();
    LogRegisterMachine(coreMachine, coreIdx, coreIdx);
    AddMachine(coreMachine);
}

MachinePtr SimSys::GetHUBCore() { return machineGroup[int(MachineType::HUB)][0]; }

void SimSys::BuildCore(DevicePtr device, AICPUPtr cpu, uint64_t idInCPU, MachineType type)
{
    int coreIdx = machineTypeSeq[type]++;
    auto coreMachine = std::make_shared<CoreMachine>(type);
    coreMachine->machineId = GetProcessID(coreMachine->machineType, coreIdx);
    coreMachine->sim = GetShared();
    coreMachine->Build();
    coreMachine->l2cacheMachine = config.mteUseL2Cache ? l2Cache : nullptr;
    coreMachine->SetQueueCounter();
    coreMachine->parentMachine = cpu; // set AICPU as parent
    coreMachine->SetTileState(device->tileState);

    cpu->subMachines.emplace_back(coreMachine); // add coreMachine to AICPU's sub_machines

    LogRegisterMachine(coreMachine, coreIdx, coreIdx);

    BuildPipes(idInCPU, coreMachine);
    wlStatus.AddMachineIndex(subMachineCount);
    subMachineCount++;
    coreMachine->pipeMachineIndex = pipeMachineIndex;
    coreMachine->numTileopSentToPipe = numTileopSentToPipe;
    coreMachine->subMachines = pipeMachines;
    AddMachine(coreMachine);
}

void SimSys::BuildAICPU(DevicePtr device, uint64_t idInDevice)
{
    uint64_t aicNum = config.cubeMachineNumberPerAICPU;
    uint64_t aivNum = config.vecMachineNumberPerAICPU;
    uint64_t mixedCoreNum = 0;
    ASSERT(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_CONFIG),
           config.coreMachineNumberPerAICPU == (aicNum + aivNum))
        << "[SIMULATION]: "
        << "The number of cores must be equal to the sum of the aic and aiv. Please reconfigure them.";
    if (config.cubeVecMixMode) {
        mixedCoreNum = config.coreMachineNumberPerAICPU;
        aicNum = 0;
        aivNum = 0;
    }
    int aicpuIdx = machineTypeSeq[MachineType::CPU]++;
    auto aicpuMachine = std::make_shared<AICPUMachine>(MachineType::CPU);
    wlStatus = AICoreWorkLoadStatus();
    wlStatus.maxLevel = config.coreMachineSmtNum;
    subMachineCount = 0;
    aicpuMachine->machineType = MachineType::CPU;
    aicpuMachine->machineId = GetProcessID(aicpuMachine->machineType, aicpuIdx);
    aicpuMachine->sim = GetShared();
    aicpuMachine->Build();
    aicpuMachine->l2cacheMachine = config.mteUseL2Cache ? l2Cache : nullptr;
    aicpuMachine->SetQueueCounter();
    AddMachine(aicpuMachine);
    aicpuMachine->parentMachine = device; // set Device as parent
    aicpuMachine->top = device;
    LogRegisterMachine(aicpuMachine, idInDevice, aicpuIdx);

    device->subMachines.emplace_back(aicpuMachine); // add AICPU to Device's subMachines

    // assign AIC
    for (uint64_t aicId = 0; aicId < aicNum; aicId++) {
        BuildCore(device, aicpuMachine, aicId, MachineType::AIC);
    }

    // assign AIV
    for (uint64_t aivId = 0; aivId < aivNum; aivId++) {
        BuildCore(device, aicpuMachine, aivId, MachineType::AIV);
    }

    // assign MIXAICORE
    for (uint64_t id = 0; id < mixedCoreNum; id++) {
        BuildCore(device, aicpuMachine, id, MachineType::MIXAICORE);
    }

    aicpuMachine->wlStatus = wlStatus;
}

void SimSys::BuildDevice()
{
    for (size_t deviceId = 0; deviceId < config.deviceMachineNumber; deviceId++) {
        int deviceIdx = machineTypeSeq[MachineType::DEVICE]++;
        auto deviceMachine = std::make_shared<DeviceMachine>();
        deviceMachine->sim = GetShared();
        deviceMachine->machineType = MachineType::DEVICE;
        deviceMachine->machineId = GetProcessID(MachineType::DEVICE, deviceIdx);
        deviceMachine->Build();
        deviceMachine->SetQueueCounter();
        AddMachine(deviceMachine);
        LogRegisterMachine(deviceMachine, deviceId, deviceIdx);

        pipeMachineIndex.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
        numTileopSentToPipe.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));

        for (size_t aicpuId = 0; aicpuId < config.aicpuMachineNumber; aicpuId++) {
            BuildAICPU(deviceMachine, aicpuId);
        }
        BuildHUBCore();
    }
}

void SimSys::BuildSystemStat()
{
    ResetStat(true);
    if (machineGroup[int(MachineType::PIPE)].size() > 0) {
        auto pipePtr = std::dynamic_pointer_cast<PipeMachine>(machineGroup[int(MachineType::PIPE)][0]);
        InitBufferThreshold(pipePtr->config);
    }
    stats->deviceMachineNum = uint64_t(machineGroup[int(MachineType::DEVICE)].size());
    stats->aicpuMachineNum = uint64_t(machineGroup[int(MachineType::CPU)].size());
    stats->coreMachineNum = uint64_t(machineGroup[int(MachineType::AIC)].size() +
                                     machineGroup[int(MachineType::AIV)].size());
    stats->cubeMachineNum = uint64_t(machineGroup[int(MachineType::AIC)].size());
    stats->vecMachineNum = uint64_t(machineGroup[int(MachineType::AIV)].size());
    stats->cvMixedCoreMachineNum = uint64_t(machineGroup[int(MachineType::MIXAICORE)].size());
}

void SimSys::AddExtraConfigs()
{
    if (config.coreMachineSmtNum > 1) {
        cfgs.emplace_back("Core.pipeMteInNum=" + std::to_string(config.coreMachineSmtNum));
        cfgs.emplace_back("Core.pipeMte1Num=" + std::to_string(config.coreMachineSmtNum));
        cfgs.emplace_back("Core.pipeMteOutNum=" + std::to_string(config.coreMachineSmtNum));
    }

    if (config.cubeVecMixMode) {
        cfgs.emplace_back("Device.submachineTypes=MIXAICORE,HUB");
    }

    if (config.genCalendarScheduleCpp) {
        cfgs.emplace_back("AICPU.completionCycles=1");
        cfgs.emplace_back("AICPU.schedulerCycles=1");
        cfgs.emplace_back("AICPU.resolveCycles=1");
    }
}

void SimSys::BuildSystem()
{
    config.OverrideDefaultConfig(&cfgs);
    AddExtraConfigs();
    // Init Trace Logger and Registe top machine
    totalTraceLogger->config.OverrideDefaultConfig(&cfgs);
    totalTraceLogger->SetProcessName("Top_Machine", topMachineViewPid, 0);

    // Init Machine Group
    Reset();
    InitMachineStartSeq();

    BuildCaches();

    // Add Device Machine
    BuildDevice();

    BuildSystemStat();
}

void SimSys::InitCoreTask() { testSingleFunc = true; }

void SimSys::Reset()
{
    globalCycles = 0;
    nextSimulationCycles = INT_MAX;
    lastSimulationCycles = INT_MAX;
    terminate = false;
    machines.clear();
    machineGroup.clear();
    pidToMachineMp.clear();
    machineGroup.resize(int(MachineType::TOTAL_MACHINE_TYPE));
    ResetStat(false);
}

void SimSys::AddMachine(std::shared_ptr<Machine> m)
{
    machines.push_back(m);
    pidToMachineMp[m->machineId] = m;
    machineGroup[int(m->machineType)].push_back(m);
    if (IsCoreMachine(m->machineType)) {
        calendarGenerator->InitAICore(m->machineId);
    }
}

void SimSys::Step()
{
    SIMULATION_LOGI("[ModelTop][Step] ========== %lu ========== [ModelTop][Step]",
                    static_cast<unsigned long>(globalCycles));
    for (const auto& machine : machines) {
        machine->Step();
    }

    for (const auto& machine : machines) {
        machine->Xfer();
    }

    terminate = CheckAllEmpty();
    stats->cycles = globalCycles;
    stats->stepCount++;
    if (globalCycles > executeCycleThreshold) {
        terminate = true;
    }
    lastSimulationCycles = globalCycles;
    globalCycles = nextSimulationCycles;
    nextSimulationCycles = INT_MAX;
}

bool SimSys::CheckAllEmpty()
{
    bool isEmpty = true;
    for (const auto& machine : machines) {
        if (!machine->IsTerminate()) {
            isEmpty = false;
            break;
        }
    }
    return isEmpty;
}

bool SimSys::IsTerminate() const { return terminate; }

void SimSys::ReportDeadlock(size_t machineId)
{
    deadlock = true;
    SIMULATION_LOGE(CostModel::ForwardSimErrorScene::DEAD_LOCK, "[ReportDeadlock] Machine %zu is deadlock at cycle %lu",
                    machineId, static_cast<unsigned long>(globalCycles));
}

bool SimSys::IsDeadlock() const { return deadlock; }

void SimSys::LoggerRecordCoreStart(std::string name, size_t machineId, std::string hint)
{
    totalTraceLogger->AddEventBegin(name, topMachineViewPid, machineId, GetCycles(), hint);
}

void SimSys::LoggerRecordCoreCompleted(size_t machineId)
{
    totalTraceLogger->AddEventEnd(topMachineViewPid, machineId, GetCycles());
}

std::string SimSys::GetFileName(const std::string& file)
{
    size_t lastSlashPos = file.find_last_of("/\\");
    if (lastSlashPos != std::string::npos) {
        return file.substr(lastSlashPos + 1);
    } else {
        return file;
    }
}

std::string SimSys::GetFileName(const std::string& dir, const std::string& inputFile, const std::string& preFix,
                                const std::string& suffix)
{
    std::string filename = GetFileName(inputFile);
    std::string outFile = preFix + "_simulate." + suffix;
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) { // 检查是否存在 "."
        filename.replace(dotPos + 1, std::string::npos, suffix);
        outFile = preFix + filename;
    }
    std::string outPath = dir + "/" + outFile;
    return outPath;
}

void SimSys::ProcessTaskMap(TaskMap& taskMap, std::string prefix)
{
    DumpTasksTopo(taskMap, prefix);
    DrawTasks(taskMap, prefix);
    GetCalendarGenerator()->InitTaskTopoInfo(taskMap);
}

void SimSys::DrawTasks(const TaskMap& taskMap, std::string prefix)
{
    if (drawGraph) {
        ModelVisualizer visualizer;
        std::string outPath = GetFileName(graphsOutdir, jsonPath, prefix, startFuncName + ".taskGraph.dot");
        visualizer.DrawTasks(taskMap, true, outPath);
        SIMULATION_LOGW("Task Graph Path: %s", outPath.c_str());
    }
}

void SimSys::DebugDrawFunc(FunctionPtr func, std::unordered_map<int, TilePtr>& tiles,
                           std::unordered_map<int, TileOpPtr>& tileOps)
{
    ModelVisualizer visualizer;
    visualizer.DebugFunction(func, tiles, tileOps, outdir);
}

void SimSys::DumpTasksTopo(const TaskMap& taskMap, std::string prefix)
{
    Json totalTopoJson;
    for (const auto& task : taskMap) {
        Json sJson;
        sJson["taskId"] = task.second->taskId;
        sJson["funcName"] = task.second->functionName;
        sJson["successors"] = Json::array();
        sJson["predecessors"] = Json::array();
        sJson["remainingPredecessors"] = task.second->remainingPredecessors;
        sJson["semanticLabel"] = task.second->semanticLabel;
        for (const auto& successor : task.second->successors) {
            sJson["successors"].push_back(successor);
        }
        for (const auto& predecessor : task.second->predecessors) {
            sJson["predecessors"].push_back(predecessor);
        }
        totalTopoJson.push_back(sJson);
    }

    std::string fileName = GetFileName(outdir, jsonPath, prefix, "topo.json");
    topoOutFile = fileName;
    SIMULATION_LOGW("Topo File Path: %s", fileName.c_str());
    std::ofstream ofs(fileName);
    ofs << totalTopoJson.dump(1) << std::endl;
    ofs.close();
}

void SimSys::OutputTrace(std::string prefix)
{
    std::string outPath = GetFileName(outdir, jsonPath, prefix, "smartperf.trace");
    std::ofstream os(outPath);
    totalTraceLogger->ToTrace(os);
    os.close();
    SIMULATION_LOGW("Please Use Smartperf For Visualization:");
    SIMULATION_LOGW("Trace Path: %s \n", outPath.c_str());
}

void SimSys::OutputPerfettoTrace(std::string prefix)
{
    std::string outPath = GetFileName(outdir, jsonPath, prefix, "pefetto.trace.json");
    std::ofstream os(outPath);
    Json trace = totalTraceLogger->ToJson();
    os << trace.dump(1) << std::endl;
    os.close();
    SIMULATION_LOGW("Please Use Perfetto For Visualization:");
    SIMULATION_LOGW("Perfetto Trace Path: %s \n", outPath.c_str());
}

void SimSys::DumpFunctionExecuteTime(std::string prefix)
{
    std::string outPath = GetFileName(outdir, jsonPath, prefix, "leafFuncs.executetime.json");
    std::ofstream os(outPath);
    Json log;
    for (auto& func : functionCache.cache) {
        log.emplace_back(func.second->DumpExecuteInfo());
    }
    os << log.dump(1) << std::endl;
    os.close();
}

void SimSys::OutputLogForPipeSwimLane(std::string prefix)
{
    if (globalCycles > config.drawPngThresholdCycle) {
        return;
    }
    std::string pipeDetailPath = GetFileName(outdir, jsonPath, prefix, "pipe.swim.json");
    std::ofstream osPipeSwim(pipeDetailPath);
    totalTraceLogger->ToPipeTrace(osPipeSwim);
    osPipeSwim.close();

    SIMULATION_LOGW("Pipe SwimLane Graph Generated (PNG & HTML): %s", pipeDetailPath.c_str());
    std::string drawScriptPath = GetPyptoLibPath() + "/scripts/draw_pipe_swim_lane.py";
    CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
          npu::tile_fwk::IsPathExist(drawScriptPath))
        << "draw_pipe_swim_lane.py does not exist. drawScriptPath: " << drawScriptPath;
    CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
          npu::tile_fwk::IsPathExist(pipeDetailPath))
        << "pipe.swim.json does not exist. pipeDetailPath: " << pipeDetailPath;
    std::string cmd = "python3 " + drawScriptPath + " " + pipeDetailPath;
    auto args = SplitString(cmd);
    int ret = SafeExecCommand(args);
    if (ret != 0) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::PYTHON_CMD_ERROR, "draw pipe swim lane cmd error");
    }
}

void SimSys::OutputLogForSwimLane(std::string prefix)
{
    std::string swimTail = "swim.json";
    std::string outSwimPath = GetFileName(outdir, jsonPath, prefix, swimTail);
    std::ofstream osSwim(outSwimPath);
    std::map<int, std::pair<std::string, std::vector<Json>>> coreTasks;
    totalTraceLogger->ToFilterTrace(osSwim, coreTasks);

    OutputLogForPipeSwimLane();
    if (config.calendarMode == static_cast<uint64_t>(CalendarMode::DEVICE)) {
        std::string calendarTail = "global.calendar.json";
        std::string outCalendarPath = GetFileName(outdir, jsonPath, prefix, calendarTail);
        std::ofstream osCalendar(outCalendarPath);
        totalTraceLogger->ToCalendarGlobalJson(osCalendar, coreTasks);
        osCalendar.close();
    }
    // Get Draw PND Python Scripts Path
    osSwim.close();
    if (globalCycles > config.drawPngThresholdCycle) {
        return;
    }
    SIMULATION_LOGW("SwimLane Graph Generated (PNG): %s", outSwimPath.c_str());
    std::string drawScriptPath = GetPyptoLibPath() + "/scripts/print_swim_lane.py";
    CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
          npu::tile_fwk::IsPathExist(drawScriptPath))
        << "draw_pipe_swim_lane.py does not exist. drawScriptPath: " << drawScriptPath;
    CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH), npu::tile_fwk::IsPathExist(outSwimPath))
        << "swim.json does not exist. outSwimPath: " << outSwimPath;
    std::string cmd = "python3 " + drawScriptPath + " " + outSwimPath + " -t";
    auto args1 = SplitString(cmd);
    int result1 = SafeExecCommand(args1);
    if (result1 != 0) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::PYTHON_CMD_ERROR, "draw swim lane cmd error.");
    }

    std::string mergeScriptPath = GetPyptoLibPath() + "/scripts/draw_swim_lane.py";
    auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(machineGroup[int(MachineType::DEVICE)][0]);
    SIMULATION_LOGW("devicePtr->config.submitTopo: %d", devicePtr->config.submitTopo);
    std::string topo_txt_path = outdir + "/../" + "dyn_topo.txt";
    SIMULATION_LOGI("topo_txt_path: %s", topo_txt_path.c_str());
    std::string program_json_path = outdir + "/../" + "program.json";
    SIMULATION_LOGI("program_json_path: %s", program_json_path.c_str());
    std::string label_type = "--label_type=1 --time_convert_denominator=1800"; // default 1.8GHz
    SIMULATION_LOGI("label_type: %s", label_type.c_str());
    CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
          npu::tile_fwk::IsPathExist(mergeScriptPath))
        << "draw_pipe_swim_lane.py does not exist. drawScriptPath: " << mergeScriptPath;
    if (devicePtr->config.submitTopo) {
        CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
              npu::tile_fwk::IsPathExist(topo_txt_path))
            << "dyn_topo.txt does not exist. topo_txt_path: " << topo_txt_path;
        CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
              npu::tile_fwk::IsPathExist(program_json_path))
            << "program.json does not exist. program_json_path: " << program_json_path;
        cmd = "python3 " + mergeScriptPath + " " + outSwimPath + " " + topo_txt_path + " " + program_json_path + " " +
              label_type;
    } else {
        CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH),
              npu::tile_fwk::IsPathExist(topoOutFile))
            << "topo.json does not exist. topoOutFile: " << topoOutFile;
        SIMULATION_LOGW("devicePtr->config.submitTopo: %d", devicePtr->config.submitTopo);
        cmd = "python3 " + mergeScriptPath + " " + outSwimPath + " " + topoOutFile;
    }
    auto args2 = SplitString(cmd);
    int result2 = SafeExecCommand(args2);
    if (result2 != 0) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::PYTHON_CMD_ERROR, "draw swim lane cmd error.");
    }
}

void SimSys::OutputCalendarScheduleCpp(std::string prefix)
{
    if (!config.genCalendarScheduleCpp) {
        return;
    }
    std::string outPath = GetFileName(outdir, jsonPath, prefix, ".calendar.cpp");
    calendarGenerator->GenCalendarCpp(outPath);
    SIMULATION_LOGW("Genearte Calendar File: %s", outPath.c_str());
}

std::map<uint64_t, std::vector<uint64_t>> SimSys::ComputeTaskAmounts(const TaskMap& taskMap)
{
    std::map<uint64_t, std::vector<uint64_t>> computeAmountMap;
    for (const auto& [taskId, task] : taskMap) {
        FunctionPtr func = nullptr;
        try {
            func = functionCache.GetFunction(task->functionHash);
        } catch (const std::out_of_range&) {
            func = nullptr;
        }
        if (func) {
            computeAmountMap[taskId] = ComputeComputeAmount(func);
            continue;
        }
        computeAmountMap[taskId] = std::vector<uint64_t>(PERF_OP_TYPE_COUNT, 0);
    }
    return computeAmountMap;
}

SimSys::LongestPathResult SimSys::ComputeLongestPaths(const TaskMap& taskMap,
                                                      const std::map<uint64_t, std::vector<uint64_t>>& computeAmountMap)
{
    const uint64_t NO_PRED = static_cast<uint64_t>(-1);

    std::map<uint64_t, std::vector<uint64_t>> longestPath;
    std::map<uint64_t, std::vector<uint64_t>> chosenPred;
    for (const auto& kv : taskMap) {
        longestPath[kv.first] = computeAmountMap.at(kv.first);
        chosenPred[kv.first] = std::vector<uint64_t>(PERF_OP_TYPE_COUNT, NO_PRED);
    }

    std::map<uint64_t, int> inDegree;
    for (const auto& [taskId, task] : taskMap) {
        inDegree[taskId] = static_cast<int>(task->predecessors.size());
    }

    std::deque<uint64_t> queue;
    for (const auto& [taskId, deg] : inDegree) {
        if (deg == 0) {
            queue.push_back(taskId);
        }
    }

    while (!queue.empty()) {
        auto vId = queue.front();
        queue.pop_front();

        auto& task = taskMap.at(vId);
        for (auto succId : task->successors) {
            if (taskMap.find(succId) == taskMap.end())
                continue;

            auto& vLp = longestPath[vId];
            auto& succLp = longestPath[succId];
            auto& succComputeAmount = computeAmountMap.at(succId);
            auto& succChosenPred = chosenPred[succId];

            for (int i = 0; i < PERF_OP_TYPE_COUNT; i++) {
                uint64_t candidate = vLp[i] + succComputeAmount[i];
                if (candidate > succLp[i]) {
                    succLp[i] = candidate;
                    succChosenPred[i] = vId;
                }
            }

            inDegree[succId]--;
            if (inDegree[succId] == 0) {
                queue.push_back(succId);
            }
        }
    }

    return {longestPath, chosenPred};
}

Json SimSys::FormatPerfMetricsJson(const std::map<uint64_t, std::vector<uint64_t>>& longestPath,
                                   const std::map<uint64_t, std::vector<uint64_t>>& chosenPred)
{
    const uint64_t NO_PRED = static_cast<uint64_t>(-1);
    Json tileOpLongestPaths;
    for (int i = 0; i < PERF_OP_TYPE_COUNT; i++) {
        uint64_t maxLp = 0;
        uint64_t endTaskId = NO_PRED;
        for (const auto& [taskId, lp] : longestPath) {
            if (lp[i] > maxLp) {
                maxLp = lp[i];
                endTaskId = taskId;
            }
        }

        if (maxLp == 0)
            continue;

        std::vector<uint64_t> chain;
        uint64_t curId = endTaskId;
        while (curId != NO_PRED) {
            chain.push_back(curId);
            curId = chosenPred.at(curId)[i];
        }
        std::reverse(chain.begin(), chain.end());

        Json entry;
        entry["longestPathChain"] = chain;
        entry["amountLp"] = maxLp;
        tileOpLongestPaths[PERF_OP_TYPE_NAMES[i]] = entry;
    }
    return tileOpLongestPaths;
}

void SimSys::OutputPerfMetrics(std::string prefix)
{
    if (machineGroup[int(MachineType::DEVICE)].empty()) {
        SIMULATION_LOGW("[PerfMetrics] No device machine, skipping.");
        return;
    }
    auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(machineGroup[int(MachineType::DEVICE)][0]);
    auto& taskMap = devicePtr->savedTaskMap;

    if (taskMap.empty()) {
        SIMULATION_LOGW("[PerfMetrics] Empty task map, skipping.");
        return;
    }

    auto computeAmountMap = ComputeTaskAmounts(taskMap);
    auto [longestPath, chosenPred] = ComputeLongestPaths(taskMap, computeAmountMap);
    auto tileOpLongestPaths = FormatPerfMetricsJson(longestPath, chosenPred);

    Json output;
    Json summary;
    summary["totalCycles"] = globalCycles;
    summary["tileOpLongestPaths"] = tileOpLongestPaths;
    output["summary"] = summary;

    std::string outPath = GetFileName(outdir, jsonPath, prefix, "perf.metrics.json");
    std::ofstream os(outPath);
    os << output.dump(1) << std::endl;
    os.close();
    SIMULATION_LOGW("[PerfMetrics] Output path: %s", outPath.c_str());
}

void SimSys::OutputConfig(std::string prefix)
{
    std::string outPath = GetFileName(outdir, jsonPath, prefix, "config.ini");
    SIMULATION_LOGW("Config Path: %s", outPath.c_str());
    std::ofstream os(outPath);
    os << config.DumpParameters() << std::endl;
    if (!machineGroup[int(MachineType::DEVICE)].empty()) {
        auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(machineGroup[int(MachineType::DEVICE)][0]);
        os << devicePtr->config.DumpParameters() << std::endl;
    }
    if (!machineGroup[int(MachineType::CPU)].empty()) {
        auto cpuPtr = std::dynamic_pointer_cast<AICPUMachine>(machineGroup[int(MachineType::CPU)][0]);
        os << cpuPtr->config.DumpParameters() << std::endl;
    }
    if (!machineGroup[int(MachineType::AIV)].empty()) {
        auto corePtr = std::dynamic_pointer_cast<CoreMachine>(machineGroup[int(MachineType::AIV)][0]);
        os << corePtr->config.DumpParameters() << std::endl;
    } else if (!machineGroup[int(MachineType::MIXAICORE)].empty()) {
        auto corePtr = std::dynamic_pointer_cast<CoreMachine>(machineGroup[int(MachineType::MIXAICORE)][0]);
        os << corePtr->config.DumpParameters() << std::endl;
    }
    if (!machineGroup[int(MachineType::PIPE)].empty()) {
        auto pipePtr = std::dynamic_pointer_cast<PipeMachine>(machineGroup[int(MachineType::PIPE)][0]);
        os << pipePtr->config.DumpParameters() << std::endl;
    }
    if (!machineGroup[int(MachineType::CACHE)].empty()) {
        auto cachePtr = std::dynamic_pointer_cast<CacheMachine>(machineGroup[int(MachineType::CACHE)][0]);
        os << cachePtr->config.DumpParameters() << std::endl;
    }

    os.close();
}

void SimSys::ResetStat(bool start)
{
    if (!start) {
        stats->Reset();
    }
    for (auto& device : machineGroup[int(MachineType::DEVICE)]) {
        auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(device);
        devicePtr->stats->Reset();
    }
    for (auto& aicpu : machineGroup[int(MachineType::CPU)]) {
        auto aicpuPtr = std::dynamic_pointer_cast<AICPUMachine>(aicpu);
        aicpuPtr->stats->Reset();
    }
    for (auto& aiCore : machineGroup[int(MachineType::AIV)]) {
        auto aiCorePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
        aiCorePtr->stats->Reset();
    }
    for (auto& aiCore : machineGroup[int(MachineType::AIC)]) {
        auto aiCorePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
        aiCorePtr->stats->Reset();
    }
    for (auto& aiCore : machineGroup[int(MachineType::MIXAICORE)]) {
        auto aiCorePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
        aiCorePtr->stats->Reset();
    }
    for (auto& cache : machineGroup[int(MachineType::CACHE)]) {
        auto cachePtr = std::dynamic_pointer_cast<CacheMachine>(cache);
        cachePtr->stats->Reset();
    }
}

void SimSys::PrintCoreStat()
{
    uint64_t cycles = GetCycles();
    std::map<int, uint64_t> totalPipeNumber;
    std::map<int, uint64_t> totalPipeUseCycles;
    std::map<int, float> averagePipeUseCycles;

    for (auto& aiCore : machineGroup[int(MachineType::AIC)]) {
        auto corePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
        for (auto& statEntry : corePtr->stats->totalPipeUseCycles) {
            if (statEntry.second != 0) {
                totalPipeNumber[statEntry.first]++;
                totalPipeUseCycles[statEntry.first] += statEntry.second;
            }
        }
    }
    for (auto& aiCore : machineGroup[int(MachineType::AIV)]) {
        auto corePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
        for (auto& statEntry : corePtr->stats->totalPipeUseCycles) {
            if (statEntry.second != 0) {
                totalPipeNumber[statEntry.first]++;
                totalPipeUseCycles[statEntry.first] += statEntry.second;
            }
        }
    }
    for (auto& aiCore : machineGroup[int(MachineType::MIXAICORE)]) {
        auto corePtr = std::dynamic_pointer_cast<CoreMachine>(aiCore);
        for (auto& statEntry : corePtr->stats->totalPipeUseCycles) {
            if (statEntry.second != 0) {
                totalPipeNumber[statEntry.first]++;
                totalPipeUseCycles[statEntry.first] += statEntry.second;
            }
        }
    }
    for (auto& pipe : totalPipeNumber) {
        if (pipe.second == 0) {
            continue;
        }
        averagePipeUseCycles[pipe.first] = (float(totalPipeUseCycles[pipe.first]) / pipe.second);
    }
    reporter.ReportTitle("Core Machine Top Statistics");

    reporter.ReportVal("Total Cycles", cycles);
    reporter.ReportVal("Used Cube Pipe Number", totalPipeNumber[int(CorePipeType::PIPE_CUBE)]);
    reporter.ReportValAndPct("Average Cube Usage", averagePipeUseCycles[int(CorePipeType::PIPE_CUBE)], cycles);
    reporter.ReportVal("Used Vector Pipe Number", totalPipeNumber[int(CorePipeType::PIPE_VECTOR_ALU)]);
    reporter.ReportValAndPct("Average Vector Usage", averagePipeUseCycles[int(CorePipeType::PIPE_VECTOR_ALU)], cycles);
    reporter.ReportVal("Used MTE_IN Pipe Number", totalPipeNumber[int(CorePipeType::PIPE_MTE_IN)]);
    reporter.ReportValAndPct("Average MTE_IN Usage", averagePipeUseCycles[int(CorePipeType::PIPE_MTE_IN)], cycles);
    reporter.ReportVal("Used MTE1 Pipe Number", totalPipeNumber[int(CorePipeType::PIPE_MTE1)]);
    reporter.ReportValAndPct("Average MTE1 Usage", averagePipeUseCycles[int(CorePipeType::PIPE_MTE1)], cycles);
    reporter.ReportVal("Used MTE_OUT Pipe Number", totalPipeNumber[int(CorePipeType::PIPE_MTE_OUT)]);
    reporter.ReportValAndPct("Average MTE_OUT Usage", averagePipeUseCycles[int(CorePipeType::PIPE_MTE_OUT)], cycles);
}

void SimSys::PrintStat()
{
    std::streambuf* coutBuf = nullptr;
    if (config.statisticReportToFile) {
        std::string outPath = GetFileName(outdir, jsonPath, "", "stat.report.txt");
        SIMULATION_LOGW("Statistic Path: %s", outPath.c_str());
        coutBuf = reporter.ReportSetOutStreamFile(outPath);
    }
    if (deadlock) {
        reporter.ReportTitle("Simulation Deadlock !!!!!!!!!");
    }
    std::string topName = "ASC++ MachineModel Report";
    stats->Report(topName);
    for (auto& device : machineGroup[int(MachineType::DEVICE)]) {
        device->Report();
    }

    PrintCoreStat();

    if (config.mteUseL2Cache) {
        l2Cache->Report();
    }

    reporter.ReportTitle("ASC++ MachineModel Report End");
    if (config.statisticReportToFile) {
        reporter.ReportResetOutStreamCout(coutBuf);
    }
}

uint64_t SimSys::GetCycles() const { return globalCycles; }

void SimSys::UpdateNextCycles(uint64_t nextCycle)
{
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, nextCycle > globalCycles)
        << "[SIMULATION]: "
        << "nextCycle is less than or equels to globalCycles. nextCycles=" << nextCycle
        << ", globalCycles=" << globalCycles;
    nextSimulationCycles = std::min(nextSimulationCycles, nextCycle);
}

void SimSys::ResetCycles(uint64_t cycles)
{
    globalCycles = cycles;
    nextSimulationCycles = cycles + 1;
}

void SimSys::AddCycles(uint64_t overTime) { globalCycles += overTime; }

std::shared_ptr<TraceLogger> SimSys::GetLogger() { return totalTraceLogger; }

std::shared_ptr<GenCalendar> SimSys::GetCalendarGenerator() { return calendarGenerator; }

CostModel::Reporter* SimSys::GetReporter() { return &reporter; }

void SimSys::LogRegisterMachine(MachinePtr machine, size_t id, int coreIdx)
{
    std::string name = MachineName(machine->machineType) + "_" + std::to_string(id);
    std::string machineViewName = MachineName(machine->machineType) + "_Machien_View";
    totalTraceLogger->SetThreadName(name, topMachineViewPid, machine->machineId);
    totalTraceLogger->SetProcessName(name, machine->machineId, coreIdx);
    totalTraceLogger->SetThreadName(machineViewName, machine->machineId, machine->coreTid);
}

uint64_t SimSys::RegisterQueuePid(std::string key)
{
    queuePidMap[key] = queuePidPtr++;
    return queuePidMap[key];
}

void SimSys::GetDeviceReadyQueueInfo(size_t& devicePid, std::set<uint64_t>& readyQueueTidSet)
{
    if (machineGroup[int(MachineType::DEVICE)].empty()) {
        return;
    }
    auto& device = machineGroup[int(MachineType::DEVICE)][0];
    auto devicePtr = std::dynamic_pointer_cast<DeviceMachine>(device);
    devicePid = devicePtr->readyQueuePid;
    readyQueueTidSet.insert(devicePtr->readyQueueTotalTid);
    for (auto& tid : devicePtr->readyQueueTid) {
        readyQueueTidSet.insert(tid.second);
    }
}

void SimSys::InitBufferThreshold(PipeConfig& pipeConfig)
{
    bufferSizeThreshold[CorePipeType::PIPE_VECTOR_BMU] = pipeConfig.ubSizeThreshold;
    bufferSizeThreshold[CorePipeType::PIPE_CUBE_BMU_L1] = pipeConfig.l1SizeThreshold;
    bufferSizeThreshold[CorePipeType::PIPE_CUBE_BMU_L0A] = pipeConfig.l0aSizeThreshold;
    bufferSizeThreshold[CorePipeType::PIPE_CUBE_BMU_L0B] = pipeConfig.l0bSizeThreshold;
    bufferSizeThreshold[CorePipeType::PIPE_CUBE_BMU_L0C] = pipeConfig.l0cSizeThreshold * L0C_THRESHOLD_MULTIPLIER;
}

uint64_t SimSys::GetBufferThreshold(CorePipeType pType) { return bufferSizeThreshold.at(pType); }
} // namespace CostModel
