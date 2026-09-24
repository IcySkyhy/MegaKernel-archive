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
 * \file ooo_schedule_statistic.cpp
 * \brief
 */

#include "ooo_schedule_statistic.h"
#include <fstream>
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"

#define MODULE_NAME "OooScheduleStatistic"
namespace npu {
namespace tile_fwk {

constexpr int32_t percent = 100;
constexpr float decimal = 10000.f; // 保留四位小数

// === ScheduleObserver callback implementations ===

void OoOScheduleStatistic::OnOpLaunch(const OpLaunchEvent& e) { pipeUsageCount[e.pipeType] += (e.cycleEnd - e.clock); }

void OoOScheduleStatistic::OnAllocExec(const AllocExecEvent& e)
{
    uint64_t size = e.addrEnd - e.addrStart;
    bufferMeta_[e.memId] = {e.memType, size};
    bufferTotalUsage[e.memType] += bufferLastUsage[e.memType] * (e.clock - lastClock[e.memType]);
    bufferLastUsage[e.memType] += size;
    lastClock[e.memType] = e.clock;
    bufferMaxUsage[e.memType] = std::max(bufferMaxUsage[e.memType], bufferLastUsage[e.memType]);
}

void OoOScheduleStatistic::OnOpRetire(const OpRetireEvent& e)
{
    for (int memId : e.freedMemIds) {
        auto it = bufferMeta_.find(memId);
        if (it == bufferMeta_.end())
            continue;
        auto memType = it->second.memType;
        auto size = it->second.size;
        bufferTotalUsage[memType] += bufferLastUsage[memType] * (e.clock - lastClock[memType]);
        bufferLastUsage[memType] -= size;
        lastClock[memType] = e.clock;
        bufferMeta_.erase(it);
    }
}

void OoOScheduleStatistic::OnSpill(const SpillEvent& e)
{
    uint64_t spillTensorSize = e.addrEnd - e.addrStart;
    SpillInfo info;
    info.spillType = e.memType;
    info.bufferCurrUsage = e.bufferCurrentUsage;
    info.bufferCapacity = e.bufferCapacity;
    info.spillTensorSize = spillTensorSize;
    info.triggerTensorSize = e.triggerTensorSize;
    info.allocOccupiedSize = e.allocOccupiedSize;
    info.spillCopyoutSize = e.spillCopyoutSize;
    info.spillTensorMagic = e.spillTensorMagic;
    spillInfoVec.emplace_back(info);

    // No OpRetire follows for spillMemId; account for the free here.
    auto it = bufferMeta_.find(e.spillMemId);
    if (it != bufferMeta_.end()) {
        bufferTotalUsage[e.memType] += bufferLastUsage[e.memType] * (e.clock - lastClock[e.memType]);
        bufferLastUsage[e.memType] -= it->second.size;
        lastClock[e.memType] = e.clock;
        bufferMeta_.erase(it);
    }
}

void OoOScheduleStatistic::OnScheduleEnd(const ScheduleEndEvent& e)
{
    clock = e.totalCycles;
    workspaceOffset = e.workspaceOffset;
}

void OoOScheduleStatistic::HealthCheckSpillInfo()
{
    int spillIdx = 0;
    Json spill = Json::array();
    for (auto spillInfo : spillInfoVec) {
        Json spillDetails;
        spillDetails["spillEventIdx"] = spillIdx++;
        spillDetails["spillBufferType"] = MemoryTypeToString(spillInfo.spillType);
        spillDetails["bufferCurrentUsage"] = spillInfo.bufferCurrUsage;
        spillDetails["bufferCurrentUsageRate"] = spillInfo.bufferCapacity == 0 ?
                                                     0.0f :
                                                     static_cast<float>(spillInfo.bufferCurrUsage) /
                                                         spillInfo.bufferCapacity;
        spillDetails["bufferOccupiedByAllocSize"] = spillInfo.allocOccupiedSize;
        spillDetails["spillTensorSize"] = spillInfo.spillTensorSize;
        spillDetails["spillTensorMagic"] = spillInfo.spillTensorMagic;
        spillDetails["triggerTensorSize"] = spillInfo.triggerTensorSize;
        spillDetails["spillCopyoutSize"] = spillInfo.spillCopyoutSize;
        spill.emplace_back(spillDetails);
    }
    report["spillDetails"] = spill;
}

double OoOScheduleStatistic::FormatUsageRate(double value) { return std::round(value * decimal) / decimal; }

Status OoOScheduleStatistic::HealthCheckOoOSchedule()
{
    auto memorySize = CommonUtils::GetLocalMemorySize();
    for (auto memType :
         {MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C}) {
        if (memorySize[memType] == 0) {
            APASS_LOG_ERROR_F(Elements::Function, "Max buffer size is 0, HealthCheckOoOSchedule failed!");
            return FAILED;
        }
    }
    if (clock == 0) {
        APASS_LOG_ERROR_F(Elements::Function, "Clock is 0, HealthCheckOoOSchedule failed!");
        return FAILED;
    }
    report["workspaceOffset"] = workspaceOffset;
    report["totalCycles"] = clock;
    ReportPipeUsage();
    ReportMemoryUsage(memorySize);
    report["spillCount"] = spillInfoVec.size();
    HealthCheckSpillInfo();
    return SUCCESS;
}

void OoOScheduleStatistic::ReportPipeUsage()
{
    static const std::vector<std::pair<PipeType, std::string>> pipeNames = {
        {PipeType::PIPE_S, "PIPE_S"},       {PipeType::PIPE_V, "PIPE_V"},       {PipeType::PIPE_M, "PIPE_M"},
        {PipeType::PIPE_MTE1, "PIPE_MTE1"}, {PipeType::PIPE_MTE2, "PIPE_MTE2"}, {PipeType::PIPE_MTE3, "PIPE_MTE3"},
        {PipeType::PIPE_FIX, "PIPE_FIX"}};
    Json pipeUsageRate;
    uint64_t maxUsage = 0;
    for (const auto& [pipeType, name] : pipeNames) {
        auto count = pipeUsageCount.at(pipeType);
        pipeUsageRate[name + "_Usage_Rate"] = FormatUsageRate(static_cast<double>(count) / clock * percent);
        maxUsage = std::max(maxUsage, count);
    }
    report["pipeUsageRate"] = pipeUsageRate;
    report["theoreticalMinimumCycles"] = maxUsage;
}

void OoOScheduleStatistic::ReportMemoryUsage(const std::unordered_map<MemoryType, int64_t>& memorySize)
{
    static const std::vector<std::pair<MemoryType, std::string>> memNames = {{MemoryType::MEM_UB, "MEM_UB"},
                                                                             {MemoryType::MEM_L1, "MEM_L1"},
                                                                             {MemoryType::MEM_L0A, "MEM_L0A"},
                                                                             {MemoryType::MEM_L0B, "MEM_L0B"},
                                                                             {MemoryType::MEM_L0C, "MEM_L0C"}};
    Json memoryUsage;
    for (const auto& [memType, name] : memNames) {
        auto maxSize = memorySize.at(memType);
        memoryUsage[name + "_Peak_Usage"] = FormatUsageRate(static_cast<double>(bufferMaxUsage.at(memType)) / maxSize *
                                                            percent);
        memoryUsage[name + "_Average_Usage"] = FormatUsageRate(static_cast<double>(bufferTotalUsage.at(memType)) /
                                                               clock / maxSize * percent);
    }
    report["memoryUsage"] = memoryUsage;
}

void OoOScheduleStatistic::HealthCheckBlockGraph(Function* function)
{
    report["totalOpCount"] = function->Operations().size();
    auto tensors = GraphUtils::GetAllTensors(*function);
    size_t maxProducers = 0;
    size_t maxConsumers = 0;
    std::vector<int> maxProducersTensors;
    std::vector<int> maxConsumersTensors;
    for (auto& tensor : tensors) {
        if (!tensor)
            continue;
        maxProducers = std::max(tensor->GetProducers().size(), maxProducers);
        maxConsumers = std::max(tensor->GetConsumers().size(), maxConsumers);
    }
    for (auto& tensor : tensors) {
        if (!tensor)
            continue;
        if (tensor->GetProducers().size() == maxProducers) {
            maxProducersTensors.emplace_back(tensor->GetMagic());
        }
        if (tensor->GetConsumers().size() == maxConsumers) {
            maxConsumersTensors.emplace_back(tensor->GetMagic());
        }
    }
    size_t maxInputs = 0;
    size_t maxOutputs = 0;
    std::vector<int> maxInputsOps;
    std::vector<int> maxOutputsOps;
    for (auto operation : function->Operations().DuplicatedOpList()) {
        maxInputs = std::max(operation->GetIOperands().size(), maxInputs);
        maxOutputs = std::max(operation->GetOOperands().size(), maxOutputs);
    }
    for (auto operation : function->Operations().DuplicatedOpList()) {
        if (operation->GetIOperands().size() == maxInputs) {
            maxInputsOps.emplace_back(operation->GetOpMagic());
        }
        if (operation->GetOOperands().size() == maxOutputs) {
            maxOutputsOps.emplace_back(operation->GetOpMagic());
        }
    }
    report["maxProducers"] = maxProducers;
    report["maxProducersTensors"] = maxProducersTensors;
    report["maxConsumers"] = maxConsumers;
    report["maxConsumersTensors"] = maxConsumersTensors;
    report["maxInputs"] = maxInputs;
    report["maxInputsOps"] = maxInputsOps;
    report["maxOutputs"] = maxOutputs;
    report["maxOutputsOps"] = maxOutputsOps;
}

Status OoOScheduleStatistic::DoHealthCheck(Function* function, const std::string& fileName)
{
    if (HealthCheckOoOSchedule() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "DoHealthCheck failed at HealthCheckOoOSchedule!");
        return FAILED;
    }
    HealthCheckBlockGraph(function);
    std::ofstream file(fileName);
    file << report.dump(1) << std::endl;
    file.close();
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
