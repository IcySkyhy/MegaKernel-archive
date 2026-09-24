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
 * \file ooo_schedule_statistic.h
 * \brief
 */

#ifndef OOO_SCHEDULE_STATISTIC_H
#define OOO_SCHEDULE_STATISTIC_H

#include <nlohmann/json.hpp>
#include "tilefwk/platform.h"
#include "interface/function/function.h"
#include "passes/statistics/schedule_observer.h"

using Json = nlohmann::json;

namespace npu {
namespace tile_fwk {
class OoOScheduleStatistic : public ScheduleObserver {
public:
    // ScheduleObserver overrides
    void OnOpLaunch(const OpLaunchEvent& e) override;
    void OnOpRetire(const OpRetireEvent& e) override;
    void OnAllocExec(const AllocExecEvent& e) override;
    void OnSpill(const SpillEvent& e) override;
    void OnScheduleEnd(const ScheduleEndEvent& e) override;
    // OnBufferRearrange / OnAllocFail intentionally use inherited no-op defaults.

    void SetOutputPrefix(const std::string& prefix) { jsonFileName = prefix; }

    Status DoHealthCheck(Function* function, const std::string& fileName);
    void HealthCheckSpillInfo();
    double FormatUsageRate(double value);
    Status HealthCheckOoOSchedule();
    void ReportPipeUsage();
    void ReportMemoryUsage(const std::unordered_map<MemoryType, int64_t>& memorySize);
    void HealthCheckBlockGraph(Function* function);
    std::string jsonFileName;
    int64_t workspaceOffset{0};
    int clock{0};
    std::unordered_map<PipeType, uint64_t> pipeUsageCount = {
        {PipeType::PIPE_S, 0},    {PipeType::PIPE_V, 0},    {PipeType::PIPE_M, 0},  {PipeType::PIPE_MTE1, 0},
        {PipeType::PIPE_MTE2, 0}, {PipeType::PIPE_MTE3, 0}, {PipeType::PIPE_FIX, 0}};
    std::unordered_map<MemoryType, uint64_t> bufferTotalUsage = {{MemoryType::MEM_UB, 0},
                                                                 {MemoryType::MEM_L1, 0},
                                                                 {MemoryType::MEM_L0A, 0},
                                                                 {MemoryType::MEM_L0B, 0},
                                                                 {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> bufferMaxUsage = {{MemoryType::MEM_UB, 0},
                                                               {MemoryType::MEM_L1, 0},
                                                               {MemoryType::MEM_L0A, 0},
                                                               {MemoryType::MEM_L0B, 0},
                                                               {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> bufferLastUsage = {{MemoryType::MEM_UB, 0},
                                                                {MemoryType::MEM_L1, 0},
                                                                {MemoryType::MEM_L0A, 0},
                                                                {MemoryType::MEM_L0B, 0},
                                                                {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> lastClock = {{MemoryType::MEM_UB, 0},
                                                          {MemoryType::MEM_L1, 0},
                                                          {MemoryType::MEM_L0A, 0},
                                                          {MemoryType::MEM_L0B, 0},
                                                          {MemoryType::MEM_L0C, 0}};
    struct SpillInfo {
        MemoryType spillType{MemoryType::MEM_UNKNOWN}; // spill buffer类型
        uint64_t bufferCurrUsage{0};                   // spill时该pool的实时占用量(现场采自buffer pool)
        uint64_t bufferCapacity{0};                    // spill时该pool的容量
        uint64_t spillTensorSize{0};                   // spill的tensor的大小
        int64_t triggerTensorSize{0};                  // 触发当前spill的tensor的大小
        uint64_t allocOccupiedSize{0};                 // 当前被alloc占用的buffer大小
        uint64_t spillCopyoutSize{0};                  // spill到ddr的数据量
        int spillTensorMagic;                          // spill tensor的magic
    };
    std::vector<SpillInfo> spillInfoVec; // size为spill的次数
    Json report;

private:
    // memId → (memType, size); OnOpRetire/OnSpill only get memIds, look up here.
    struct BufferMeta {
        MemoryType memType;
        uint64_t size;
    };
    std::unordered_map<int, BufferMeta> bufferMeta_;
};

} // namespace tile_fwk
} // namespace npu
#endif // OOO_SCHEDULE_STATISTIC_H
