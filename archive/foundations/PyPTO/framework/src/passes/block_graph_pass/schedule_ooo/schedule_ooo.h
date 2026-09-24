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
 * \file schedule_ooo.h
 * \brief
 */

#ifndef PASS_SCHEDULE_OOO_H
#define PASS_SCHEDULE_OOO_H

#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/common/iso_matcher.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/ooo_scheduler.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/remove_alloc.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/add_alloc.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/core_assign.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/latency_estimator.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/optimize_sort.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/task_splitter.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/statistics/memory_tracer.h"
#include "passes/statistics/ooo_schedule_statistic.h"

namespace npu::tile_fwk {
struct ScheduleUnit {
    int earliestStartTime;
    std::vector<Operation*> mergedOps;
    bool isMerged{false};
};

const std::unordered_map<TargetCoreType, CoreLocationType> targetCoreTypeMap{
    {TargetCoreType::AIC, CoreLocationType::AIC},
    {TargetCoreType::AIV0, CoreLocationType::AIV0},
    {TargetCoreType::AIV1, CoreLocationType::AIV1}};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_ONE_AIV = {CoreLocationType::AIC,
                                                                                 CoreLocationType::AIV0};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_TWO_AIV = {
    CoreLocationType::AIC, CoreLocationType::AIV0, CoreLocationType::AIV1};

class OoOSchedule : public Pass {
public:
    OoOSchedule() : Pass("OoOSchedule") {}
    ~OoOSchedule() override {}

private:
    Status RunOnFunction(Function& function) override;
    bool IsAicpuProgram(std::vector<Operation*> opList);
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    void DoHealthCheckAfter(Function& function, const std::string& folderPath) override;
    void CollectStatistic(OoOScheduleStatistic& oooHealthCheck, Function& function,
                          std::pair<uint64_t, Function*>& program);
    Status TaskSchedule(std::vector<Operation*>& opList, Function& function, TaskSplitter& splitter);
    Status ConcatTaskOpLists(TaskSplitter& splitter, std::vector<Operation*>& newOpList, Function& function);
    void StableUnique(std::vector<Operation*>& newOpList);
    Status ModifyTaskOplist(std::vector<Operation*>& taskList, const std::unordered_map<int, Operation*>& allocMap);
    Status DoOoOSchedule(std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
                         int64_t& workeSpaceSize, bool enableDualDst);
    Status Schedule(std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
                    int64_t& maxWorkeSpaceSize);
    void CollectLocalMemIds(Operation* op, std::vector<int>& memIds);
    Status ModifyAllocOrder(std::vector<Operation*>& opList);
    std::vector<ScheduleUnit> BuildScheduleUnits(const std::vector<TaskNode>& taskNodeList,
                                                 const std::vector<std::pair<int, int>>& cyclePairs);
    void CollectAivTasksByStart(const std::vector<TaskNode>& tasks,
                                std::unordered_map<int, std::vector<const TaskNode*>>& aiv0ByStart,
                                std::unordered_map<int, std::vector<const TaskNode*>>& aiv1ByStart);
    size_t CountNonAllocOps(const std::vector<Operation*>& ops);
    bool CheckAndRecordDualDstPair(int start, const TaskNode* ta, const TaskNode* tb);
    bool ShouldEnableDualDst(TaskSplitter& splitter);
    bool MergeIsoPairs(const std::vector<IsoPair>& pairs, std::unordered_map<Operation*, Operation*>& target);
    std::unordered_map<Operation*, Operation*> dualDstPairs_;   // AIV0 alloc -> AIV1 alloc
    std::unordered_map<Operation*, Operation*> dualDstOpPairs_; // AIV0 op -> AIV1 op, non-alloc
    std::vector<Function*> oriFunctions;
    std::map<uint64_t, OoOScheduleStatistic> statisticMap_;
    // Per-program tracers, populated on SUCCESS only; failure path flushes inline.
    std::map<uint64_t, MemoryTracer> tracerMap_;
    void CollectMemoryTrace(MemoryTracer& tracer, Function& function, std::pair<uint64_t, Function*>& program);
    void FlushMemoryTraceOnFailure(MemoryTracer& tracer, Function& function, std::pair<uint64_t, Function*>& program);
    Status CheckAllocOp(std::vector<Operation*> list);
    void UpdateAllocMap(Operation* op, std::map<int, Operation*>& allocMap);
    OoOScheduleChecker checker;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_OOO_H
