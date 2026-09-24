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
 * \file latency_estimator.h
 * \brief
 */

#ifndef PASS_ESTIMATE_LATENCY_H
#define PASS_ESTIMATE_LATENCY_H

#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include <vector>

namespace npu::tile_fwk {
class LatencyEstimator {
public:
    LatencyEstimator() {}
    LatencyEstimator(std::vector<Operation*> newTaskList, std::vector<Operation*> newOperations)
        : taskList(std::move(newTaskList)), operations(std::move(newOperations))
    {
        InitMemWithoutAlloc();
        state_.Init(taskList);
    }
    ~LatencyEstimator() {}

    ScheduleState state_;

    std::map<MemoryType, OpQueue> allocIssueQueue; // alloc执行 顺序
    std::map<PipeType, OpQueue> opQueues;
    std::map<Operation*, bool> opRetiredInfo;
    std::vector<Operation*> taskList;
    std::vector<Operation*> operations;
    std::set<int> spillblockMemIds;

    void LaunchReadyIssue();
    Status FreeBuffer(Operation* op);
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status PreMainLoop();
    Status PostMainLoop();
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, OpQueue& pipe);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status LaunchIssueStage(int& nextCycle);
    Status SpillOnBlock();
    void initLatencyEstimatorOpQueues();
    void InitMemWithoutAlloc();
    Status LatencyEstimatorMainLoop();
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_BASE_H
