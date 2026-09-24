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
 * \file PipeMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <deque>
#include <vector>
#include <climits>
#include <unordered_map>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/config/PipeConfig.h"
#include "Scheduler.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/config/CoreConfig.h"
#include "cost_model/simulation/statistics/CoreStats.h"
#include "cost_model/simulation/arch/PipeMachineImpl.h"

namespace CostModel {
class PipeMachine : public Machine {
public:
    CorePipeType pipeType = CorePipeType::TOTAL_CORE_PIPE_TYPE;
    int pipeId = -1;
    int magic = -1;
    uint64_t executingTaskId = 0;
    uint64_t retireCycle = 0;
    UnifiedPipeMachinePtr pipeImpl;
    PipeMachine();
    PipeMachine(CostModel::MachineType mType, CostModel::CorePipeType pType, int pId);
    PipeConfig config;
    std::shared_ptr<CoreStats> stats = nullptr;

    TileOpPtr tileOp = nullptr;
    TilePtr tile = nullptr;
    bool waitL2CacheResponse = false;

    void Step() override;
    void Xfer() override;
    void Build() override;
    void Reset() override;
    std::shared_ptr<SimSys> GetSim() override;
    void Report() override;
    void InitQueueDelay() override;
    void StepQueue() override;
    bool IsTerminate() override;

    void RunAtBegin();
    void RunAtEnd();
    void ReceivePacket();
    void ReceiveL2Packet();
    void ProcessTileOp();
    void SendCachePacket(bool read);
    void PushCompletion(int taskId, int curMagic);
    uint64_t GetQueueNextCycles();
    void LoggerRecordPipeWL(size_t pId, CounterType type);
};
} // namespace CostModel
