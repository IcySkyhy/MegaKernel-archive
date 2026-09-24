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
 * \file CacheMachine.h
 * \brief
 */

#pragma once

#include <list>
#include <unordered_map>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/config/PipeConfig.h"
#include "cost_model/simulation/machine/Scheduler.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/config/CacheConfig.h"
#include "cost_model/simulation/statistics/CacheStats.h"
#include "cost_model/simulation/arch/CacheMachineImpl.h"

namespace CostModel {
class CacheMachine : public Machine {
public:
    SimQueue<CachePacket> dataRequestQueue;
    std::unique_ptr<CacheMachineImpl> cacheImpl;
    CacheType cacheType = CacheType::TOTAL_CACHE_TYPE;

    CacheMachine(CacheType type, std::string aType);

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
    void ReceivePacket();
    uint64_t GetQueueNextCycles();
    void RequestData(CachePacket pkt, uint64_t extraDelay = 0);

    CacheConfig config;
    std::shared_ptr<CacheStats> stats = nullptr;

private:
    // Miss Status Holding Register
    struct MSHR {
        uint64_t addr;
        uint64_t readyCycle;
        std::list<CachePacket> inflyMisses;
        MSHR(uint64_t address, uint64_t rdyCycle, const CachePacket& req) : addr(address), readyCycle(rdyCycle)
        {
            inflyMisses.emplace_back(req);
        }
    };
    std::unordered_map<uint64_t, MSHR> misses;
    void AddMSHR(const CachePacket& req, uint64_t curCycle);
    void ProcessMSHR();

    using LRUIter = std::list<uint64_t>::iterator;
    struct CacheEntry {
        uint64_t addr;
        uint64_t lastAccess;
        LRUIter lruPos;
    };
    std::unordered_map<uint64_t, CacheEntry> cache;
    std::list<uint64_t> lru;
    void AllocateCache(uint64_t addr, uint64_t currentCycle);
    bool AccessCache(uint64_t addr, uint64_t currentCycle);
    void EvictLRU();

    // Hold the pipeline to the response.
    std::list<std::pair<CachePacket, uint64_t>> responseQueue;
    void ProcessResp();
};
} // namespace CostModel
