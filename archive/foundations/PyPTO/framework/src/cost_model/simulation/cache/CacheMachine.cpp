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
 * \file CacheMachine.cpp
 * \brief
 */

#include "cost_model/simulation/cache/CacheMachine.h"

#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/arch/PipeFactory.h"

namespace CostModel {
CacheMachine::CacheMachine(CacheType type, std::string aType)
{
    machineType = MachineType::CACHE;
    cacheType = type;
    cacheImpl = PipeFactory::CreateCache(CacheType::L2CACHE, aType);
}

void CacheMachine::Step()
{
    RunAtBegin();
    ReceivePacket();
    ProcessMSHR();
    ProcessResp();
}

void CacheMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    stats = std::make_shared<CacheStats>(GetSim()->GetReporter());
    InitQueueDelay();
}

void CacheMachine::Reset() {}

void CacheMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();

    // Updat next simulation cycles
    nextCycles = INT_MAX;
    nextCycles = std::min(nextCycles, GetQueueNextCycles());
    for (auto iter = misses.begin(), end = misses.end(); iter != end;) {
        auto& mshr = iter->second;
        nextCycles = std::min(nextCycles, mshr.readyCycle);
        iter++;
    }
    if (!responseQueue.empty()) {
        nextCycles = std::min(nextCycles, responseQueue.front().second);
    }
    GetSim()->UpdateNextCycles(nextCycles);
}

std::shared_ptr<SimSys> CacheMachine::GetSim() { return sim; }

void CacheMachine::Report()
{
    std::string name = "L2";
    stats->Report(name);
}

void CacheMachine::RunAtBegin() { nextCycles = INT_MAX; }

void CacheMachine::InitQueueDelay()
{
    submissionQueue.SetWriteDelay(0);
    submissionQueue.SetReadDelay(0);
    completionQueue.SetWriteDelay(0);
    completionQueue.SetReadDelay(0);
    outcastReferenceQueue.SetWriteDelay(0);
    outcastReferenceQueue.SetReadDelay(0);
    incastReferenceQueue.SetWriteDelay(0);
    incastReferenceQueue.SetReadDelay(0);
    releaseQueue.SetWriteDelay(0);
    releaseQueue.SetReadDelay(0);
    cacheRespQueue.SetWriteDelay(0);
    cacheRespQueue.SetReadDelay(0);
    dataRequestQueue.SetWriteDelay(0);
    dataRequestQueue.SetReadDelay(0);
}

void CacheMachine::StepQueue()
{
    uint64_t intervalCycles = GetSim()->GetCycles() - lastCycles;
    dataRequestQueue.UpdateIntervalCycles(intervalCycles);
    dataRequestQueue.Step();
}

uint64_t CacheMachine::GetQueueNextCycles()
{
    uint64_t res = INT_MAX;
    uint64_t gCycles = GetSim()->GetCycles();
    res = std::min(res, gCycles + dataRequestQueue.GetMinWaitCycles());
    return res;
}

bool CacheMachine::IsTerminate() { return (!executingTask && dataRequestQueue.IsTerminate() && responseQueue.empty()); }

void CacheMachine::ReceivePacket()
{
    if (dataRequestQueue.Empty() || executingTask) {
        return;
    }
    CachePacket packet;
    dataRequestQueue.Dequeue(packet);
    auto addr = packet.addr;
    auto curCycle = GetSim()->GetCycles();
    packet.cycleInfo.cacheRecvCycle = curCycle;
    if (packet.requestType == CacheRequestType::DATA_READ_REQ) {
        stats->totalReadNum++;
    } else if (packet.requestType == CacheRequestType::DATA_WRITE_REQ) {
        stats->totalWriteNum++;
    }

    auto hit = AccessCache(addr, curCycle);
    if (hit) {
        // Simply put into the response queue with some hit latency.
        auto simCycle = cacheImpl->Simulate(packet);
        responseQueue.emplace_back(packet, curCycle + simCycle + config.l2HitLatency);
    } else {
        // We need to allocate MSHR.
        AddMSHR(packet, curCycle);
    }
}

void CacheMachine::ProcessMSHR()
{
    auto curCycle = GetSim()->GetCycles();
    // Iterate through all misses.
    for (auto iter = misses.begin(), end = misses.end(); iter != end;) {
        auto& mshr = iter->second;
        if (mshr.readyCycle <= curCycle) {
            // We are ready.
            for (const auto& req : mshr.inflyMisses) {
                uint64_t cycle = curCycle + config.l2HitLatency;
                responseQueue.emplace_back(req, cycle);
            }
            iter = misses.erase(iter);
        } else {
            ++iter;
        }
    }
}

void CacheMachine::ProcessResp()
{
    auto curCycle = GetSim()->GetCycles();
    // Iterate through all misses.
    while (!responseQueue.empty() && responseQueue.front().second <= curCycle) {
        // We have a response to send.
        auto& req = responseQueue.front().first;
        auto machine = GetSim()->pidToMachineMp.at(req.pid);
        req.cycleInfo.cacheRespCycle = GetSim()->GetCycles();
        stats->totalResponseLatency += (req.cycleInfo.cacheRespCycle - req.cycleInfo.cacheRecvCycle);
        machine->cacheRespQueue.Enqueue(req);

        responseQueue.pop_front();
    }
}

void CacheMachine::AddMSHR(const CachePacket& req, uint64_t curCycle)
{
    auto addr = req.addr;
    auto iter = misses.find(addr);
    if (iter != misses.end()) {
        iter->second.inflyMisses.emplace_back(req);
        return;
    }

    // Add new MSHR.
    misses.emplace(std::piecewise_construct, std::forward_as_tuple(addr),
                   std::forward_as_tuple(addr, curCycle + config.l2MissExtraLatency, req));
}

void CacheMachine::AllocateCache(uint64_t addr, uint64_t currentCycle)
{
    // Check if we have to evict.
    if (cache.size() >= config.l2Size / config.l2LineSize) {
        EvictLRU();
    }
    stats->totalInsertNum++;
    auto& entry = cache.emplace(addr, CacheEntry()).first->second;
    entry.addr = addr;
    entry.lastAccess = currentCycle;
    lru.push_back(addr);
    entry.lruPos = lru.end();
    --entry.lruPos;
}

bool CacheMachine::AccessCache(uint64_t addr, uint64_t currentCycle)
{
    stats->totalQueryNum++;
    auto iter = cache.find(addr);
    if (iter == cache.end()) {
        stats->totalMissNum++;
        return false;
    }
    // Hit in cache.
    stats->totalHitNum++;
    auto& entry = iter->second;
    entry.lastAccess = currentCycle;
    lru.erase(entry.lruPos);
    lru.push_back(addr);
    entry.lruPos = lru.end();
    --entry.lruPos;
    return true;
}

void CacheMachine::EvictLRU()
{
    stats->totalInsertNum++;
    auto addr = lru.front();
    lru.pop_front();
    cache.erase(addr);
}

void CacheMachine::RequestData(CachePacket pkt, uint64_t extraDelay)
{
    lastCycles = GetSim()->GetCycles();
    dataRequestQueue.Enqueue(pkt, extraDelay);
}
} // namespace CostModel
