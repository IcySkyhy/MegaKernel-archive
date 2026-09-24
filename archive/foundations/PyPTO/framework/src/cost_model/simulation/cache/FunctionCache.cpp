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
 * \file FunctionCache.cpp
 * \brief
 */

#include "cost_model/simulation/cache/FunctionCache.h"

#include <utility>
#include "cost_model/simulation/base/ModelTop.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {

void FunctionCache::Insert(FunctionPtr func)
{
    // Key is Function hash
    uint64_t index = func->functionHash;
    funcNameToKey[func->funcName] = index;
    cache[index] = func;
    funcLastUseTime[index] = ++cacheTblTime;
    inCacheFunction.insert({funcLastUseTime[index], index});
    while (uint64_t(inCacheFunction.size()) > maxCacheSize) {
        inCacheFunction.erase(inCacheFunction.begin());
    }
}

void FunctionCache::CountFunctionCache(uint64_t key, CostModel::Pid pid, CostModel::Tid tid, bool hit)
{
    if (pid != LLONG_MAX) {
        if (hit) {
            GetSim()->GetLogger()->AddCounterEvent(pid, tid, CostModel::CounterType::CACHE_HIT);
        } else {
            GetSim()->GetLogger()->AddCounterEvent(pid, tid, CostModel::CounterType::CACHE_MISS);

            SIMULATION_LOGI(
                "[Cycle: %lu][CoreMachine][ReceivePacket] CoreMachine: %lu Function Not Exist In Function Cache.",
                GetSim()->GetCycles(), key);
        }
    }
}

bool FunctionCache::Lookup(uint64_t key, CostModel::Pid pid, CostModel::Tid tid)
{
    bool hit = !inCacheFunction.empty() && funcLastUseTime[key] >= inCacheFunction.begin()->first;
    if (hit) {
        inCacheFunction.erase({funcLastUseTime[key], key});
    }
    funcLastUseTime[key] = ++cacheTblTime;
    inCacheFunction.insert({funcLastUseTime[key], key});
    while (uint64_t(inCacheFunction.size()) > maxCacheSize) {
        inCacheFunction.erase(inCacheFunction.begin());
    }
    CountFunctionCache(key, pid, tid, hit);
    return hit;
}

FunctionPtr FunctionCache::GetFunction(uint64_t key) { return cache.at(key); }

std::shared_ptr<SimSys> FunctionCache::GetSim() { return sim; }

void FunctionCache::SetSim(std::shared_ptr<CostModel::SimSys> simPtr) { sim = std::move(simPtr); }

void FunctionCache::SetMaxCacheSize(uint64_t cacheSize) { maxCacheSize = cacheSize; }
} // namespace CostModel
