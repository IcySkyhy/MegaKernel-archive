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
 * \file BaseQueue.h
 * \brief
 */

#pragma once
#ifndef QUEUE_H
#define QUEUE_H

#include <climits>
#include <queue>
#include <iostream>
#include "cost_model/simulation/base/SimObj.h"
#include "cost_model/simulation/statistics/TraceLogger.h"

namespace CostModel {

template <typename T>
class SimQueue : public SimObj {
private:
    uint64_t maxQueueSize = 1024;
    uint64_t tick = 0;
    uint64_t wDelay = 0;
    uint64_t rDelay = 0;
    std::deque<T> wQueue;
    std::deque<T> rQueue;
    std::deque<uint64_t> wTimeQueue;
    std::deque<uint64_t> rTimeQueue;
    uint64_t rWaitLatency = INT_MAX;
    uint64_t wWaitLatency = INT_MAX;
    uint64_t intervalCycles = 1;

    std::shared_ptr<TraceLogger> mLogger = nullptr;
    CostModel::Pid pid = 0;
    CostModel::Tid tid = 0;

public:
    SimQueue() = default;
    ~SimQueue() override = default;
    void Build() override { Reset(); }
    void Xfer() override {}

    void UpdateIntervalCycles(uint64_t deltaCycles) { intervalCycles = deltaCycles; }

    void Step() override
    {
        if (wTimeQueue.empty() && rTimeQueue.empty()) {
            tick = 0;
            rWaitLatency = INT_MAX;
            wWaitLatency = INT_MAX;
        } else {
            tick += intervalCycles;
        }
        while (!wTimeQueue.empty() && tick >= wTimeQueue.front()) {
            rQueue.push_back(std::move(wQueue.front()));
            rTimeQueue.push_back(tick + rDelay);
            wQueue.pop_front();
            wTimeQueue.pop_front();
        }
        if (!wTimeQueue.empty()) {
            wWaitLatency = wTimeQueue.front() - tick;
        }
        if (!rTimeQueue.empty()) {
            rWaitLatency = rTimeQueue.front() - tick;
        }
    }

    void Reset() override
    {
        wQueue.clear();
        rQueue.clear();
        wTimeQueue.clear();
        rTimeQueue.clear();
    }
    std::shared_ptr<SimSys> GetSim() override { return nullptr; }

    // Disable copy and assignment
    SimQueue(const SimQueue&) = delete;
    SimQueue& operator=(const SimQueue&) = delete;

    uint64_t GetMinWaitCycles()
    {
        uint64_t res = std::min(rWaitLatency, wWaitLatency);
        if (res == 0) {
            return 1;
        }
        return res;
    }

    // Enqueue an element into the queue
    void Enqueue(T element, uint64_t extraDelay = 0)
    {
        wQueue.push_back(std::move(element));
        wTimeQueue.push_back(tick + extraDelay + wDelay);
        this->LoggerRecordQueueEvent(CounterType::QUEUE_PUSH);
    }

    // Dequeue an element from the queue
    bool Dequeue(T& element)
    {
        if (rQueue.empty()) {
            return false;
        }
        this->LoggerRecordQueueEvent(CounterType::QUEUE_POP);
        element = std::move(rQueue.front());
        rQueue.pop_front();
        rTimeQueue.pop_front();
        return true;
    }

    // Read the front from the queue
    bool Front(T& element)
    {
        if (rQueue.empty()) {
            return false;
        }
        element = rQueue.front();
        return true;
    }

    // Pop front from the queue
    bool PopFront()
    {
        if (rQueue.empty()) {
            return false;
        }
        rQueue.pop_front();
        rTimeQueue.pop_front();
        return true;
    }

    size_t Size() const { return rQueue.size(); }

    size_t WriteQueueSize() const { return wQueue.size(); }

    // Check if the queue is empty
    bool Empty() const { return (rTimeQueue.empty() || tick < rTimeQueue.front()); }

    // Check if the queue is full
    bool Full() const { return wTimeQueue.size() >= maxQueueSize; }

    void SetMaxSize(uint64_t maxSize) { maxQueueSize = maxSize; }

    void SetWriteDelay(uint64_t delay) { wDelay = delay; }

    void SetReadDelay(uint64_t delay) { rDelay = delay; }

    bool IsTerminate() { return (wQueue.empty() && rQueue.empty()); }

    void SetCounterInfo(std::shared_ptr<TraceLogger> logger, CostModel::Pid pId, CostModel::Tid tId)
    {
        mLogger = logger;
        pid = pId;
        tid = tId;
    }
    void LoggerRecordQueueEvent(CounterType type)
    {
        if (mLogger != nullptr) {
            mLogger->AddCounterEvent(pid, tid, type);
        }
    }
    // Pop front from the queue
    int CalendarPopFront()
    {
        if (rQueue.empty()) {
            return false;
        }
        int number = rQueue.front();
        rQueue.pop_front();
        rTimeQueue.pop_front();
        return number;
    }
};

} // namespace CostModel

#endif
