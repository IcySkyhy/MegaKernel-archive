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
 * \file parallel_execute.cpp
 * \brief
 */

#include "parallel_execute.h"

#include <cstddef>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace npu::tile_fwk {
namespace {
class ThreadSafeIndexQueue {
public:
    explicit ThreadSafeIndexQueue(std::deque<size_t> indices) { q_ = std::move(indices); }

    std::optional<size_t> GetIndex()
    {
        const std::lock_guard<std::mutex> taskLock(m_);

        if (q_.empty()) {
            return std::nullopt;
        }

        size_t index = q_.front();
        q_.pop_front();
        return index;
    }

private:
    std::deque<size_t> q_;
    std::mutex m_;
};

using PackagedTask = std::packaged_task<void()>;

struct TaskBatch {
    std::vector<PackagedTask> packagedTasks;
    std::vector<std::future<void>> futures;

    explicit TaskBatch(std::deque<Task> tasks)
    {
        packagedTasks.reserve(tasks.size());
        futures.reserve(tasks.size());
        while (!tasks.empty()) {
            packagedTasks.emplace_back(std::move(tasks.front()));
            tasks.pop_front();
            futures.push_back(packagedTasks.back().get_future());
        }
    }
};

void TaskRunner(ThreadSafeIndexQueue& indexQueue, TaskBatch& batch)
{
    while (true) {
        auto indexMaybe = indexQueue.GetIndex();
        if (!indexMaybe) {
            break;
        }

        batch.packagedTasks[indexMaybe.value()]();
    }
}

void WaitTaskFutures(std::vector<std::future<void>>& futures)
{
    for (auto& future : futures) {
        if (future.valid()) {
            future.get();
        }
    }
}
} // namespace

void ParallelExecuteAndWait(unsigned threadNum, std::deque<Task> tasks)
{
    if (threadNum == 0) {
        threadNum = 1;
    }

    TaskBatch batch(std::move(tasks));

    if (threadNum == 1) {
        for (size_t i = 0; i < batch.packagedTasks.size(); ++i) {
            batch.packagedTasks[i]();
            batch.futures[i].get();
        }
        return;
    }

    std::deque<size_t> indices;
    for (size_t i = 0; i < batch.packagedTasks.size(); ++i) {
        indices.push_back(i);
    }

    ThreadSafeIndexQueue indexQueue(std::move(indices));

    std::vector<std::thread> threadPool;
    threadPool.reserve(threadNum);
    for (unsigned i = 0; i < threadNum; i++) {
        threadPool.emplace_back(TaskRunner, std::ref(indexQueue), std::ref(batch));
    }

    for (auto& worker : threadPool) {
        worker.join();
    }

    WaitTaskFutures(batch.futures);
}
} // namespace npu::tile_fwk
