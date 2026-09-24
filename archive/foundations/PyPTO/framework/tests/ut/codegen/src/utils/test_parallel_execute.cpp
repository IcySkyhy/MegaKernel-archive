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
 * \file test_parallel_execute.cpp
 * \brief Unit test for ParallelExecuteAndWait exception propagation.
 */

#include <atomic>
#include <chrono>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "codegen/utils/parallel_execute.h"

namespace npu::tile_fwk {
namespace {

std::deque<Task> MakeIncrementTasks(int count, std::atomic<int>& counter)
{
    std::deque<Task> tasks;
    for (int i = 0; i < count; ++i) {
        tasks.emplace_back([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    return tasks;
}

} // namespace

TEST(TestParallelExecute, SingleThreadRunsAllTasks)
{
    std::atomic<int> counter{0};
    ParallelExecuteAndWait(1, MakeIncrementTasks(5, counter));
    EXPECT_EQ(counter.load(), 5);
}

TEST(TestParallelExecute, MultiThreadRunsAllTasks)
{
    std::atomic<int> counter{0};
    ParallelExecuteAndWait(4, MakeIncrementTasks(20, counter));
    EXPECT_EQ(counter.load(), 20);
}

TEST(TestParallelExecute, ZeroThreadNumTreatedAsOne)
{
    std::atomic<int> counter{0};
    ParallelExecuteAndWait(0, MakeIncrementTasks(3, counter));
    EXPECT_EQ(counter.load(), 3);
}

TEST(TestParallelExecute, EmptyTaskQueueCompletes)
{
    std::deque<Task> tasks;
    EXPECT_NO_THROW(ParallelExecuteAndWait(4, std::move(tasks)));
}

TEST(TestParallelExecute, SingleThreadPropagatesStdException)
{
    std::deque<Task> tasks;
    tasks.emplace_back([]() { throw std::runtime_error("single thread boom"); });
    EXPECT_THROW(ParallelExecuteAndWait(1, std::move(tasks)), std::runtime_error);
}

TEST(TestParallelExecute, SingleThreadPreservesExceptionMessage)
{
    const std::string msg = "specific parallel task error";
    std::deque<Task> tasks;
    tasks.emplace_back([&msg]() { throw std::runtime_error(msg); });

    try {
        ParallelExecuteAndWait(1, std::move(tasks));
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), msg.c_str());
    }
}

TEST(TestParallelExecute, SingleThreadStopsSubsequentTasksOnError)
{
    std::atomic<int> counter{0};
    std::deque<Task> tasks;
    tasks.emplace_back([]() { throw std::runtime_error("fail early"); });
    tasks.emplace_back([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });

    EXPECT_THROW(ParallelExecuteAndWait(1, std::move(tasks)), std::runtime_error);
    EXPECT_EQ(counter.load(), 0);
}

TEST(TestParallelExecute, SingleThreadPropagatesUnknownException)
{
    std::deque<Task> tasks;
    tasks.emplace_back([]() { throw 42; });

    bool caught = false;
    try {
        ParallelExecuteAndWait(1, std::move(tasks));
    } catch (...) {
        caught = true;
    }
    EXPECT_TRUE(caught);
}

TEST(TestParallelExecute, MultiThreadPropagatesStdException)
{
    std::deque<Task> tasks;
    tasks.emplace_back([]() { throw std::runtime_error("multi thread boom"); });
    tasks.emplace_back([]() {});
    EXPECT_THROW(ParallelExecuteAndWait(4, std::move(tasks)), std::runtime_error);
}

TEST(TestParallelExecute, MultiThreadPropagatesUnknownException)
{
    std::deque<Task> tasks;
    tasks.emplace_back([]() { throw 42; });
    tasks.emplace_back([]() {});

    bool caught = false;
    try {
        ParallelExecuteAndWait(4, std::move(tasks));
    } catch (...) {
        caught = true;
    }
    EXPECT_TRUE(caught);
}

TEST(TestParallelExecute, MultiThreadExceptionPropagatesAfterWorkersJoin)
{
    std::atomic<int> startedAfterFailure{0};
    std::deque<Task> tasks;
    tasks.emplace_back([]() { throw std::runtime_error("trigger stop"); });
    for (int i = 0; i < 8; ++i) {
        tasks.emplace_back([&startedAfterFailure]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            startedAfterFailure.fetch_add(1, std::memory_order_relaxed);
        });
    }

    EXPECT_THROW(ParallelExecuteAndWait(4, std::move(tasks)), std::runtime_error);
}

} // namespace npu::tile_fwk
