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
 * \file thread_pool.h
 * \brief
 */

#pragma once

#include <atomic>
#include <vector>
#include <thread>
#include <deque>
#include <condition_variable>
#include <functional>

namespace npu::tile_fwk {
namespace util {

class ThreadPool {
    struct Task {
        Task() {};
        Task(void* ctx, void (*entry)(void*)) : ctx_(ctx), entry_(entry) {}

        void Run() { entry_(ctx_); }

        bool Empty() const { return ctx_ == nullptr && entry_ == nullptr; }

    private:
        void* ctx_ = nullptr;
        void (*entry_)(void* ctx) = nullptr;
    };

    struct TaskQueue {
        bool Empty()
        {
            std::lock_guard<std::mutex> guard(taskListMutex_);
            return taskList_.empty();
        };
        void Push(const Task& t)
        {
            std::lock_guard<std::mutex> guard(taskListMutex_);
            taskList_.push_back(t);
        }
        Task Pop()
        {
            std::lock_guard<std::mutex> guard(taskListMutex_);
            Task t;
            if (!taskList_.empty()) {
                t = taskList_.front();
                taskList_.pop_front();
            }
            return t;
        }

    private:
        std::deque<Task> taskList_;
        std::mutex taskListMutex_;
    };

private:
    enum class State {
        T_STARTING,
        T_WAITING,
        T_RUNNING,
        T_JOINING,
    };

public:
    ThreadPool(int threadCount) : threadCount_(threadCount), stateList_(threadCount_)
    {
        for (int i = 0; i < threadCount; i++) {
            stateList_[i] = static_cast<int>(State::T_STARTING);
        }
        for (int i = 0; i < threadCount; i++) {
            threadList_.emplace_back([this, i]() { this->Run(i); });
        }
    }
    ~ThreadPool()
    {
        Stop();
        for (int i = 0; i < threadCount_; i++) {
            threadList_[i].join();
        }
    }
    void SubmitTask(void* ctx, void (*entry)(void* ctx)) { taskQueue_.Push(Task(ctx, entry)); }

    void NotifyAll() { taskReadyNotifier_.notify_all(); }

    static void Yield() { std::this_thread::sleep_for(std::chrono::milliseconds(0)); }

    void WaitForAll()
    {
        waiting_ = true;
        while (!taskQueue_.Empty()) {
            Yield();
        }
        for (int i = 0; i < threadCount_; i++) {
            while (stateList_[i] != static_cast<int>(State::T_WAITING)) {
                Yield();
            }
        }
        waiting_ = false;
    }

    void Stop()
    {
        waiting_ = true;
        stopped_ = true;
        while (!taskQueue_.Empty()) {
            Yield();
        }
        for (int i = 0; i < threadCount_; i++) {
            while (stateList_[i] != static_cast<int>(State::T_JOINING)) {
                taskReadyNotifier_.notify_all();
                Yield();
            }
        }
    }

    int GetThreadCount() const { return threadCount_; }

private:
    void Run(int threadIndex)
    {
        stateList_[threadIndex] = static_cast<int>(State::T_STARTING);
        while (!stopped_) {
            stateList_[threadIndex] = static_cast<int>(State::T_WAITING);
            {
                std::unique_lock<std::mutex> lk(taskReadyNotifierMutex_);
                taskReadyNotifier_.wait(lk, [this] { return stopped_ || !taskQueue_.Empty(); });
            }
            if (stopped_)
                break;
            stateList_[threadIndex] = static_cast<int>(State::T_RUNNING);
            while (true) {
                auto task = taskQueue_.Pop();
                if (task.Empty()) {
                    if (waiting_) {
                        break;
                    } else {
                        Yield();
                        continue;
                    }
                } else {
                    task.Run();
                }
            }
        }
        stateList_[threadIndex] = static_cast<int>(State::T_JOINING);
    }

    std::atomic<bool> stopped_{false};
    std::atomic<bool> waiting_{false};
    std::condition_variable taskReadyNotifier_;
    std::mutex taskReadyNotifierMutex_;
    int threadCount_;
    std::vector<std::atomic<int>> stateList_;
    std::vector<std::thread> threadList_;
    TaskQueue taskQueue_;
};

} // namespace util
} // namespace npu::tile_fwk
