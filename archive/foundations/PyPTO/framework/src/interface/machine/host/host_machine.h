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
 * \file host_machine.h
 * \brief
 */

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <cstdint>
#include <tuple>
#include <nlohmann/json.hpp>
#include "interface/machine/host/machine_task.h"
#include "interface/configs/config_manager.h"
#include "interface/cache/function_cache.h"

namespace npu::tile_fwk {
enum class HostMachineMode {
    SERVER = 0, // server扩展模式，host machine内部完成端到端调度上板执行，submit task & compile & run 不对外暴露
    API = 1, // api 模式，当前torch对接使用此模式，对外暴露submit task  & compile & run api供外部调用
};

template <typename T>
class SafeQueue {
public:
    void Push(T value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
    }

    T Pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    bool Empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    uint64_t Size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void Clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_ = std::queue<T>();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
};

class HostMachine {
public:
    static HostMachine& GetInstance();

    bool Init(const HostMachineMode mode); // init resource & launch device machine core machine
    void Destroy();                        // release resource & stop device machine core machine

    void SubTask(Function* function);
    void WaitTaskFinish(); // wait all task finish

    void StashTask(Function* function);
    void SubAllStashedTask();

    void ClearStashFuncQueue();

    void ResetAllPasses();

public: // api mode
    MachineTask* Compile(MachineTask* task = nullptr) const;

private:
    HostMachine() : initialized_(false), mode_(HostMachineMode::SERVER) {}
    ~HostMachine() { DestroyThread(); }
    void InitThread();
    void DestroyThread();

    void CompileFunction(Function* func) const;
    /* 线程处理函数 */
    void CompileThreadFunc();
    void AgentThreadFunc();

    void PushAgentQueue(std::unique_ptr<MachineTask> task);
    void PushFinishQueue(std::unique_ptr<MachineTask> task);

    static std::string GetCacheKeyFromFunction(Function* function);

private:
    std::atomic<bool> initialized_;
    HostMachineMode mode_;
    MachineTask* curTask;

    std::atomic<uint64_t> curTaskId_{0};
    std::atomic<bool> stopFlag_{false};

    /* 线程管理 */
    int compileThreadCount_{1};
    int agentThreadCount_{1};
    std::mutex compileQueueMutex_;
    std::mutex agentQueueMutex_;
    std::mutex stashQueueMutex_;
    std::condition_variable compileQueueCv_;
    std::condition_variable agentQueueCv_;
    std::vector<std::thread> compileThreads_;
    std::vector<std::thread> agentThreads_;
    SafeQueue<std::unique_ptr<MachineTask>> compileQueue_; // 待编译任务
    SafeQueue<std::unique_ptr<MachineTask>> agentQueue_;   // 待device agent处理任务
    SafeQueue<std::unique_ptr<MachineTask>> finishQueue_;  // device machine 处理结束任务
    SafeQueue<std::tuple<Function*, std::shared_ptr<ConfigScope>, InternalGlobalConfig,
                         nlohmann::json>>
        stashedFuncQueue_; // stash func
};

} // namespace npu::tile_fwk
