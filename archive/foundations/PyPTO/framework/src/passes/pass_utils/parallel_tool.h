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
 * \file pass_utils/parallel_tool.h
 * \brief
 */

#ifndef PASS_PARALLEL_TOOL_H
#define PASS_PARALLEL_TOOL_H

#include <vector>
#include <thread>
#include <functional>
#include <iostream>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <unordered_set>
#include <unordered_map>

namespace npu::tile_fwk {

class WorkItem {
public:
    int start_;
    int end_;
    std::function<void(int, int, int)>* bodyPtr_;
    WorkItem() : start_(0), end_(0), bodyPtr_(nullptr) {}
    WorkItem(int start, int end, std::function<void(int, int, int)>* bodyPtr)
        : start_(start), end_(end), bodyPtr_(bodyPtr)
    {}
};

class ParallelQueue {
public:
    void Clear();
    void Insert(int start, int end, std::function<void(int, int, int)>* bodyPtr);
    bool Empty();
    bool Get(WorkItem* workItem);
    std::deque<WorkItem> workItems;
    std::mutex get_mtx;
};

class ParallelTool {
public:
    explicit ParallelTool(unsigned int num_threads);
    ~ParallelTool();
    static std::unique_ptr<ParallelTool> parallelToolPtr;
    static void Init();
    static ParallelTool& Instance();
    void ExecTaskVec(int threadIdx);
    int GetThreadNum();
    void Parallel_for(int start, int end, int step, std::function<void(int, int, int)> body);              // static
    void Parallel_for(int start, int end, int step, int numWork, std::function<void(int, int, int)> body); // dynamic
    std::vector<std::thread> workers;
    std::vector<std::atomic<bool>> taskReadys;
    int numThread;
    std::vector<std::mutex> waiting_mtx;
    std::vector<std::condition_variable> waiting_cv;
    ParallelQueue workQueue;
    std::atomic<int> workingThreads{0};
    std::atomic<bool> killThreads{false};
    std::condition_variable completion_cv;
    std::mutex completion_mtx;
};
} // namespace npu::tile_fwk
#endif
