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
 * \file parallel_tool.cpp
 * \brief
 */

#include "passes/pass_utils/parallel_tool.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void ParallelQueue::Clear() { workItems.clear(); }

void ParallelQueue::Insert(int start, int end, std::function<void(int, int, int)>* bodyPtr)
{
    std::unique_lock lock(get_mtx);
    workItems.emplace_back(start, end, bodyPtr);
}

bool ParallelQueue::Empty() { return workItems.empty(); }

bool ParallelQueue::Get(WorkItem* workItem)
{
    std::unique_lock lock(get_mtx);
    if (Empty()) {
        return false;
    }
    *workItem = workItems.front();
    workItems.pop_front();
    return true;
}

void ParallelTool::Init()
{
    if (parallelToolPtr == nullptr) {
        const int default_concurrency = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
        parallelToolPtr = std::make_unique<ParallelTool>(default_concurrency);
    }
}

ParallelTool& ParallelTool::Instance()
{
    Init();
    return *parallelToolPtr;
}

ParallelTool::ParallelTool(unsigned int num_threads)
    : taskReadys(num_threads), numThread(num_threads), waiting_mtx(num_threads), waiting_cv(num_threads)
{
    if (num_threads <= 1) {
        return;
    }
    for (int i = 0; i < static_cast<int>(num_threads); i++) {
        taskReadys[i] = false;
        workers.emplace_back([this, i]() {
            while (true) {
                std::unique_lock lock(waiting_mtx[i]);
                waiting_cv[i].wait(lock, [this, i] { return taskReadys[i] || killThreads; });
                if (killThreads) {
                    break;
                }
                ExecTaskVec(i);
                std::unique_lock completion_lock(completion_mtx);
                taskReadys[i] = false;
                workingThreads--;
                if (workingThreads == 0) {
                    completion_cv.notify_one();
                }
            }
        });
    }
}

ParallelTool::~ParallelTool()
{
    if (numThread > 1) {
        killThreads = true;
        for (int i = 0; i < numThread; i++) {
            waiting_cv[i].notify_one();
        }
        for (int i = 0; i < numThread; i++) {
            workers[i].join();
        }
    }
}

void ParallelTool::ExecTaskVec(int threadIdx)
{
    (void)threadIdx;
    WorkItem workItem;
    while (workQueue.Get(&workItem)) {
        (*workItem.bodyPtr_)(workItem.start_, workItem.end_, threadIdx);
    }
}

int ParallelTool::GetThreadNum() { return numThread; }

void ParallelTool::Parallel_for(int start, int end, int step, std::function<void(int, int, int)> body)
{
    Parallel_for(start, end, step, numThread, body);
}

void ParallelTool::Parallel_for(int start, int end, int step, int numWork, std::function<void(int, int, int)> body)
{
    if (numThread <= 1) {
        body(start, end, 0);
        return;
    }
    if (end >= start && step > 0) {
        int numTask = (end - start) / step;
        int numTaskSingle = (numTask + numWork - 1) / numWork;
        int stepSize = step * numTaskSingle;
        for (int i = start; i < end; i += stepSize) {
            int boundEnd = std::min(end, i + stepSize);
            workQueue.Insert(i, boundEnd, &body);
        }
    } else if (end <= start && step < 0) {
        int numTask = (start - end) / (-step);
        int numTaskSingle = (numTask + numWork - 1) / numWork;
        int stepSize = step * numTaskSingle;
        for (int i = start; i > end; i += stepSize) {
            int boundEnd = std::max(end, i + stepSize);
            workQueue.Insert(i, boundEnd, &body);
        }
    }
    std::unique_lock lock(completion_mtx);
    workingThreads = numThread;
    for (int i = 0; i < numThread; i++) {
        taskReadys[i] = true;
    }
    for (int i = 0; i < numThread; i++) {
        waiting_cv[i].notify_one();
    }
    completion_cv.wait(lock, [this]() { return workingThreads == 0; });
}

std::unique_ptr<ParallelTool> ParallelTool::parallelToolPtr = nullptr;

} // namespace npu::tile_fwk
