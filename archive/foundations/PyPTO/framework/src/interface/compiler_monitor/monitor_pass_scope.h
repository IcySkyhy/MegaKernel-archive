/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include "interface/compiler_monitor/monitor_manager.h"

namespace npu::tile_fwk {

class MonitorPassCompileScope {
public:
    MonitorPassCompileScope(const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
                            const std::string& functionName, int functionIndex, int functionOpSize)
        : strategy_(strategy),
          passIdentifier_(passIdentifier),
          passIndex_(passIndex),
          functionName_(functionName),
          functionIndex_(functionIndex),
          functionOpSize_(functionOpSize),
          start_(std::chrono::high_resolution_clock::now())
    {
        MonitorManager::Instance().StartPassCompile(strategy_, passIdentifier_, passIndex_, functionName_,
                                                    functionIndex_, functionOpSize_);
    }

    ~MonitorPassCompileScope()
    {
        if (!finished_) {
            Finish(false);
        }
    }

    MonitorPassCompileScope(const MonitorPassCompileScope&) = delete;
    MonitorPassCompileScope& operator=(const MonitorPassCompileScope&) = delete;

    std::chrono::time_point<std::chrono::high_resolution_clock> GetStartTime() const { return start_; }

    void Finish(bool success)
    {
        auto end = std::chrono::high_resolution_clock::now();
        FinishAt(success, end);
    }

    void FinishAt(bool success, const std::chrono::time_point<std::chrono::high_resolution_clock>& end)
    {
        if (finished_) {
            return;
        }
        finished_ = true;
        double elapsed = std::chrono::duration<double>(end - start_).count();
        MonitorManager::Instance().RecordPassCompileTime(strategy_, passIdentifier_, passIndex_, functionName_,
                                                         functionIndex_, functionOpSize_, elapsed, success);
        MonitorManager::Instance().EndPassCompile(strategy_, passIdentifier_, passIndex_, functionName_,
                                                  functionIndex_);
    }

private:
    std::string strategy_;
    std::string passIdentifier_;
    size_t passIndex_;
    std::string functionName_;
    int functionIndex_;
    int functionOpSize_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool finished_{false};
};

} // namespace npu::tile_fwk
