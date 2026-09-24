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

#include <string>
#include "interface/compiler_monitor/monitor_manager.h"

namespace npu::tile_fwk {

class MonitorStageScope {
public:
    explicit MonitorStageScope(const std::string& stageName) : stageName_(stageName)
    {
        MonitorManager::Instance().StartStage(stageName_);
    }

    MonitorStageScope(const std::string& stageName, int rootFuncIndex, const std::string& rootFuncName,
                      int rootFuncOpSize)
        : stageName_(stageName),
          rootFuncIndex_(rootFuncIndex),
          rootFuncName_(rootFuncName),
          rootFuncOpSize_(rootFuncOpSize),
          isRootFunc_(true)
    {
        MonitorManager::Instance().StartStage(stageName_, rootFuncIndex_, rootFuncName_, rootFuncOpSize_);
    }

    ~MonitorStageScope()
    {
        if (isRootFunc_) {
            MonitorManager::Instance().EndStage(stageName_, rootFuncIndex_, rootFuncName_, rootFuncOpSize_);
        } else {
            MonitorManager::Instance().EndStage(stageName_);
        }
    }

    MonitorStageScope(const MonitorStageScope&) = delete;
    MonitorStageScope& operator=(const MonitorStageScope&) = delete;

private:
    std::string stageName_;
    int rootFuncIndex_{0};
    std::string rootFuncName_;
    int rootFuncOpSize_{0};
    bool isRootFunc_{false};
};

} // namespace npu::tile_fwk
