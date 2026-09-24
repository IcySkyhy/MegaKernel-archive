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

#include <cstdlib>
#include "gtest/gtest.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"

namespace npu::tile_fwk {

class MonitorTestFixtureBase : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        unsetenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED");
        unsetenv("PYPTO_COMPILER_MONITOR_CURRENT");
        unsetenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC");
        unsetenv("PYPTO_COMPILER_MONITOR_TIMEOUT_SEC");
        unsetenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC");
    }
    void TearDown() override
    {
        MonitorManager::Instance().NotifyCompilationFinished();
        MonitorManager::Instance().SetProcessingThresholdSec(60);
    }
};

} // namespace npu::tile_fwk
