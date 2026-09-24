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
 * \file ModelConfig.h
 * \brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "cost_model/simulation/base/Config.h"

namespace CostModel {
struct ModelConfig : public Config {
    ModelConfig();
    bool statisticReportToFile = false;
    uint64_t heartInterval = 50000;
    uint64_t drawPngThresholdCycle = 500000;
    bool testDeadLock = false;
    std::string startFunctionLabel = "root";
    bool useOOOPassSeq = true;
    bool genCalendarScheduleCpp = false;
    uint64_t deviceMachineNumber = 1;
    uint64_t aicpuMachineNumber = 6;
    uint64_t aicpuMachineSmtNum = 1;
    uint64_t coreMachineNumberPerAICPU = 12;
    uint64_t cubeMachineNumberPerAICPU = 4;
    uint64_t vecMachineNumberPerAICPU = 8;
    uint64_t coreMachineSmtNum = 1;
    bool cubeVecMixMode = false;
    bool mteUseL2Cache = false;
    uint64_t functionCacheSize = 8;
    std::string deviceArch = "A2A3";
    bool simulationFixedLatencyTask = false;
    std::string fixedLatencyTaskInfoPath = "";
    uint64_t fixedLatencyTimeConvert = 1;
    uint64_t pipeBoardVibration = 0;
    uint64_t calendarMode = 0;
    std::string calendarFile = "";
};
} // namespace CostModel
