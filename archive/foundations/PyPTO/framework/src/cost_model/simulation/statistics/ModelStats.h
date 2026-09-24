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
 * \file ModelStats.h
 * \brief
 */

#pragma once

#include <map>

#include "cost_model/simulation/base/BaseStats.h"

class ModelStats : public CostModel::BaseStats {
public:
    uint64_t cycles;
    uint64_t stepCount;
    uint64_t hostMachineNum;
    uint64_t deviceMachineNum;
    uint64_t aicpuMachineNum;
    uint64_t coreMachineNum;
    uint64_t cubeMachineNum;
    uint64_t vecMachineNum;
    uint64_t cvMixedCoreMachineNum;
    uint64_t pipeGroupNum;
    uint64_t totalFunctionNum;
    uint64_t totalFunctionCube;
    uint64_t totalFunctionVec;
    uint64_t totalFunctionMix;
    uint64_t totalFunctionTileOps;
    std::map<int, uint64_t> coreUseCycles;

    using BaseStats::BaseStats;
    void Reset() override;
    void Report(std::string& name) override;
};
