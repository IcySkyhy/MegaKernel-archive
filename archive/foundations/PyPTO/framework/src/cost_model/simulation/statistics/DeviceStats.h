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
 * \file DeviceStats.h
 * \brief
 */

#pragma once

#include <map>

#include "cost_model/simulation/base/BaseStats.h"

class DeviceStats : public CostModel::BaseStats {
public:
    uint64_t totalSubmitNum;
    uint64_t cubeSubmitNum;
    uint64_t vectorSubmitNum;

    uint64_t totalTaskExecuteCycles;
    uint64_t minTaskExecuteCycles;
    uint64_t maxTaskExecuteCycles;

    uint64_t resolveNum;
    uint64_t pollingNum;

    using BaseStats::BaseStats;
    void Reset() override;
    void Report(std::string& name) override;
};
