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
 * \file CacheStats.h
 * \brief
 */

#pragma once

#include <map>

#include "cost_model/simulation/base/BaseStats.h"

class CacheStats : public CostModel::BaseStats {
public:
    uint64_t totalInsertNum;
    uint64_t totalEvictNum;
    uint64_t totalQueryNum;
    uint64_t totalHitNum;
    uint64_t totalMissNum;
    uint64_t totalReadNum;
    uint64_t totalWriteNum;

    uint64_t totalResponseLatency;

    using BaseStats::BaseStats;
    void Reset() override;
    void Report(std::string& name) override;
};
