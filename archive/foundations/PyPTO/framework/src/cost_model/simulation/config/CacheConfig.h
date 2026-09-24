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
 * \file CacheConfig.h
 * \brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "cost_model/simulation/base/Config.h"

namespace CostModel {
struct CacheConfig : public Config {
    CacheConfig();
    uint64_t l2InputPortNum = 6;
    uint64_t l2Size = 20971520;
    uint64_t l2LineSize = 512;
    uint64_t l2HitLatency = 50;
    uint64_t l2MissExtraLatency = 150;
};
} // namespace CostModel
