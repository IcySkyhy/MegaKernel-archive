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
 * \file CoreConfig.h
 * \brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "cost_model/simulation/base/Config.h"

namespace CostModel {
struct CoreConfig : public Config {
    CoreConfig();
    uint64_t pipeTileAllocNum = 1;
    uint64_t pipeVectorBmuNum = 1;
    uint64_t pipeCubeBmuL1NUM = 1;
    uint64_t pipeCubeBmuL0ANUM = 1;
    uint64_t pipeCubeBmuL0BNUM = 1;
    uint64_t pipeCubeBmuL0CNUM = 1;
    uint64_t pipeMteInNum = 1;
    uint64_t pipeMte1Num = 1;
    uint64_t pipeVectorAluNum = 1;
    uint64_t pipeCubeNum = 1;
    uint64_t pipeMteOutNum = 1;
    uint64_t pipeSimCallNum = 1;
    uint64_t tileopSentToPipeThreshold = 1;
    uint64_t calendarSetQueueWDelay = 140;
    bool bufferBackPressure = true;
    uint64_t logLabelMode = 1;
    bool enableTileOpFlow = false;
};
} // namespace CostModel
