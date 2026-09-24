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
 * \file CycleInfo.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace CostModel {
/* Tile Operation Execute Info */
struct CycleInfo {
    // Execute Cycle Info
    uint64_t fetchCycle = 0;
    uint64_t decodeCycle = 0;
    uint64_t renameCycle = 0;
    uint64_t dispatchCycle = 0;
    uint64_t insertIqCycle = 0;
    uint64_t readyCycle = 0;
    uint64_t pickedCycle = 0;
    uint64_t issueCycle = 0;
    uint64_t executeStartCycle = 0;
    uint64_t executeEndCycle = 0;
    uint64_t completedCycle = 0;
    uint64_t retireCycle = 0;
    uint64_t allocCycle = 0;
    uint64_t writeCycle = 0;
    uint64_t freeCycle = 0;

    // Cache Access Cycle Info
    uint64_t pktSendCycle = 0;
    uint64_t cacheRecvCycle = 0;
    uint64_t cacheQueryCycle = 0;
    uint64_t cacheMissCycle = 0;
    uint64_t cacheHitCycle = 0;
    uint64_t cacheRespCycle = 0;
    uint64_t pktRetToSenderCycle = 0;

    uint64_t relativeStartCycle = 0;
    uint64_t relativeEndCycle = 0;
    void Reset();
};

// Task Execute Cycle Info
struct TaskCycleInfo {
    uint64_t taskGenCycle = 0;
    uint64_t taskSentCycle = 0;
    uint64_t taskRecvCycle = 0;
    uint64_t taskExecuteStartCycle = 0;
    uint64_t taskExecuteEndCycle = 0;
};

} // namespace CostModel
