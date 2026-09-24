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

#include "device_common.h"
#include "device_utils.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk::dynamic {

enum CtrlWaitLevel : int {
    CTRL_WAIT_ARBITRATING = -2, // spin 等待中
    CTRL_WAIT_FAILED = -1,      // scheNum==1 时超时
    CTRL_WAIT_UNSET = 0,        // 未决策
    CTRL_WAIT_OK = 1,           // ctrl 已起来，无需丢弃
    CTRL_WAIT_DROPPED = 2,      // 需丢弃一个线程
};

static inline uint64_t CalcWaitTimeout(bool isOnlyOneSche = false, bool isWaitCtrlLevel = false)
{
    uint64_t waitTimeout = isWaitCtrlLevel ? TIMEOUT_A2A3_1SEC : TIMEOUT_A2A3_100US;
    if (!IsDeviceMode()) {
        waitTimeout = isWaitCtrlLevel ? TIMEOUT_A2A3_10SEC : TIMEOUT_A2A3_1SEC;
        return waitTimeout;
    }
    DEV_IF_INFO { waitTimeout = isWaitCtrlLevel ? TIMEOUT_A2A3_1SEC : TIMEOUT_A2A3_2MS; }
    DEV_IF_DEBUG { waitTimeout = isWaitCtrlLevel ? TIMEOUT_A2A3_1SEC : TIMEOUT_A2A3_3MS; }
    // 避免 OS 节流导致 aicpu 拉起时间较长，超时报错
    if (isOnlyOneSche && !isWaitCtrlLevel) {
        waitTimeout = TIMEOUT_A2A3_55MS;
    }
    return waitTimeout;
}

// spin 等待 ctrl 线程起来（ctrlRound >= scheRound），成功返回 true，超时返回 false
static inline bool WaitCtrlRoundReady(ArchInfo archInfo, std::atomic<uint64_t>& ctrlRound,
                                      std::atomic<uint64_t>& scheRound, int arbitratedScheNum)
{
    bool isOnlyOneSche = arbitratedScheNum == 1;
    uint64_t waitTimeout = CalcWaitTimeout(isOnlyOneSche, false);
    TIMEOUT_CHECK_INIT(archInfo, waitTimeout);
    while (ctrlRound.load(std::memory_order_acquire) <= scheRound.load(std::memory_order_acquire)) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return false, "#ctrl.wait: ctrl thread not start.");
    }
    return true;
}

// follower spin 等待决策终态 level
static inline int WaitCtrlDecision(ArchInfo archInfo, std::atomic<int>& ctrlWaitLevel, int& outLevel)
{
    uint64_t waitTimeout = CalcWaitTimeout(false, true);
    TIMEOUT_CHECK_INIT(archInfo, waitTimeout);
    int level = ctrlWaitLevel.load(std::memory_order_acquire);
    while (level == CTRL_WAIT_UNSET || level == CTRL_WAIT_ARBITRATING) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR,
                                               "Wait for ctrl thread timeout, ctrl wait level: %d", level);
        level = ctrlWaitLevel.load(std::memory_order_acquire);
    }
    outLevel = level;
    return DEVICE_MACHINE_OK;
}

// 对于 Dav2201,通过 ctrlRound/scheRound 判断 ctrl 是否已起来，决定是否丢弃一个线程
// 对于其他类型, 直接返回 DEVICE_MACHINE_OK
static inline int WaitForCtrlDecision(ArchInfo archInfo, int& curThreadIdx, int& arbitratedScheNum,
                                      std::atomic<int>& ctrlWaitLevel, std::atomic<uint64_t>& ctrlRound,
                                      std::atomic<uint64_t>& scheRound)
{
    if (archInfo != ArchInfo::DAV_2201) {
        return DEVICE_MACHINE_OK;
    }
    int level = CTRL_WAIT_UNSET;
    // CAS 抢占：成功者 spin 等 ctrl 起来并发布终态；失败者 spin 等终态
    int expected = CTRL_WAIT_UNSET;
    if (ctrlWaitLevel.compare_exchange_strong(expected, CTRL_WAIT_ARBITRATING, std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
        if (WaitCtrlRoundReady(archInfo, ctrlRound, scheRound, arbitratedScheNum)) {
            level = CTRL_WAIT_OK;
        } else {
            level = (arbitratedScheNum == 1) ? CTRL_WAIT_FAILED : CTRL_WAIT_DROPPED;
        }
        ctrlWaitLevel.store(level, std::memory_order_release);
    } else {
        int ret = WaitCtrlDecision(archInfo, ctrlWaitLevel, level);
        if (ret != DEVICE_MACHINE_OK) {
            DEV_ERROR(SchedErr::WAIT_CTRL_TIMEOUT,
                      "Thread %d encountered timeout when waiting for ctrl level, current level: %d", curThreadIdx,
                      ctrlWaitLevel.load());
            return ret;
        }
    }
    // 决策后统一分发
    if (level == CTRL_WAIT_FAILED) {
        DEV_ERROR(SchedErr::WAIT_CTRL_TIMEOUT,
                  "Wait for ctrl thread timeout,"
                  "only one sched thread exists, id is %d.",
                  curThreadIdx);
        return DEVICE_MACHINE_ERROR;
    }
    if (level == CTRL_WAIT_OK) {
        return DEVICE_MACHINE_OK;
    }
    // CTRL_WAIT_DROPPED：arbitratedScheNum 全局一致地减 1
    int scheNum = arbitratedScheNum;
    arbitratedScheNum -= 1;
    if (curThreadIdx == scheNum) {
        DEV_INFO("Waiting for ctrl thread timed out, release sched thread %d.", curThreadIdx);
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }
    DEV_INFO("Thread %d wait ctrl thread timeout, another sched thread released, arbitrated sche num: %d", curThreadIdx,
             arbitratedScheNum);
    return DEVICE_MACHINE_OK;
}
} // namespace npu::tile_fwk::dynamic
