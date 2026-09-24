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
#include <cstdint>
#include <cstring>
#include "aicore_constants.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk::dynamic {

// CoreStatusManager: 设备侧动态调度核心状态管理器
// 在 AICPU 调度热路径中维护各 AICore 的就绪状态，供调度器查询可用核心并分配任务。
// 每个 AICore 有 running/pending 两个任务槽位（见 aicore_manager.h 的 runningIds_/pendingIds_）：
//   - PendReady: pending 槽位为空、可接收新任务的核心计数（含完全空闲的核心 +
//     running 在跑但 pending 已释放的核心）。初始化时等于全部核心数。
//   - RunReady: 已完全就绪（running/pending 均空）可立即被取用的核心，是 PendReady 的子集，
//     用紧凑数组 + 位置反查表实现 O(1) 增删
class CoreStatusManager {
public:
    CoreStatusManager()
    {
        size_t positionSize = sizeof(coreIdxPosition_);
        auto ret = memset_s(coreIdxPosition_, positionSize, INVALID_COREIDX_POSITION, positionSize);
        if (ret != 0) {
            DEV_ERROR(DevCommonErr::MEMCPY_FAILED, "#sche.init: coreIdxPosition_ init failed: %d", ret);
        }
    }

    inline uint32_t& WaitTaskCnt(int type) { return waitTaskCnt[type]; }

    inline uint8_t GetCorePendReadyCnt(int type) const { return corePendReadyCnt_[type]; }
    inline void SetCorePendReadyCnt(int type, uint8_t value) { corePendReadyCnt_[type] = value; }

    inline uint8_t GetCoreRunReadyCnt(int type) const { return coreRunReadyCnt_[type]; }
    inline void SetCoreRunReadyCnt(int type, uint8_t value) { coreRunReadyCnt_[type] = value; }

    inline uint8_t GetRunReadyCoreIdx(int type, int idx) const { return runReadyCoreIdx_[type][idx]; }

    inline uint8_t GetLastPendReadyCoreIdx(int type) const { return lastPendReadyCoreIdx_[type]; }
    inline void SetLastPendReadyCoreIdx(int type, uint8_t value) { lastPendReadyCoreIdx_[type] = value; }

    inline uint8_t GetCoreIdxPosition(int coreIdx) const { return coreIdxPosition_[coreIdx]; }
    inline void SetCoreIdxPosition(int coreIdx, uint8_t value) { coreIdxPosition_[coreIdx] = value; }

    inline void RemoveRunAndPendCoreIdx(int coreIdx, int type)
    {
        RemoveRunReadyCoreIdx(coreIdx, type);
        RemovePendReadyCoreIdx(type);
    }

    inline void RemovePendReadyCoreIdx(int type) { corePendReadyCnt_[type]--; }

    inline void RemoveRunReadyCoreIdx(int coreIdx, int type)
    {
        uint8_t pos = coreIdxPosition_[coreIdx];
        if (pos == INVALID_COREIDX_POSITION) {
            DEV_WARN("RemoveRunReadyCoreIdx: coreIdx %d not in runReady list (pos=INVALID)", coreIdx);
            return;
        }

        uint32_t tail = --coreRunReadyCnt_[type];
        if (pos != tail) {
            runReadyCoreIdx_[type][pos] = runReadyCoreIdx_[type][tail];
            coreIdxPosition_[runReadyCoreIdx_[type][pos]] = pos;
        }
        coreIdxPosition_[coreIdx] = INVALID_COREIDX_POSITION;
    }

    inline void BatchRemovePendReadyCoreIdx(int type, uint32_t batch) { corePendReadyCnt_[type] -= batch; }

    inline void RemoveReadyCoreIdxTail(int coreIdx, int type)
    {
        if (coreIdxPosition_[coreIdx] == INVALID_COREIDX_POSITION) {
            DEV_WARN("RemoveReadyCoreIdxTail: coreIdx %d not in runReady list (pos=INVALID)", coreIdx);
            return;
        }
        coreRunReadyCnt_[type]--;
        coreIdxPosition_[coreIdx] = INVALID_COREIDX_POSITION;
    }

    inline void AddRunAndPendCoreIdx(int coreIdx, int type)
    {
        AddRunReadyCoreIdx(coreIdx, type);
        AddPendReadyCoreIdx(type);
    }

    inline void AddRunReadyCoreIdx(int coreIdx, int type)
    {
        coreIdxPosition_[coreIdx] = coreRunReadyCnt_[type];
        runReadyCoreIdx_[type][coreRunReadyCnt_[type]++] = coreIdx;
    }

    inline void AddPendReadyCoreIdx(int type) { corePendReadyCnt_[type]++; }

private:
    // 各核心类型的待处理任务计数
    uint32_t waitTaskCnt[AICORE_TYPE_NUM]{0, 0};
    // 各核心类型可接收新任务（pending 槽位空闲）的核心计数。pending槽空且running槽非空，表示任务可下发进入pending状态
    uint8_t corePendReadyCnt_[AICORE_TYPE_NUM]{0, 0};
    // 各核心类型已完全就绪可立即调度的核心计数
    uint8_t coreRunReadyCnt_[AICORE_TYPE_NUM]{0, 0};
    // 各核心类型的 RunReady 核心索引紧凑数组，有效元素为 [0, coreRunReadyCnt_)
    uint8_t runReadyCoreIdx_[AICORE_TYPE_NUM][MAX_MANAGER_AIV_NUM];
    // 各核心类型最近一次派任务的 PendReady 核心索引（Round-Robin 扫描起点）
    uint8_t lastPendReadyCoreIdx_[AICORE_TYPE_NUM]{0, 0};

    // 核心索引 -> 在 runReadyCoreIdx_ 中的位置反查表，用于 O(1) 删除
    // INVALID_COREIDX_POSITION 表示该核心不在 RunReady 列表中
    uint8_t coreIdxPosition_[MAX_AICORE_NUM]{INVALID_COREIDX_POSITION};
};

} // namespace npu::tile_fwk::dynamic
