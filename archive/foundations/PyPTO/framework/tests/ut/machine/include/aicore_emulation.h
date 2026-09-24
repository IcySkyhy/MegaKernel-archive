/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_emulation.h
 * \brief
 */

#ifndef AICORE_EMULATION_H
#define AICORE_EMULATION_H

#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "tilefwk/aikernel_define.h"
#include "tilefwk/aikernel_data.h"
#include "tilefwk/aikernel_runtime.h"
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::machine {

class AicoreEmulationBase {
public:
    virtual ~AicoreEmulationBase() = default;

    uint64_t AicoreGetSysCnt()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        return ns;
    }

    virtual int64_t AicoreGetPhyIdx() { return 0; }

    virtual int AicoreGetCoreIdx() { return 0; }

    virtual void AicoreSetCond(uint64_t cond) { UNUSED(cond); }

    virtual uint64_t AicoreGetData() { return 0; }

    virtual void AicoreCallSubFuncTask(uint64_t funcIdx, npu::tile_fwk::CoreFuncParam* param, int64_t gmStackAddr,
                                       __gm__ int64_t* hcclContext, TaskStat* taskStat)
    {
        UNUSED(funcIdx);
        UNUSED(param);
        UNUSED(gmStackAddr);
        UNUSED(hcclContext);
        UNUSED(taskStat);
    }
};

class ThreadAicoreInfo {
public:
    ThreadAicoreInfo(std::shared_ptr<std::thread> thread, int phyIdx, int coreIdx)
        : thread_(thread), phyIdx_(phyIdx), coreIdx_(coreIdx)
    {}

    int GetPhyIdx() const { return phyIdx_; }
    int GetCoreIdx() const { return coreIdx_; }

    uint64_t GetCond() { return cond_; }
    void SetCond(uint64_t cond) { cond_ = cond; }

    uint64_t GetData() { return data_; }
    void SetData(uint64_t data) { data_ = data; }

private:
    std::shared_ptr<std::thread> thread_;
    int phyIdx_{-1};
    int coreIdx_{-1};

    uint64_t cond_{0};
    uint64_t data_{0};
};

class ThreadAicoreEmulation : public AicoreEmulationBase {
public:
    void AppendAicore(std::shared_ptr<std::thread> thread, int phyIdx, int coreIdx)
    {
        std::lock_guard<std::mutex> guard(aicoreInfoMutex_);

        std::shared_ptr<ThreadAicoreInfo> info = std::make_shared<ThreadAicoreInfo>(thread, phyIdx, coreIdx);
        aicoreInfoDict_[thread->get_id()] = info;

        if (phyIdx >= (int)phyIdxAicoreInfoDict_.size()) {
            phyIdxAicoreInfoDict_.resize(phyIdx + 1);
        }
        phyIdxAicoreInfoDict_[phyIdx] = info;

        if (coreIdx >= (int)coreIdxAicoreInfoDict_.size()) {
            coreIdxAicoreInfoDict_.resize(coreIdx + 1);
        }
        coreIdxAicoreInfoDict_[coreIdx] = info;
    }

    std::shared_ptr<ThreadAicoreInfo> GetAicoreInfoByThread()
    {
        std::lock_guard<std::mutex> guard(aicoreInfoMutex_);

        auto threadId = std::this_thread::get_id();
        if (aicoreInfoDict_.count(threadId)) {
            auto info = aicoreInfoDict_[threadId];
            return info;
        } else {
            return nullptr;
        }
    }

    std::shared_ptr<ThreadAicoreInfo> GetAicoreInfoByPhyIdx(int phyIdx)
    {
        std::lock_guard<std::mutex> guard(aicoreInfoMutex_);

        if (phyIdx < (int)phyIdxAicoreInfoDict_.size()) {
            return phyIdxAicoreInfoDict_[phyIdx];
        } else {
            return nullptr;
        }
    }

    std::shared_ptr<ThreadAicoreInfo> GetAicoreInfoByCoreIdx(int coreIdx)
    {
        std::lock_guard<std::mutex> guard(aicoreInfoMutex_);

        if (coreIdx < (int)coreIdxAicoreInfoDict_.size()) {
            return coreIdxAicoreInfoDict_[coreIdx];
        } else {
            return nullptr;
        }
    }

    virtual int64_t AicoreGetPhyIdx() override
    {
        auto info = GetAicoreInfoByThread();
        return info->GetPhyIdx();
    }

    virtual int AicoreGetCoreIdx() override
    {
        auto info = GetAicoreInfoByThread();
        return info->GetCoreIdx();
    }

    virtual void AicoreSetCond(uint64_t cond) override
    {
        auto info = GetAicoreInfoByThread();
        info->SetCond(cond);
    }

    virtual uint64_t AicoreGetData() override
    {
        auto info = GetAicoreInfoByThread();
        return info->GetData();
    }

    uint64_t AicpuGetCond(int coreIdx)
    {
        auto info = GetAicoreInfoByCoreIdx(coreIdx);
        return info->GetCond();
    }
    void AicpuSetData(int coreIdx, uint64_t data)
    {
        auto info = GetAicoreInfoByCoreIdx(coreIdx);
        info->SetData(data);
    }

private:
    std::unordered_map<std::thread::id, std::shared_ptr<ThreadAicoreInfo>> aicoreInfoDict_;
    std::vector<std::shared_ptr<ThreadAicoreInfo>> phyIdxAicoreInfoDict_;
    std::vector<std::shared_ptr<ThreadAicoreInfo>> coreIdxAicoreInfoDict_;
    std::mutex aicoreInfoMutex_;
};

class AicoreEmulationManager {
public:
    static AicoreEmulationManager& GetInstance();

    AicoreEmulationManager() { base_ = std::make_shared<AicoreEmulationBase>(); }

    void SetupAicoreEmulation(std::shared_ptr<AicoreEmulationBase> curr) { curr_ = curr; }
    void Reset() { curr_ = nullptr; }

    std::shared_ptr<AicoreEmulationBase> GetEmulation()
    {
        if (curr_) {
            return curr_;
        } else {
            return base_;
        }
    }

private:
    std::shared_ptr<AicoreEmulationBase> base_;
    std::shared_ptr<AicoreEmulationBase> curr_;
};

} // namespace npu::tile_fwk::machine

#define dcci(...)
#define dsb(...)
#define set_flag(...)
#define wait_flag(...)
#define set_mask_norm(...)

static inline uint64_t get_sys_cnt()
{
    auto cnt = npu::tile_fwk::machine::AicoreEmulationManager::GetInstance().GetEmulation()->AicoreGetSysCnt();
    return cnt;
}

static inline int64_t get_coreid()
{
    auto coreid = npu::tile_fwk::machine::AicoreEmulationManager::GetInstance().GetEmulation()->AicoreGetPhyIdx();
    return coreid;
}

static inline int get_block_idx()
{
    auto blockIdx = npu::tile_fwk::machine::AicoreEmulationManager::GetInstance().GetEmulation()->AicoreGetCoreIdx();
    return blockIdx;
}

static inline void set_cond(uint64_t cond)
{
    npu::tile_fwk::machine::AicoreEmulationManager::GetInstance().GetEmulation()->AicoreSetCond(cond);
}

static inline uint64_t GetDataMainBase()
{
    auto mainBase = npu::tile_fwk::machine::AicoreEmulationManager::GetInstance().GetEmulation()->AicoreGetData();
    return mainBase;
}

static inline void CallSubFuncTask(uint64_t funcIdx, npu::tile_fwk::CoreFuncParam* param, int64_t gmStackAddr,
                                   __gm__ int64_t* hcclContext, TaskStat* taskStat)
{
    npu::tile_fwk::machine::AicoreEmulationManager::GetInstance().GetEmulation()->AicoreCallSubFuncTask(
        funcIdx, param, gmStackAddr, hcclContext, taskStat);
}

template <typename T, typename V>
static inline T atomicAdd(__gm__ T* address, V value)
{
    return __atomic_fetch_add(address, static_cast<T>(value), __ATOMIC_RELAXED);
}

template <typename T>
static inline T atomicCAS(__gm__ T* address, T compare, T value)
{
    __atomic_compare_exchange_n(address, &compare, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return compare;
}

template <typename T>
static inline T atomicExch(__gm__ T* address, T value)
{
    return __atomic_exchange_n(address, value, __ATOMIC_RELAXED);
}

#define __HAS_SUB_FUNC__
// don't need head file in emulation.
#define __HEAD_FILE__ stdint.h
#include "tilefwk/aicore_entry.h"
#endif
