/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file perf_event_sampler.h
 * \brief Linux perf_event_open based PMU sampler for AICPU performance analysis
 *        Uses PERF_TYPE_HARDWARE standard events for better compatibility
 */

#pragma once

#include <cstdint>
#include <sys/types.h>

#include "machine/utils/device_switch.h"
#include "machine/utils/device_log.h"

#define MAX_PERF_EVENT_NUM 8

namespace npu::tile_fwk {

struct PerfCacheMetrics {
    bool valid{false};
    double missRate{0.0};
};

struct PerfDerivedMetrics {
    double ipc{0.0};
    double cpi{0.0};
    double branchMissRate{0.0};
    PerfCacheMetrics l1dCache;
    PerfCacheMetrics llCache;
};

// Event indices for standard PERF_TYPE_HARDWARE events
enum PerfEventIdx {
    IDX_CPU_CYCLES = 0,
    IDX_INSTRUCTIONS,
    IDX_BRANCH_INST,
    IDX_BRANCH_MISS,
    IDX_L1D_CACHE_REFS,
    IDX_L1D_CACHE_MISSES,
    IDX_LL_CACHE_REFS,
    IDX_LL_CACHE_MISSES,
    PERF_EVENT_COUNT
};

double DividePerfCounter(uint64_t dividend, uint64_t divisor);
double PercentPerfCounter(uint64_t dividend, uint64_t divisor);
PerfCacheMetrics BuildPerfCacheMetrics(uint64_t refs, uint64_t misses);
PerfDerivedMetrics BuildPerfDerivedMetrics(const uint64_t* counts);

#if __PYPTO_AICPU_PMU_EVENT_ENABLE
struct PerfEventRecord {
    int type_{0};
    uint64_t config_{0};
    int fd_{-1};
    const char* name_{nullptr};
    bool valid_{false};
};

class PerfEventGroup {
public:
    explicit PerfEventGroup(pid_t tid);
    ~PerfEventGroup();

    PerfEventGroup(const PerfEventGroup&) = delete;
    PerfEventGroup& operator=(const PerfEventGroup&) = delete;

    int GetNrEvent() const;
    int GetValidEventCount() const;
    int AddEvent(int type, uint64_t config, const char* name);
    bool Enable();
    void Disable();
    int Read(uint64_t* counts);

private:
    pid_t tid_;
    int nrEvent_{0};
    int validEventCount_{0};
    int groupFd_{-1};
    PerfEventRecord events_[MAX_PERF_EVENT_NUM];
};

class AicpuPerfEventSampler {
public:
    AicpuPerfEventSampler();

    void Begin();
    void End();
    void Dump();

private:
    void TryAddEvent(int type, uint64_t config, const char* name);
    void TryAddCacheEvent(uint64_t cacheId, uint64_t opId, uint64_t resultId, const char* name);
    void DumpSummary(const char* title);
    void DumpReport(const uint64_t* counts);
    void DumpElapsedCycles();
    void DumpSectionHeader(const char* title);
    void DumpRawCounters(const uint64_t* counts);
    PerfDerivedMetrics BuildDerivedMetrics(const uint64_t* counts);
    PerfCacheMetrics BuildCacheMetrics(uint64_t refs, uint64_t misses);
    void DumpDerivedMetrics(const PerfDerivedMetrics& metrics);
    void DumpCacheDerivedMetric(const char* name, const PerfCacheMetrics& metrics);
    static double Divide(uint64_t dividend, uint64_t divisor);
    static double Percent(uint64_t dividend, uint64_t divisor);

    uint64_t cycles{0};
    PerfEventGroup events;
    bool pmuAvailable{true};
    bool pmuEnabled{false};
};

static inline AicpuPerfEventSampler& GetAicpuPerfEventSampler()
{
    static thread_local AicpuPerfEventSampler sampler;
    return sampler;
}

class AicpuPerfScopedSampler {
public:
    explicit AicpuPerfScopedSampler(const char* sectionName);
    ~AicpuPerfScopedSampler();

private:
    const char* sectionName_{"unnamed"};
    AicpuPerfEventSampler& sampler_;
};

#define AICPU_PMU_SCOPE(section_name_literal) \
    ::npu::tile_fwk::AicpuPerfScopedSampler aicpuPerfScopedSampler_##__LINE__(section_name_literal)

#define AICPU_PMU_BEGIN(sampler_name)                                 \
    auto& sampler_name = ::npu::tile_fwk::GetAicpuPerfEventSampler(); \
    (sampler_name).Begin()

#define AICPU_PMU_END(sampler_name, section_name_literal)                         \
    do {                                                                          \
        (sampler_name).End();                                                     \
        DEV_ERROR(MachineError::UNKNOWN, "[AICPU_PMU] %s", section_name_literal); \
        (sampler_name).Dump();                                                    \
    } while (0)

// 外部对象式采样（跨函数场景）
#define AICPU_PMU_BEGIN_EXTERNAL(sampler_ptr) \
    do {                                      \
        (sampler_ptr)->Begin();               \
    } while (0)

#define AICPU_PMU_END_EXTERNAL(sampler_ptr, section_name_literal)                 \
    do {                                                                          \
        (sampler_ptr)->End();                                                     \
        DEV_ERROR(MachineError::UNKNOWN, "[AICPU_PMU] %s", section_name_literal); \
        (sampler_ptr)->Dump();                                                    \
    } while (0)
#else
class AicpuPerfEventSampler {
public:
    void Begin() {}
    void End() {}
    void Dump() {}
};

static inline AicpuPerfEventSampler& GetAicpuPerfEventSampler()
{
    static thread_local AicpuPerfEventSampler sampler;
    return sampler;
}

class AicpuPerfScopedSampler {
public:
    explicit AicpuPerfScopedSampler(const char* sectionName) { (void)sectionName; }
};

#define AICPU_PMU_SCOPE(section_name_literal)
#define AICPU_PMU_BEGIN(sampler_name)
#define AICPU_PMU_END(sampler_name, section_name_literal)
#define AICPU_PMU_BEGIN_EXTERNAL(sampler_ptr)
#define AICPU_PMU_END_EXTERNAL(sampler_ptr, section_name_literal)
#endif

} // namespace npu::tile_fwk
