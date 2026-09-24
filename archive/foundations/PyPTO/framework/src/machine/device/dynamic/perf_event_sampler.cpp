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
 * \file perf_event_sampler.cpp
 * \brief
 */

#include "machine/device/dynamic/perf_event_sampler.h"

namespace npu::tile_fwk {

double DividePerfCounter(uint64_t dividend, uint64_t divisor)
{
    return divisor > 0 ? static_cast<double>(dividend) / divisor : 0.0;
}

double PercentPerfCounter(uint64_t dividend, uint64_t divisor) { return DividePerfCounter(dividend, divisor) * 100.0; }

PerfCacheMetrics BuildPerfCacheMetrics(uint64_t refs, uint64_t misses)
{
    PerfCacheMetrics metrics;
    if (refs == 0) {
        return metrics;
    }
    metrics.valid = true;
    metrics.missRate = PercentPerfCounter(misses, refs);
    return metrics;
}

PerfDerivedMetrics BuildPerfDerivedMetrics(const uint64_t* counts)
{
    PerfDerivedMetrics metrics;
    metrics.ipc = DividePerfCounter(counts[IDX_INSTRUCTIONS], counts[IDX_CPU_CYCLES]);
    metrics.cpi = DividePerfCounter(counts[IDX_CPU_CYCLES], counts[IDX_INSTRUCTIONS]);
    metrics.branchMissRate = PercentPerfCounter(counts[IDX_BRANCH_MISS], counts[IDX_BRANCH_INST]);
    metrics.l1dCache = BuildPerfCacheMetrics(counts[IDX_L1D_CACHE_REFS], counts[IDX_L1D_CACHE_MISSES]);
    metrics.llCache = BuildPerfCacheMetrics(counts[IDX_LL_CACHE_REFS], counts[IDX_LL_CACHE_MISSES]);
    return metrics;
}

} // namespace npu::tile_fwk

#if __PYPTO_AICPU_PMU_EVENT_ENABLE
#include <cinttypes>
#include <cstddef>
#include <cerrno>
#include <linux/perf_event.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "machine/device/dynamic/device_utils.h"
#include "securec.h"

namespace npu::tile_fwk {
namespace {

uint64_t MakeCacheEventConfig(uint64_t cacheId, uint64_t opId, uint64_t resultId)
{
    return cacheId | (opId << 8) | (resultId << 16);
}

std::string FormatNumber(uint64_t n)
{
    char buf[32] = {0};
    int ret = snprintf_s(buf, sizeof(buf), sizeof(buf) - 1, "%" PRIu64, n);
    return ret < 0 ? std::string() : std::string(buf);
}

pid_t GetCurrentTid() { return static_cast<pid_t>(syscall(__NR_gettid)); }

} // namespace

PerfEventGroup::PerfEventGroup(pid_t tid) : tid_(tid) {}

PerfEventGroup::~PerfEventGroup()
{
    for (int i = 0; i < nrEvent_; i++) {
        if (events_[i].fd_ >= 0) {
            close(events_[i].fd_);
        }
    }
}

int PerfEventGroup::GetNrEvent() const { return nrEvent_; }

int PerfEventGroup::GetValidEventCount() const { return validEventCount_; }

int PerfEventGroup::AddEvent(int type, uint64_t config, const char* name)
{
    if (nrEvent_ == MAX_PERF_EVENT_NUM) {
        return -EINVAL;
    }

    struct perf_event_attr pe;
    if (memset_s(&pe, sizeof(pe), 0, sizeof(pe)) != EOK) {
        DEV_WARN("[AICPU_PMU] memset_s perf_event_attr failed");
        return -1;
    }
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    if (groupFd_ == -1) {
        pe.read_format = PERF_FORMAT_GROUP;
    }

    int fd = syscall(__NR_perf_event_open, &pe, tid_, -1, groupFd_, 0);
    if (fd < 0) {
        DEV_WARN("[AICPU_PMU] Failed to register event '%s' (type=%d, config=%lu, errno=%d), "
                 "event will be marked as unavailable",
                 name, type, config, errno);
        events_[nrEvent_++] = {type, config, -1, name, false};
        return -1;
    }
    if (groupFd_ == -1) {
        groupFd_ = fd;
    }
    events_[nrEvent_++] = {type, config, fd, name, true};
    validEventCount_++;
    return 0;
}

bool PerfEventGroup::Enable()
{
    if (groupFd_ == -1) {
        DEV_WARN("[AICPU_PMU] Enable failed: no valid perf event group (all events failed to register)");
        return false;
    }
    if (ioctl(groupFd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0) {
        DEV_WARN("[AICPU_PMU] PERF_EVENT_IOC_RESET failed, errno=%d", errno);
        return false;
    }
    if (ioctl(groupFd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) {
        DEV_WARN("[AICPU_PMU] PERF_EVENT_IOC_ENABLE failed, errno=%d", errno);
        return false;
    }
    return true;
}

void PerfEventGroup::Disable()
{
    if (groupFd_ != -1) {
        ioctl(groupFd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }
}

int PerfEventGroup::Read(uint64_t* counts)
{
    uint64_t buf[MAX_PERF_EVENT_NUM + 1];
    if (groupFd_ == -1 || validEventCount_ == 0) {
        return 0;
    }
    ssize_t len = read(groupFd_, buf, sizeof(buf));
    if (len < 0) {
        DEV_WARN("[AICPU_PMU] read perf event group failed, errno=%d", errno);
        return 0;
    }
    size_t expectedLen = (validEventCount_ + 1) * sizeof(uint64_t);
    if (static_cast<size_t>(len) != expectedLen) {
        DEV_WARN("[AICPU_PMU] read perf event group length mismatch, actual=%zd, expected=%zu", len, expectedLen);
        return 0;
    }

    int validIdx = 0;
    for (int i = 0; i < nrEvent_; i++) {
        if (events_[i].valid_) {
            counts[i] = buf[validIdx + 1];
            validIdx++;
        } else {
            counts[i] = 0;
        }
    }
    return buf[0];
}

AicpuPerfEventSampler::AicpuPerfEventSampler() : events(GetCurrentTid())
{
    // 使用 PERF_TYPE_HARDWARE 标准事件，权限要求较低且兼容性更好。
    TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cpu_cycles");
    TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions");
    TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branch_inst");
    TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch_miss");
    TryAddCacheEvent(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                     "l1d_cache_refs");
    TryAddCacheEvent(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS,
                     "l1d_cache_misses");
    TryAddCacheEvent(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                     "ll_cache_refs");
    TryAddCacheEvent(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS,
                     "ll_cache_misses");

    if (events.GetValidEventCount() > 0) {
        DEV_INFO("[AICPU_PMU] Registered %d/%d PMU events successfully", events.GetValidEventCount(),
                 events.GetNrEvent());
    } else {
        DEV_WARN("[AICPU_PMU] PMU events unavailable (errno=%d). "
                 "Possible causes: container restrictions or missing capabilities. "
                 "Fallback to time-only mode.",
                 errno);
        pmuAvailable = false;
    }
}

void AicpuPerfEventSampler::Begin()
{
    pmuEnabled = events.Enable();
    cycles = dynamic::GetCycles();
}

void AicpuPerfEventSampler::End()
{
    cycles = dynamic::GetCycles() - cycles;
    events.Disable();
}

void AicpuPerfEventSampler::Dump()
{
    if (!pmuAvailable || !pmuEnabled || events.GetValidEventCount() == 0) {
        DumpSummary("ExecDyn Summary (PMU unavailable)");
        DEV_ERROR(MachineError::UNKNOWN, "  Note: PMU events disabled due to permission restrictions");
        return;
    }

    uint64_t counts[MAX_PERF_EVENT_NUM] = {0};
    int n = events.Read(counts);
    if (n == 0) {
        DumpSummary("ExecDyn Summary");
        return;
    }

    DumpReport(counts);
}

void AicpuPerfEventSampler::TryAddEvent(int type, uint64_t config, const char* name)
{
    int ret = events.AddEvent(type, config, name);
    if (ret < 0) {
        DEV_DEBUG("[AICPU_PMU] Failed to register: %s (type=%d, config=%lu, errno=%d)", name, type, config, errno);
    }
}

void AicpuPerfEventSampler::TryAddCacheEvent(uint64_t cacheId, uint64_t opId, uint64_t resultId, const char* name)
{
    TryAddEvent(PERF_TYPE_HW_CACHE, MakeCacheEventConfig(cacheId, opId, resultId), name);
}

void AicpuPerfEventSampler::DumpSummary(const char* title)
{
    DEV_ERROR(MachineError::UNKNOWN, "[AICPU_PMU] %s", title);
    DumpElapsedCycles();
}

void AicpuPerfEventSampler::DumpReport(const uint64_t* counts)
{
    DEV_ERROR(MachineError::UNKNOWN, "[AICPU_PMU] Performance Report");
    DEV_ERROR(MachineError::UNKNOWN, "============================================================");
    DumpElapsedCycles();
    DumpRawCounters(counts);
    DumpDerivedMetrics(BuildDerivedMetrics(counts));
    DEV_ERROR(MachineError::UNKNOWN, "============================================================");
}

void AicpuPerfEventSampler::DumpElapsedCycles()
{
    DEV_ERROR(MachineError::UNKNOWN, "Elapsed Cycles: %s", FormatNumber(cycles).c_str());
}

void AicpuPerfEventSampler::DumpSectionHeader(const char* title)
{
    DEV_ERROR(MachineError::UNKNOWN, "------------------------------------------------------------");
    DEV_ERROR(MachineError::UNKNOWN, "%s", title);
    DEV_ERROR(MachineError::UNKNOWN, "------------------------------------------------------------");
}

void AicpuPerfEventSampler::DumpRawCounters(const uint64_t* counts)
{
    DumpSectionHeader("PMU Event Counters");
    DEV_ERROR(MachineError::UNKNOWN, "  CPU Cycles:         %s", FormatNumber(counts[IDX_CPU_CYCLES]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  Instructions:       %s", FormatNumber(counts[IDX_INSTRUCTIONS]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  Branch Instructions:%s", FormatNumber(counts[IDX_BRANCH_INST]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  Branch Misses:      %s", FormatNumber(counts[IDX_BRANCH_MISS]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  L1D Cache Refs:     %s", FormatNumber(counts[IDX_L1D_CACHE_REFS]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  L1D Cache Misses:   %s", FormatNumber(counts[IDX_L1D_CACHE_MISSES]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  LL Cache Refs:      %s", FormatNumber(counts[IDX_LL_CACHE_REFS]).c_str());
    DEV_ERROR(MachineError::UNKNOWN, "  LL Cache Misses:    %s", FormatNumber(counts[IDX_LL_CACHE_MISSES]).c_str());
}

PerfDerivedMetrics AicpuPerfEventSampler::BuildDerivedMetrics(const uint64_t* counts)
{
    return BuildPerfDerivedMetrics(counts);
}

PerfCacheMetrics AicpuPerfEventSampler::BuildCacheMetrics(uint64_t refs, uint64_t misses)
{
    return BuildPerfCacheMetrics(refs, misses);
}

void AicpuPerfEventSampler::DumpDerivedMetrics(const PerfDerivedMetrics& metrics)
{
    DumpSectionHeader("Derived Metrics");
    DEV_ERROR(MachineError::UNKNOWN, "  IPC:                %.2f", metrics.ipc);
    DEV_ERROR(MachineError::UNKNOWN, "  CPI:                %.2f", metrics.cpi);
    DEV_ERROR(MachineError::UNKNOWN, "  Branch Miss Rate:   %.2f%%", metrics.branchMissRate);
    DumpCacheDerivedMetric("L1D Cache", metrics.l1dCache);
    DumpCacheDerivedMetric("LL Cache", metrics.llCache);
}

void AicpuPerfEventSampler::DumpCacheDerivedMetric(const char* name, const PerfCacheMetrics& metrics)
{
    if (!metrics.valid) {
        DEV_ERROR(MachineError::UNKNOWN, "  %s Hit Rate: N/A", name);
        DEV_ERROR(MachineError::UNKNOWN, "  %s Miss Rate:N/A", name);
        return;
    }
    DEV_ERROR(MachineError::UNKNOWN, "  %s Hit Rate: %.2f%%", name, 100.0 - metrics.missRate);
    DEV_ERROR(MachineError::UNKNOWN, "  %s Miss Rate:%.2f%%", name, metrics.missRate);
}

double AicpuPerfEventSampler::Divide(uint64_t dividend, uint64_t divisor)
{
    return DividePerfCounter(dividend, divisor);
}

double AicpuPerfEventSampler::Percent(uint64_t dividend, uint64_t divisor)
{
    return PercentPerfCounter(dividend, divisor);
}

AicpuPerfScopedSampler::AicpuPerfScopedSampler(const char* sectionName)
    : sectionName_(sectionName), sampler_(GetAicpuPerfEventSampler())
{
    sampler_.Begin();
}

AicpuPerfScopedSampler::~AicpuPerfScopedSampler()
{
    sampler_.End();
    DEV_ERROR(MachineError::UNKNOWN, "[AICPU_PMU] %s", sectionName_);
    sampler_.Dump();
}

} // namespace npu::tile_fwk
#endif
