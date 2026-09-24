/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <string>

namespace npu::tile_fwk {

#define TRACEPHASE_LIST                      \
    PERF_DEFINE(RunDeviceInit)               \
    PERF_DEFINE(RunDeviceSetCapture)         \
    PERF_DEFINE(RunDevEnvReady)              \
    PERF_DEFINE(RunDevInitTiling)            \
    PERF_DEFINE(RunDevInitInOutTensor)       \
    PERF_DEFINE(RunDevRegistKernelBin)       \
    PERF_DEFINE(RunDevKernelInitErrCallBack) \
    PERF_DEFINE(RunDevKernelInitAicpuSo)     \
    PERF_DEFINE(RunDevKernelInitKernelArgs)  \
    PERF_DEFINE(RunDevKernelInitH2DMemCopy)  \
    PERF_DEFINE(RunDevKernelInitRunPrepare)  \
    PERF_DEFINE(RunDevKernelLaunchAicpuInit) \
    PERF_DEFINE(RunDevKernelLaunchAicpuRun)  \
    PERF_DEFINE(RunDevKernelLaunchAIcore)    \
    PERF_DEFINE(RunDevRunProfile)            \
    PERF_DEFINE(LaunchInit)                  \
    PERF_DEFINE(LaunchGetKernel)             \
    PERF_DEFINE(LaunchAllocWorkSpace)        \
    PERF_DEFINE(LaunchAttachStream)          \
    PERF_DEFINE(FindCtrlFlowCache)           \
    PERF_DEFINE(Launch)                      \
    PERF_DEFINE(MAX_TRACE_PHASES)

enum class TracePhase {
#define PERF_DEFINE(trace) trace,
    TRACEPHASE_LIST
#undef PERF_DEFINE
};

inline const std::string g_perfTraceName[] = {
#define PERF_DEFINE(trace) #trace,
    TRACEPHASE_LIST
#undef PERF_DEFINE
};

#define EVTPHASE_LIST               \
    PERF_DEFINE(INVALID)            \
    PERF_DEFINE(BuildCtrlFlowCache) \
    PERF_DEFINE(RunDevice)          \
    PERF_DEFINE(LaunchKernel)       \
    PERF_DEFINE(MAX_EVENT_PHASES)

enum class EventPhase {
#define PERF_DEFINE(evt) evt,
    EVTPHASE_LIST
#undef PERF_DEFINE
};

inline const std::string g_perfEventName[] = {
#define PERF_DEFINE(evt) #evt,
    EVTPHASE_LIST
#undef PERF_DEFINE
};

struct PerfData {
    uint64_t totalTimeNs{0};
    uint64_t count{0};
    uint64_t maxTimeNs{0};
    uint64_t minTimeNs{UINT64_MAX};
    std::string name;
    uint64_t ignoreHeaderCnt{0};

    PerfData() : totalTimeNs(0), count(0), maxTimeNs(0), minTimeNs(UINT64_MAX) {}

    uint64_t AvgTimeNs() const { return count > 0 ? totalTimeNs / count : 0; }

    void AddStat(uint64_t durationNs, bool isTrace)
    {
        count++;
        if (isTrace) {
            if (count <= ignoreHeaderCnt) {
                return;
            } else if (count == ignoreHeaderCnt + 1) {
                count = 1;
                ignoreHeaderCnt = 0;
            }
        }

        totalTimeNs += durationNs;
        if (durationNs > maxTimeNs)
            maxTimeNs = durationNs;
        if (durationNs < minTimeNs)
            minTimeNs = durationNs;
    }
};

#define HOST_PERF_SWITCH 0
#if HOST_PERF_SWITCH
#define HOST_PERF_TRACE_START() PerfAnalysis::Get().TraceStart()
#define HOST_PERF_TRACE(type) PerfAnalysis::Get().Trace(type)
#define HOST_PERF_EVT_BEGIN(type) PerfAnalysis::Get().EventBegin(type)
#define HOST_PERF_EVT_END(type) PerfAnalysis::Get().EventEnd(type)
#else
#define HOST_PERF_TRACE_START()
#define HOST_PERF_TRACE(type)
#define HOST_PERF_EVT_BEGIN(type)
#define HOST_PERF_EVT_END(type)
#endif
class PerfAnalysis {
private:
    PerfAnalysis()
    {
        traceData_.resize(static_cast<size_t>(TracePhase::MAX_TRACE_PHASES));
        eventData_.resize(static_cast<size_t>(EventPhase::MAX_EVENT_PHASES));
        InitDataNames();
        eventStartTimes_.resize(static_cast<size_t>(EventPhase::MAX_EVENT_PHASES));
        Reset();
    }

    PerfAnalysis(const PerfAnalysis&) = delete;
    PerfAnalysis& operator=(const PerfAnalysis&) = delete;

    std::vector<PerfData> traceData_;
    std::vector<PerfData> eventData_;
    std::vector<std::chrono::high_resolution_clock::time_point> eventStartTimes_;
    std::chrono::high_resolution_clock::time_point initTime_;
    std::chrono::high_resolution_clock::time_point lastTraceTime_;
    bool isTraceInitialized_{false};

    void InitDataNames()
    {
        for (int i = 0; i < static_cast<int>(TracePhase::MAX_TRACE_PHASES); i++) {
            traceData_[i].name = std::string(g_perfTraceName[i]);
        }

        for (int i = 0; i < static_cast<int>(EventPhase::MAX_EVENT_PHASES); i++) {
            eventData_[i].name = std::string(g_perfEventName[i]);
        }
    }

    uint64_t CalculateTraceAvgSumNs()
    {
        uint64_t traceAvgSumNs = 0;
        for (int i = 0; i < static_cast<int>(TracePhase::MAX_TRACE_PHASES); i++) {
            const auto& data = traceData_[i];
            if (data.count > 0) {
                traceAvgSumNs += data.AvgTimeNs();
            }
        }
        return traceAvgSumNs;
    }

    uint64_t CalculateEventAvgSumNs()
    {
        uint64_t eventAvgSumNs = 0;
        for (int i = 0; i < static_cast<int>(EventPhase::MAX_EVENT_PHASES); i++) {
            const auto& data = eventData_[i];
            if (data.count > 0) {
                eventAvgSumNs += data.AvgTimeNs();
            }
        }
        return eventAvgSumNs;
    }

    void PrintPerfTable(std::ostream& out, const std::vector<PerfData>& dataVec, size_t dataSize, uint64_t totalTimeNs,
                        uint64_t avgSumNs, const std::string& title)
    {
        out << "\n--- " << title << " Statistics ---" << std::endl;

        out << std::left << std::setw(40) << title + " Name" << std::setw(12) << "Count" << std::setw(15)
            << "Total Time(us)" << std::setw(18) << "Total Percent(%)" << std::setw(15) << "Avg Time(us)"
            << std::setw(18) << "Avg Percent(%)" << std::setw(15) << "Max Time(us)" << std::setw(15) << "Min Time(us)"
            << std::endl;

        out << std::string(140, '-') << std::endl;

        for (size_t i = 0; i < dataSize; i++) {
            const auto& data = dataVec[i];
            if (data.count > 0) {
                double totalTimeUs = static_cast<double>(data.totalTimeNs) / 1000.0;
                double avgTimeUs = static_cast<double>(data.AvgTimeNs()) / 1000.0;
                double maxTimeUs = static_cast<double>(data.maxTimeNs) / 1000.0;
                double minTimeUs = static_cast<double>(data.minTimeNs) / 1000.0;

                double totalPercent = (totalTimeNs > 0) ?
                                          (static_cast<double>(data.totalTimeNs) / totalTimeNs * 100.0) :
                                          0.0;
                double avgPercent = (avgSumNs > 0) ? (static_cast<double>(data.AvgTimeNs()) / avgSumNs * 100.0) : 0.0;

                out << std::left << std::fixed << std::setprecision(3) << std::setw(40) << data.name << std::setw(12)
                    << data.count << std::setw(15) << totalTimeUs << std::setw(18) << totalPercent << std::setw(15)
                    << avgTimeUs << std::setw(18) << avgPercent << std::setw(15) << maxTimeUs << std::setw(15)
                    << minTimeUs << std::endl;
            }
        }

        out << std::string(140, '-') << std::endl;

        out << std::left << std::setw(40) << "TOTAL" << std::setw(12) << "-" << std::setw(15)
            << static_cast<double>(totalTimeNs) / 1000.0 << std::setw(15) << "-" << std::setw(15)
            << static_cast<double>(avgSumNs) / 1000.0 << std::setw(15) << "-" << std::setw(15) << "-" << std::setw(15)
            << "-" << std::endl;
    }

public:
    static PerfAnalysis& Get();

    void TraceStart()
    {
        lastTraceTime_ = std::chrono::high_resolution_clock::now();
        isTraceInitialized_ = true;
    }

    void Trace(TracePhase phase)
    {
        if (!isTraceInitialized_) {
            TraceStart();
            auto idx = static_cast<size_t>(phase);
            if (idx < traceData_.size()) {
                traceData_[idx].AddStat(0, true);
            }
            return;
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - lastTraceTime_).count();

        auto idx = static_cast<size_t>(phase);
        if (idx < traceData_.size()) {
            traceData_[idx].AddStat(durationNs, true);
        }

        lastTraceTime_ = now;
    }

    void EventBegin(EventPhase phase)
    {
        auto idx = static_cast<size_t>(phase);
        eventStartTimes_[idx] = std::chrono::high_resolution_clock::now();
    }

    void EventEnd(EventPhase phase)
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto idx = static_cast<size_t>(phase);

        if (eventStartTimes_[idx] != std::chrono::high_resolution_clock::time_point()) {
            auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - eventStartTimes_[idx]).count();

            if (idx < eventData_.size()) {
                eventData_[idx].AddStat(durationNs, false);
            }

            eventStartTimes_[idx] = std::chrono::high_resolution_clock::time_point();
        }
    }

    void ResetTrace()
    {
        for (auto& data : traceData_) {
            data.totalTimeNs = 0;
            data.count = 0;
            data.maxTimeNs = 0;
            data.minTimeNs = UINT64_MAX;
        }
        isTraceInitialized_ = false;
    }

    void ResetEvent()
    {
        for (auto& data : eventData_) {
            data.totalTimeNs = 0;
            data.count = 0;
            data.maxTimeNs = 0;
            data.minTimeNs = UINT64_MAX;
        }
        for (auto& timePoint : eventStartTimes_) {
            timePoint = std::chrono::high_resolution_clock::time_point();
        }
    }

    void Reset()
    {
        ResetTrace();
        ResetEvent();
        initTime_ = std::chrono::high_resolution_clock::now();
    }

    uint64_t GetTotalTimeUs()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto totalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - initTime_).count();
        return totalNs / 1000;
    }

    uint64_t GetTraceTotalTimeUs()
    {
        uint64_t totalNs = 0;
        for (const auto& data : traceData_) {
            totalNs += data.totalTimeNs;
        }
        return totalNs / 1000;
    }

    uint64_t GetEventTotalTimeUs()
    {
        uint64_t totalNs = 0;
        for (const auto& data : eventData_) {
            totalNs += data.totalTimeNs;
        }
        return totalNs / 1000;
    }

    uint64_t GetAllTotalTimeUs() { return GetTraceTotalTimeUs() + GetEventTotalTimeUs(); }

    void Dump(bool toFile = false, const std::string& filename = "perf_stats.txt")
    {
        auto totalTimeUs = GetTotalTimeUs();
        auto traceTotalUs = GetTraceTotalTimeUs();
        auto eventTotalUs = GetEventTotalTimeUs();
        auto allTotalUs = GetAllTotalTimeUs();

        std::ostream* output = &std::cout;
        std::ofstream fileStream;

        if (toFile) {
            fileStream.open(filename);
            if (fileStream.is_open()) {
                output = &fileStream;
            } else {
                std::cerr << "Failed to open file: " << filename << ", outputting to console instead." << std::endl;
            }
        }

        std::ostream& out = *output;

        out << "========== Perf Statistics ==========" << std::endl;
        uint64_t traceTotalNs = traceTotalUs * 1000;
        uint64_t traceAvgSumNs = CalculateTraceAvgSumNs();
        PrintPerfTable(out, traceData_, static_cast<size_t>(TracePhase::MAX_TRACE_PHASES), traceTotalNs, traceAvgSumNs,
                       "Trace");

        uint64_t eventTotalNs = eventTotalUs * 1000;
        uint64_t eventAvgSumNs = CalculateEventAvgSumNs();
        PrintPerfTable(out, eventData_, static_cast<size_t>(EventPhase::MAX_EVENT_PHASES), eventTotalNs, eventAvgSumNs,
                       "Event");

        out << std::endl << "--- Summary ---" << std::endl;
        out << "Total time since initialization: " << std::fixed << std::setprecision(3) << totalTimeUs / 1000.0
            << " ms"
            << " (" << totalTimeUs << " us)" << std::endl;

        out << "\nCombined Total Time: " << std::fixed << std::setprecision(3) << allTotalUs / 1000.0 << " ms"
            << " (" << allTotalUs << " us)" << std::endl;

        if (allTotalUs > 0 && totalTimeUs > 0) {
            double percentage = (double)allTotalUs / totalTimeUs * 100.0;
            out << "Percentage of total time: " << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
        }

        out << "============================================" << std::endl;

        if (toFile && fileStream.is_open()) {
            fileStream.close();
            std::cout << "Statistics dumped to file: " << filename << std::endl;
        }
    }
};
} // namespace npu::tile_fwk
