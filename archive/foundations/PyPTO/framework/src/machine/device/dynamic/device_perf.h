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
 * \file device_perf.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
struct PerfettoMgr {
    static const int MAX_THEAD_NUM = 200;
    static const int TRUNK_SIZE = 32768;
    static const int MAX_EVT_DEPTH = 64;

    struct Record {
        int type = 0;
        int tid = 0;
        uint64_t start = 0;
        uint64_t end = 0;
        uint32_t index = 0;
        uint32_t pIndex = 0;
        std::string name = "-";
    };

    template <typename T, int N>
    struct Array {
        size_t used{0};
        T data[N];

        inline void Push(const T& t) { data[used++] = t; }
        inline void Pop() { used--; }
        inline bool Full() { return used == N; }
        inline bool Empty() { return used == 0; }
        inline T& Top() { return data[used - 1]; }
        inline T* Alloc() { return &data[used++]; }
    };

    using Trunk = Array<Record, TRUNK_SIZE>;
    using EvtStack = Array<Record*, MAX_EVT_DEPTH>;

    Record* allocRecord(int tid)
    {
        if (trunk_[tid] == nullptr || trunk_[tid]->Full()) {
            trunk_[tid] = new Trunk;
            mutex_.lock();
            trunks_.push_back(trunk_[tid]);
            mutex_.unlock();
        }
        return trunk_[tid]->Alloc();
    }

    void PerfBegin(int type, int tid)
    {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto r = allocRecord(tid);
        auto& stack = evtStack[tid];
        r->type = type;
        r->tid = tid;
        r->start = GetCycles();
        r->index = evtIndex++;
        r->pIndex = stack.Empty() ? -1 : stack.Top()->index;
        stack.Push(r);
#endif
        (void)type;
        (void)tid;
    }

    void PerfEnd(int type, int tid)
    {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto& stack = evtStack[tid];
        auto r = stack.Top();
        r->end = GetCycles();
        stack.Pop();
#endif
        (void)type;
        (void)tid;
    }

    void PerfEvent(int type, int tid, uint64_t start, uint64_t end, std::string name)
    {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto r = allocRecord(tid);
        auto& stack = evtStack[tid];
        r->type = type;
        r->tid = tid;
        r->name = name;
        r->start = start;
        r->end = end;
        r->index = evtIndex++;
        r->pIndex = stack.Empty() ? -1 : stack.Top()->index;
#endif
        (void)type;
        (void)tid;
        (void)start;
        (void)end;
        (void)name;
    }

    static PerfettoMgr& Instance()
    {
        static PerfettoMgr recorder;
        return recorder;
    }

    void Dump(const std::string& file)
    {
        std::ofstream os(file);
        for (auto& trunk : trunks_) {
            for (size_t i = 0; i < trunk->used; i++) {
                auto& r = trunk->data[i];
                os << PerfEventName[r.type] << " ";
                os << r.name << " ";
                os << r.index << " ";
                os << r.start << " ";
                os << r.end << " ";
                os << r.tid << ";";
                os << r.pIndex << " ";
                os << std::endl;
            }
        }
    }

private:
    PerfettoMgr() = default;
    ~PerfettoMgr()
    {
        for (auto* trunk : trunks_) {
            delete trunk;
        }
        trunks_.clear();
    }

private:
    EvtStack evtStack[MAX_THEAD_NUM];
    Trunk* trunk_[MAX_THEAD_NUM] = {nullptr};

    std::mutex mutex_;
    std::vector<Trunk*> trunks_;
    std::atomic<int32_t> evtIndex;
};
struct PerfEvtMgr {
    struct Counter {
        int64_t start;
        int64_t total;
        int64_t count;
    };

    bool GetIsOpenProf() { return isOpenProf_; }

    void SetIsOpenProf(bool isOpenProf, uint64_t aicpuPerf = 0)
    {
        if (ctrlTurn_ >= MAX_ROUND_NUM) {
            aicpuPerf_ = 0;
            isOpenProf_ = false;
            DEV_WARN("Aicpu perf info more than maxTurnNum=%u, some info would be lost", MAX_ROUND_NUM);
            return;
        }
        isOpenProf_ = isOpenProf;
        aicpuPerf_ = aicpuPerf;
    }

    void AddCtrlTurn() { ctrlTurn_++; }

    void AddScheduleTurn() { schTurn_++; }

    void PerfBegin(int type) { counters[type].start = static_cast<int64_t>(GetCycles()); }

    void PerfEnd(int type)
    {
        auto& c = counters[type];
        c.count++;
        c.total += static_cast<int64_t>(GetCycles() - c.start);
    }

    static PerfEvtMgr& Instance()
    {
        static PerfEvtMgr recorder;
        return recorder;
    }

    static void RepeatPuts(char c, size_t count)
    {
        char buf[80];
        for (size_t i = 0; i < count; i++) {
            buf[i] = c;
        }
        buf[count] = '\0';
        DEV_ERROR(0, "%s.", buf);
    }

    void Dump()
    {
        uint64_t freq = GetFreq();
        static constexpr size_t SHEET_WIDTH = 40 + 3 + 10 + 3 + 10 + 3 + 10;

        RepeatPuts('=', SHEET_WIDTH);
        DEV_ERROR(0, "%40s | %10s | %10s | %10s.", "EventType", "Count", "Total(us)", "Avg(us)");
        RepeatPuts('-', SHEET_WIDTH);

        for (int i = 0; i < PERF_EVT_MAX; i++) {
            auto evt = counters[i];
            if (evt.count != 0) {
                uint64_t total = evt.total * NSEC_PER_SEC / freq / NSEC_PER_USEC;
                float avg = static_cast<float>(total / evt.count);
                DEV_ERROR(0, "%-40s | %10ld | %10lu | %10.1f.", PerfEventName[i], evt.count, total, avg);
            }
        }

        RepeatPuts('=', SHEET_WIDTH);
    }

    void PerfTrace(uint32_t type, uint32_t tid, uint64_t cycle)
    {
        if (unlikely(tid >= MAX_USED_AICPU_NUM || aicpuPerf_ == 0)) {
            return;
        }

        MetricPerf* aicpuMetrics = (MetricPerf*)(aicpuPerf_ + (tid == 0 ? ctrlTurn_ : schTurn_) * sizeof(MetricPerf));

        if (IsDevTaskType(type)) {
            uint32_t idx = DEVTASK_PERF_ARRY_INDEX(type);
            DevTaskPerf& perfSlot = aicpuMetrics->devTaskPerfs[tid][idx];
            uint64_t& devCnt = perfSlot.cnt;
            if (unlikely(devCnt >= PERF_TRACE_COUNT_DEVTASK_MAX_NUM)) {
                DEV_EVENT("Dev task num larger than: %u, the excess part will not be recorded",
                          PERF_TRACE_COUNT_DEVTASK_MAX_NUM);
                return;
            }

            perfSlot.timeStamp[devCnt] = cycle == 0 ? static_cast<uint64_t>(GetCycles()) : cycle;
            devCnt++;
        } else {
            aicpuMetrics->perfAicpuTrace[tid][type] = cycle == 0 ? static_cast<uint64_t>(GetCycles()) : cycle;
        }
    }

private:
    PerfEvtMgr()
    {
#if ENABLE_PERF_EVT
        memset_s(counters, sizeof(counters), 0, sizeof(counters));
#endif
    };

private:
    Counter counters[PERF_EVT_MAX];
    bool isOpenProf_{false};
    uint64_t aicpuPerf_{0};
    uint32_t ctrlTurn_{0};
    uint32_t schTurn_{0};
};

inline void PerfBegin(int type)
{
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfBegin(type);
#else
    (void)type;
#endif
}

inline void PerfEnd(int type)
{
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfEnd(type);
#else
    (void)type;
#endif
}

inline void PerfMtBegin(int type, int tid)
{
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfBegin(type + tid);
#else
    (void)type;
    (void)tid;
#endif
}

inline void PerfMtEnd(int type, int tid)
{
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfEnd(type + tid);
#else
    (void)type;
    (void)tid;
#endif
}

inline void PerfMtEvent(int type, int tid, uint64_t start, uint64_t end, std::string name = "-")
{
    if (PerfEvtEnable[type]) {
        PerfettoMgr::Instance().PerfEvent(type, tid, start, end, name);
    }
}

inline void PerfMtTrace(uint32_t type, uint32_t tid, uint64_t cycle = 0)
{
    PerfEvtMgr::Instance().PerfTrace(type, tid, cycle);
}

struct AutoScopedPerf {
    explicit AutoScopedPerf(int type) : type_(type) { PerfBegin(type); }
    ~AutoScopedPerf() { PerfEnd(type_); }
    int type_;
};
} // namespace npu::tile_fwk::dynamic
