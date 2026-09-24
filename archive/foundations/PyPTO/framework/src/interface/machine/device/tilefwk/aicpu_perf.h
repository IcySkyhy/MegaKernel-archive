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
 * \file aicpu_perf.h
 * \brief
 */

#ifndef AICPU_PREF_H
#define AICPU_PREF_H

#include <cstdint>

namespace npu::tile_fwk::dynamic {
constexpr int MAX_LAUNCH_SCHEDULE_AICPU_NUM = 6;
constexpr uint32_t MAX_SCHEDULE_AICPU_NUM = 5;     // 真正负责调度aicore的最大aicpu个数
constexpr uint32_t MAX_CONTROL_FLOW_AICPU_NUM = 1; // contral flow aicpu数量
constexpr uint32_t MAX_USED_AICPU_NUM = MAX_SCHEDULE_AICPU_NUM + MAX_CONTROL_FLOW_AICPU_NUM;
constexpr uint32_t FREQ_DAV_2201 = 50;
constexpr uint32_t FREQ_DAV_3510 = 1000;

#define PERF_TRACES                         \
    X(BEGIN)                                \
    X(ALLOC_THREAD_ID)                      \
    X(INIT)                                 \
    X(CORE_HAND_SHAKE)                      \
    XDEVTASK(DEV_TASK_BUILD)                \
    XDEVTASK(DEV_TASK_RCV)                  \
    XDEVTASK(DEV_TASK_SEND_FIRST_LEAF_TASK) \
    XDEVTASK(DEV_TASK_SCHED_EXEC)           \
    XDEVTASK(DEV_TASK_SYNC_CORE_STOP)       \
    XDEVTASK(DEV_TASK_RSP)                  \
    X(WAIT_ALL_DEV_TASK_FINISH)             \
    X(WAIT_CORE_EXIT)                       \
    X(EXIT)                                 \
    X(MAX)

enum PerfTraceType {
#define X(trace) PERF_TRACE_##trace,
#define XDEVTASK(trace) PERF_TRACE_##trace,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

#define X(trace)
#define XDEVTASK(trace) | (1ULL << PERF_TRACE_##trace)
constexpr uint64_t DEV_TASK_TYPE_MASK = 0ULL PERF_TRACES;
#undef XDEVTASK
#undef X

inline constexpr bool IsDevTaskType(uint32_t type) { return (DEV_TASK_TYPE_MASK >> type) & 1ULL; }

inline const char* PerfTraceName[] = {
#define X(trace) #trace,
#define XDEVTASK(trace) #trace,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

#define DEVTASK_PERF_ARRY_INDEX(type) (type - PERF_TRACE_DEV_TASK_BUILD)
inline constexpr uint32_t DEVTASK_PERF_TYPE_NUM = (PERF_TRACE_DEV_TASK_RSP - PERF_TRACE_DEV_TASK_BUILD + 1);
inline constexpr uint32_t PERF_TRACE_COUNT_DEVTASK_MAX_NUM = 200;

#undef PERF_TRACES
} // namespace npu::tile_fwk::dynamic
#endif
