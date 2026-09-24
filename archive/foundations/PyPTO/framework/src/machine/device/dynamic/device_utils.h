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
 * \file device_utils.h
 * \brief
 */

#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <time.h>
#include <atomic>
#include <vector>
#include <mutex>
#include <sstream>
#include <fstream>
#include <ostream>
#include "securec.h"
#include "tilefwk/device_error_code.h"
#include "machine/utils/device_log.h"
#include "interface/machine/device/tilefwk/aicpu_perf.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"

#ifndef CONFIG_MAX_DEVICE_TASK_NUM
#define CONFIG_MAX_DEVICE_TASK_NUM 1024
#endif

#define MAX_DEVICE_TASK_NUM CONFIG_MAX_DEVICE_TASK_NUM

#define BITS_PER_BYTES 8
#define BITS_PER_INT 32
#define BITS_PER_LONG 64

namespace npu::tile_fwk::dynamic {
inline constexpr bool IsDeviceMode()
{
#ifdef __DEVICE__
    return true;
#else
    return false;
#endif // __DEVICE__
}

constexpr int32_t DEVICE_MACHINE_INVALID_RUN_MODE = PYPTO_DEVICE_ERROR_PARAM_INVALID;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH = PYPTO_DEVICE_ERROR_AICPU_TIMEOUT;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH = PYPTO_DEVICE_ERROR_AICORE_TIMEOUT;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_CORETASK = PYPTO_DEVICE_ERROR_TASK_TIMEOUT;
constexpr int32_t DEVICE_MACHINE_ERROR = PYPTO_DEVICE_ERROR_INTERNAL_ERROR;
constexpr int32_t DEVICE_MACHINE_OK = PYPTO_DEVICE_SUCCESS;

constexpr uint32_t CTRL_CPU_THREAD_IDX = 0;
constexpr int32_t START_AICPU_NUM = 3;
constexpr uint64_t NUM_FIFTY = 50;
constexpr uint64_t US_PER_SEC = 1000000;
constexpr uint64_t NSEC_PER_USEC = 1000;
constexpr uint64_t NSEC_PER_SEC = 1000000000;
constexpr int32_t MAX_MNG_AICORE_AVG_NUM = 8;
constexpr uint32_t CORE_IDX_AIV = 0;
constexpr uint32_t CORE_IDX_AIC = 1;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const int INVALID_CORE_IDX = 0xFF;

// ============================================================
// AICPU 超时常量
// ============================================================
constexpr uint64_t TIMEOUT_A2A3_50US = 2500ULL;                     // 50us
constexpr uint64_t TIMEOUT_A2A3_100US = 5000ULL;                    // 100us
constexpr uint64_t TIMEOUT_A2A3_1MS = 50000ULL;                     // 1ms
constexpr uint64_t TIMEOUT_A2A3_2MS = 100000ULL;                    // 2ms
constexpr uint64_t TIMEOUT_A2A3_3MS = 150000ULL;                    // 3ms
constexpr uint64_t TIMEOUT_A2A3_55MS = 2750000ULL;                  // 55ms
constexpr uint64_t TIMEOUT_A2A3_1SEC = 50000000ULL;                 // 1s
constexpr uint64_t TIMEOUT_A2A3_10SEC = 500000000;                  // 10s
constexpr uint64_t TIMEOUT_A2A3_1MIN = 3000000000ULL;               // 1min
constexpr uint64_t TIMEOUT_A2A3_20MIN = 60000000000ULL;             // 20min
constexpr uint64_t TIMEOUT_A5_50US = 50000ULL;                      // 50us
constexpr uint64_t TIMEOUT_A5_1SEC = 1000000000ULL;                 // 1s
constexpr uint64_t TIMEOUT_A5_5SEC = 5000000000ULL;                 // 5s
constexpr uint64_t TIMEOUT_A5_10SEC = 10000000000;                  // 10s
constexpr uint64_t TIMEOUT_A5_1MIN = 60000000000ULL;                // 1min
constexpr uint64_t TIMEOUT_A5_20MIN = 1200000000000ULL;             // 20min
constexpr uint64_t HAND_SHAKE_TIMEOUT_A2A3_CYCLES = 48000000000ULL; // 16min
constexpr uint64_t HAND_SHAKE_TIMEOUT_A5_CYCLES = 960000000000ULL;  // 16min

#define PERF_LEVEL 0
#define PERF_AICORE_THREAD_START 100

#define PERF_EVENTS                  \
    X(0, EXEC_DYN)                   \
    X(1, INIT)                       \
    X(2, CONTROL_FLOW_MAPEXE)        \
    X(3, CONTROL_FLOW_MAPEXE_MEMCPY) \
    X(1, CONTROL_FLOW_CALL)          \
    X(1, CONTROL_FLOW_INIT)          \
    X(1, CONTROL_FLOW)               \
    X(2, ROOT_FUNC)                  \
    X(3, STAGE_DUP_ROOT)             \
    X(3, ALLOCATE_WORKSPACE)         \
    X(3, STAGE_STITCH)               \
    X(4, FAST_STITCH)                \
    X(4, UPDATE_SLOT)                \
    X(3, SUBMIT_AICORE)              \
    X(4, DECIDE_SLOT_ADDRESS)        \
    X(4, DECIDE_INCAST_ADDRESS)      \
    X(4, RELEASE_FINISH_TASK)        \
    X(4, DEALLOCATE_TASK)            \
    X(4, STAGE_BUILD_TASK)           \
    X(5, ALLOCATE_TASK)              \
    X(5, BUILD_TASK_DATA)            \
    X(6, READY_QUEUE)                \
    X(7, READY_QUEUE_IN)             \
    X(6, RESOLVE_EARLY)              \
    X(6, CORE_FUNCDATA)              \
    X(5, SLAB_MEM_SUBMIT)            \
    X(4, DEALLOCATE_WORKSPACE)       \
    X(4, STAGE_PUSH_TASK)            \
    X(1, STAGE_TASK_SYNC)            \
    X(2, RELEASE_FINISH_TASK_INSYNC) \
    X(3, DEALLOCATE_TASK_INSYNC)     \
    X(1, STAGE_STOP_AICORE)          \
    X(1, DEVICE_MACHINE_INIT_DYN)    \
    X(1, DEVICE_MACHINE_SERVER_DYN)  \
    X5(1, STAGE_SCHEDULE)            \
    X5(2, RUN_TASK)                  \
    X5(3, POLLING_AICORES)           \
    X5(3, RESOLVE_DEPENDENCE)        \
    X5(3, SEND_AIC_TASK)             \
    X5(3, SEND_AIV_TASK)             \
    X5(3, WAIT_AICORE_FINISH)        \
    X5(3, DISPATCH_TASK)             \
    X5(2, SYNC_AICORE)               \
    X5(1, TASK)                      \
    X5(2, RECV_TASK)                 \
    X5(2, SEND_TASK)                 \
    X5(2, RESOLVE_DEP)               \
    X_L2(1, WSALLOC_CORE_A)          \
    X_L2(1, WSALLOC_CORE_D)          \
    X_L2(1, WSALLOC_CPU_A)           \
    X_L2(1, WSALLOC_CPU_D)           \
    X_L2(1, WSRTALLOC_CPU_A)         \
    X_L2(1, WSRTALLOC_CPU_D)         \
    X(0, MAX)

#define X5(a, b) \
    X(a, b)      \
    X(a, b##1)   \
    X(a, b##2)   \
    X(a, b##3)   \
    X(a, b##4)

enum PerfEventType {
#define X(ind, evt) PERF_EVT_##evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
};

inline const char* PerfEventName[] = {
#define SPACE_0 ""
#define SPACE_1 "  " SPACE_0
#define SPACE_2 "  " SPACE_1
#define SPACE_3 "  " SPACE_2
#define SPACE_4 "  " SPACE_3
#define SPACE_5 "  " SPACE_4
#define SPACE_6 "  " SPACE_5
#define SPACE_7 "  " SPACE_6
#define X(ind, evt) SPACE_##ind #evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
#undef SPACE_7
#undef SPACE_6
#undef SPACE_5
#undef SPACE_4
#undef SPACE_3
#undef SPACE_2
#undef SPACE_1
#undef SPACE_0
};

inline bool PerfEvtEnable[] = {
#define X(ind, evt) PERF_LEVEL >= 0,
#define X_L2(ind, evt) PERF_LEVEL >= 2,
    PERF_EVENTS
#undef X_L2
#undef X
};

// common of ptr
template <typename TI, typename TO>
inline TO* PtrToPtr(TI* const ptr)
{
    return reinterpret_cast<TO*>(ptr);
}

template <typename TI, typename TO>
inline const TO* PtrToPtr(const TI* const ptr)
{
    return reinterpret_cast<const TO*>(ptr);
}

inline uint64_t PtrToValue(const void* const ptr) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)); }
inline uint64_t PtrToValue(const volatile void* const ptr)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(const_cast<const void*>(ptr)));
}

inline void* ValueToPtr(const uint64_t value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }

inline std::vector<uint64_t> VPtrToValue(const std::vector<void*> v_ptr)
{
    std::vector<uint64_t> v_value;
    for (const auto& ptr : v_ptr) {
        v_value.emplace_back(PtrToValue(ptr));
    }
    return v_value;
}

inline uint64_t GetTimeMonotonic()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

inline uint64_t GetCycles()
{
    uint64_t cycles;
#ifdef __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#else
    cycles = GetTimeMonotonic();
#endif
    return cycles;
}

inline uint64_t GetFreq()
{
    uint64_t freq;
#ifdef __aarch64__
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
#else
    freq = NSEC_PER_SEC;
#endif
    return freq;
}

inline uint64_t CurrentTime()
{
    uint64_t mono = GetTimeMonotonic();
    return mono / NSEC_PER_USEC;
}

// Timeout check macros
// TIMEOUT_CHECK_INIT(arch, timeout_cycles_val) - initializes timeout_map, start, timeout_cycles, warn_interval(1/10)
// TIMEOUT_CHECK_INIT_WARN_ONLY(arch) - for infinite wait scenarios (no timeout, warn_interval = 1MIN)

#define TIMEOUT_CHECK_INIT(arch, timeout_cycles_val)                                                \
    const uint64_t* timeout_map = (arch == ArchInfo::DAV_3510) ? TIMEOUT_MAP_A5 : TIMEOUT_MAP_A2A3; \
    uint64_t start = GetCycles();                                                                   \
    uint64_t timeout_cycles = timeout_cycles_val;                                                   \
    uint64_t warn_interval = timeout_cycles / 10;                                                   \
    (void)timeout_map;

#define TIMEOUT_CHECK_INIT_WARN_ONLY(arch) \
    uint64_t start = GetCycles();          \
    uint64_t warn_interval = (arch == ArchInfo::DAV_3510) ? TIMEOUT_A5_1MIN : TIMEOUT_A2A3_1MIN

constexpr uint64_t TIMEOUT_INDEX_50US = 0;
constexpr uint64_t TIMEOUT_INDEX_1SEC = 1;
constexpr uint64_t TIMEOUT_INDEX_10SEC = 2;
constexpr uint64_t TIMEOUT_INDEX_1MIN = 3;
constexpr uint64_t TIMEOUT_INDEX_20MIN = 4;
constexpr uint64_t TIMEOUT_INDEX_HAND_SHAKE = 5;
constexpr uint64_t TIMEOUT_INDEX_DISPATCH = 6;

static constexpr uint64_t TIMEOUT_MAP_A2A3[7] = {TIMEOUT_A2A3_50US,          TIMEOUT_A2A3_1SEC,
                                                 TIMEOUT_A2A3_10SEC,         TIMEOUT_A2A3_1MIN,
                                                 TIMEOUT_A2A3_20MIN,         HAND_SHAKE_TIMEOUT_A2A3_CYCLES,
                                                 AICPU_DISPATCH_TIMEOUT_A2A3};

static constexpr uint64_t TIMEOUT_MAP_A5[7] = {TIMEOUT_A5_50US,          TIMEOUT_A5_1SEC,  TIMEOUT_A5_10SEC,
                                               TIMEOUT_A5_1MIN,          TIMEOUT_A5_20MIN, HAND_SHAKE_TIMEOUT_A5_CYCLES,
                                               AICPU_DISPATCH_TIMEOUT_A5};

#define TIMEOUT_10SEC (timeout_map[TIMEOUT_INDEX_10SEC])
#define TIMEOUT_1MIN (timeout_map[TIMEOUT_INDEX_1MIN])
#define TIMEOUT_20MIN (timeout_map[TIMEOUT_INDEX_20MIN])
#define TIMEOUT_HAND_SHAKE (timeout_map[TIMEOUT_INDEX_HAND_SHAKE])
#define TIMEOUT_DISPATCH (timeout_map[TIMEOUT_INDEX_DISPATCH])

// Only warning, never exit - for infinite wait scenarios (uses warn_interval from TIMEOUT_CHECK_INIT_WARN_ONLY)
#define __PYPTO_TIMEOUT_CHECK_WARN_ONLY(fmt, ...)           \
    do {                                                    \
        if ((GetCycles() - start) % warn_interval == 0) {   \
            DEV_WARN(fmt " still waiting.", ##__VA_ARGS__); \
        }                                                   \
    } while (0)

// Only exit, no warning - for short timeout scenarios like 10sec (uses timeout_cycles from TIMEOUT_CHECK_INIT)
#define __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(error_code, action, fmt, ...) \
    do {                                                              \
        (void)warn_interval;                                          \
        if ((GetCycles() - start) > timeout_cycles) {                 \
            DEV_ERROR(error_code, fmt " timeout.", ##__VA_ARGS__);    \
            action;                                                   \
        }                                                             \
    } while (0)

// Warn and exit - for arbitration scenarios where failure is normal degradation (uses timeout_cycles from
// TIMEOUT_CHECK_INIT)
#define __PYPTO_TIMEOUT_CHECK_WARN_EXIT(action, fmt, ...) \
    do {                                                  \
        (void)warn_interval;                              \
        if ((GetCycles() - start) > timeout_cycles) {     \
            DEV_WARN(fmt " timeout.", ##__VA_ARGS__);     \
            action;                                       \
        }                                                 \
    } while (0)

// Both exit and warning - for long timeout scenarios (uses timeout_cycles and warn_interval from TIMEOUT_CHECK_INIT)
#define __PYPTO_TIMEOUT_CHECK(error_code, action, fmt, ...)        \
    do {                                                           \
        if ((GetCycles() - start) > timeout_cycles) {              \
            DEV_ERROR(error_code, fmt " timeout.", ##__VA_ARGS__); \
            action;                                                \
        }                                                          \
        if ((GetCycles() - start) % warn_interval == 0) {          \
            DEV_WARN(fmt " still waiting.", ##__VA_ARGS__);        \
        }                                                          \
    } while (0)

// Reset timeout start time - used when a new batch of parallel tasks arrives,
// giving each batch its own full timeout window.
#define __PYPTO_TIMEOUT_CHECK_RESET start = GetCycles();
} // namespace npu::tile_fwk::dynamic
#endif
