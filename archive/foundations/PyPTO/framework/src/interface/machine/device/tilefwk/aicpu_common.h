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
 * \file aicpu_common.h
 * \brief
 */

#ifndef RUNTIME_COMMON_DEF_H
#define RUNTIME_COMMON_DEF_H

#include <cstddef>
#include <cstdint>
#include "aicpu_perf.h"
#include "aikernel_data.h"

using ArchInfo = npu::tile_fwk::ArchInfo;

const uint64_t AICORE_TASK_INIT = 0xFFFFFFFF;
const uint64_t AICORE_TASK_STOP = 0xF0000001;
const uint64_t AICORE_FUNC_STOP = 0xF0000002;
const uint64_t AICORE_TASK_ABNORMAL_STOP = 0xF0000004;
const uint64_t AICORE_TASK_FETCH_CONFLICT = 0xF0000008;
const uint64_t AICORE_TASK_NO_INCOME = 0xF0000009;
const uint64_t AICORE_TASK_ALL_FINISH = 0xF000000a;
const uint64_t AICORE_FIN_MASK = 0x80000000;
const uint64_t AICORE_TASK_MAX = 0x70000000;

const uint64_t AICORE_SAY_HELLO = 0x80000000;
const uint64_t AICORE_SAY_ACK = 0x80000000;
const uint64_t AICORE_SAY_GOODBYE = 0x88888888;
const int64_t PRO_LEVEL1 = 2;
const int64_t PRO_LEVEL2 = 3;

constexpr int REG_LOW_TASK_PING = 0;
constexpr int REG_LOW_TASK_PONG = 1;
constexpr int MAX_DFX_TASK_NUM_PER_CORE = 10000;

// AICPU dispatch/sync 阶段超时时间
#define AICPU_DISPATCH_TIMEOUT_A2A3 50000000ULL * 10
#define AICPU_DISPATCH_TIMEOUT_A5 1000000000ULL * 10

constexpr int SHAK_BUF_PRINT_BUFFER_INDEX = 5;
constexpr int SHAK_BUF_COREFUNC_DATA_INDEX = 6;
constexpr int SHAK_BUF_DFX_DATA_INDEX = 7;

constexpr int CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX = 0;
constexpr int CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX = 0;

constexpr int FUNC_ID_BATCH = 0x7FF;

const uint64_t SHARED_BUFFER_SIZE = 512;
const uint64_t PMU_BUFFER_SIZE = 4096;
const uint64_t DEVICE_QUEUE_SIZE = 512;
const uint64_t PRINT_BUFFER_SIZE = 131072; // 128KB

constexpr const int DEV_SHAPE_DIM_NUM_2 = 2;
constexpr const int DEV_SHAPE_DIM_NUM_3 = 3;
constexpr const int DEV_SHAPE_DIM_NUM_4 = 4;
constexpr const int DEV_SHAPE_DIM_NUM_5 = 5;

constexpr const uint32_t MAX_ROUND_NUM = 200;

#define DEVICE_TASK_STOP 0x7FFFFFFE

#define DEVICE_TASK_TYPE_STATIC 0
#define DEVICE_TASK_TYPE_DYN 1
#define DEVICE_TASK_TYPE_INVALID 0xf

template <typename DerivedType, typename UnderlyingType>
class BitmaskBase {
public:
    using underlying_type = UnderlyingType;
    underlying_type value{0};
    constexpr BitmaskBase(underlying_type v = 0) : value(v) {}
    constexpr bool Empty() const { return value == 0; }
    constexpr bool Contains(underlying_type mask) const { return (value & mask) == mask; }
    constexpr bool Overlaps(underlying_type mask) const { return (value & mask) != 0; }
    constexpr void Add(underlying_type mask) { value |= mask; }
    constexpr void Remove(underlying_type mask) { value &= ~mask; }
    friend constexpr DerivedType operator|(DerivedType lhs, DerivedType rhs)
    {
        return DerivedType(lhs.value | rhs.value);
    }
    friend constexpr DerivedType operator&(DerivedType lhs, DerivedType rhs)
    {
        return DerivedType(lhs.value & rhs.value);
    }
    friend constexpr DerivedType operator^(DerivedType lhs, DerivedType rhs)
    {
        return DerivedType(lhs.value ^ rhs.value);
    }
    friend constexpr DerivedType operator~(DerivedType lhs) { return DerivedType(~lhs.value); }
    constexpr operator underlying_type() const { return value; }
};

struct ProfConfig : public BitmaskBase<ProfConfig, uint32_t> {
    using BitmaskBase::BitmaskBase;
    enum : underlying_type {
        OFF = 0x0,
        AICPU_FUNC = 0x1 << 0,
        AICORE_TIME = 0x1 << 1,
        AICORE_PMU = 0x1 << 2,
    };
};

struct ToSubMachineConfig {
    ProfConfig profConfig{ProfConfig::OFF};
};

enum DeviceKernelRunMode : uint32_t {
    RUN_INVALID = 0,
    RUN_SPLITTED_STREAM_CTRL = 1,
    RUN_SPLITTED_STREAM_SCHE = 2,
};

struct DeviceKernelArgsParameter {
    uint32_t runMode{RUN_INVALID};
    uint32_t p1;
    uint64_t globalRound{0};
    uint64_t ctrlBlockNum{0};
};
static_assert(sizeof(DeviceKernelArgsParameter) == sizeof(uint64_t) * 0x3, "Invalid parameter size");

struct DeviceRuntimeOffset {
    uint64_t startArgsOffset{0};
    uint64_t taskCtrlPoolOffset{0};
    uint64_t taskQueueOffset{0};
    uint64_t deviceTaskReadyQueueOffset{0};
    uint64_t generalOffset{0};
    uint64_t stitchPoolOffset{0};
    uint64_t size{0};
    uint64_t count{0};
};

struct DeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint64_t opaque{0};         // store device global data, must be init with zero
    uint64_t devQueueAddr;      // pcie/XLink mem, used between host and device, `DEVICE_QUEUE_SIZE`
    uint64_t sharedBuffer;      // SHARED_BUFFER_SIZE per core, aics first
    uint64_t coreRegAddr;       // core reg addr, uint64_t per core, aic first
    uint64_t corePmuRegAddr;    // pmu reg addr, uint64_t per core, aic first
    uint64_t corePmuAddr;       // pmu data addr, PAGE_SIZE per core, aic first
    uint64_t pmuEventAddr;      // pmu event addr
    uint64_t taskType : 4;      // initial task type
    uint64_t machineConfig : 8; // machine config
    uint64_t taskId : 52;       // initial task id
    uint64_t taskData;          // initial task data
    uint64_t taskWastTime{0};
    uint64_t aicpuSoBin{0};                // server so Bin
    uint64_t aicpuSoLen{0};                // server so len
    uint64_t deviceId{0};                  // for device copy fileName
    uint64_t runtimeDataRingBufferAddr{0}; // DevStartArgs addr
    uint64_t dynamicCellMatchAddr{0};      // standalone metadata pool for dynamic cell match
    uint64_t dynamicCellMatchCapacity{0};  // allocated bytes for dynamicCellMatchAddr
    uint32_t hostPid{0};                   // for dump tensor
    uint32_t scheCpuNum{0};                // sche cpu num calc by host
    uint32_t enableCtrl : 2;               // if enable builtin ctrl
    uint32_t validGetPgMask : 30;          // mark pgmask is invalid
    uint64_t aicpuPerfAddr{0};             // aicpuPer Gm addr
    uint64_t devDfxArgAddr{0};             // devDfx
    uint64_t GetBlockNum() const { return nrValidAic * (nrAiv / nrAic + 1); }
    uint32_t die0MaxCpuid{0};
    uint32_t launchSchedAicpuNum{7};    // user configures the number of launch schedules for aicpu
    bool launchSchedSameCluster = true; // user configures whether enables binding aicpu within the same cluster
    bool enableVFFusion = false;
    bool enableEslModel = false;
    bool hasAicpuTask = false;
    bool enableAicoreResolve = false; // runtime switch, driven by env ENABLE_AICORE_RESOLVE=true
    ArchInfo archInfo{ArchInfo::DAV_2201};
    ToSubMachineConfig toSubMachineConfig;
};

#define TO_ENTRY_IMPL(name, sep1, key1, sep2, key2, type) (name##sep1##key1##sep2##key2##type)
#define TO_ENTRY(name, key1, key2, type) TO_ENTRY_IMPL(name, _, key1, _, key2, type)

#ifdef __MIX__
#ifdef __AIV__
#define KERNEL_ENTRY(x, y, z) TO_ENTRY(x, y, z, _mix_aiv)
#else
#define KERNEL_ENTRY(x, y, z) TO_ENTRY(x, y, z, _mix_aic)
#endif
#else
#define KERNEL_ENTRY(x, y, z) x
#endif

const uint64_t AICORE_REG_SAY_HELLO = 0xF000000080000000;
constexpr uint32_t REG_HIGH_DTASKID_SHIFT = 32;
enum class TASK_POS : size_t { LOW_REG = 0, HIGH_REG = 1, ALL_REG = 2, REG_POS_BUTT = 3 };
constexpr uint32_t
    MAX_SYNC_EVENT_NUM = 48; // the max set/wait insts in a mix subgraph leaffunction is 48, can set larger manually

struct TaskStat {
    int16_t seqNo;
    int16_t subGraphId;
    int32_t taskId;
    int64_t execStart;
    int64_t execEnd;
    uint64_t perfDataBaseAddr;
    uint64_t setEventAddr;
    uint64_t waitEventAddr;
    int16_t setEventNum;
    int16_t waitEventNum;
};

struct DevDfxArgs {
    int32_t logLevel{-1};
    int32_t isOpenPerfTrace{0};
    uint32_t deviceId{0};
    uint8_t profLevel{0};
};

constexpr uint32_t PERF_TRACE_INST_MAX_NUM_EVERY_TYPE = 200;
constexpr uint32_t INVALID_DEV_TASK_ID = 0xFFFFFFFF;
enum AicorePerfTrace {
    PERF_TRACE_CORE_BEGIN = 0,
    PERF_TRACE_CORE_INIT,
    PERF_TRACE_CORE_DEV_TASK_RCV_MODEL,
    PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK,
    PERF_TRACE_CORE_DEV_TASK_LEAF_TASK_EXEC,
    PERF_TRACE_CORE_DEV_TASK_WAIT_SYNC_STOP_NOTIFY,
    PERF_TRACE_CORE_WAIT_ALL_DEV_TASK_LEAF_TASK_EXEC_FINISH,
    PERF_TRACE_CORE_WAIT_EXIT_NOTIFY,
    PERF_TRACE_CORE_MAX
};

struct AicoreDevTaskPerf {
    uint64_t type = 0;
    uint64_t devTaskIdx = 0;
    uint64_t aicoreDevTimeStamp = 0;
    AicoreDevTaskPerf() = default;
    AicoreDevTaskPerf(const AicoreDevTaskPerf&) = default;
    AicoreDevTaskPerf& operator=(const AicoreDevTaskPerf&) = default;
};

struct AicoreMetric {
    uint64_t cnt = 0;
    AicoreDevTaskPerf aicoreEveryDevTypeTimeStamp[PERF_TRACE_CORE_MAX * PERF_TRACE_INST_MAX_NUM_EVERY_TYPE];
    AicoreMetric() = default;
    AicoreMetric(const AicoreMetric&) = default;
    AicoreMetric& operator=(const AicoreMetric&) = default;
};

struct Metrics {
    int16_t isMetricStop;
    int16_t scheCpuIdx;
    int32_t coreType = -1;
    int64_t taskCount;
    int64_t turnNum;
    AicoreMetric aicoreDevTaskInfo[MAX_ROUND_NUM];
    TaskStat tasks[];
};

constexpr uint32_t AVG_EVENTS_PER_TASK = 16;
constexpr uint32_t SET_WAIT_EVENT_DATA_SIZE = MAX_DFX_TASK_NUM_PER_CORE * AVG_EVENTS_PER_TASK * sizeof(uint64_t);
constexpr uint32_t PERF_DATA_SYNC_EVENT_SIZE = SET_WAIT_EVENT_DATA_SIZE * 2;
constexpr uint32_t PERF_DATA_BASE_SIZE = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
constexpr uint32_t PERF_DATA_TOTAL_SIZE = PERF_DATA_BASE_SIZE + PERF_DATA_SYNC_EVENT_SIZE;

struct DevTaskPerf {
    uint64_t cnt{0};
    uint64_t timeStamp[npu::tile_fwk::dynamic::PERF_TRACE_COUNT_DEVTASK_MAX_NUM] = {0};
};

struct MetricPerf {
    uint64_t perfAicpuTrace[npu::tile_fwk::dynamic::MAX_USED_AICPU_NUM][npu::tile_fwk::dynamic::PERF_TRACE_MAX] = {{0}};
    DevTaskPerf devTaskPerfs[npu::tile_fwk::dynamic::MAX_USED_AICPU_NUM][npu::tile_fwk::dynamic::DEVTASK_PERF_TYPE_NUM];
};

inline const char* AicorePerfTraceName[] = {"BEGIN",
                                            "INIT",
                                            "DEV_TASK_RCV_MODEL",
                                            "DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK",
                                            "DEV_TASK_ALL_LEAF_TASK_EXEC",
                                            "DEV_TASK_WAIT_SYNC_STOP_NOTIFY",
                                            "WAIT_ALL_DEV_TASK_LEAF_TASK_EXEC_FINISH",
                                            "WAIT_EXIT_NOTIFY"};

// use ring buffer to control parallel multi devtask
struct ParallelDevTask {
    uint32_t front{0};
    uint32_t rear{0};
    uint32_t version;
    uint32_t reserver;
    uint32_t idElements[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; // device task id
    int64_t ptrElements[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; // device task ptr
};

struct TaskEntry {
    int32_t subGraphId;
    int32_t taskId;
    int64_t funcAddr;
    int64_t tensorAddrs;
    int64_t gmStackSize;
    int64_t gmStackBase;
    int64_t reserved2[2];
    uint32_t tensorSize;
    uint32_t reserved[1];
};

struct KernelArgs {
    int64_t shakeBuffer[8];
    int64_t shakeBufferCpuToCore[8];
    int64_t waveBufferCpuToCore[8];
    int64_t dfxBuffer[8];
    struct ParallelDevTask parallelDevTask;
    TaskEntry taskEntry;
};

union KernelSharedBuffer {
    struct KernelArgs args;
    uint8_t sharedBuffer[SHARED_BUFFER_SIZE];
    KernelSharedBuffer() {};
};

static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);
#endif
