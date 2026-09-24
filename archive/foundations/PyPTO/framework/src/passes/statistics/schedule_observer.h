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
 * \file schedule_observer.h
 * \brief Observer interface and event structs for OoO schedule instrumentation.
 *        Consumers (HealthCheck, MemoryTracer) aggregate what they need.
 *        See docs/passes/memory_visualization_design.md.
 */

#ifndef SCHEDULE_OBSERVER_H
#define SCHEDULE_OBSERVER_H

#include <cstdint>
#include <string>
#include <vector>
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"                                       // MemoryType
#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h" // BufferAddrChange

namespace npu::tile_fwk {

// Mapped from CoreLocationType at the Notify site: AIC→{AIC,0}, AIV0→{AIV,0}, AIV1→{AIV,1}.
enum class CoreClass : int { AIC = 0, AIV = 1, UNKNOWN = 2 };

struct CoreLocation {
    CoreClass coreType{CoreClass::UNKNOWN};
    int coreIdx{0};
};

struct LogicalTensorBrief {
    int magic{0};
    uint64_t validSize{0};
};

enum class DDRBufferKind : int {
    FUNCTION_TEMP = 0, // workspace alloc for function I/O / intermediate DDR
    SPILL_TEMP = 1,    // CreateGMTensor spill destination
    INCAST = 2,        // cross-device input, machine layer owns the buffer
    OUTCAST = 3,       // cross-device output
};

// INCAST/OUTCAST have no OoO-side address, so addr/size stay 0.
struct DDRRef {
    int memId{0};
    DDRBufferKind kind{DDRBufferKind::FUNCTION_TEMP};
    MemoryType memType{MemoryType::MEM_DEVICE_DDR};
    uint64_t addrStart{0};
    uint64_t addrEnd{0};
    uint64_t size{0};
};

struct OpLaunchEvent {
    int clock;
    int cycleEnd;
    int opMagic;
    PipeType pipeType;
    CoreLocation coreLocation;
    std::vector<int> inputMemIds; // memId snapshot at launch time
    std::vector<int> outputMemIds;
    std::vector<DDRRef> ddrRefs;
};

struct OpRetireEvent {
    int clock;
    int opMagic;
    std::vector<int> freedMemIds; // memIds whose refcount dropped to 0
};

struct AllocExecEvent {
    int clock;
    int memId;
    MemoryType memType;
    CoreLocation coreLocation;
    uint64_t addrStart;
    uint64_t addrEnd;
    std::vector<LogicalTensorBrief> logicalTensors;
};

struct SpillEvent {
    int clock;
    int beginClock;
    int endClock;
    int spillMemId;
    MemoryType memType;
    CoreLocation coreLocation;
    uint64_t addrStart;
    uint64_t addrEnd;
    int triggerOpMagic;
    int64_t triggerTensorSize;
    int spillCopyoutOpMagic; // -1 = COPY_IN reuse path, no copyout
    int reloadAllocOpMagic;
    int reloadCopyInOpMagic;
    int spillTensorMagic;
    uint64_t spillCopyoutSize;
    uint64_t allocOccupiedSize{0};  // buffer held by owners still being alloc ops (not yet consumed)
    uint64_t bufferCurrentUsage{0}; // pool occupancy sampled at spill (valid in both phases)
    uint64_t bufferCapacity{0};
    // DDR landing buffer, valid when spillCopyoutOpMagic != -1.
    int spillDdrMemId{-1}; // >= 0x3f000000
    DDRBufferKind ddrKind{DDRBufferKind::SPILL_TEMP};
    MemoryType ddrMemType{MemoryType::MEM_DEVICE_DDR};
    uint64_t ddrAddrStart{0};
    uint64_t ddrAddrEnd{0};
    uint64_t ddrSize{0};
};

struct BufferRearrangeEvent {
    using Change = BufferAddrChange;
    int clock;
    MemoryType memType;
    CoreLocation coreLocation;
    int triggerOpMagic;
    std::vector<Change> changes;
};

struct AllocFailEvent {
    struct OccupiedSlice {
        int memId;
        uint64_t addrStart;
        uint64_t addrEnd;
        uint64_t size;
        int ownerOpMagic;
    };
    struct FreeInterval {
        uint64_t start;
        uint64_t size;
    };
    int clock;
    int triggerOpMagic;
    MemoryType memType;
    CoreLocation coreLocation;
    uint64_t requestSize;
    uint64_t capacity;
    std::vector<OccupiedSlice> occupiedSlices;
    std::vector<FreeInterval> freeIntervals;
};

struct ScheduleEndEvent {
    int totalCycles;
    int64_t workspaceOffset;
    bool success;
};

// incast/outcast registration: machine layer owns the buffers, so only magic +
// shape are surfaced (no OoO address) for IDE placeholders and graph lookup.
struct InitDDRBufferEvent {
    int clock; // always -1, predates the timeline
    int memId;
    DDRBufferKind kind;
    int magic;
    DataType dtype;
    std::vector<std::string> shape; // dynamic axes carry SymbolicScalar names
};

class ScheduleObserver {
public:
    virtual ~ScheduleObserver() = default;

    virtual void OnOpLaunch(const OpLaunchEvent&) {}
    virtual void OnOpRetire(const OpRetireEvent&) {}
    virtual void OnAllocExec(const AllocExecEvent&) {}
    virtual void OnSpill(const SpillEvent&) {}
    virtual void OnBufferRearrange(const BufferRearrangeEvent&) {}
    virtual void OnAllocFail(const AllocFailEvent&) {}
    virtual void OnScheduleEnd(const ScheduleEndEvent&) {}
    virtual void OnInitDDRBuffer(const InitDDRBufferEvent&) {}
    // Fired from OoOScheduler::PreMainLoop / PostMainLoop and LatencyEstimator::PreMainLoop / PostMainLoop.
    virtual void OnMainLoopBegin() {}
    virtual void OnMainLoopEnd() {}
};

} // namespace npu::tile_fwk
#endif // SCHEDULE_OBSERVER_H
