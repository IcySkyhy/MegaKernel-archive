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
 * \file machine_ws_intf.h
 * \brief
 */

#ifndef MACHINE_WS_INTF_H
#define MACHINE_WS_INTF_H

#include "tilefwk/aicpu_common.h"
#include "interface/utils/common.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/error_code.h"
#include "machine/utils/device_log.h"
#include "machine/utils/queues.h"

namespace npu::tile_fwk {
enum class MachineStatus { START = 0, FINISH = 1, STOP = 2 };

#define MAX_WRAP_TASK_NUM 3 // 最多1C2V任务
#define WRAP_IDX_AIC 0
#define WRAP_IDX_AIV0 1
#define WRAP_IDX_AIV1 2

struct StaticReadyCoreFunctionQueue {
    uint32_t head;
    uint32_t tail;
    uint64_t* elem;
};

struct WrapInfo {
    uint32_t wrapId;
    uint32_t tasklist[MAX_WRAP_TASK_NUM]; // 顺序C、V1、V2
    uint8_t mixResourceType;
};

struct WrapInfoQueue {
    uint32_t head;
    uint32_t tail;
    uint32_t capacity;
    WrapInfo* elem;
    size_t lock;
    uint64_t Size() { return tail - head; }
};

enum class BinDataType {
    READY_STATUS,        // CoreFunction ready_status(no need update)
    READY_AIC_CORE_FUNC, // ready aic CoreFunction id list(no need update)
    READY_AIV_CORE_FUNC, // ready aiv CoreFunction id list(no need update)
    CACHE_HEADER,        // cache header
    CCE_BIN,             // all cce bin
    TOPO,                // TOPO
    INVOKE_OFFSET_TABLE, // invoke offset
    INVODE_TENSOR_INDEX, // invoke tensor index
    INVODE_TENSOR_INFO,  // invoke tensor
    INVOKE_PARA_OFFSET,  // invoke para offset
    CORE_FUNC_WS_ADDR,   // corefunc args
    END
};
#pragma pack(8)
struct DeviceTaskBin {
    BaseArgs baseArgs;
    DeviceTask deviceTask; // initial task data
    uint64_t dataSize[static_cast<size_t>(BinDataType::END)];
    uint64_t dataOffset[static_cast<size_t>(BinDataType::END)];
    uint8_t data[0];
};
#pragma pack()

constexpr int64_t DEVICE_QUEUE_SIZE = 512;
#define DEVICE_TASK_STOP 0x7FFFFFFE

struct DeviceKernelArgs {
    int64_t* ctrlFlowCache{nullptr};
    int64_t* inputs{nullptr};
    int64_t* outputs{nullptr};
    int64_t* workspace{nullptr};
    int64_t* tilingdata{nullptr};
    int64_t* cfgdata{nullptr};
    int64_t* commContexts{nullptr};
    // following 4 paras need remove to binary
    void* costmodeldata{nullptr};
    void* aicoreModel{nullptr};
    uint64_t taskWastTime{0};
    uint64_t maxDynamicAssembleOutcastMem{0};
    uint64_t maxDynamicCellMatchTableMem{0};
    uint64_t runtimeDynamicCellMatchAddr{0};
    uint64_t runtimeDynamicCellMatchCapacity{0};
    uint8_t machineConfig;
    uint8_t schedSyncMode{0}; // 0: AsyncMode 1: SyncMode
    ToSubMachineConfig toSubMachineConfig;
    DeviceKernelArgsParameter parameter;
};

struct AiCpuArgs {
    DeviceKernelArgs kArgs;
    const char kernelName[32] = {"DynTileFwkKernelServer"};
    const char soName[32] = {"libaicpu_extend_kernels.so"};
    const char opName[32] = {""};
};

struct LogHead {
    int type;
    int len;
    int64_t data[];
};
} // namespace npu::tile_fwk
#endif
