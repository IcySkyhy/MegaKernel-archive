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
 * \file aikernel_device_task.h
 * \brief
 */

#ifndef AIKERNEL_DEVICE_TASK_H
#define AIKERNEL_DEVICE_TASK_H

#include <atomic>
#include "tilefwk/aikernel_tensor.h"
#include "tilefwk/aikernel_drco_leaf_task_ready_queue.h"
#include "tilefwk/aikernel_root_function.h"

namespace npu::tile_fwk {

struct DevStartArgsBase;

struct DrcoRootFuncData {
    __gm__ int32_t* predCount;
    __gm__ int32_t* succStaticList;
    __gm__ DevAscendFunctionDuppedStitchNode** succStitchList;
    __gm__ DevAscendFunctionOperationSuccInfo* succInfoList;
};

struct DynFuncData {
    uint64_t exprNum;              // static
    __gm__ uint64_t* opAttrs;      // static
    __gm__ int32_t* opAtrrOffsets; // static
    __gm__ uint64_t* exprTbl;      // dyn
    __gm__ DevRawTensorDesc* rawTensorDesc;
    __gm__ uint64_t* rawTensorAddr;
    uint64_t opAttrSize;
    uint64_t rawTensorDescSize;
    uint64_t rawTensorAddrSize;
    uint64_t workspaceAddr;
    __gm__ int* cceBinaryIndexList;
    DrcoRootFuncData drcoRootFuncData;
};

struct DynFuncBin {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    int32_t wrapVecId{-1};
    uint8_t mixResourceType{0};
};

constexpr uint32_t MAX_AICORE_NUM_FOR_QUEUE = 108;
constexpr uint32_t LOCAL_GROUP_SIZE = 6;
constexpr uint32_t NUM_LOCAL_GROUPS = (MAX_AICORE_NUM_FOR_QUEUE + LOCAL_GROUP_SIZE - 1) / LOCAL_GROUP_SIZE;
constexpr uint32_t NUM_CORE_TYPES = 2;

struct DynFuncHeader {
    uint64_t seqNo;
    uint32_t funcNum;
    uint32_t funcSize;
    __gm__ DynFuncBin* cceBinary;
    uint64_t stackWorkSpaceAddr;
    uint64_t stackWorkSpaceSize;
    __gm__ DevStartArgsBase* startArgs;

    INLINE uint64_t GetIndex() { return seqNo; }
    INLINE uint32_t Size() { return funcNum; }
    INLINE DynFuncData& At(int index) { return (reinterpret_cast<DynFuncData*>(this + 1))[index]; }
};

constexpr uint32_t DRCO_QUEUE_AIV = 0;
constexpr uint32_t DRCO_QUEUE_AIC = 1;
constexpr uint32_t DRCO_QUEUE_M1V = 2;
constexpr uint32_t DRCO_QUEUE_M2V = 3;
constexpr uint32_t DRCO_QUEUE_MAX = 4;

struct DrcoRootFuncList {
    DrcoGlobalReadyQueuePtr globalReadyQueueList[DRCO_QUEUE_MAX];
    uint32_t globalQueueInitTail[DRCO_QUEUE_MAX];

    __gm__ PerCorePendingQueue* perCorePendingQueueArray[MAX_AICORE_NUM_FOR_QUEUE];
    __gm__ DrcoLocalReadyQueue* localReadyQueueArray[NUM_CORE_TYPES][NUM_LOCAL_GROUPS];
    alignas(64) uint32_t totalTaskCount;
    alignas(64) uint32_t devTaskFinished;
    alignas(64) uint32_t executedTaskCount;
    alignas(64) uint8_t pad[64];
};

constexpr int32_t DEVICE_TASK_QUEUE_SIZE = 100;
struct DrcoDeviceTask {
    __gm__ DynFuncHeader* dynFuncDataList;
    __gm__ DrcoRootFuncList* drcoRootFuncList;
};
struct DrcoDeviceTaskReadyQueue {
    uint32_t head;
    uint32_t size;
    DrcoDeviceTask dynFuncDataListList[DEVICE_TASK_QUEUE_SIZE];
#ifdef __TILE_FWK_HOST__
    void Reset()
    {
        head = 0;
        size = 0;
    }
    void Append(DynFuncHeader* dynFuncDataList, DrcoRootFuncList* rootFuncList)
    {
        dynFuncDataListList[size].dynFuncDataList = dynFuncDataList;
        dynFuncDataListList[size].drcoRootFuncList = rootFuncList;
        size++;
    }
#endif
};

} // namespace npu::tile_fwk

#endif
