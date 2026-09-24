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
 * \file aikernel_drco_leaf_task_ready_queue.h
 * \brief
 */

#ifndef AIKERNEL_DRCO_LEAF_TASK_READY_QUEUE_H
#define AIKERNEL_DRCO_LEAF_TASK_READY_QUEUE_H

#include "tilefwk/aikernel_tensor.h"

namespace npu::tile_fwk {

/*
 * DRCO = Dependency Resolving by aiCOre
 * DRCU = Dependency Resolving by aiCpU
 */

using LeafTaskId = uint32_t;

struct PerCoreReadyList {
    uint32_t size;
    LeafTaskId taskList[0];
};

/* per core pending queue mechanism can be used in both codr and cudr */
struct PerCorePendingQueue {
    uint32_t head;
    uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    PerCorePendingQueue() : head(0), tail(0), size(0) {}

    void UnsafeEnqueue(LeafTaskId task) { taskList[size++] = task; }
#endif
};

struct DrcoLocalReadyQueue {
    alignas(64) uint32_t head;
    alignas(64) uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    explicit DrcoLocalReadyQueue(uint32_t capacity) : head(0), tail(0), size(capacity) {}
#endif
};

#define DRCO_ENCODE_TASK(task) ((task) + 1)
#define DRCO_DECODE_TASK(task) ((task) - 1)

struct DrcoGlobalReadyQueue {
    alignas(64) uint32_t head;
    alignas(64) uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    DrcoGlobalReadyQueue() : head(0), tail(0), size(0) {}

    void UnsafeEnqueue(LeafTaskId task) { taskList[tail++] = DRCO_ENCODE_TASK(task); }
#endif
};

struct DrcoGlobalReadyQueuePtr {
    __gm__ DrcoGlobalReadyQueue* ptr;
};

} // namespace npu::tile_fwk

#endif
