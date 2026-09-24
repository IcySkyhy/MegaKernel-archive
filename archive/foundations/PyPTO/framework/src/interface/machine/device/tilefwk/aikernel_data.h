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
 * \file aikernel_data.h
 * \brief
 */

#ifndef AIKERNEL_DATA_H
#define AIKERNEL_DATA_H
#include <atomic>
#include "tilefwk/aikernel_define.h"
#include "tilefwk/aikernel_tensor.h"
#include "tilefwk/aikernel_device_task.h"
#include "tilefwk/aikernel_runtime_data_ring_buffer.h"

struct LogContext;

namespace npu::tile_fwk {

constexpr uint32_t HCCL_GROUP_NUM = 2;

enum class CoreType { AIV = 0, AIC = 1, MIX = 2, AICPU = 3, HUB = 4, GMATOMIC = 5, HUB_MIX = 6, INVALID = 20 };

struct CoreFuncParam {
    __gm__ npu::tile_fwk::DynFuncData* funcData;
    __gm__ npu::tile_fwk::DynFuncHeader* funcHeader;
    __gm__ uint64_t* opAttrs;
    __gm__ uint64_t* exprTbl;
    uint32_t taskId;
    LogContext* ctx;
};

/*
    |--------16bit-------------|----16bit----|----1bit----|-----1bit------|------1bit-----|-----3bit--------|---10bit---|---16bit--|
    |-parallel ctx modifyflag--|--devtaskid--|----rspflag-|--pingpongflag-|---dcci flag---|--prallel index--|--func
   id--|--opindex-|
*/
#define TASKID_TASK_BITS 16
#define TASKID_TASK_MASK ((1 << TASKID_TASK_BITS) - 1)

#define TASKID_FUNC_BITS 10
#define TASKID_FUNC_MASK ((1 << TASKID_FUNC_BITS) - 1)

#define TASKID_PARALLEL_INDEX_BITS 3
#define TASKID_PARALLEL_INDEX_MASK ((1 << TASKID_PARALLEL_INDEX_BITS) - 1)

#define TASKID_DEVTASK_DCCI_BITS 1
#define TASKID_DEVTASK_DCCI_MASK ((1 << TASKID_DEVTASK_DCCI_BITS) - 1)

#define TASKID_SHIFT32 32
#define TASKID_FROM_CTRL_TOPO_MASK ((1 << (TASKID_TASK_BITS + TASKID_FUNC_BITS)) - 1)

const uint32_t SCH_DEVTASK_MAX_PARALLELISM = (1 << TASKID_PARALLEL_INDEX_BITS);

INLINE uint32_t FuncID(uint32_t taskId) { return (taskId >> TASKID_TASK_BITS) & TASKID_FUNC_MASK; }

INLINE uint32_t TaskID(uint32_t taskId) { return taskId & TASKID_TASK_MASK; }

INLINE uint32_t MakeTaskID(uint32_t rootId, uint32_t leafId) { return (rootId << TASKID_TASK_BITS) | leafId; }

INLINE uint32_t ParallelIndex(uint32_t taskId)
{
    return (taskId >> (TASKID_TASK_BITS + TASKID_FUNC_BITS)) & TASKID_PARALLEL_INDEX_MASK;
}

INLINE uint32_t DevTaskDcciFlag(uint32_t taskId)
{
    return (taskId >> (TASKID_TASK_BITS + TASKID_FUNC_BITS + TASKID_PARALLEL_INDEX_BITS)) & TASKID_DEVTASK_DCCI_MASK;
}

#define REG_VAL_DEVTASK_ID_BITS 24
#define REG_VAL_DEVTASK_ID_MASK ((1 << REG_VAL_DEVTASK_ID_BITS) - 1)

#define REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_BITS 8
#define REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_MASK ((1 << REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_BITS) - 1)

INLINE uint32_t DevTaskId(uint64_t highRegValue) { return highRegValue & REG_VAL_DEVTASK_ID_MASK; }

INLINE uint32_t ParallelDevTaskModifyFlag(uint64_t highRegValue)
{
    return (highRegValue >> REG_VAL_DEVTASK_ID_BITS) & REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_MASK;
}

} // namespace npu::tile_fwk

#endif
