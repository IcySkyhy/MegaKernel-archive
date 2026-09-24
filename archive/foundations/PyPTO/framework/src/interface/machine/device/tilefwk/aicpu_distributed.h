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
 * \file aicpu_distributed.h
 * \brief
 */

#pragma once
#include "aicpu_runtime.h"
#include "tileop/distributed/comm_context.h"

#define RUNTIME_GetHcclRankId(groupIndex) ((TileOp::CommContext*)(startArgs->commContexts[groupIndex]))->rankId

#define RUNTIME_BindTensor(groupIndex, memType, size, maxTileNum, index)                                         \
    [&](void* ctx, uint64_t tgroupIndex, uint64_t tmemType, uint64_t tsize, uint64_t tmaxTileNum) -> uint64_t {  \
        uint64_t param[] = {tgroupIndex, tmemType, tsize, tmaxTileNum};                                          \
        return (uint64_t)runtimeCallList[RuntimeCallStage::T_RUNTIME_CALL_SHMEM_ALLOC](ctx, (uint64_t)(&param)); \
    }(ctx, groupIndex, memType, size, maxTileNum)
