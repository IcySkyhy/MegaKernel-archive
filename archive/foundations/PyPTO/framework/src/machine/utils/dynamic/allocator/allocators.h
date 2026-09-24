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
 * \file allocators.h
 * \brief
 */

#pragma once

#include "slab_ws_allocator.h"
#include "ws_allocator_basics.h"
#include "ws_slot_allocator.h"
#include "seq_ws_allocator.h"
#include "ws_allocator_counter.h"
#include "ws_metadata_allocator.h"
#include "tilefwk/aikernel_data.h"

namespace npu::tile_fwk::dynamic {
struct MetadataAllocator {
    WsMetadataAllocator general;      // aicpu coherent for small suballocation, not support recycle
    SlabWsAllocator generalSlab;      // aicpu meta memory, support reclamation
    SlabWsAllocator stitchSlab;       // aicpu stitched data support reclamation
    WsSlotAllocator dynamicCellMatch; // runtime dynamic cell match table slots
};

struct TensorAllocator {
    SeqWsAllocator rootInner;
    SeqWsAllocator devTaskInnerExclusiveOutcasts;
    WsSlotAllocator devTaskBoundaryOutcasts;
    WsSlotAllocator devTaskInnerTemporalOutcasts;
};
struct RuntimeReuseInfo {
    uint32_t poolResetTimes;
};
} // namespace npu::tile_fwk::dynamic
