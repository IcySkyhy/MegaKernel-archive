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
 * \file device_slot_context.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"

namespace npu::tile_fwk::dynamic {
using StitchedList = Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator>;
struct DeviceSlotContext {
    void InitAllocator(DeviceWorkspaceAllocator& workspace, uint64_t slotSize);

    void FillInputOutputSlot(DevAscendProgram* devProg, DevStartArgs* args);

    uint32_t UpdateSlots(DevAscendFunctionDupped& devRootDup, uint32_t devTaskId, uint32_t devNextIdx,
                         uint32_t cellMatchTagSeq);

    DeviceExecuteSlot* GetSlotList() { return slotList_.data(); }
    size_t GetSlotSize() { return slotList_.size(); }

    void ClearDirty()
    {
        const size_t n = slotList_.size();
        DeviceExecuteSlot* slots = slotList_.data();
        for (size_t i = 0; i < n; i++) {
            slots[i].stitchDupIdx = INVALID_STITCH_IDX;
        }
    }

public:
    void FillInputOutputSlot(DeviceExecuteSlot* slotList, [[maybe_unused]] size_t slotSize, DevAscendProgram* devProg,
                             DevStartArgs* args);

private:
    Vector<DeviceExecuteSlot, WsMemCategory::VECTOR_SLOT_LIST> slotList_;
    DeviceWorkspaceAllocator* workspace_{nullptr};
};
} // namespace npu::tile_fwk::dynamic
