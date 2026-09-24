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
 * \file ws_slot_allocator.h
 * \brief
 */

#pragma once

#include "ws_allocator_basics.h"
#include "ws_metadata_allocator.h"

#include <cinttypes>
#include <limits>

namespace npu::tile_fwk::dynamic {

class WsSlotAllocator {
public:
    using uintdevptr_t = uint64_t;

    struct BlockHeader {
        BlockHeader* listNext;
        uintdevptr_t ptr;
    };

    // Relative-pointer sentinel used by CF-cache Backup/Restore for nullptr list heads.
    static constexpr uintptr_t kNullBlockHeaderIndex = std::numeric_limits<uintptr_t>::max();

public:
    // [workspaceAddr, workspaceAddr + workspaceSize)
    // -> [root function internal workspace | slot pool]
    // Lazy freelist: virgin BlockHeaders are materialized on first Allocate (ItemPool-style),
    // so Init stays O(1) instead of O(slotNum) list wiring each ExecDyn.
    void InitTensorAllocator(uintdevptr_t workspaceAddr, size_t slotNum, uint64_t slotStandardMemReq,
                             WsMetadataAllocator& allocator)
    {
        workspaceAddr_ = workspaceAddr;
        slotNum_ = slotNum;
        slotStandardMemReq_ = slotStandardMemReq;

        availableSlots_ = slotNum_;
        initReadyCount_ = 0;
        freeListHeader_ = nullptr;
        notInUseHeaders_ = nullptr;

        allocator_ = &allocator;
        allocation_ = allocator_->Allocate<BlockHeader>(slotNum_, WsMemCategory::WS_SLOT_MEM_BLOCK);
    }

    BlockHeader* GetBlockHeaderBase() { return allocation_.As<BlockHeader>(); }

    size_t InitReadyCount() const { return initReadyCount_; }

    bool IsValidSlotMemRequirement(uint64_t memReq) const { return memReq <= slotStandardMemReq_; }

    WsAllocation Allocate()
    {
        DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, availableSlots_ > 0, "Available slot: %zu/%zu",
                       availableSlots_, slotNum_);

        BlockHeader* node = nullptr;
        if (freeListHeader_ != nullptr) {
            node = freeListHeader_;
            freeListHeader_ = freeListHeader_->listNext;
        } else {
            DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, initReadyCount_ < slotNum_,
                           "Virgin slot overflow: ready=%zu/%zu", initReadyCount_, slotNum_);
            node = allocation_.As<BlockHeader>() + initReadyCount_;
            node->ptr = workspaceAddr_ + initReadyCount_ * slotStandardMemReq_;
            initReadyCount_++;
        }

        WsAllocation allocation;
        allocation.ptr = node->ptr;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        allocation.category_ = WsMemCategory::TENSOR_ROOTFUNC_OUTCAST_SLOT;
        allocation.rawMemReq_ = slotStandardMemReq_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        dfx_.historicalAllocated_++;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT

        InsertList(node, notInUseHeaders_);
        availableSlots_--;

        return allocation;
    }

    /* allocate at most n elements, and store the result into allocateList.
     * Return false if out of memory.
     */
    bool Allocate(int n, WsAllocation* allocateList)
    {
        for (int i = 0; i < n; i++) {
            allocateList[i] = Allocate();
            if (allocateList[i].ptr == 0) {
                for (int j = 0; j < i; j++) {
                    Deallocate(allocateList[j]);
                }
                return false;
            }
        }
        return true;
    }

    void Deallocate(uintdevptr_t ptr)
    {
        DEV_ASSERT_MSG(WsErr::WS_TENSOR_ADDRESS_OUT_OF_RANGE,
                       workspaceAddr_ <= ptr && ptr < workspaceAddr_ + slotNum_ * slotStandardMemReq_,
                       "Pointer to deallocate is out of range");
        DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, notInUseHeaders_ != nullptr,
                       "Blocks are all free, there shouldn't be any deallocation request.");

        BlockHeader* node = notInUseHeaders_;
        notInUseHeaders_ = notInUseHeaders_->listNext;

        node->ptr = ptr;

        InsertList(node, freeListHeader_);
        availableSlots_++;
    }

    void Deallocate(const WsAllocation& allocation) { Deallocate(allocation.ptr); }

    size_t AvailableSlots() const { return availableSlots_; }

    uint64_t SlotByteSize() const { return slotStandardMemReq_; }

    bool ContainsPtr(uintdevptr_t ptr) const
    {
        return workspaceAddr_ <= ptr && ptr < workspaceAddr_ + slotNum_ * slotStandardMemReq_;
    }

    void DumpMemoryUsage(const char* hint) const
    {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        DEV_MEM_DUMP("Slot tensor memory usage (%s)\n", hint);
        DEV_MEM_DUMP("            Memory pool size: %10lu bytes (%zu x %lu bytes)\n", slotNum_ * slotStandardMemReq_,
                     slotNum_, slotStandardMemReq_);
        DEV_MEM_DUMP("    Total memory requirement: %10lu bytes (%zu x %lu bytes)\n",
                     dfx_.historicalAllocated_ * slotStandardMemReq_, dfx_.historicalAllocated_, slotStandardMemReq_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)hint;
    }

private:
    void InsertList(BlockHeader* node, BlockHeader*& listHead)
    {
        node->listNext = listHead;
        listHead = node;
    }

private:
    WsMetadataAllocator* allocator_{nullptr};
    WsAllocation allocation_;

    size_t availableSlots_{0};
    size_t slotNum_{0};
    size_t initReadyCount_{0};
    uint64_t slotStandardMemReq_{0};

    uintdevptr_t workspaceAddr_{0};
    BlockHeader* freeListHeader_{nullptr};
    BlockHeader* notInUseHeaders_{nullptr};

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    struct DfxInfo {
        size_t historicalAllocated_{0};
    } dfx_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT

    friend class DevControlFlowCache;
};

} // namespace npu::tile_fwk::dynamic
