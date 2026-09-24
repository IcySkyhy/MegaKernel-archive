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
 * \file item_pool.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/allocator/allocators.h"
#include <limits>

namespace npu::tile_fwk::dynamic {

using ItemPoolIter = int64_t;

struct ItemPoolMeta {
    size_t size;
    size_t freeCount;
    ItemPoolIter freeListHeadIndex;
    size_t initReadyCount;
};

static constexpr ItemPoolIter ITEM_POOL_INVALID_INDEX = std::numeric_limits<int64_t>::max();
static constexpr ItemPoolIter ITEM_POOL_NON_FREE_INDEX = std::numeric_limits<int64_t>::max() - 1;

template <typename T, typename WsAllocator_T = WsMetadataAllocator>
class ItemPool {
public:
    struct ItemBlock {
        char buf[sizeof(T)];
        ItemPoolIter freeListNextIndex;

        T& Item() { return *reinterpret_cast<T*>(buf); }

        const T& Item() const { return *reinterpret_cast<const T*>(buf); }
    };

public:
    ItemPool() = default;
    ItemPool(WsAllocator_T& allocator, size_t size, WsMemCategory category = WsMemCategory::UNCLASSIFIED_ITEMPOOL)
    {
        Init(allocator, size, category);
    }

    ~ItemPool()
    {
        if (allocation_) {
            // Only [0, initReadyCount_) have ever been constructed (or recycled onto freelist).
            ItemBlock* itemBase = allocation_.As<ItemBlock>();
            for (size_t i = 0; i < initReadyCount_; i++) {
                if (itemBase[i].freeListNextIndex == ITEM_POOL_NON_FREE_INDEX) {
                    (reinterpret_cast<T*>(itemBase + i))->~T();
                }
            }

            DEV_ASSERT(DevDataErr::ITEM_POOL_UNINITIALIZED, allocator_);
            allocator_->Deallocate(allocation_);
        }
    }

    void Init(WsAllocator_T& allocator, size_t size, WsMemCategory category = WsMemCategory::UNCLASSIFIED_ITEMPOOL)
    {
        DEV_ASSERT_MSG(DevDataErr::ITEM_POOL_UNINITIALIZED, !allocator_, "ItemPool has been initialized already");
        allocator_ = &allocator;
        size_ = size;
        category_ = category;
        freeCount_ = size_;
        initReadyCount_ = 0;
        freeListHeadIndex_ = ITEM_POOL_INVALID_INDEX;
        allocation_ = allocator_->template Allocate<ItemBlock>(size_, category);
    }

    template <typename... Args>
    T* Create(Args&&... args)
    {
        DEV_ASSERT_MSG(DevDataErr::ITEM_POOL_FREE_LIST_INVALID, freeCount_ > 0, "Available items: %zu/%zu", freeCount_,
                       size_);
        ItemBlock* item = nullptr;
        if (freeListHeadIndex_ != ITEM_POOL_INVALID_INDEX) {
            // Recycled slot from Destroy.
            item = &ItemAt(freeListHeadIndex_);
            freeListHeadIndex_ = item->freeListNextIndex;
        } else {
            // Virgin slot: bump high-water mark; never touch freelist.
            item = &ItemAt(static_cast<ItemPoolIter>(initReadyCount_));
            initReadyCount_++;
        }
        item->freeListNextIndex = ITEM_POOL_NON_FREE_INDEX;
        freeCount_--;

        T* newItem = reinterpret_cast<T*>(item->buf);
        new (newItem) T(std::forward<Args>(args)...);
        return newItem;
    }

    template <typename... Args>
    ItemPoolIter Allocate(Args&&... args)
    {
        T* item = Create(args...);
        return reinterpret_cast<ItemBlock*>(item) - allocation_.As<ItemBlock>();
    }

    // Virgin bump: Fill I/O is the first Allocate on a freshly Init()'d pool, so the
    // freelist is empty. Skip ItemAt bounds check and freelist branch per item.
    template <typename... Args>
    ItemPoolIter AllocateBump(Args&&... args)
    {
        DEV_ASSERT_MSG(DevDataErr::ITEM_POOL_FREE_LIST_INVALID,
                       freeCount_ > 0 && initReadyCount_ < size_ && freeListHeadIndex_ == ITEM_POOL_INVALID_INDEX,
                       "Available items: %zu/%zu ready=%zu", freeCount_, size_, initReadyCount_);
        ItemBlock* base = allocation_.As<ItemBlock>();
        ItemBlock* item = base + initReadyCount_;
        item->freeListNextIndex = ITEM_POOL_NON_FREE_INDEX;
        initReadyCount_++;
        freeCount_--;
        new (item->buf) T(std::forward<Args>(args)...);
        return static_cast<ItemPoolIter>(item - base);
    }

    void Destroy(T* item)
    {
        item->~T();
        ItemBlock* block = (ItemBlock*)item;
        DEV_ASSERT_MSG(DevDataErr::ITEM_POOL_FREE_LIST_INVALID, block->freeListNextIndex == ITEM_POOL_NON_FREE_INDEX,
                       "Double free detected in ItemPool");
        AppendFreeList(block);
        freeCount_++;
    }

    T& At(ItemPoolIter index) { return ItemAt(index).Item(); }

    void DestroyAt(ItemPoolIter index) { Destroy(&At(index)); }

    size_t FreeItemNum() const { return freeCount_; }

    size_t InitReadyCount() const { return initReadyCount_; }

    size_t Size() const { return size_; }

    void RestoreMetaData(const ItemPoolMeta& meta)
    {
        size_ = meta.size;
        freeCount_ = meta.freeCount;
        freeListHeadIndex_ = meta.freeListHeadIndex;
        initReadyCount_ = meta.initReadyCount;
    }

    ItemPoolMeta GetMetaData() const { return {size_, freeCount_, freeListHeadIndex_, initReadyCount_}; }

private:
    inline void AppendFreeList(ItemBlock* block)
    {
        block->freeListNextIndex = freeListHeadIndex_;
        freeListHeadIndex_ = block - allocation_.As<ItemBlock>();
    }

    inline ItemBlock& ItemAt(ItemPoolIter index)
    {
        DEV_ASSERT_MSG(DevDataErr::ITEM_POOL_INDEX_OUT_OF_RANGE, index >= 0 && static_cast<size_t>(index) < size_,
                       "Index %" PRId64 " out of range [0, %zu)", index, size_);
        return allocation_.As<ItemBlock>()[index];
    }

private:
    WsMemCategory category_{WsMemCategory::UNCLASSIFIED_ITEMPOOL};
    WsAllocator_T* allocator_{nullptr};
    WsAllocation allocation_;
    size_t size_{0};
    size_t freeCount_{0};
    size_t initReadyCount_{0};
    ItemPoolIter freeListHeadIndex_{ITEM_POOL_INVALID_INDEX};
};

} // namespace npu::tile_fwk::dynamic
