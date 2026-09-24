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
 * \file ws_allocator_basics.h
 * \brief
 */

#pragma once

#include "interface/utils/common.h"
#include "machine/utils/device_switch.h"
#include "machine/utils/device_log.h"

#include <cstdint>

namespace npu::tile_fwk::dynamic {

#define X(value) value,
enum class WsMemCategory : uint8_t {
#include "ws_categories.in"
};
#undef X

inline constexpr const char* GetCategoryName(WsMemCategory category)
{
#define X(value) #value,
    constexpr const char* WS_MEM_CATEGORY_NAMELIST[] = {
#include "ws_categories.in"
    };
#undef X

    return WS_MEM_CATEGORY_NAMELIST[ToUnderlying(category)];
}
struct WsAllocation {
    friend class WsSlotAllocator;
    friend class SeqWsAllocator;
    friend class WsAllocatorCounter;
    friend class DelayedDumper;

    using uintdevptr_t = uint64_t;

    uintdevptr_t ptr{0};
    uint32_t parallelWsId{0};

    WsAllocation() {}
    WsAllocation(uintdevptr_t addr, uint32_t wsid) : ptr(addr), parallelWsId(wsid) {}

    operator bool() const { return ptr != 0; }

    template <typename T>
    T* As()
    {
        return reinterpret_cast<T*>(ptr);
    }

    template <typename T>
    const T* As() const
    {
        return reinterpret_cast<const T*>(ptr);
    }

    void Invalidate() { ptr = 0; }

    const char* GetCategoryName() const
    {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return ::npu::tile_fwk::dynamic::GetCategoryName(category_);
#else
        return "";
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    size_t rawMemReq_{0};
    WsMemCategory category_{WsMemCategory::UNCLASSIFIED};
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
};

enum class WsAllocatorProperty {
    TENSOR_MEM,
    METADATA_MEM,
};
inline constexpr const char* GetWsAllocatorPropertyName(WsAllocatorProperty property)
{
    switch (property) {
        case WsAllocatorProperty::TENSOR_MEM:
            return "Tensordata";
        case WsAllocatorProperty::METADATA_MEM:
            return "Metadata";
        default:
            return "Undefined";
    }
}
enum class WsMemoryState {
    INSIDE,
    OUTSIDE,
    CROSS_BOUNDARY,
};

class WsMemoryVerifier {
    using uintdevptr_t = uint64_t;

public:
    void Init(uintdevptr_t workspaceAddr, uint64_t workspaceSize)
    {
        workspaceAddr_ = workspaceAddr;
        workspaceSize_ = workspaceSize;
    }

    WsMemoryState Verify(uintdevptr_t ptr, size_t size) const
    {
        if (workspaceAddr_ <= ptr && ptr + size <= workspaceAddr_ + workspaceSize_) {
            return WsMemoryState::INSIDE;
        }
        if (ptr + size <= workspaceAddr_ || workspaceAddr_ + workspaceSize_ <= ptr) {
            return WsMemoryState::OUTSIDE;
        }
        return WsMemoryState::CROSS_BOUNDARY;
    }

private:
    uintdevptr_t workspaceAddr_{0};
    uint64_t workspaceSize_{0};
};

} // namespace npu::tile_fwk::dynamic
