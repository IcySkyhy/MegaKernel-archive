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
 * \file id_gen.h
 * \brief
 */
#include <atomic>
#include <mutex>

#pragma once

namespace npu::tile_fwk {
enum class IdType {
    RAW_TENSOR,
    FUNCTION,
    LOGICAL_TENSOR,
    TENSOR_INDEX,
};

template <IdType T>
class IdGen {
public:
    static auto& Inst()
    {
        static IdGen<T> inst;
        return inst;
    }

    auto NewId() { return id_.fetch_add(1, std::memory_order_relaxed); }

    auto CurId() const { return id_.load(std::memory_order_acquire); }

    void Reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id_.store(0, std::memory_order_release);
    }

    void SetId(int id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id_.store(id, std::memory_order_release);
    }

private:
    IdGen() = default;
    std::atomic<int> id_{0};
    std::mutex mutex_;
};

} // namespace npu::tile_fwk
