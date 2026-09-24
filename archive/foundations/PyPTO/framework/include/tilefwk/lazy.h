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
 * \file lazy.h
 * \brief
 */

#pragma once

#include <atomic>

namespace npu::tile_fwk {

template <class T>
class LazyValue {
public:
    virtual ~LazyValue() = default;
    virtual const T& Get() const = 0;
};

template <typename T>
class LazyShared {
public:
    LazyShared() = default;

    LazyShared(const LazyShared& rhs)
    {
        if (auto val = rhs.value_.load(std::memory_order_acquire)) {
            value_ = new T(*val);
        }
    }

    LazyShared(LazyShared&& rhs)
    {
        if (auto val = rhs.value_.exchange(nullptr, std::memory_order_acq_rel)) {
            value_ = new T(*val);
        }
    }

    virtual ~LazyShared() { Reset(); }

    template <typename Factory>
    T& Ensure(const Factory& factory)
    {
        if (auto val = value_.load(std::memory_order_acquire)) {
            return *val;
        }
        auto val = new T(factory());
        T* old = nullptr;
        if (!value_.compare_exchange_strong(old, val, std::memory_order_release, std::memory_order_acquire)) {
            delete val;
            return *old;
        }
        return *val;
    }

    void Reset()
    {
        if (auto old = value_.load(std::memory_order_relaxed)) {
            value_.store(nullptr, std::memory_order_relaxed);
            delete old;
        }
    }

    LazyShared& operator=(const LazyShared& rhs)
    {
        if (this != &rhs)
            *this = LazyValue(rhs);
        return *this;
    }

    LazyShared& operator=(const LazyShared&& rhs)
    {
        if (this != &rhs)
            *this = LazyValue(std::move(rhs));
        return *this;
    }

private:
    std::atomic<T*> value_{nullptr};
};
} // namespace npu::tile_fwk
