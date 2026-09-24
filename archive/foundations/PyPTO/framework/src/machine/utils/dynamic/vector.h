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
 * \file vector.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/utils/device_log.h"
#include "securec.h"
#include <type_traits>
#include <cstring>
#include <ostream>

namespace npu::tile_fwk::dynamic {

template <typename T, WsMemCategory category = WsMemCategory::UNCLASSIFIED_VECTOR,
          typename WsAllocator_T = WsMetadataAllocator>
class Vector {
public:
    using value_type = T;
    using size_type = uint32_t;

public:
    Vector() = default;
    explicit Vector(WsAllocator_T& allocator) : allocator_{&allocator} {}

    Vector(const Vector& oth) : allocator_(oth.allocator_)
    {
        if (oth.empty()) {
            return;
        }
        ExpandCapacity(oth.size_);
        if constexpr (std::is_trivially_copyable_v<value_type>) {
            DevMemcpyS(data(), sizeof(value_type) * oth.size_, oth.data(), sizeof(value_type) * oth.size_);
        } else {
            for (size_type i = 0; i < oth.size_; i++) {
                new (data() + i) value_type(oth[i]);
            }
        }
        size_ = oth.size_;
    }

    Vector(Vector&& oth)
        : allocator_(oth.allocator_), dataAllocation_(oth.dataAllocation_), capacity_(oth.capacity_), size_(oth.size_)
    {
        oth.dataAllocation_.Invalidate();
        oth.capacity_ = 0;
        oth.size_ = 0;
    }

    ~Vector()
    {
        clear();
        if (dataAllocation_) {
            DEV_ASSERT(DevDataErr::VECTOR_UNINITIALIZED, allocator_);
            allocator_->Deallocate(dataAllocation_);
        }
    }

    Vector& operator=(const Vector& oth)
    {
        if (this == &oth) {
            return *this;
        }

        this->~Vector();
        new (this) Vector(oth);

        return *this;
    }

    Vector& operator=(Vector&& oth)
    {
        if (this == &oth) {
            return *this;
        }

        this->~Vector();
        new (this) Vector(std::move(oth));

        return *this;
    }

    void InitAllocator(WsAllocator_T& allocator)
    {
        // InitAllocator or passing allocator via constructor is allowed to happen only once
        DEV_ASSERT_MSG(DevDataErr::VECTOR_UNINITIALIZED, !allocator_, "Vector has been initialized already");
        allocator_ = &allocator;
    }

    value_type* data() { return dataAllocation_.As<value_type>(); }
    const value_type* data() const { return dataAllocation_.As<value_type>(); }

    value_type& operator[](size_type idx)
    {
        DEV_ASSERT_MSG(DevDataErr::VECTOR_INDEX_OUT_OF_RANGE, idx < size_, "idx=%u >= size_=%u", idx, size_);
        return data()[idx];
    }
    const value_type& operator[](size_type idx) const
    {
        DEV_ASSERT_MSG(DevDataErr::VECTOR_INDEX_OUT_OF_RANGE, idx < size_, "idx=%u >= size_=%u", idx, size_);
        return data()[idx];
    }

    value_type* begin() { return data(); }
    const value_type* begin() const { return data(); }

    value_type* end() { return data() + size_; }
    const value_type* end() const { return data() + size_; }

    value_type& front()
    {
        DEV_ASSERT(DevDataErr::VECTOR_EMPTY_ACCESS, !empty());
        return *begin();
    }
    const value_type& front() const
    {
        DEV_ASSERT(DevDataErr::VECTOR_EMPTY_ACCESS, !empty());
        return *begin();
    }

    value_type& back()
    {
        DEV_ASSERT(DevDataErr::VECTOR_EMPTY_ACCESS, !empty());
        return *(end() - 1);
    }
    const value_type& back() const
    {
        DEV_ASSERT(DevDataErr::VECTOR_EMPTY_ACCESS, !empty());
        return *(end() - 1);
    }

    size_type capacity() const { return capacity_; }
    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void push_back(const value_type& value) { emplace_back(value); }
    void push_back(value_type&& value) { emplace_back(std::move(value)); }

    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        if (size_ == capacity_) {
            ExpandCapacity(size_ + 1);
        }
        if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible_v<value_type>) {
            memset_s(data() + size_, sizeof(value_type), 0, sizeof(value_type));
        } else {
            new (data() + size_) value_type(std::forward<Args>(args)...);
        }
        size_++;
    }

    void pop_back()
    {
        DEV_ASSERT(DevDataErr::VECTOR_EMPTY_ACCESS, !empty());
        if constexpr (!std::is_trivially_destructible_v<value_type>) {
            data()[size_ - 1].~value_type();
        }
        size_--;
    }

    void reserve(size_type newCapacity)
    {
        if (newCapacity <= capacity_) {
            return;
        }

        InternalReserve(newCapacity);
    }

    void resize(size_type newSize)
    {
        if (newSize > size_) {
            InternalAppend(newSize - size_);
        } else {
            InternalPopBack(size_ - newSize);
        }
    }

    void resize(size_type newSize, const value_type& val)
    {
        if (newSize > size_) {
            InternalAppend(newSize - size_, val);
        } else {
            InternalPopBack(size_ - newSize);
        }
    }

    void clear() { InternalPopBack(size_); }

private:
    void ExpandCapacity(size_type capacityReq)
    {
        static constexpr size_type MIN_CAPACITY = 8;

        DEV_ASSERT_MSG(DevDataErr::VECTOR_INDEX_OUT_OF_RANGE, capacityReq > capacity_,
                       "Unexpected ExpandCapacity call: capacityReq %u <= capacity_ %u", capacityReq, capacity_);

        size_type newCapacity = std::max(MIN_CAPACITY, capacityReq);
        newCapacity = std::max(newCapacity, capacity_ << 1); // multiplier: 2.0
        InternalReserve(newCapacity);
    }

    void InternalReserve(size_type capacity)
    {
        DEV_ASSERT(DevDataErr::VECTOR_UNINITIALIZED, allocator_);
        WsAllocation alloc = allocator_->template Allocate<value_type>(capacity, category);
        if (dataAllocation_) {
            value_type* newData = alloc.As<value_type>();
            value_type* oldData = data();
            if constexpr (std::is_trivially_copyable_v<value_type>) {
                DevMemcpyS(newData, sizeof(value_type) * size_, oldData, sizeof(value_type) * size_);
            } else {
                for (size_type i = 0; i < size_; i++) {
                    new (newData + i) value_type(std::move(oldData[i]));
                }
            }
            if constexpr (!std::is_trivially_destructible_v<value_type>) {
                for (size_type i = 0; i < size_; i++) {
                    oldData[i].~value_type();
                }
            }
            allocator_->Deallocate(dataAllocation_);
        }
        dataAllocation_ = alloc;
        capacity_ = capacity;
    }

    template <typename... Args>
    void InternalAppend(size_type n, Args&&... args)
    {
        if (n == 0) {
            return;
        }

        if (size_ + n > capacity_) {
            ExpandCapacity(size_ + n);
        }

        if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible_v<value_type>) {
            memset_s(data() + size_, sizeof(value_type) * n, 0, sizeof(value_type) * n);
        } else {
            new (data() + size_) value_type(std::forward<Args>(args)...);
            for (size_type i = size_ + 1; i < size_ + n; i++) {
                new (data() + i) value_type(data()[size_]);
            }
        }
        size_ += n;
    }

    void InternalPopBack(size_type n)
    {
        if (n == 0) {
            return;
        }

        DEV_ASSERT_MSG(DevDataErr::VECTOR_INDEX_OUT_OF_RANGE, n <= size_, "n=%u > size_=%u", n, size_);
        if constexpr (!std::is_trivially_destructible_v<value_type>) {
            for (size_type i = size_ - n; i < size_; i++) {
                data()[i].~value_type();
            }
        }
        size_ -= n;
    }

private:
    WsAllocator_T* allocator_{nullptr};
    WsAllocation dataAllocation_;
    size_type capacity_{0};
    size_type size_{0};
};

template <typename T, WsMemCategory category, typename WsAllocator_T>
inline std::ostream& operator<<(std::ostream& os, const Vector<T, category, WsAllocator_T>& vector)
{
    os << "[";
    if (!vector.empty()) {
        os << vector[0];
        for (size_t i = 1; i < vector.size(); i++) {
            os << ", " << vector[i];
        }
    }
    os << "]";
    return os;
}

} // namespace npu::tile_fwk::dynamic
