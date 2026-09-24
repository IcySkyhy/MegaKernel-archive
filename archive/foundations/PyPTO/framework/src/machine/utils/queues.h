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
 * \file queues.h
 * \brief
 */

#ifndef QUEUES_H
#define QUEUES_H

#include "tilefwk/aicpu_common.h"
#include "interface/utils/common.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/error_code.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_encode_types.h"
#include <string>
#include <sstream>
#include <mutex>

namespace npu::tile_fwk {

template <class T>
struct QueueGeneric {
    QueueGeneric(uint32_t capacity, T* elem) : head_(0), tail_(0), elem_(elem), capacity_(capacity) {}

    QueueGeneric& operator=(const QueueGeneric& rhs)
    {
        head_ = 0;
        tail_ = rhs.Size();
        if (Capacity() == 0) {
            return *this;
        }
        ASSERT(SchedErr::READY_QUEUE_OVERFLOW, rhs.Size() <= Capacity());
        std::copy(rhs.begin(), rhs.end(), elem_);
        return *this;
    }

    __attribute__((always_inline)) uint32_t Capacity() const { return capacity_; }

    __attribute__((always_inline)) uint32_t Size() const { return tail_ - head_; }

    std::string Str() const
    {
        std::stringstream ss;
        ss << "Queue at " << this << " head=" << head_ << " tail=" << tail_ << " capacity=" << Capacity();
        return ss.str();
    }

    std::string Dump() const
    {
        std::stringstream ss;
        for (const ValueType* it = begin(); it != end(); ++it) {
            ss << *it << " ";
        }
        return ss.str();
    }

    const T* begin() const { return elem_ + head_; }

    const T* end() const { return elem_ + tail_; }

    typedef T ValueType;

    void Reloc(dynamic::RelocRange& r) { r.Reloc(this->elem_); }

protected:
    uint32_t head_;
    uint32_t tail_;
    ValueType* elem_;

private:
    uint32_t capacity_;
};

template <class T>
struct LockableQueueGeneric : public QueueGeneric<T> {
    using DataRange = std::pair<const T*, const T*>;
    using QueueGeneric<T>::operator=;
    typedef T ValueTypeV32 __attribute__((vector_size(32)));

    LockableQueueGeneric(uint32_t capacity = 0, T* elem = nullptr) : QueueGeneric<T>(capacity, elem), lockFlag_(0) {}

    __attribute__((always_inline)) inline void lock()
    {
        while (!__sync_bool_compare_and_swap(&lockFlag_, 0, 1)) {
        }
    }

    __attribute__((always_inline)) inline void unlock()
    {
        while (!__sync_bool_compare_and_swap(&lockFlag_, 1, 0)) {
        }
    }

    __attribute__((always_inline)) inline uint32_t UnsafeSize() const { return this->Size(); }

    __attribute__((always_inline)) inline uint32_t UnsafeAtomicSize() const
    {
        return __atomic_load_n(&this->tail_, std::memory_order_relaxed) -
               __atomic_load_n(&this->head_, std::memory_order_relaxed);
    }

    __attribute__((always_inline)) inline void UnsafeEnqueue(T x)
    {
        ASSERT(SchedErr::READY_QUEUE_OVERFLOW, this->tail_ < this->Capacity());
        this->elem_[this->tail_] = x;
        this->tail_++;
    }

    __attribute__((always_inline)) inline void UnsafeEnqueue(T* x, uint32_t count)
    {
        DevMemcpyS(this->elem_ + this->tail_, sizeof(T) * (this->Capacity() - this->tail_), x, sizeof(T) * count);
        this->tail_ += count;
    }

    __attribute__((always_inline)) inline void UnsafeEnqueueSimd(const ValueTypeV32& taskidv)
    {
        constexpr auto n = sizeof(ValueTypeV32) / sizeof(T);
        ASSERT(SchedErr::READY_QUEUE_OVERFLOW, this->tail_ + n <= this->Capacity());
        *reinterpret_cast<ValueTypeV32*>(&this->elem_[this->tail_]) = taskidv;
        this->tail_ += n;
    }

    __attribute__((always_inline)) inline bool TryEnqueue(T x)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_fetch_add(&this->tail_, 1, std::memory_order_release);
        if (unlikely(t >= this->Capacity())) {
            __atomic_store_n(&this->tail_, t, std::memory_order_release);
            return false;
        }
        this->elem_[t] = x;
        return true;
    }

    __attribute__((always_inline)) inline bool TryEnqueue(const T* x, uint32_t count)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_fetch_add(&this->tail_, count,
                                        std::memory_order_release); // Is release order necessary here?
        if (unlikely(t + count > this->Capacity())) {
            __atomic_store_n(&this->tail_, t, std::memory_order_release); // Is release order necessary here?
            return false;
        }
        DevMemcpyS(this->elem_ + t, sizeof(T) * count, x, sizeof(T) * count);
        return true;
    }

    __attribute__((always_inline)) inline std::pair<const T*, const T*> DequeueAll()
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_load_n(&this->tail_, std::memory_order_relaxed);
        uint32_t h = __atomic_exchange_n(&this->head_, t, std::memory_order_relaxed);
        return std::make_pair(this->elem_ + h, this->elem_ + t);
    }

    __attribute__((always_inline)) inline DataRange Dequeue(uint32_t max_count)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_load_n(&this->tail_, std::memory_order_relaxed);
        uint32_t h = __atomic_load_n(&this->head_, std::memory_order_relaxed);
        uint32_t cnt = std::min(t - h, max_count);
        if (cnt == 0) {
            return DataRange(nullptr, nullptr);
        }
        __atomic_store_n(&this->head_, h + cnt, std::memory_order_release); // Is release order necessary here?
        return DataRange(this->elem_ + h, this->elem_ + h + cnt);
    }

    __attribute__((always_inline)) inline DataRange DequeueTail(uint32_t max_count, T* out)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_load_n(&this->tail_, std::memory_order_relaxed);
        uint32_t h = __atomic_load_n(&this->head_, std::memory_order_relaxed);
        uint32_t cnt = std::min(t - h, max_count);
        if (cnt == 0) {
            return DataRange(nullptr, nullptr);
        }
        __atomic_store_n(&this->tail_, t - cnt, std::memory_order_release); // Is release order necessary here?
        DevMemcpyS(out, sizeof(T) * max_count, this->elem_ + t - cnt, sizeof(T) * cnt);
        return DataRange(out, out + cnt);
    }

private:
    size_t lockFlag_;

    using QueueGeneric<T>::Size;
};

// aic aiv 已经ready的core function id队列
typedef LockableQueueGeneric<uint32_t> ReadyCoreFunctionQueue;
typedef QueueGeneric<uint32_t> ReadyCoreFunctionQueueUnsafe;

} // namespace npu::tile_fwk

#endif // QUEUES_H
