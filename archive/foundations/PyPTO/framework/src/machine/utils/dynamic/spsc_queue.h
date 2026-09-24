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
 * \file spsc_queue.h
 * \brief
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

template <typename T, int N>
class SPSCQueue {
    constexpr static int ALIGN_SIZE = 512;

public:
    inline void Enqueue(const T& val)
    {
        while (!TryEnqueue(val))
            ;
    }

    inline bool TryEnqueue(const T& val)
    {
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_relaxed);
        if (tail - head == N) {
            return false;
        }
        pools_[tail % N] = val;
        tail_.fetch_add(1, std::memory_order_release);
        return true;
    }

    inline T Dequeue()
    {
        T val;
        while (!TryDequeue(val))
            ;
        return val;
    }

    inline bool TryDequeue(T& val)
    {
        auto head = head_.load(std::memory_order_relaxed);
        auto tail = tail_.load(std::memory_order_acquire);
        if (tail - head == 0) {
            return false;
        }
        val = pools_[head % N];
        head_.fetch_add(1, std::memory_order_release);
        return true;
    }

    inline void PopFront()
    {
        if (tail_ - head_ > 0) {
            head_.fetch_add(1, std::memory_order_release);
        }
    }

    inline bool TempDequeue(T& val)
    {
        auto head = head_.load(std::memory_order_relaxed);
        auto tail = tail_.load(std::memory_order_acquire);
        if (tail - head == 0) {
            return false;
        }
        val = pools_[head % N];
        return true;
    }

    inline bool FreeUntil(std::function<bool(const T&, bool&)> checker)
    {
        bool anyCanfree = false;
        while (true) {
            size_t head = head_.load(std::memory_order_acquire);
            size_t tail = tail_.load(std::memory_order_acquire);
            if (head == tail)
                break;

            size_t i = 0;
            bool popped = false;
            while (head + i < tail) {
                size_t idx = (head + i) % N;
                const T& elem = pools_[idx];

                bool continueFlag = false;
                bool canfree = checker(elem, continueFlag);
                if (canfree) {
                    anyCanfree = true;
                    if (i == 0) {
                        head_.fetch_add(1, std::memory_order_release);
                        popped = true;
                        break;
                    } else {
                        pools_[idx] = nullptr;
                        ++i;
                        continue;
                    }
                } else {
                    if (continueFlag) {
                        ++i;
                        continue;
                    } else {
                        return anyCanfree;
                    }
                }
            }

            if (!popped) {
                break;
            }
        }
        return anyCanfree;
    }

    inline bool IsEmpty() { return (head_ == tail_); }

    inline void ResetEmpty()
    {
        head_ = 0;
        tail_ = 0;
    }

private:
    alignas(ALIGN_SIZE) std::atomic<uint64_t> head_ = {0};
    alignas(ALIGN_SIZE) std::atomic<uint64_t> tail_ = {0};
    T pools_[N];
};
