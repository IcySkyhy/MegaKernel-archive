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
 * \file large_bm.cpp
 * \brief
 */

#include "large_bm.h"

#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "GlobalMemoryReuse"

namespace npu::tile_fwk {
constexpr size_t BITS_EACH_VALUE = 64UL;
constexpr size_t RIGHT_SHIFT_SIZE = 6UL;

constexpr size_t AlignBitSize(size_t bitSize) { return bitSize + BITS_EACH_VALUE - 1; }

constexpr size_t AlignArraySize(size_t bitSize) { return AlignBitSize(bitSize) >> RIGHT_SHIFT_SIZE; }

void LargeBitmap::ResizeBits(const size_t newSize)
{
    if (newSize < size_) {
        return;
    }

    size_t newByteSize = AlignArraySize(newSize);
    if (newByteSize == AlignArraySize(size_)) {
        size_ = newSize;
        return;
    }

    this->bits_.resize(newByteSize, 0);
    for (size_t i = size_; i < AlignBitSize(size_); ++i) {
        ClearBit(i);
    }

    size_ = newSize;
}

// Shifting right by 6 bits is equivalent to dividing by 64
void LargeBitmap::ClearBit(const size_t bitIdx)
{
    if (bitIdx >= size_) {
        APASS_LOG_WARN_F(Elements::Function,
                         "Func LargeBitmap::ClearBit bitIdx %zu is not valid, total size is %zu bits.", bitIdx, size_);
        return;
    }
    bits_[bitIdx >> RIGHT_SHIFT_SIZE] &= ~(1UL << (bitIdx % BITS_EACH_VALUE));
}

LargeBitmap::LargeBitmap(const size_t& size) : size_(size), bits_(AlignArraySize(size), 0UL) {}

bool LargeBitmap::operator==(const LargeBitmap& anotherBm) const { return bits_ == anotherBm.bits_; }

bool LargeBitmap::operator!=(const LargeBitmap& anotherBm) const { return bits_ != anotherBm.bits_; }

void LargeBitmap::SetValues(const uint64_t& value) { std::fill(bits_.begin(), bits_.end(), value); }

void LargeBitmap::SetBit(const size_t& index)
{
    if (index >= size_) {
        APASS_LOG_WARN_F(Elements::Function, "Index %zu is not valid, total size is %zu bits.", index, size_);
        return;
    }
    bits_[index / BITS_EACH_VALUE] |= 1UL << (index % BITS_EACH_VALUE);
}

bool LargeBitmap::GetBit(const size_t& index) const
{
    if (index >= size_) {
        APASS_LOG_WARN_F(Elements::Function, "Index %zu is not valid, total size is %zu bits.", index, size_);
        return false;
    }
    return static_cast<bool>(bits_[index / BITS_EACH_VALUE] & (1UL << (index % BITS_EACH_VALUE)));
}

void LargeBitmap::Or(const LargeBitmap& anotherBm)
{
    size_t index = 0UL;
    const size_t anotherSize = anotherBm.bits_.size();
    for (auto& bit : bits_) {
        if (index >= anotherSize) {
            return;
        }
        bit |= anotherBm.bits_[index];
        ++index;
    }
}

void LargeBitmap::And(const LargeBitmap& anotherBm)
{
    size_t index = 0UL;
    const size_t anotherSize = anotherBm.bits_.size();
    for (auto& bit : bits_) {
        if (index >= anotherSize) {
            return;
        }
        bit &= anotherBm.bits_[index];
        ++index;
    }
}
} // namespace npu::tile_fwk
