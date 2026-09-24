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
 * \file small_array.h
 * \brief
 */

#pragma once

#include <array>
#include <cstring>
#include <cassert>
#include "securec.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk {
template <typename T, int N>
class FixedArray {
public:
    inline size_t size() const { return size_; }

    inline void clear() { size_ = 0; }
    inline bool empty() { return size_ == 0; }

    inline T* back() { return &data_[size_]; }
    inline T* data() { return &data_.data(); }

    inline void push_back(T val) { data_[size_++] = val; }
    inline void CopyTo(T* ptr, int n, int offset = 0) const
    {
        DevMemcpyS(ptr, n * sizeof(T), data_.data() + offset, n * sizeof(T));
    }

    inline void resize(size_t size)
    {
        if (size > N) {
            DEV_ERROR(DevDataErr::SMALL_ARRAY_RESIZE_OOB,
                      "#array.resize: resize %zu exceeds maximum allowed value %d\n", size, N);
        }
        DEV_ASSERT(DevDataErr::SMALL_ARRAY_RESIZE_OOB, size <= N);
        size_ = size;
    }

    inline T& operator[](size_t index) { return data_[index]; }

    inline const T& operator[](size_t index) const { return data_[index]; }

private:
    size_t size_{0};
    std::array<T, N> data_;
};
} // namespace npu::tile_fwk
