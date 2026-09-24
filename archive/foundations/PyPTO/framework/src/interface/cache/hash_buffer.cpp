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
 * \file hash_buffer.cpp
 * \brief
 */

#include "interface/inner/hash_buffer.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {

template <>
uint64_t HashBuffer::Get<uint64_t>(int index) const
{
    FE_ASSERT(index >= 0 && static_cast<size_t>(index + 1) < this->size())
        << "HashBuffer::Get index out of range: index=" << index << ", size=" << this->size();
    uint64_t l = this->at(index);
    uint64_t h = this->at(index + 1);
    return l + (h << 32); // h takes high 32 bits
}

template <>
int64_t HashBuffer::Get<int64_t>(int index) const
{
    FE_ASSERT(index >= 0 && static_cast<size_t>(index + 1) < this->size())
        << "HashBuffer::Get index out of range: index=" << index << ", size=" << this->size();
    int64_t l = this->at(index);
    int64_t h = this->at(index + 1);
    return l + (h << 32); // h takes high 32 bits
}

} // namespace npu::tile_fwk
