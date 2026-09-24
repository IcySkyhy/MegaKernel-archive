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
 * \file connection_matrix.cpp
 * \brief
 */

#include "connection_matrix.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "GlobalMemoryReuse"

namespace npu::tile_fwk {
ConnectionMatrix::ConnectionMatrix(Function* func) : func_(func)
{
    auto operations = func->Operations(false);
    size_ = operations.size();
    invalidBitmap_.ResizeBits(size_);
    bitMaps_.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
        bitMaps_.emplace_back(size_);
    }
}

void ConnectionMatrix::Generate(Function* func)
{
    if (func == nullptr) {
        return;
    }
    func_ = func;

    for (auto& op : func->Operations(false)) {
        std::unordered_set<Operation*> producers = op.ProducerOps();
        SetConnectivity(producers, op);
    }
    return;
}

void ConnectionMatrix::SetConnectivity(const std::unordered_set<Operation*>& producers, Operation& op)
{
    LargeBitmap& bitmap = GetMutableBitMap(op);
    if (producers.count(&op) == 0) {
        bitmap.SetValues(0U);
    }
    bitmap.SetBit(static_cast<size_t>(GetIndex(op)));
    for (Operation* producer : producers) {
        if (producer != &op) {
            bitmap.Or(GetBitMap(*producer));
            APASS_LOG_DEBUG_F(Elements::Function, "SetConnectivity for op %s %d and op %s %d.",
                              producer->GetOpcodeStr().c_str(), producer->opmagic, op.GetOpcodeStr().c_str(),
                              op.opmagic);
        }
    }
}

uint64_t ConnectionMatrix::GetIndex(const Operation& op) const
{
    return static_cast<uint64_t>(func_->Operations(false).GetOpPosition(op));
}

bool ConnectionMatrix::IsConnected(const Operation& a, const Operation& b) const
{
    return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

bool ConnectionMatrix::IsConnected(uint64_t indexA, uint64_t indexB) const
{
    if (indexA >= size_ || indexB >= size_) {
        APASS_LOG_WARN_F(Elements::Function,
                         "Func ConnectionMatrix::IsConnected invalid index: indexA %lu, indexB, %lu.", indexA, indexB);
        return false;
    }
    return GetBitMap(indexB).GetBit(static_cast<size_t>(indexA));
}

const LargeBitmap& ConnectionMatrix::GetBitMap(const Operation& op) const
{
    return bitMaps_[static_cast<uint64_t>(GetIndex(op))];
}

LargeBitmap& ConnectionMatrix::GetMutableBitMap(const Operation& op)
{
    return bitMaps_[static_cast<uint64_t>(GetIndex(op))];
}

const LargeBitmap& ConnectionMatrix::GetBitMap(uint64_t index) const
{
    if (index >= size_) {
        APASS_LOG_WARN_F(Elements::Function, "Func ConnectionMatrix::GetBitMap invalid index: index %lu.", index);
        return invalidBitmap_;
    }
    return bitMaps_[index];
}
} // namespace npu::tile_fwk
