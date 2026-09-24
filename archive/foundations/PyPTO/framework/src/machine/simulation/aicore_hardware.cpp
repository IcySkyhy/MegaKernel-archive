/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_hardware.cpp
 * \brief host-side aicore hardware simulation for mainbase/cond/reg32 communication
 */

#include "machine/simulation/aicore_hardware.h"

namespace npu::tile_fwk::dynamic {

AicoreHardware& AicoreHardware::Global()
{
    static AicoreHardware hardware;
    return hardware;
}

void AicoreHardware::Reset(size_t coreNum)
{
    std::lock_guard<std::mutex> lock(mutex_);
    coreNum_ = coreNum;
    mainBase_ = std::make_unique<std::atomic<uint64_t>[]>(coreNum_);
    cond_ = std::make_unique<std::atomic<uint64_t>[]>(coreNum_);
    reg32_ = std::make_unique<std::atomic<uint64_t>[]>(coreNum_ + 3);
    for (size_t i = 0; i < coreNum_; ++i) {
        mainBase_[i].store(0, std::memory_order_relaxed);
        cond_[i].store(0, std::memory_order_relaxed);
    }
}

bool AicoreHardware::IsValid(size_t blockId) const { return blockId < coreNum_ && mainBase_ && cond_; }

void AicoreHardware::WriteMainBase(size_t blockId, uint64_t value)
{
    if (!IsValid(blockId)) {
        return;
    }
    mainBase_[blockId].store(value, std::memory_order_release);
}

uint64_t AicoreHardware::ReadMainBase(size_t blockId) const
{
    if (!IsValid(blockId)) {
        return 0;
    }
    return mainBase_[blockId].load(std::memory_order_acquire);
}

void AicoreHardware::WriteCond(size_t blockId, uint64_t value)
{
    if (!IsValid(blockId)) {
        return;
    }
    cond_[blockId].store(value, std::memory_order_release);
}

uint64_t AicoreHardware::ReadCond(size_t blockId) const
{
    if (!IsValid(blockId)) {
        return 0;
    }
    return cond_[blockId].load(std::memory_order_acquire);
}

size_t AicoreHardware::CoreNum() const { return coreNum_; }

uint64_t AicoreHardware::GetReg32Addr(size_t blockId) const
{
    if (!IsValid(blockId)) {
        return 0;
    }
    return reinterpret_cast<uint64_t>(&reg32_[blockId]);
}

} // namespace npu::tile_fwk::dynamic
