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
 * \file aicore_hardware.h
 * \brief host-side aicore hardware simulation for mainbase/cond/reg32 communication
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <memory>
#include <mutex>

namespace npu::tile_fwk::dynamic {

class AicoreHardware {
public:
    static AicoreHardware& Global();

    void Reset(size_t coreNum);

    void WriteMainBase(size_t blockId, uint64_t value);
    uint64_t ReadMainBase(size_t blockId) const;

    void WriteCond(size_t blockId, uint64_t value);
    uint64_t ReadCond(size_t blockId) const;
    uint64_t GetReg32Addr(size_t blockId) const;

    size_t CoreNum() const;

private:
    bool IsValid(size_t blockId) const;

private:
    mutable std::mutex mutex_;
    size_t coreNum_{0};
    std::unique_ptr<std::atomic<uint64_t>[]> mainBase_;
    std::unique_ptr<std::atomic<uint64_t>[]> cond_;
    std::unique_ptr<std::atomic<uint64_t>[]> reg32_;
};

} // namespace npu::tile_fwk::dynamic
