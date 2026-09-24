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
 * \file aot_binary.h
 * \brief Process-global AOT control-flow code pool.
 *
 * Policy: each pool entry records the DevProg hash that currently owns it.
 * EnsureCached(hash) — if that hash is still in the pool, skip memcpy; otherwise
 * overwrite a free / LRU entry. No refCount / pin / Detach.
 *
 * Pool size is small (AOT_CODE_POOL_NUM=16), so metadata uses a flat array +
 * monotonic LRU clock rather than open-hash / linked lists.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <climits>
#include <vector>
#include <tuple>
#include "securec.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/device_perf.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "tilefwk/aicpu_runtime.h"
#include "tilefwk/aikernel_data.h"

#ifndef STR
#define STR_(n) #n
#define STR(n) STR_(n)
#endif

#define AOT_CODE_POOL_NUM 16
#define AOT_CODE_POOL_CODE_SIZE (4096 * 0x1000)
#define AOT_CODE_POOL_TOTAL_SIZE (AOT_CODE_POOL_NUM * AOT_CODE_POOL_CODE_SIZE)
#define AOT_POOL_ENTRY_INVALID 0xFFU

// Weak comdat BSS (shared across TUs). Symbol must be at global scope so the
// C++ reference name matches the unmangled asm label (same pattern as the
// single-slot aotCodePoolCode pool). Do not put this extern inside a namespace.
#define AOT_DEFINE_BSS_POOL(name, totalSize)                                                                     \
    extern uint8_t name[totalSize];                                                                              \
    asm("\n\t.pushsection .bss." STR(name) ",\"axwG\",@nobits," STR(                                             \
        name) ",comdat"                                                                                          \
              "\n\t.p2align 12"                                                                                  \
              "\n\t.weak " STR(name) "\n\t.type " STR(name) ", @gnu_unique_object"                               \
                                                            "\n\t.size " STR(name) ", " STR(totalSize) "\n" STR( \
                                                                name) ":"                                        \
                                                                      "\n\t.zero " STR(totalSize) "\n\t.popsection")

AOT_DEFINE_BSS_POOL(aotCodePoolCodes, AOT_CODE_POOL_TOTAL_SIZE);

namespace npu::tile_fwk::dynamic {

// hashKey is written at encode time in EncodeDevAscendProgram::Init (dev_encode.cpp).
// 0 is reserved as the empty-entry sentinel for the AOT code pool.
inline uint64_t GetAOTCacheKey(const DevAscendProgram* prog) { return prog->hashKey; }

// Shared host/device pool (16 entries x 16MiB BSS).
struct AOTCodePoolManager {
    uint64_t ownerHashKey_[AOT_CODE_POOL_NUM]{};
    uint64_t lruSeq_[AOT_CODE_POOL_NUM]{};
    uint64_t lruClock_{0};

    static AOTCodePoolManager& Instance();

    static uintptr_t EntryCodeBase(int entryId)
    {
        return reinterpret_cast<uintptr_t>(aotCodePoolCodes) +
               static_cast<uintptr_t>(entryId) * static_cast<uintptr_t>(AOT_CODE_POOL_CODE_SIZE);
    }

    // lastId: DevProg hint of the previous pool entry that held this hash.
    int EnsureCached(uint64_t hashKey, uint8_t& lastId, const void* data, uint64_t size);

private:
    int FindEntry(uint64_t hashKey, uint8_t lastId) const;
    int SelectVictimEntry() const;
    void LoadEntry(int entryId, uint64_t hashKey, const void* data, uint64_t size);
};

struct AOTBinary {
    AOTBinary() {}

    void InitCodeSizeCached(uint64_t hashKey, uint8_t& lastId, const void* data, uint64_t size)
    {
        // Empty CF binary must not occupy / evict a pool entry (LoadEntry still writes ownerHashKey_).
        if (size == 0) {
            code_ = nullptr;
            size_ = 0;
            return;
        }
        if (size > AOT_CODE_POOL_CODE_SIZE) {
            DEV_ERROR(DevCommonErr::MEMCPY_FAILED, "AOTBinary code size %zu is too large, max %d", size,
                      AOT_CODE_POOL_CODE_SIZE);
            DEV_ASSERT(DevCommonErr::MEMCPY_FAILED, false);
            return;
        }
        const int entryId = AOTCodePoolManager::Instance().EnsureCached(hashKey, lastId, data, size);
        code_ = reinterpret_cast<unsigned char*>(AOTCodePoolManager::EntryCodeBase(entryId));
        size_ = size;
    }

    void InitCode(const void* data) { code_ = reinterpret_cast<const unsigned char*>(data); }

    const unsigned char* code_{nullptr};
    size_t size_{0};
};

struct DeviceExecuteContext;

struct AOTBinaryControlFlow : AOTBinary {
    typedef void (*controlFlowEntry)(struct DeviceExecuteContext* ctx, int64_t* symbolTable,
                                     RuntimeCallEntryType runtimeCallList[T_RUNTIME_CALL_MAX],
                                     DevStartArgsBase* startArgsBase);

    AOTBinaryControlFlow() = default;

    AOTBinaryControlFlow(const AOTBinaryControlFlow&) = delete;
    AOTBinaryControlFlow& operator=(const AOTBinaryControlFlow&) = delete;

    AOTBinaryControlFlow(AOTBinaryControlFlow&& other) noexcept
    {
        code_ = other.code_;
        size_ = other.size_;
        other.code_ = nullptr;
        other.size_ = 0;
    }

    AOTBinaryControlFlow& operator=(AOTBinaryControlFlow&& other) noexcept
    {
        if (this != &other) {
            code_ = other.code_;
            size_ = other.size_;
            other.code_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    AOTBinaryControlFlow(const std::tuple<const void*, uint64_t>& code, DevAscendProgram* prog,
                         AOTBinaryControlFlow::controlFlowEntry entry = nullptr)
        : AOTBinaryControlFlow(std::get<0>(code), std::get<1>(code), prog, entry)
    {}

    AOTBinaryControlFlow(const std::vector<uint8_t>& code, DevAscendProgram* prog,
                         AOTBinaryControlFlow::controlFlowEntry entry = nullptr)
        : AOTBinaryControlFlow(code.data(), code.size(), prog, entry)
    {}

    AOTBinaryControlFlow(const void* code, uint64_t codeSize, DevAscendProgram* prog,
                         AOTBinaryControlFlow::controlFlowEntry entry = nullptr)
    {
        if (entry != nullptr) {
            InitCode(reinterpret_cast<void*>(entry));
        } else {
            InitCodeSizeCached(GetAOTCacheKey(prog), prog->aotPoolLastId, code, codeSize);
        }
    }

    void CallControlFlow(struct DeviceExecuteContext* ctx, int64_t* symbolTable,
                         RuntimeCallEntryType runtimeCallList[T_RUNTIME_CALL_MAX], DevStartArgsBase* startArgsBase)
    {
        (reinterpret_cast<controlFlowEntry>(const_cast<unsigned char*>(code_)))(ctx, symbolTable, runtimeCallList,
                                                                                startArgsBase);
    }
};

struct DeviceExecuteProgram {
    DevAscendProgram* prog{nullptr};

    AOTBinaryControlFlow controlFlowBinary;

    DeviceExecuteProgram() {}
    DeviceExecuteProgram(DevAscendProgram* prog_, AOTBinaryControlFlow::controlFlowEntry entry = nullptr)
        : prog(prog_),
          controlFlowBinary(IsDeviceMode() ? prog_->GetDevControlFlowBinary() : prog_->GetHostControlFlowBinary(),
                            prog_, entry)
    {}

    DeviceExecuteProgram(const DeviceExecuteProgram&) = delete;
    DeviceExecuteProgram& operator=(const DeviceExecuteProgram&) = delete;

    DeviceExecuteProgram(DeviceExecuteProgram&& other) noexcept
        : prog(other.prog), controlFlowBinary(std::move(other.controlFlowBinary))
    {
        other.prog = nullptr;
    }

    DeviceExecuteProgram& operator=(DeviceExecuteProgram&& other) noexcept
    {
        if (this != &other) {
            prog = other.prog;
            controlFlowBinary = std::move(other.controlFlowBinary);
            other.prog = nullptr;
        }
        return *this;
    }

    const void* GetControlFlowEntry() { return controlFlowBinary.code_; }
};
} // namespace npu::tile_fwk::dynamic
