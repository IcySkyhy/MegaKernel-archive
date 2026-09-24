/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicpu_instrumentation.h
 * \brief
 */

#pragma once

#include <stdint.h>

namespace npu::tile_fwk::dynamic {
#define ARCH_AMD64_NOP "nopl 0(%%eax, %%eax)"
#define ARCH_AMD64_NOP_SIZE 5
#define ARCH_AMD64_NOP_OPCODE 0x0f
#define ARCH_AMD64_NOP_OPDATA 0x0000441f
#define ARCH_AMD64_JUMP_OPCODE 0xe9

static inline void ArchAmd64CreateJump(uint8_t* addr, uint64_t source, uint64_t target)
{
    *addr = ARCH_AMD64_JUMP_OPCODE;
    *(uint32_t*)(addr + 1) = (uint32_t)(target - (source + ARCH_AMD64_NOP_SIZE));
}
static inline void ArchAmd64CreateNop(uint8_t* addr)
{
    *addr = ARCH_AMD64_NOP_OPCODE;
    *(uint32_t*)(addr + 1) = (uint32_t)ARCH_AMD64_NOP_OPDATA;
}

#define ARCH_ARM64_NOP "nop"
#define ARCH_ARM64_NOP_SIZE 4
#define ARCH_ARM64_NOP_OPCODE 0xd503201f
#define ARCH_ARM64_JUMP_OPCODE (0x5 << 26)
#define ARCH_ARM64_JUMP_MASK ((1 << 26) - 1)
static inline void ArchArm64CreateJump(uint8_t* addr, uint64_t source, uint64_t target)
{
    *(uint32_t*)addr = ARCH_ARM64_JUMP_OPCODE | (((target - source) / ARCH_ARM64_NOP_SIZE) & ARCH_ARM64_JUMP_MASK);
}
static inline void ArchArm64CreateNop(uint8_t* addr) { *(uint32_t*)addr = ARCH_ARM64_NOP_OPCODE; }

#if defined(__x86_64__)
#define ARCH_NOP ARCH_AMD64_NOP
#define ARCH_NOP_SIZE ARCH_AMD64_NOP_SIZE
#define ArchCreateJump ArchAmd64CreateJump
#define ArchCreateNop ArchAmd64CreateNop

#elif defined(__aarch64__)
#define ARCH_NOP ARCH_ARM64_NOP
#define ARCH_NOP_SIZE ARCH_ARM64_NOP_SIZE
#define ArchCreateJump ArchArm64CreateJump
#define ArchCreateNop ArchArm64CreateNop

#else
#error "Unknow architecture"
#endif

__attribute__((always_inline)) inline bool HardBranchTrueAndRecordLabel()
{
    bool ret;
    asm goto("\n5:"
             "\n\t" ARCH_NOP "\n6:"
             "\n\t"
             ".previous"
             "\n\t"
             ".8byte 5b, %l0"
             "\n\t"
             ".popsection" :: ::dst);
    ret = true;
    for (;;) {
        break;
    dst:
        ret = false;
        break;
    }
    return ret;
}

#define HardBranchTrue(name)                                                                                       \
    ({                                                                                                             \
        asm volatile("\n\t"                                                                                        \
                     ".pushsection _hard_branch_" #name ", \"awG\", @progbits, \"_hard_branch_" #name "\", comdat" \
                     "\n\t"                                                                                        \
                     ".previous");                                                                                 \
        npu::tile_fwk::dynamic::HardBranchTrueAndRecordLabel();                                                    \
    })

struct HardBranch {
    uint64_t source;
    uint64_t target;
};

class HardBranchGroup {
public:
    HardBranchGroup() = default;
    HardBranchGroup(HardBranch* begin, HardBranch* end)
    {
        base_ = begin;
        size_ = end - begin;
    }

    HardBranch& At(uint64_t index) { return base_[index]; }
    uint64_t Size() { return size_; }

private:
    HardBranch* base_{nullptr};
    uint64_t size_{0};
};

#define HardBranchGroupDefine(name)                                                                              \
    extern "C" npu::tile_fwk::dynamic::HardBranch __start__hard_branch_##name[];                                 \
    extern "C" npu::tile_fwk::dynamic::HardBranch __stop__hard_branch_##name[];                                  \
    static inline npu::tile_fwk::dynamic::HardBranchGroup HardBranchGroup##name()                                \
    {                                                                                                            \
        return npu::tile_fwk::dynamic::HardBranchGroup(__start__hard_branch_##name, __stop__hard_branch_##name); \
    }

#define HardBranchGroupCreate(name) HardBranchGroup##name()

class HardBranchManager {
public:
    void AddGroup(const HardBranchGroup& group)
    {
        groupList_[groupSize_] = group;
        groupSize_++;
    }

    HardBranchGroup& At(const uint64_t index) { return groupList_[index]; }
    uint64_t Size() const { return groupSize_; }
    void Clear() { groupSize_ = 0; }

    int ProtectGetRange(uint64_t& base, uint64_t& size);
    int ProtectRange(uint64_t base, uint64_t size, bool enableWrite);

    int SwitchToJump() { return SwitchTo(true); }
    int SwitchToNop() { return SwitchTo(false); }

    static HardBranchManager& GetInstance();

private:
    static void Placeholder();
    int SwitchTo(bool toJump);

    HardBranchGroup groupList_[0x10];
    uint64_t groupSize_{0};
};
} // namespace npu::tile_fwk::dynamic
