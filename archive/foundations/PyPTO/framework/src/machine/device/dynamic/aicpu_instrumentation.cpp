/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/device/dynamic/aicpu_instrumentation.h"
#include <sys/mman.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

namespace npu::tile_fwk::dynamic {
HardBranchManager& HardBranchManager::GetInstance()
{
    static HardBranchManager manager;
    return manager;
}

int HardBranchManager::ProtectGetRange(uint64_t& base, uint64_t& size)
{
    uint64_t minAddr = ~0x0;
    uint64_t maxAddr = 0;
    int total = 0;
    for (size_t groupIndex = 0; groupIndex < Size(); groupIndex++) {
        HardBranchGroup& group = At(groupIndex);
        for (size_t branchIndex = 0; branchIndex < group.Size(); branchIndex++) {
            HardBranch& branch = group.At(branchIndex);
            if (minAddr > branch.source) {
                minAddr = branch.source;
            }
            if (maxAddr < branch.source + ARCH_NOP_SIZE) {
                maxAddr = branch.source + ARCH_NOP_SIZE;
            }
            total++;
        }
    }

    if (total != 0) {
        uint64_t begin = minAddr & ~(PAGE_SIZE - 1);
        uint64_t end = (maxAddr + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        base = begin;
        size = end - begin;
    }
    return total;
}

int HardBranchManager::ProtectRange(uint64_t base, uint64_t size, bool enableWrite)
{
    int prot = PROT_READ | PROT_EXEC;
    if (enableWrite) {
        prot |= PROT_WRITE;
    }
    if (mprotect((void*)base, size, prot) != 0) {
        return -1;
    }
    return 0;
}

int HardBranchManager::SwitchTo(bool toJump)
{
    uint64_t base = 0;
    uint64_t size = 0;
    if (ProtectGetRange(base, size) == 0) {
        return 0;
    }
    if (const int error = ProtectRange(base, size, true)) {
        return error;
    }
    for (size_t groupIndex = 0; groupIndex < Size(); groupIndex++) {
        HardBranchGroup& group = At(groupIndex);
        for (size_t branchIndex = 0; branchIndex < group.Size(); branchIndex++) {
            HardBranch& branch = group.At(branchIndex);
            if (toJump) {
                ArchCreateJump(reinterpret_cast<uint8_t*>(branch.source), branch.source, branch.target);
            } else {
                ArchCreateNop(reinterpret_cast<uint8_t*>(branch.source));
            }
        }
    }
    if (const int error = ProtectRange(base, size, false)) {
        return error;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(base), reinterpret_cast<char*>(base) + size);
    return 0;
}

void HardBranchManager::Placeholder()
{
    HardBranchTrue(verboseInfo);
    HardBranchTrue(verboseDebug);
}
} // namespace npu::tile_fwk::dynamic
