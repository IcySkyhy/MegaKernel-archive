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
 * \file workspace_desc.h
 * \brief Shared workspace descriptor for interface and machine layers.
 */

#pragma once

#include "tilefwk/symbolic_scalar.h"

#include <cstdint>
#include <string>
#include <vector>

namespace npu::tile_fwk {

class Function;

struct WorkspaceDesc {
    struct WorkspaceConfig {
        uint64_t parallelism{0};
    } config;

    struct WorkspacePlatform {
        uint64_t aicoreCount{0};
    } platform;

    struct WorkspacePerRootFunctionDesc {
        Function* func{nullptr};
        std::string devFuncName;
        uint64_t unroll{0};

        uint64_t rootInnerSpilledRawMem{0};
        uint64_t leafPerCoreSpilledMem{0};
        uint64_t rootTotalExclusiveOutcastRawMem{0};

        uint64_t rootMaxExclusiveOutcastMem{0};
        int64_t rootMaxExclusiveOutcastIdx{-1};
    };

    std::vector<WorkspacePerRootFunctionDesc> rootFuncDescList;

    uint64_t maxRootInnerSpilledMem{0};
    uint64_t maxLeafPerCoreSpilledMem{0};
    uint64_t maxRootTotalExclusiveOutcastMem{0};

    uint64_t maxStaticOutcastMem{0};
    SymbolicScalar maxDynamicAssembleOutcastMem;

    uint64_t totalExclusiveOutcastSlot{0};
    uint64_t totalAssembleOutcastSlot{0};
    uint64_t devTaskBoundaryOutcastNum{0};
    uint64_t devTaskInnerTemporalOutcastNum{0};

    struct WorkspaceCellMatch {
        uint64_t dynamicCellMatchSlotNum{0};
        SymbolicScalar maxDynamicCellMatchTableMem;
    } cellMatch;

    // Max unroll across encoded dev roots (diagnostics only; encode budget uses root count).
    uint32_t maxUnrollTimes{1};
};

} // namespace npu::tile_fwk
