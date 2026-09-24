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
 * \file control_flow_codegen.h
 * \brief Host-side control-flow C++ source emission (BuildControlFlow / SetExprBatch).
 */

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/cache/function_cache.h"
#include "machine/host/backend.h"
#include "machine/host/expr_generator.h"

namespace npu::tile_fwk {

struct ControlFlowEmitCtx {
    FunctionCache& cache;
    Linker& linker;
    const std::string& sectionName;
    std::unordered_map<int, int>& slotIdxMapping;
    DyndevFunctionAttribute::FunctionGroup& group;
    std::unordered_map<Function*, Function*>& rootTileDict;
    std::ostringstream& controlFlowOss;
    std::ostringstream& expressionOss;
    std::ostringstream& exprHeaderOss;
    const std::string& expName;
    std::vector<std::string>& exprSrcFiles;
    ValDependTensorMeta& valDependTensorMeta;
    const GetInputCse* getInputCse;
};

void BuildControlFlow(ControlFlowEmitCtx& ctx, Function* func, int indent);

void GenerateExpression(SymbolicExpressionTable* exprTable, int devRootKey, const std::string& expName,
                        std::vector<std::string>& exprSrcFiles, std::ostringstream& controlFlowOss,
                        std::ostringstream& exprHeaderOss, int indent, const GetInputCse* getInputCse);

template <typename HandleSlot>
inline void ForEachNeedAllocAssembleOutcastSlot(Function* tile, const IncastOutcastSlot& ioslot,
                                                const std::unordered_set<int>& assembleSlotIndexSet,
                                                HandleSlot handleSlot)
{
    const auto& tileOutcasts = tile->GetOutcast();
    size_t outcastCount = std::min(ioslot.outcastSlot.size(), tileOutcasts.size());
    for (size_t outcastIdx = 0; outcastIdx < outcastCount; ++outcastIdx) {
        if (!tile->IsOutcastNeedAlloc(tileOutcasts[outcastIdx])) {
            continue;
        }
        for (int slot : ioslot.outcastSlot[outcastIdx]) {
            if (assembleSlotIndexSet.count(slot) == 0) {
                continue;
            }
            handleSlot(slot);
        }
    }
}

} // namespace npu::tile_fwk
