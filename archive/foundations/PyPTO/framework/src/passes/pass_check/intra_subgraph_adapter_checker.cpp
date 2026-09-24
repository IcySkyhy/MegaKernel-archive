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
 * \file intra_subgraph_adapter_checker.cpp
 * \brief
 */

#include "intra_subgraph_adapter_checker.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/platform.h"
#define MODULE_NAME "IntraSubgraphAdapterChecker"

namespace npu {
namespace tile_fwk {

Status IntraSubgraphAdapterChecker::PostCheckSubgraphTensor(const std::vector<std::vector<Operation*>>& subgraphs)
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        return SUCCESS;
    }
    for (const auto& subgraph : subgraphs) {
        if (subgraph.empty()) {
            continue;
        }
        int32_t aicMemoryCount = 0;
        int32_t aivMemoryCount = 0;
        std::unordered_set<std::shared_ptr<LogicalTensor>> tensorList;
        int32_t subgraphId = subgraph[0]->GetSubgraphID();
        for (const auto& op : subgraph) {
            for (const auto& iTensor : op->GetIOperands()) {
                if (tensorList.find(iTensor) != tensorList.end()) {
                    continue;
                }
                tensorList.insert(iTensor);
                if (iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
                    aicMemoryCount++;
                    continue;
                }
                if (iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                    aivMemoryCount++;
                }
            }
        }
        if (aicMemoryCount > 0 && aivMemoryCount > 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Subgraph %d has both ub(%d) and l0/l1(%d) memory type tensor.",
                              subgraphId, aivMemoryCount, aicMemoryCount);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status IntraSubgraphAdapterChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for IntraSubgraphAdapter.");
    std::vector<std::vector<Operation*>> subgraphs(function.GetTotalSubGraphCount());
    for (auto& op : function.Operations()) {
        int32_t curSubgraphID = op.GetSubgraphID();
        if (curSubgraphID == -1) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Operation (opmagic: %d) is not in any subgraph; Please review the error messages generated during the "
                "processing procedure.%s",
                op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (curSubgraphID < 0 || curSubgraphID >= static_cast<int32_t>(function.GetTotalSubGraphCount())) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Operation (opmagic: %d) has illegal SubgraphID; Please review the error messages generated during the "
                "processing procedure.%s",
                op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        subgraphs[curSubgraphID].push_back(&op);
    }
    if (PostCheckSubgraphTensor(subgraphs) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Subgraph post check failed.");
        return FAILED;
    }

    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
