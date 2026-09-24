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
 * \file copy_out_resolve.cpp
 * \brief
 */

#include "copy_out_resolve.h"

#include "interface/program/program.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

static int copyOutScheduleDefaultDistance = 5;

std::vector<Operation*> CopyOutResolve::LookupOutcastLastCopyOut(Function* leafFunc) const
{
    // For each outcast, find the last op that writes it.
    std::vector<Operation*> outcastLastCopyOut(leafFunc->GetOutcast().size());
    for (auto& op : leafFunc->Operations(false)) {
        Opcode opcode = op.GetOpcode();
        if (!OpcodeManager::Inst().IsCopyOut(opcode)) {
            continue;
        }
        std::shared_ptr<LogicalTensor> outcast = op.GetOOperands()[0];
        int outcastIndex = leafFunc->GetOutcastIndex(outcast);
        if (outcastIndex != -1) {
            outcastLastCopyOut[outcastIndex] = &op;
        }
    }
    return outcastLastCopyOut;
}

void CopyOutResolve::CheckOutcastProducer(Function* leafFunc) const
{
    std::shared_ptr<LeafFuncAttribute> leafAttr = leafFunc->GetLeafFuncAttribute();
    for (size_t outcastIndex = 0; outcastIndex < leafFunc->GetOutcast().size(); outcastIndex++) {
        if (leafAttr->outcastCopyOutResolveCounterList[outcastIndex] != -1) {
            // outcast produced by any copy out
        } else {
            std::shared_ptr<LogicalTensor> outcast = leafFunc->GetOutcast()[outcastIndex];
            if (leafFunc->GetOutcastIndex(outcast) < (int)outcastIndex) {
                // duplicated outcast in the leaf function
            } else {
                for (const auto& producer : outcast->GetProducers()) {
                    Opcode producerOpCode = producer->GetOpcode();
                    if (!OpcodeManager::Inst().IsCopyOut(producerOpCode)) {
                        // not a copyout outcast, which should be ignored
                    } else {
                        ASSERT(FunctionErr::FUNCTION_GRAPH_STRUCTURE, false) << "Outcast not filled by any operation!";
                    }
                }
            }
        }
    }
}

template <typename T>
static std::vector<T*> DecreaseDictCount(std::unordered_map<T*, int>& dict)
{
    std::vector<T*> zeroCountList;
    for (auto& [key, distance] : dict) {
        distance -= 1;
        if (distance == 0) {
            zeroCountList.push_back(key);
        }
    }
    for (auto& key : zeroCountList) {
        dict.erase(key);
    }
    return zeroCountList;
}

void CopyOutResolve::InsertCopyOutResolveForLeaf(int copyOutResolveCoalescing, Function* leafFunc) const
{
    std::vector<Operation*> outcastLastCopyOut = LookupOutcastLastCopyOut(leafFunc);
    std::unordered_map<Operation*, int> resolveCopyOutDict;
    for (size_t k = 0; k < outcastLastCopyOut.size(); k++) {
        resolveCopyOutDict[outcastLastCopyOut[k]] = (int)k;
    }

    // For each outcast, mark the corresponding counter
    std::shared_ptr<LeafFuncAttribute> leafAttr = leafFunc->GetLeafFuncAttribute();
    leafAttr->outcastCopyOutResolveCounterList.resize(leafFunc->GetOutcast().size(), -1);
    uint32_t copyOutResolveCounter = 0;

    auto subgraphID = leafFunc->Operations(false).begin()->GetSubgraphID();

    std::vector<Operation*> opList;
    std::vector<OperationPtr> leafFuncOpList = leafFunc->GetProgramOp();
    std::unordered_map<Operation*, int> aicpuCallCopyOutFinishDistanceDict;
    std::unordered_map<Operation*, int> aicpuCallCopyOutCoalescingDistanceDict;
    for (auto& leafFuncOp : leafFuncOpList) {
        Operation* op = leafFuncOp.get();
        if (!resolveCopyOutDict.count(op)) {
            opList.push_back(op);
        } else {
            std::shared_ptr<LogicalTensor> outcast = op->GetOOperands()[0];
            int outcastIndex = resolveCopyOutDict[op];

            leafAttr->outcastCopyOutResolveCounterList[outcastIndex] = copyOutResolveCounter;
            std::ostringstream oss;
            oss << "outcast=" << outcastIndex << " copy_out_resolve_counter=" << copyOutResolveCounter;
            op->GetCommentList().push_back("copyout: " + oss.str());
            opList.push_back(op);

            Opcode opcode = leafFunc->IsCube() ? Opcode::OP_AICPU_CALL_AIC : Opcode::OP_AICPU_CALL_AIV;
            auto& aicpuCall = PassOperationUtils::AddOperation(
                *leafFunc, opcode, {outcast}, {}, [&copyOutResolveCounter](Operation& newOp) {
                    newOp.SetAttribute(OpAttributeKey::aicpuCall,
                                       (int64_t)(uint32_t)((AICPU_CALL_NUM_COPYOUT_RESOLVE << AICPU_CALL_ARG_BIT) +
                                                           copyOutResolveCounter));
                });
            aicpuCall.UpdateSubgraphID(subgraphID);
            aicpuCall.GetCommentList().push_back("aicpuCall: " + oss.str());
            aicpuCallCopyOutFinishDistanceDict[&aicpuCall] = copyOutScheduleDefaultDistance;

            copyOutResolveCounter++;
        }

        // Find to aicpu call that should be emitted
        std::vector<Operation*> emittedAicpuCallList = DecreaseDictCount(aicpuCallCopyOutFinishDistanceDict);
        DecreaseDictCount(aicpuCallCopyOutCoalescingDistanceDict);

        for (auto& emittedAicpuCall : emittedAicpuCallList) {
            opList.push_back(emittedAicpuCall);
            if (aicpuCallCopyOutCoalescingDistanceDict.size() != 0) {
                // still under previous aicpu call's coalescing region, so don't emit this.
                emittedAicpuCall->SetAsDeleted();
            } else {
                // already out of previous aicpu call's coalescing region, add this call's coalescing region.
                aicpuCallCopyOutCoalescingDistanceDict[emittedAicpuCall] = copyOutResolveCoalescing;
            }
        }
    }
    // Append the rest to tail
    for (auto& [aicpuCall, dis] : aicpuCallCopyOutFinishDistanceDict) {
        (void)dis;
        opList.push_back(aicpuCall);
        // tailing call should be ignored. However, because the op is already added by AddOperation, we have to firstly
        // add it, and then remove it.
        aicpuCall->SetAsDeleted();
    }

    // insert all aicpu call
    leafFunc->ScheduleBy(opList);

    // remove coalescing region covered aicpu call
    leafFunc->EraseOperations(false, false);

    leafAttr->copyOutResolveSize = copyOutResolveCounter;
    CheckOutcastProducer(leafFunc);
}

void CopyOutResolve::CopyOutResolveCall(Function& function) const
{
    if (function.paramConfigs_.copyOutResolveCoalescing == 0) {
        return;
    }
    for (auto& leaf : function.rootFunc_->programs_) {
        Function* leafFunc = leaf.second;
        InsertCopyOutResolveForLeaf(function.paramConfigs_.copyOutResolveCoalescing, leafFunc);
    }
}
} // namespace npu::tile_fwk
