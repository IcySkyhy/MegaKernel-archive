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
 * \file reschedule_utils.h
 * \brief
 */

#ifndef RESCHEDULE_UTILS_H_
#define RESCHEDULE_UTILS_H_

#include <unordered_map>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class RescheduleUtils {
public:
    struct TensorDependencyInfo {
        std::vector<int> producerOps;
        std::vector<int> producerColors;
        std::vector<int> consumerColors;
    };
    using TensorDependencyMap = std::unordered_map<const LogicalTensor*, TensorDependencyInfo>;
    struct TopoHashContribution {
        uint64_t power{1};
        uint64_t polynomial{0};
    };

    static bool isAllocOp(Operation* op);
    static void SortUnique(std::vector<int>& values);
    static TensorDependencyMap BuildTensorDependencyInfo(const OperationsViewer& opList,
                                                         bool collectConsumerColors = true);
    static void BuildInputTopoHashList(const OperationsViewer& opList, const TensorDependencyMap& tensorDeps,
                                       std::vector<uint64_t>& hashList);
    static void BuildInputTopoHashList(const OperationsViewer& opList, std::vector<uint64_t>& hashList);
    static uint64_t GetOpcodeHash(const std::string& op);
    static TopoHashContribution BuildTopoHashContribution(const std::vector<int>& producerOps,
                                                          const std::vector<uint64_t>& hashList);
    static uint64_t ApplyTopoHashContribution(uint64_t hash, const TopoHashContribution& contribution);

    // vector size == 2
    // 对于ingraph outgraph color
    // colornodes等信息可以之后用一个单例管理起来，避免每个pass遍历图去重新收集这些信息，导致编译时间变长
    static std::vector<std::vector<std::vector<int>>> GetInOutGraphs(const std::vector<Operation*>& opList);

    static PipeType GetOpPipeType(const Operation* op);

    // 计算op的offset无关hash（op name + params + input shape/dtype + output shape/dtype），用于Cycle数校准。
    static unsigned long ComputeOperationHash(const Operation* op);

    static void UpdateTensorConsProd(Function* funcPtr);
    static void ClearInputConsProd(Operation& op, Function* funcPtr,
                                   const std::unordered_set<LogicalTensorPtr>& incastSet);
    static void ClearOutputConsProd(Operation& op, Function* funcPtr,
                                    const std::unordered_set<LogicalTensorPtr>& outcastSet);
    static void EraseOpsBelongToFunc(std::set<Operation*, LogicalTensor::CompareOp>& ops, Function* funcPtr);
    static void PrintColorNode(Function& func);
};
} // namespace npu::tile_fwk
#endif
