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
 * \file common_operation_eliminate_utils.h
 * \brief utils of common operation elimination
 */

#ifndef PASS_COMMON_OPERATION_ELIMINATE_UTILS_H_
#define PASS_COMMON_OPERATION_ELIMINATE_UTILS_H_

#include <cstdint>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class CommonOperationEliminateUtils {
public:
    CommonOperationEliminateUtils() = default;
    ~CommonOperationEliminateUtils() = default;

    static Status EliminateCommonOperation(Function& function);

    Status Process(Function& function);

private:
    static const std::unordered_set<Opcode>& GetSkipEliminateOpcodes();
    void SortedProducer(std::vector<Operation*>& sortedProducers) const;
    void CollectProducerInfo(const std::vector<Operation*>& sortedProducers, const LogicalTensorPtr& curTensor,
                             std::vector<std::string>& opStrList, std::stringstream& ss) const;
    unsigned long ComputeHash(const std::vector<Operation*>& producers, LogicalTensorPtr curTensor) const;
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> GetTensorProducers(
        Function& function, std::vector<LogicalTensorPtr>& sequence);
    void UpdateConnection(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor);
    uint32_t GetTensorCoreFlag(const LogicalTensorPtr& tensor) const;
    void CollectSubgraphIds(const std::set<Operation*, LogicalTensor::CompareOp>& ops,
                            std::unordered_set<int>& subgraphIds) const;
    void UpdateInternalTensorCoreFlag(const LogicalTensorPtr& tensor,
                                      std::unordered_map<int, uint32_t>& subgraphCoreFlags) const;
    std::unordered_set<int> GetMixSubgraphIds(Function& function) const;
    MemoryType GetAssembleInputMemoryType(Operation* consumer, const LogicalTensorPtr& input) const;
    MemoryType GetAssembleOutputMemoryType(Operation* consumer) const;
    bool HasSameMemoryAssembleUse(const LogicalTensorPtr& tensor, std::unordered_set<int>& visitedTensorMagics) const;
    bool ShouldSkipSameMemoryAssembleMerge(const LogicalTensorPtr& oldTensor) const;
    bool WouldExposeMixInternalTensorAfterMerge(const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
                                                const std::unordered_set<int>& mixSubgraphIds) const;
    std::pair<LogicalTensorPtr, std::vector<Operation*>> TensorHashExist(
        const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
        const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap);
    bool TensorProducersMerge(Function& function, const LogicalTensorPtr orderedTensor,
                              std::unordered_set<Operation*>& cacheProducers,
                              const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap);
    void UpdateView(ViewOpAttribute* viewOpAttribute, const std::shared_ptr<LogicalTensor> oldTensor,
                    const std::shared_ptr<LogicalTensor> newTensor) const;
    void UpdateCopy(CopyOpAttribute* copyOpAttribute, const std::shared_ptr<LogicalTensor> oldTensor,
                    const std::shared_ptr<LogicalTensor> newTensor) const;

    std::unordered_map<uint64_t, std::pair<LogicalTensorPtr, std::vector<Operation*>>> hashCache_;
    std::unordered_set<int> mixSubgraphIds_;
};
} // namespace npu::tile_fwk
#endif // PASS_COMMON_OPERATION_ELIMINATE_UTILS_H_
