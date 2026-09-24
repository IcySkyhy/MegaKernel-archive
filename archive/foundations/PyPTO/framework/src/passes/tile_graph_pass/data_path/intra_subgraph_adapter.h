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
 * \file intra_subgraph_adapter.h
 * \brief
 */

#ifndef PASS_INTRA_SUBGRAPH_ADAPTER_H_
#define PASS_INTRA_SUBGRAPH_ADAPTER_H_

#include <unordered_set>
#include <vector>
#include "interface/operation/opcode.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
class IntraSubgraphAdapter : public Pass {
public:
    IntraSubgraphAdapter() : Pass("IntraSubgraphAdapter") {}
    ~IntraSubgraphAdapter() override = default;
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;

private:
    Status CheckColorCount(Function& function, LogicalTensorPtr tensor, size_t commonColorCount);
    Status CheckBoundaryTensor(LogicalTensorPtr tensor);
    Status SplitBoundaryTensor(Function& function, LogicalTensorPtr tensor, int mainSubgraphID,
                               LogicalTensors& newBoundaryTensors);
    LogicalTensors CollectBoundaryTensors(Function& function);
    Status ProcessBoundaryTensors(Function& function, LogicalTensors tensors);
    Status ProcessBoundaryTensor(Function& function, LogicalTensorPtr tensor);
    Status AdapteTensorProducers(Function& function, LogicalTensorPtr tensor);
    Status AdapteTensorConsumers(Function& function, LogicalTensorPtr tensor);
    LogicalTensorPtr InsertOpBetween(Function& function, Opcode opcode, Operation& op, LogicalTensorPtr tensor);
    LogicalTensorPtr InsertOpBetween(Function& function, Opcode opcode, LogicalTensorPtr tensor,
                                     const std::vector<Operation*>& ops, int newOpSubgraphID = -1);
    void CollectProducerColors(LogicalTensorPtr tensor, std::set<int>& colors);
    void CollectConsumerColors(LogicalTensorPtr tensor, std::set<int>& colors);
    std::set<int> SetIntersection(std::set<int>& a, std::set<int>& b);
    std::vector<Operation*> newOps;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_INTRA_SUBGRAPH_ADAPTER_H_
