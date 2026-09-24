/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <memory>
#include <vector>

#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
struct L1DataMoveGraphNodes {
    std::shared_ptr<LogicalTensor> inputCast1;
    std::shared_ptr<LogicalTensor> inputCast2;
    std::shared_ptr<LogicalTensor> inputCast1View;
    std::shared_ptr<LogicalTensor> inputCast2View;
    std::shared_ptr<LogicalTensor> redundantViewOut1;
    std::shared_ptr<LogicalTensor> redundantViewOut2;
    std::shared_ptr<LogicalTensor> opViewL1Out1;
    std::shared_ptr<LogicalTensor> opViewL1Out2;
    std::shared_ptr<LogicalTensor> viewOut1;
    std::shared_ptr<LogicalTensor> viewOut2;
    std::shared_ptr<LogicalTensor> viewOut3;
    std::shared_ptr<LogicalTensor> viewOut4;
    std::shared_ptr<LogicalTensor> l0aOut1;
    std::shared_ptr<LogicalTensor> l0aOut2;
    std::shared_ptr<LogicalTensor> l0bOut1;
    std::shared_ptr<LogicalTensor> l0bOut2;
    std::shared_ptr<LogicalTensor> aMulBOut1;
    std::shared_ptr<LogicalTensor> aMulBOut2;
};

inline L1DataMoveGraphNodes CreateL1DataMoveGraphNodes(const std::shared_ptr<Function>& currFunctionPtr)
{
    L1DataMoveGraphNodes graph;
    graph.inputCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 64});
    graph.inputCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64, 16});
    graph.inputCast1View = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 64});
    graph.inputCast2View = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64, 16});
    graph.redundantViewOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 64});
    graph.redundantViewOut2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64, 16});
    graph.opViewL1Out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 64});
    graph.opViewL1Out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64, 16});
    graph.viewOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 32});
    graph.viewOut2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 32});
    graph.viewOut3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 16});
    graph.viewOut4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 16});
    graph.l0aOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 32});
    graph.l0aOut2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 32});
    graph.l0bOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 16});
    graph.l0bOut2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 16});
    graph.aMulBOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 16});
    graph.aMulBOut2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32, 16});
    return graph;
}

inline std::shared_ptr<ViewOpAttribute> CreateL1ViewAttribute()
{
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    return viewAttribute;
}

inline void BuildL1DataMoveTail(const std::shared_ptr<Function>& currFunctionPtr, const L1DataMoveGraphNodes& graph)
{
    auto& viewOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {graph.opViewL1Out1},
                                                   {graph.viewOut1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& viewOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {graph.opViewL1Out1},
                                                   {graph.viewOut2});
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 32}));
    auto& viewOp3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {graph.opViewL1Out2},
                                                   {graph.viewOut3});
    viewOp3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& viewOp4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {graph.opViewL1Out2},
                                                   {graph.viewOut4});
    viewOp4.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 0}));

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {graph.viewOut1}, {graph.l0aOut1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {graph.viewOut2}, {graph.l0aOut2});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {graph.viewOut3}, {graph.l0bOut1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {graph.viewOut4}, {graph.l0bOut2});

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {graph.l0aOut1, graph.l0bOut1},
                                   {graph.aMulBOut1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {graph.l0aOut2, graph.l0bOut2},
                                   {graph.aMulBOut2});

    currFunctionPtr->inCasts_.push_back(graph.inputCast1);
    currFunctionPtr->inCasts_.push_back(graph.inputCast2);
    currFunctionPtr->outCasts_.push_back(graph.aMulBOut1);
    currFunctionPtr->outCasts_.push_back(graph.aMulBOut2);
}
} // namespace tile_fwk
} // namespace npu
