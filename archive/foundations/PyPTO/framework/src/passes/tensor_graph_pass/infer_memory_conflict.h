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
 * \file infer_memory_conflict.h
 * \brief
 */

#ifndef PASS_INFER_MEMORY_CONFLICT_H_
#define PASS_INFER_MEMORY_CONFLICT_H_

#include <vector>
#include <queue>
#include <unordered_map>

#include "passes/pass_interface/pass.h"
#include "passes/tensor_graph_pass/derivation_tile_shape.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class InferMemoryConflict : public Pass {
public:
    InferMemoryConflict() : Pass("InferMemoryConflict") {}
    ~InferMemoryConflict() override = default;

private:
    Status RunOnFunction(Function& function) override;
    Status Init(Function& function);
    Status ForwardPropagation(Function& function);
    Status UpdateForwardTensor(Function& function, const LogicalTensorPtr& curTensor, Operation* consumer,
                               std::queue<LogicalTensorPtr>& curTensors);
    Status BackwardPropagation(Function& function);
    Status UpdateBackwardTensor(Function& function, const LogicalTensorPtr& curTensor, Operation* producer,
                                std::queue<LogicalTensorPtr>& curTensors);
    Status InsertPrecededCopys(Function& function);
    Status InsertPostCopys(Function& function);
    Status InsertCopys(Function& function);
    Status UpdateViewTypeTileShape(Function& function);
    Status ObtainReshapeTile(Operation& op, Shape& inTileShape, Shape& outTileShape);
    Status InferTileShape(Operation& op, const LogicalTensorPtr& tensor, TileShape parentTile, Shape& reshapeTile);
    Status SetDefaultShape(const LogicalTensorPtr& tensor, std::vector<int64_t>& defaultTile);

    TileShape ObtainTileShape(const std::unordered_set<Operation*>& origOp);

    bool CheckTransmit(Operation& curOp);
    bool CheckConflict(const LogicalTensorPtr& inTensor, const LogicalTensorPtr& outTensor);
    bool CheckRawShapeConflict(const LogicalTensorPtr& inTensor, const LogicalTensorPtr& outTensor);
    bool IsValidTileShape(const Operation& op) const;
    bool ShouldSkipOutcastInput(const LogicalTensorPtr& inputTensor, Function& function);
    Status HandleReshapeBackward(Function& function, const LogicalTensorPtr& curTensor,
                                 const LogicalTensorPtr& inputTensor, Operation* producer,
                                 const LogicalTensorPtr& reshapeOutput, bool& needSkip);
    Status HandleConflictBackward(Function& function, const LogicalTensorPtr& curTensor,
                                  const LogicalTensorPtr& inputTensor, Operation* producer,
                                  const LogicalTensorPtr& reshapeOutput);
    bool TryInsertPreMergeCopy(Function& function, const LogicalTensorPtr& tensor,
                               const LogicalTensorPtr& conflictTensor, Operation* finalOp);
    Status MatchReshapePattern(Function& function, const LogicalTensorPtr& reshapeInput,
                               const LogicalTensorPtr& reshapeOut, bool& matched);
    bool CheckReshapeContext(const LogicalTensorPtr& reshapeInput, const LogicalTensorPtr& reshapeOut);

    std::set<Operation*> preregcopys;
    std::set<Operation*> postregcopys;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> memoryInfo;
    std::unordered_map<DataType, int> viewTypeTable = {{DT_INT8, 1},    {DT_BF16, 2},    {DT_FP16, 2},   {DT_FP32, 4},
                                                       {DT_FP8E4M3, 1}, {DT_FP8E5M2, 1}, {DT_FP8E8M0, 1}};
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_INFER_MEMORY_CONFLICT_H_
