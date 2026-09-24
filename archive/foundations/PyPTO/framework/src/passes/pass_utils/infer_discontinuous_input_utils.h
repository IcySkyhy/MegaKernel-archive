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
 * \file infer_discontinuous_input_utils.h
 * \brief utils of infer discontinuous input
 */

#ifndef PASS_INFER_DISCONTINUOUS_INPUT_UTILS_H_
#define PASS_INFER_DISCONTINUOUS_INPUT_UTILS_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class InferDiscontinuousInputUtils {
public:
    InferDiscontinuousInputUtils() = default;
    ~InferDiscontinuousInputUtils() = default;

    Status Process(Function& function, bool checkViewConflict = true);

    static std::vector<std::pair<LogicalTensorPtr, Operation*>> GetInplacedTileTensors(LogicalTensorPtr targetTensor);
    static std::vector<size_t> GetInputTileConflict(
        Function& function, const std::vector<std::pair<LogicalTensorPtr, Operation*>>& inplaceTensors,
        bool checkViewConflict);
    static void DDRTensorAssignUB(Function& function,
                                  std::unordered_map<LogicalTensorPtr, std::unordered_set<Operation*>>& insertedNodes);

private:
    void Init(Function& function);
    Status InferFromIncast(Function& function, bool checkViewConflict);
    std::vector<std::pair<LogicalTensorPtr, Operation*>> FilterCopyScenes(
        Function& function, const std::vector<std::pair<LogicalTensorPtr, Operation*>>& inplaceTensors,
        bool checkViewConflict);
    void InsertViewOp(Function& function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    void InsertAssembleOp(Function& function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    void InsertCopyOp(Function& function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Status InsertTensorCopy(Function& function);

    std::unordered_map<LogicalTensorPtr, std::vector<std::pair<LogicalTensorPtr, Operation*>>> insertCopys_;
    std::unordered_map<LogicalTensorPtr, size_t> tensorProducers_;
    std::vector<Operation*> newOps_;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_INFER_DISCONTINUOUS_INPUT_UTILS_H_
