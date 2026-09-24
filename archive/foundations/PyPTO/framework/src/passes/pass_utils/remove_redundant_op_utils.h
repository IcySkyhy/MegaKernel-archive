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
 * \file remove_redundant_op_utils.h
 * \brief utils for redundant view/assemble and slice/contract elimination
 */

#ifndef PASS_REMOVE_REDUNDANT_OP_UTILS_H_
#define PASS_REMOVE_REDUNDANT_OP_UTILS_H_

#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class RemoveRedundantOpUtils {
public:
    RemoveRedundantOpUtils() = default;
    ~RemoveRedundantOpUtils() = default;

    static Status Process(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    static Status ProcessViewAssembleLike(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    static Status ProcessContractSlice(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    static Status ProcessViewCopyout(Function& function, bool& operationUpdated);
    static Status ProcessCopyinAssemble(Function& function, bool& operationUpdated);

private:
    Status ProcessImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessViewAssembleLikeImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessContractSliceImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessMultiContractSingleSlice(Function& function, bool& operationUpdated);
    Status ProcessSingleContractMultiSlice(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);

    void ProcessPerfectMatch(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor, bool endTensorIsOutcast,
                             bool& operationUpdated);
    void RemoveViewAssembleForOutcast(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                                      bool& operationUpdated);
    void CalculateViewOffset(Operation& op, LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                             std::vector<int64_t>& newOffset, std::vector<SymbolicScalar>& newDynOffset);
    void GenerateNewViewLike(Function& function, Operation& op, LogicalTensorPtr& startTensor,
                             LogicalTensorPtr& endTensor, Opcode newOpcode, std::vector<Operation*>& newOps,
                             bool& operationUpdated);
    void GenerateContractSliceView(Function& function, Operation& sliceOp, const LogicalTensorPtr& inputTensor,
                                   const std::vector<int64_t>& fromOffset,
                                   const std::vector<SymbolicScalar>& fromDynOffset, std::vector<Operation*>& newOps,
                                   bool& operationUpdated);

    bool IsNotSameViewInput(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const;
    bool IsDataReplace(LogicalTensorPtr& endTensor) const;
    bool IsValidViewAssemble(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const;

    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_REMOVE_REDUNDANT_OP_UTILS_H_
