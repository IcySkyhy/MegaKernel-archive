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
 * \file remove_redundant_assemble.h
 * \brief
 */

#ifndef PASS_REMOVE_REDUNDANT_ASSEMBLE_H
#define PASS_REMOVE_REDUNDANT_ASSEMBLE_H
#include "pre_graph_common.h"

namespace npu::tile_fwk {
class RemoveRedundantAssemble {
public:
    RemoveRedundantAssemble() {}
    ~RemoveRedundantAssemble() = default;

    bool IsCandidateAssembleOp(Function& function, Operation& op) const;
    Status DeleteRedundantAssemble(Function& function) const;
    void HandleForAssembleToOutcast(Function& function, Operation& assembleOp,
                                    std::set<Operation*, LogicalTensor::CompareOp>& producersBackup) const;
    void HandleForAssembleFromInOut(Function& function, Operation& AssembleOp,
                                    std::set<Operation*, LogicalTensor::CompareOp>& producersBackup) const;
    void HandleForReshapeToOutcast(Function& function) const;
    void HanldeForMultiAssemble(Function& function, std::unordered_set<Operation*>& concurrentAssembles) const;
    Status HanldeForSingleAssemble(Function& function, LogicalTensorPtr input, LogicalTensorPtr output,
                                   Operation& op) const;
    Status ProcessViewToCopyIn(Function& function) const;
    Status ProcessView(Function& function) const;

private:
    void UpdateReshapeShape(Operation& reshapeOp, LogicalTensorPtr tensorPtr, const Shape& newRawShape,
                            const std::vector<SymbolicScalar>& newDynRawShape) const;
    Status SplitMultiConsumerReshape(Function& function,
                                     std::vector<std::pair<Operation*, Operation*>>& multiReshapeVector) const;
    Status ProcessReshape(Function& function, Operation*& operation,
                          std::vector<std::pair<Operation*, Operation*>>& multiReshapeVector) const;
    Status RemoveViewMultiReshape(const std::vector<std::pair<Operation*, Operation*>>& multiReshapeVector) const;
    Status RemoveViewSingleReshape(Function& function) const;
    Status HandleCopyOutToAssemble(Operation& assembleOp, LogicalTensorPtr output,
                                   const std::set<Operation*, LogicalTensor::CompareOp>& producers) const;
    bool CalculateReshapeToAssembleInfo(Operation& assembleOp, Operation& reshapeOp,
                                        const std::vector<SymbolicScalar>& dynOffset, std::vector<int64_t>& newRawShape,
                                        std::vector<SymbolicScalar>& newDynRawShape,
                                        std::vector<SymbolicScalar>& newDynOffset) const;
    Status UpdateCopyOutBeforeReshape(LogicalTensorPtr reshapeInput, const std::vector<int64_t>& newRawShape,
                                      const std::vector<SymbolicScalar>& newDynOffset) const;
    Status HandleReshapeToAssemble(Operation& assembleOp,
                                   const std::set<Operation*, LogicalTensor::CompareOp>& producers) const;
};
} // namespace npu::tile_fwk
#endif // PASS_REMOVE_REDUNDANT_ASSEMBLE_H
