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
 * \file remove_unaligned_reshape_op.h
 * \brief
 */

#ifndef REMOVE_UNALIGNED_RESHAPE_OP_H_
#define REMOVE_UNALIGNED_RESHAPE_OP_H_

#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

using OverlaprawMagic = int;
struct CopyOutOpMemUnalign {
    MemoryType from;
    std::vector<int64_t> toOffset;
    std::shared_ptr<LogicalTensor> input;
    std::shared_ptr<LogicalTensor> output;
};

struct CopyInOpMemUnalign {
    MemoryType to;
    std::vector<int64_t> fromOffset;
    std::shared_ptr<LogicalTensor> input;
    std::shared_ptr<LogicalTensor> output;
};
/*
 移除尾轴非对齐的reshape，插入copy_out, copy_in
*/
class RemoveUnalignedReshape : public Pass {
public:
    RemoveUnalignedReshape() : Pass("RemoveUnalignedReshape") {}
    ~RemoveUnalignedReshape() override = default;
    Status RunOnFunction(Function& function) override;

private:
    void CollectReshapeOps(Function& function);
    void ReplaceDynUnalignedReshapeOps(Function& function);
    void ReplaceDynUnalignedReshapeOpsForUB(Function& function, Operation& op);
    void ReplaceDynUnalignedReshapeOpsForDDR(Function& function, Operation& op);
    void InsertReshapeCopy(Function& function, Operation& op);
    bool ProcessCopyOutOfDDRReshape(Operation* copyOutOp);
    bool ProcessMultipleCopyOuts(std::vector<Operation*>& copyOutOps);
    void ProcessCopyInOfDDRReshape(std::vector<Operation*>& copyInOps);
    std::unordered_set<int> processedReshapeOps;
    Operation* CopyBranchBetweenCopyOut2Reshape(
        Function& function, const std::vector<std::pair<Operation*, LogicalTensorPtr>>& toCopyProducerTensor,
        const int& consumerIndex);
    LogicalTensorPtr HandleNoCopyOutInProducer(Function& function, Operation& op, bool& checkOverUbSize);
    int FindConsumerIndex(LogicalTensorPtr input, Operation* consumerOp);
    void GetPathBetweenSingleCopyOutAndReshape(
        Operation* op, std::vector<std::pair<Operation*, LogicalTensorPtr>>& toCopyProducerTensor, bool& findCopyOut,
        bool& needToCopy, int& index);
    void FindAllProducerCopyOuts(LogicalTensorPtr tensor, std::vector<Operation*>& copyOutOps);
    bool checkNonCopyInConsumerExists(LogicalTensorPtr tensor, std::vector<Operation*>& copyInOps);
    void HandleNoCopyInConsumer(Function& function, Operation& op, LogicalTensorPtr output,
                                std::vector<Operation*>& copyInOps, bool& checkOverUbSize);
    bool CheckUnaligned(Operation& op);
    bool CheckAllCopyOutInputsNonUb(const std::vector<Operation*>& copyOutOps);
    void CollectSiblingCopyInConsumersOfDDRReshapeInput(LogicalTensorPtr reshapeInput, Operation& reshapeOp,
                                                        std::vector<Operation*>& copyInOps);
    LogicalTensorPtr InsertIOTensor(Function& function, Operation& op,
                                    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>>& rawIO,
                                    LogicalTensorPtr& ioTensor);
    std::vector<CopyOutOpMemUnalign> copyOuts;
    std::vector<CopyInOpMemUnalign> copyIns;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawOutputs;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawInputs;
    std::vector<Operation*> newOps;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_REMOVE_UNALIGNED_RESHAPE_OP_H_
