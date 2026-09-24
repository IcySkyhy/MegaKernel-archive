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
 * \file merge_src_dst_buffer.h
 * \brief
 */

#ifndef PASS_MERGE_SRC_DST_BUFFER_H
#define PASS_MERGE_SRC_DST_BUFFER_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
class SrcDstBufferMergeImpl {
public:
    SrcDstBufferMergeImpl() = default;
    ~SrcDstBufferMergeImpl() = default;
    Status Run(Function& func);

private:
    void InitializeTensorMemorymap(Operation& op) const;
    void InitTensorMaxSize(const LogicalTensorPtr& output);
    void InitOpOutput(const Operation& op);
    Status CheckOpValid(const Operation* op, int opId);
    Status Init(const std::vector<Operation*>& opList);
    bool CheckIgnoreScene(const Operation& oriOps);
    Status CheckHasInplaced(const Operation& oriOps, const Operation& ops,
                            std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
                            bool& hasInplaced);
    Status FindReplaced(const Operation& oriOps, const Operation& ops,
                        std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors, bool& hasFound);
    void NotFindReplacedProcess(const Operation& ops,
                                const std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors);
    bool CheckAssembleReuse(const LogicalTensorPtr& outOperand);
    bool CanSrcDstReuse(const Operation& ops, std::shared_ptr<LogicalTensor> iOperand,
                        std::shared_ptr<LogicalTensor> oOperand);
    bool IsL1ToL0Transfer(const Operation& op);
    bool IsL0CToL1Transfer(const Operation& op);
    Status ProcessInplaceReuse(const Operation& oriOps, const Operation& ops,
                               std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
                               bool& hasFound);
    Status ProcessL0MemoryReuse(const Operation& op,
                                std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
                                bool& hasFound);
    Status FindReuseableL0Tensor(const Operation& op,
                                 std::unordered_map<int, std::shared_ptr<LogicalTensor>>& replacedTensors,
                                 LogicalTensorPtr needReplacedTensor, bool& hasFound);

    std::map<int, std::set<int>> tensorConsumers_;
    std::map<int, int64_t> tensorMaxSize_;
    std::set<int> hasReusedL0Tensors_;
};

class SrcDstBufferMerge : public Pass {
public:
    SrcDstBufferMerge() : Pass("SrcDstBufferMerge") {}

private:
    Status RunOnFunction(Function& function) override
    {
        SrcDstBufferMergeImpl merge;
        if (merge.Run(function) != SUCCESS) {
            return SUCCESS;
        }
        return SUCCESS;
    }
};
} // namespace npu::tile_fwk
#endif // PASS_MERGE_SRC_DST_BUFFER_H
