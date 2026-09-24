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
 * \file mix_call_operation_builder.h
 * \brief 用于提供所需的接口
 */

#ifndef MIX_CALL_OPERATION_BUILDER_H
#define MIX_CALL_OPERATION_BUILDER_H

#include "interface/tensor/irbuilder.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {
struct ExtractInfo {
    std::vector<OperandAttribute>& iOffsets;
    std::vector<OperandAttribute>& oOffsets;
    std::set<LogicalTensorPtr>& processedIncasts;
    std::set<LogicalTensorPtr>& processedOutcasts;
};

struct CallOpCreationInfo {
    Function* leafFunc;
    uint64_t newProgramID;
    size_t componentIndex;
    Operation* originalCallOp;
    uint64_t wrapId;
    std::vector<OperandAttribute> iOffsets;
    std::vector<OperandAttribute> oOffsets;
    Operation* createdCallOp = nullptr;
};

class MixCallOperationBuilder {
public:
    MixCallOperationBuilder() : nextWrapId_(0) {}
    Status CreateCallOps(Function& rootFunc, const std::vector<Operation*>& originalCallOps, Function* originalMixFunc,
                         const std::vector<InternalComponentInfo>& components,
                         const std::vector<uint64_t>& newProgramIDs, SubgraphToFunction& subgraphToFunction,
                         std::vector<Function*>& newFunctions);

private:
    // 在root function中创建call op
    Status CreateCallOpInRootFunction(Function& rootFunc, Function& leafFunc, uint64_t newProgramID,
                                      uint64_t componentIndex, Operation* originalCallOp, Function* originalMixFunc,
                                      SubgraphToFunction& subgraphToFunction, CallOpCreationInfo& info);
    void FindNewIOperandsInOriginalIncast(const std::vector<LogicalTensorPtr>& originalIOperands,
                                          const std::vector<std::shared_ptr<LogicalTensor>>& originalIncasts,
                                          const SubfuncInvokeInfoTy& invokeInfo,
                                          std::vector<LogicalTensorPtr>& newIOperands,
                                          std::set<LogicalTensorPtr>& processedIncasts) const;
    void FindNewOOperandsInOriginalOutcast(const std::vector<LogicalTensorPtr>& originalOOperands,
                                           const std::vector<std::shared_ptr<LogicalTensor>>& originalOutcasts,
                                           const SubfuncInvokeInfoTy& invokeInfo,
                                           std::vector<LogicalTensorPtr>& newOOperands,
                                           std::set<LogicalTensorPtr>& processedOutcasts) const;
    void FindNewIOperandsAndOOperandsInPropagateInOutcast(
        const std::vector<LogicalTensorPtr>& originalIOperands, const std::vector<LogicalTensorPtr>& originalOOperands,
        const std::vector<std::shared_ptr<LogicalTensor>>& originalIncasts,
        const std::vector<std::shared_ptr<LogicalTensor>>& originalOutcasts,
        const std::vector<std::shared_ptr<LogicalTensor>>& actualIncasts,
        const std::vector<std::shared_ptr<LogicalTensor>>& actualOutcasts, std::vector<LogicalTensorPtr>& newIOperands,
        std::vector<LogicalTensorPtr>& newOOperands, std::set<LogicalTensorPtr>& processedIncasts,
        std::set<LogicalTensorPtr>& processedOutcasts) const;
    int FindTensorIndexInList(int tensorMagic, const std::vector<LogicalTensorPtr>& tensorList) const;
    // 参数提取函数
    void FindIOpAttrOffsetAndOOpAttrOffset(Function& leafFunc, const SubfuncInvokeInfoTy& invokeInfo,
                                           std::vector<OperandAttribute>& iOffsets,
                                           std::vector<OperandAttribute>& oOffsets, Function* originalMixFunc) const;
    bool FindIOpAttrOffsetFromIncast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc,
                                     ExtractInfo& extractInfo) const;
    bool FindOOpAttrOffsetFromOutcast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc,
                                      ExtractInfo& extractInfo) const;
    bool FindIOOpAttrOffsetGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc,
                                        ExtractInfo& extractInfo) const;
    bool FindIOpAttrOffsetFromActualIncasts(const std::vector<std::shared_ptr<LogicalTensor>>& actualIncasts,
                                            ExtractInfo& extractInfo, Function* originalMixFunc) const;
    bool FindOOpAttrOffsetFromActualOutcasts(const std::vector<std::shared_ptr<LogicalTensor>>& actualOutcasts,
                                             ExtractInfo& extractInfo, Function* originalMixFunc) const;

    OperandAttribute GetOperandAttr(int opMagic, int operandIdx, Function& leafFunc, bool isOutput) const;
    OperandAttribute FindOriginalAttrInMixFunction(LogicalTensorPtr tensor, Function* originalMixFunc) const;
    OperandAttribute FindOriginalAttrInMixFunctionByRawMagic(LogicalTensorPtr tensor, Function* originalMixFunc) const;
    void SetCallOpAttribute(Function& leafFunc, Operation& callOp, Operation* originalCallOp,
                            CallOpAttribute* originalCallAttr, uint64_t newProgramID, uint64_t componentIndex,
                            SubgraphToFunction& subgraphToFunction, CallOpCreationInfo& info);

    uint64_t nextWrapId_;
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_CALL_OPERATION_BUILDER_H
