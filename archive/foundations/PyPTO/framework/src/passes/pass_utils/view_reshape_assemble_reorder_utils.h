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
 * \file view_reshape_assemble_reorder_utils.h
 * \brief utils of view/assemble and reshape operation reordering
 */

#ifndef PASS_VIEW_RESHAPE_ASSEMBLE_REORDER_UTILS_H_
#define PASS_VIEW_RESHAPE_ASSEMBLE_REORDER_UTILS_H_

#include <any>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class ViewReshapeAssembleReorderUtils {
public:
    ViewReshapeAssembleReorderUtils() = default;
    ~ViewReshapeAssembleReorderUtils() = default;

    static Status ReorderViewReshapeAssemble(Function& function);

    static bool RemapOffsetBackwardThroughReshape(
        const LogicalTensorPtr& reshapeInput, const LogicalTensorPtr& reshapeOutput,
        const std::vector<int64_t>& outputBaseShape, const std::vector<SymbolicScalar>& outputBaseDynShape,
        const std::vector<int64_t>& outputOffset, const std::vector<SymbolicScalar>& outputDynOffset,
        std::vector<int64_t>& inputBaseShape, std::vector<SymbolicScalar>& inputBaseDynShape,
        std::vector<int64_t>& inputOffset, std::vector<SymbolicScalar>& inputDynOffset);

    Status Process(Function& function);

private:
    struct AxisGroup {
        size_t srcBegin = 0;
        size_t srcEnd = 0;
        size_t dstBegin = 0;
        size_t dstEnd = 0;
    };

    struct RemapResult {
        std::vector<int64_t> staticOffset;
        std::vector<SymbolicScalar> dynOffset;
    };

    struct ChainMatch {
        LogicalTensorPtr input;
        LogicalTensorPtr middle;
        Operation* secondOp = nullptr;
        LogicalTensorPtr output;
    };

    struct ViewReshapeRecord {
        Operation* viewOp = nullptr;
        Operation* reshapeOp = nullptr;
        LogicalTensorPtr input;
        LogicalTensorPtr output;
        LogicalTensorPtr reshapeOutput;
        std::vector<int64_t> viewOffset;
        std::vector<SymbolicScalar> viewDynOffset;
        std::vector<SymbolicScalar> reshapeDynShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode = false;
        std::any copyInModeValue;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct FanoutViewRecord {
        Operation* viewOp = nullptr;
        LogicalTensorPtr output;
        std::vector<int64_t> viewOffset;
        std::vector<SymbolicScalar> viewDynOffset;
        std::vector<SymbolicScalar> outputDynShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode = false;
        std::any copyInModeValue;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct ViewReshapeFanoutRecord {
        Operation* viewOp = nullptr;
        Operation* reshapeOp = nullptr;
        LogicalTensorPtr input;
        LogicalTensorPtr reshapeOutput;
        std::vector<SymbolicScalar> reshapeDynShape;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
        std::vector<FanoutViewRecord> fanoutViews;
    };

    struct ReshapeAssembleRecord {
        Operation* reshapeOp = nullptr;
        Operation* assembleOp = nullptr;
        LogicalTensorPtr input;
        LogicalTensorPtr output;
        LogicalTensorPtr assembleOutput;
        std::vector<int64_t> assembleOffset;
        std::vector<SymbolicScalar> assembleDynOffset;
        std::vector<SymbolicScalar> reshapeDynShape;
        std::vector<SymbolicScalar> outputDynShape;
        MemoryType fromType = MemoryType::MEM_UNKNOWN;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct FaninAssembleRecord {
        Operation* assembleOp = nullptr;
        LogicalTensorPtr input;
        std::vector<int64_t> assembleOffset;
        std::vector<SymbolicScalar> assembleDynOffset;
        std::vector<SymbolicScalar> inputDynShape;
        MemoryType fromType = MemoryType::MEM_UNKNOWN;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct ReshapeAssembleFaninRecord {
        Operation* reshapeOp = nullptr;
        Operation* assembleOp = nullptr;
        LogicalTensorPtr output;
        LogicalTensorPtr reshapeInput;
        std::vector<SymbolicScalar> reshapeDynShape;
        std::vector<SymbolicScalar> outputDynShape;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
        std::vector<FaninAssembleRecord> faninAssembles;
    };

    void ClearRecords();
    bool HasRecords() const;
    static bool HasCascadedPattern(Function& function);
    Status ProcessOperations(Function& function);
    Status TryRecordViewReshape(Function& function, Operation& viewOp);
    Status TryRecordReshapeAssemble(Function& function, Operation& reshapeOp);
    Status TryRecordViewReshapeFanout(Function& function, Operation& viewOp, Operation& reshapeOp,
                                      const ChainMatch& match, const ViewOpAttribute& viewAttr,
                                      const std::vector<int64_t>& reshapeOutputShape,
                                      const std::vector<SymbolicScalar>& reshapeDynShape,
                                      const std::vector<SymbolicScalar>& inputDynShape);
    Status TryCollectFanoutViewRecord(
        Operation& reshapeOp, Operation& consumer, const ChainMatch& match, const ViewOpAttribute& viewAttr,
        const std::vector<SymbolicScalar>& compactDynShape, const std::vector<SymbolicScalar>& middleDynShape,
        const std::vector<SymbolicScalar>& inputDynShape, const std::vector<int64_t>& reshapeOutputShape,
        const std::vector<SymbolicScalar>& reshapeDynShape, FanoutViewRecord& fanoutRecord, bool& canReorder);
    Status TryRecordDirectReshapeAssemble(Function& function, Operation& reshapeOp, Operation& assembleOp,
                                          const ChainMatch& match, const AssembleOpAttribute& assembleAttr,
                                          const std::vector<int64_t>& assembleOutputShape,
                                          const std::vector<SymbolicScalar>& assembleDynShape,
                                          const std::vector<SymbolicScalar>& middleDynShape,
                                          const std::vector<SymbolicScalar>& outputDynShape);
    Status TryRecordReshapeAssembleFanin(Function& function, Operation& reshapeOp, Operation& assembleOp,
                                         const ChainMatch& match, const AssembleOpAttribute& assembleAttr,
                                         const std::vector<AxisGroup>& axisPlan);
    Status TryCollectFaninAssembleRecord(Operation& reshapeOp, Operation& assembleOp, Operation& producer,
                                         const ChainMatch& match, const AssembleOpAttribute& assembleAttr,
                                         const std::vector<SymbolicScalar>& inputDynShape,
                                         const std::vector<SymbolicScalar>& middleDynShape,
                                         const std::vector<SymbolicScalar>& outputDynShape,
                                         const std::vector<int64_t>& reshapeInputShape,
                                         const std::vector<SymbolicScalar>& reshapeDynShape,
                                         FaninAssembleRecord& faninRecord, bool& canReorder);
    void AppendViewReshapeRecords(Function& function);
    void AppendViewReshapeFanoutRecords(Function& function);
    void AppendReshapeAssembleRecords(Function& function);
    void AppendReshapeAssembleFaninRecords(Function& function);
    void CleanUp(Function& function);
    void MarkViewReshapeFanoutVisited(Operation& viewOp, Operation& reshapeOp, const ViewReshapeFanoutRecord& record);
    void MarkReshapeAssembleFaninVisited(Operation& reshapeOp, Operation& assembleOp,
                                         const ReshapeAssembleFaninRecord& record);
    void CreateMetadataReshape(Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                               const std::vector<SymbolicScalar>& dynShape, const ir::Span& span,
                               const Operation::ScopeInfo& scopeInfo, Operation& srcOp);
    static bool InferInputDynRawShapeFromOutput(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                                std::vector<SymbolicScalar>& inferredInputDynRawShape);
    static bool BuildAxisPlanAllowFirstUnknown(const std::vector<int64_t>& srcShape,
                                               const std::vector<int64_t>& dstShape, std::vector<AxisGroup>& axisPlan);
    void CreateView(Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                    const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset,
                    const std::vector<SymbolicScalar>& outputDynShape, MemoryType toType, bool hasCopyInMode,
                    const std::any& copyInModeValue, const ir::Span& span, const Operation::ScopeInfo& scopeInfo);
    void CreateAssemble(Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                        const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset,
                        const std::vector<SymbolicScalar>& inputDynShape, MemoryType fromType, const ir::Span& span,
                        const Operation::ScopeInfo& scopeInfo, Operation& srcOp);

    static bool GetChainMatch(Operation& firstOp, Opcode secondOpcode, ChainMatch& match);
    static Operation* GetPrecedingViewOp(Operation& reshapeOp);
    static Operation* GetFollowingAssembleOp(Operation& reshapeOp);
    static bool ValidateChainShapes(const ChainMatch& match);
    static bool BuildAxisPlan(const std::vector<int64_t>& srcShape, const std::vector<int64_t>& dstShape,
                              std::vector<AxisGroup>& axisPlan);
    static bool ApplyForwardShape(const std::vector<int64_t>& baseShape,
                                  const std::vector<SymbolicScalar>& baseDynShape,
                                  const std::vector<AxisGroup>& axisPlan, std::vector<int64_t>& newShape,
                                  std::vector<SymbolicScalar>& newDynShape);
    static bool ApplyBackwardShape(const std::vector<int64_t>& baseShape,
                                   const std::vector<SymbolicScalar>& baseDynShape,
                                   const std::vector<AxisGroup>& axisPlan, std::vector<int64_t>& newShape,
                                   std::vector<SymbolicScalar>& newDynShape);
    static bool RemapOffset(const std::vector<int64_t>& oldOffset, const std::vector<SymbolicScalar>& oldDynOffset,
                            const std::vector<int64_t>& oldShape, const std::vector<SymbolicScalar>& oldDynShape,
                            const std::vector<int64_t>& newShape, const std::vector<SymbolicScalar>& newDynShape,
                            RemapResult& result);
    static bool RemapFanoutViewOffset(
        const std::vector<int64_t>& baseViewOffset, const std::vector<SymbolicScalar>& baseViewDynOffset,
        const std::vector<int64_t>& fanoutOffset, const std::vector<SymbolicScalar>& fanoutDynOffset,
        const std::vector<int64_t>& compactShape, const std::vector<SymbolicScalar>& compactDynShape,
        const std::vector<int64_t>& middleShape, const std::vector<SymbolicScalar>& middleDynShape,
        const std::vector<int64_t>& inputShape, const std::vector<SymbolicScalar>& inputDynShape,
        const std::vector<int64_t>& newShape, const std::vector<SymbolicScalar>& newDynShape, RemapResult& result);
    static bool RemapFaninAssembleOffset(
        const std::vector<int64_t>& inputAssembleOffset, const std::vector<SymbolicScalar>& inputAssembleDynOffset,
        const std::vector<int64_t>& outputAssembleOffset, const std::vector<SymbolicScalar>& outputAssembleDynOffset,
        const std::vector<int64_t>& compactShape, const std::vector<SymbolicScalar>& compactDynShape,
        const std::vector<int64_t>& middleShape, const std::vector<SymbolicScalar>& middleDynShape,
        const std::vector<int64_t>& outputShape, const std::vector<SymbolicScalar>& outputDynShape,
        const std::vector<int64_t>& newShape, const std::vector<SymbolicScalar>& newDynShape, RemapResult& result);
    static bool IsContiguousRegion(const std::vector<int64_t>& offset, const std::vector<int64_t>& regionShape,
                                   const std::vector<int64_t>& baseShape);
    static bool IsLinearizedContiguousRegion(const std::vector<int64_t>& offset,
                                             const std::vector<int64_t>& regionShape,
                                             const std::vector<int64_t>& baseShape);
    static bool AreCollapsedGroupsContiguous(const std::vector<int64_t>& offset,
                                             const std::vector<int64_t>& regionShape,
                                             const std::vector<int64_t>& baseShape,
                                             const std::vector<AxisGroup>& axisPlan, bool useSrcGroup);
    static bool GetSymbolicShape(const LogicalTensorPtr& tensor, std::vector<SymbolicScalar>& dynShape);
    static bool GetChainSymbolicShapes(const ChainMatch& match, std::vector<SymbolicScalar>& inputDynShape,
                                       std::vector<SymbolicScalar>& middleDynShape,
                                       std::vector<SymbolicScalar>& outputDynShape);
    static std::vector<SymbolicScalar> GetSymbolicShapeOrStatic(const LogicalTensorPtr& tensor);
    static std::vector<SymbolicScalar> NormalizeDynOffset(const std::vector<int64_t>& offset,
                                                          const std::vector<SymbolicScalar>& dynOffset);
    static bool BuildAssembledValidShape(const std::vector<int64_t>& offset,
                                         const std::vector<SymbolicScalar>& dynOffset,
                                         const std::vector<SymbolicScalar>& inputDynShape, size_t outputRank,
                                         std::vector<SymbolicScalar>& outputDynShape);
    static bool MergeValidShape(const std::vector<SymbolicScalar>& candidate, std::vector<SymbolicScalar>& merged);
    static ir::Span GetFirstSpan(Operation& first, Operation& second);
    static Operation::ScopeInfo GetChainScopeInfo(Operation& first, Operation& second);
    static bool IsScopeCompatible(Operation& first, Operation& second);

    std::unordered_set<int> visitedOp_;
    std::vector<ViewReshapeRecord> viewReshapeRecords_;
    std::vector<ViewReshapeFanoutRecord> viewReshapeFanoutRecords_;
    std::vector<ReshapeAssembleRecord> reshapeAssembleRecords_;
    std::vector<ReshapeAssembleFaninRecord> reshapeAssembleFaninRecords_;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_VIEW_RESHAPE_ASSEMBLE_REORDER_UTILS_H_
