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
 * \file remove_redundant_op_utils.cpp
 * \brief utils for redundant view/assemble and slice/contract elimination
 */

#include "remove_redundant_op_utils.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "interface/operation/attribute.h"
#include "interface/tensor/tensor_offset.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "RemoveRedundantOpUtils"

namespace npu {
namespace tile_fwk {
namespace {
bool IsViewLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE; }

bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT; }

bool IsMatmulOpcode(Opcode opcode) { return opcode == Opcode::OP_A_MUL_B || opcode == Opcode::OP_A_MULACC_B; }

Operation* GetSingleProducer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetProducers().size() != 1) {
        return nullptr;
    }
    return *tensor->GetProducers().begin();
}

bool IsMatmulBackedContractInput(const LogicalTensorPtr& contractInput)
{
    auto* inputProducer = GetSingleProducer(contractInput);
    if (inputProducer == nullptr) {
        return false;
    }
    if (IsMatmulOpcode(inputProducer->GetOpcode())) {
        return true;
    }
    if (inputProducer->GetOpcode() != Opcode::OP_SLICE || inputProducer->GetIOperands().size() != 1) {
        return false;
    }
    auto* materializeOp = GetSingleProducer(inputProducer->GetIOperands().front());
    if (materializeOp == nullptr || materializeOp->GetOpcode() != Opcode::OP_CONTRACT ||
        materializeOp->GetIOperands().size() != 1) {
        return false;
    }
    auto* sourceOp = GetSingleProducer(materializeOp->GetIOperands().front());
    return sourceOp != nullptr && IsMatmulOpcode(sourceOp->GetOpcode());
}

bool IsSupportedViewAssembleCascade(Opcode viewOpcode, Opcode assembleOpcode)
{
    return (viewOpcode == Opcode::OP_VIEW && assembleOpcode == Opcode::OP_ASSEMBLE) ||
           (viewOpcode == Opcode::OP_SLICE && assembleOpcode == Opcode::OP_CONTRACT);
}

bool IsLegacyViewAssembleCascade(Opcode viewOpcode, Opcode assembleOpcode)
{
    return viewOpcode == Opcode::OP_VIEW && assembleOpcode == Opcode::OP_ASSEMBLE;
}

bool IsZeroOffset(const std::vector<int64_t>& offset)
{
    return std::all_of(offset.begin(), offset.end(), [](int64_t value) { return value == 0; });
}

bool IsZeroDynOffset(const std::vector<SymbolicScalar>& dynOffset)
{
    return std::all_of(dynOffset.begin(), dynOffset.end(),
                       [](const SymbolicScalar& value) { return value.ConcreteValid() && value.Concrete() == 0; });
}

bool IsZeroTensorOffset(const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset)
{
    return IsZeroOffset(offset) && (dynOffset.empty() || IsZeroDynOffset(dynOffset));
}

bool IsInitialDynOffset(const std::vector<SymbolicScalar>& dynOffset)
{
    return std::all_of(dynOffset.begin(), dynOffset.end(), [](const SymbolicScalar& value) {
        return value.ConcreteValid() && value.Concrete() == INT_MAX;
    });
}

std::shared_ptr<ViewOpAttribute> GetViewAttr(const Operation& op);
std::shared_ptr<AssembleOpAttribute> GetAssembleAttr(const Operation& op);

struct RegionMapping {
    std::vector<int64_t> sourceOffset;
    std::vector<int64_t> targetOffset;
    std::vector<int64_t> shape;
    std::vector<int64_t> translation;
    bool hasDynOffset{false};
};

bool IsConcreteDynOffsetConsistent(const std::vector<int64_t>& staticOffset,
                                   const std::vector<SymbolicScalar>& dynOffset)
{
    if (dynOffset.empty()) {
        return true;
    }
    if (staticOffset.size() != dynOffset.size()) {
        return false;
    }
    for (size_t idx = 0; idx < staticOffset.size(); ++idx) {
        auto simplified = dynOffset[idx].Simplify();
        if (!simplified.ConcreteValid() || simplified.Concrete() != staticOffset[idx]) {
            return false;
        }
    }
    return true;
}

bool IsConcreteDynValidShapeWithinShape(const std::vector<int64_t>& shape,
                                        const std::vector<SymbolicScalar>& dynValidShape)
{
    if (dynValidShape.empty()) {
        return true;
    }
    if (shape.size() != dynValidShape.size()) {
        return false;
    }
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        auto simplified = dynValidShape[idx].Simplify();
        if (shape[idx] <= 0 || !simplified.ConcreteValid() || simplified.Concrete() <= 0 ||
            simplified.Concrete() > shape[idx]) {
            return false;
        }
    }
    return true;
}

// 获取 tensor 的 concrete valid shape（真实搬运 extent）。
// 若 valid shape 为动态符号（如 view 的 len），说明该 copy 是部分搬运，无法安全与 assemble 合并。
bool GetConcreteValidShape(const LogicalTensorPtr& tensor, std::vector<int64_t>& validShape)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& dynValidShape = tensor->GetDynValidShape();
    validShape.clear();
    if (dynValidShape.empty()) {
        validShape = tensor->GetShape();
        return !validShape.empty();
    }
    if (dynValidShape.size() != tensor->GetShape().size()) {
        return false;
    }
    validShape.reserve(dynValidShape.size());
    for (const auto& dim : dynValidShape) {
        auto simplified = dim.Simplify();
        if (!simplified.ConcreteValid()) {
            return false;
        }
        validShape.push_back(simplified.Concrete());
    }
    return true;
}

bool IsTensorOffset32BAligned(const LogicalTensorPtr& tensor)
{
    constexpr int64_t kOffsetAlignmentBytes = 32;
    if (tensor == nullptr || tensor->GetRawTensor() == nullptr) {
        return false;
    }
    const auto& rawShape = tensor->GetRawTensor()->GetRawShape();
    const auto& offset = tensor->GetOffset();
    if (rawShape.empty() || rawShape.size() != offset.size() ||
        !IsConcreteDynOffsetConsistent(offset, tensor->GetDynOffset())) {
        return false;
    }
    const auto dataSize = static_cast<int64_t>(BytesOf(tensor->Datatype()));
    if (dataSize <= 0) {
        return false;
    }

    int64_t linearOffsetModulo = 0;
    for (size_t idx = 0; idx < rawShape.size(); ++idx) {
        if (rawShape[idx] <= 0 || offset[idx] < 0 || offset[idx] >= rawShape[idx]) {
            return false;
        }
        linearOffsetModulo = (linearOffsetModulo * (rawShape[idx] % kOffsetAlignmentBytes) + offset[idx]) %
                             kOffsetAlignmentBytes;
    }
    return (linearOffsetModulo * dataSize) % kOffsetAlignmentBytes == 0;
}

bool IsRegionWithinShape(const std::vector<int64_t>& offset, const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& outerShape)
{
    if (offset.size() != shape.size() || shape.size() != outerShape.size()) {
        return false;
    }
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (offset[idx] < 0 || shape[idx] <= 0 || outerShape[idx] <= 0 || offset[idx] > outerShape[idx] - shape[idx]) {
            return false;
        }
    }
    return true;
}

bool RegionsOverlap(const RegionMapping& lhs, const RegionMapping& rhs)
{
    for (size_t idx = 0; idx < lhs.shape.size(); ++idx) {
        if (lhs.targetOffset[idx] + lhs.shape[idx] <= rhs.targetOffset[idx] ||
            rhs.targetOffset[idx] + rhs.shape[idx] <= lhs.targetOffset[idx]) {
            return false;
        }
    }
    return true;
}

bool TryCalculateVolume(const std::vector<int64_t>& shape, uint64_t& volume)
{
    volume = 1;
    for (auto dim : shape) {
        if (dim <= 0 || volume > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            return false;
        }
        volume *= static_cast<uint64_t>(dim);
    }
    return true;
}

bool AreTranslatedEndpointsCompatible(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor)
{
    return startTensor != nullptr && endTensor != nullptr &&
           startTensor->GetShape().size() == endTensor->GetShape().size() &&
           startTensor->Datatype() == endTensor->Datatype() && startTensor->Format() == endTensor->Format() &&
           IsConcreteDynValidShapeWithinShape(startTensor->GetShape(), startTensor->GetDynValidShape()) &&
           IsConcreteDynValidShapeWithinShape(endTensor->GetShape(), endTensor->GetDynValidShape());
}

bool HasConsumerOutsideViewAssembleCascade(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor)
{
    return std::any_of(startTensor->GetConsumers().begin(), startTensor->GetConsumers().end(),
                       [&endTensor](const Operation* startConsumer) {
                           if (startConsumer == nullptr || !IsViewLikeOpcode(startConsumer->GetOpcode())) {
                               return true;
                           }
                           const auto consumerOps = startConsumer->ConsumerOps();
                           return consumerOps.empty() ||
                                  std::any_of(consumerOps.begin(), consumerOps.end(),
                                              [startConsumer, &endTensor](const Operation* endProducer) {
                                                  return endProducer == nullptr ||
                                                         endProducer->GetOOperands().empty() ||
                                                         endProducer->GetOOperands().front() != endTensor ||
                                                         !IsSupportedViewAssembleCascade(startConsumer->GetOpcode(),
                                                                                         endProducer->GetOpcode());
                                              });
                       });
}

bool GetTranslatedMappingChain(const Operation* assembleOp, const LogicalTensorPtr& startTensor,
                               const LogicalTensorPtr& endTensor, LogicalTensorPtr& tempTensor, Operation*& viewOp)
{
    if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
        assembleOp->GetIOperands().size() != 1 || assembleOp->GetOOperands().empty() ||
        assembleOp->GetOOperands().front() != endTensor) {
        return false;
    }
    tempTensor = assembleOp->GetIOperands().front();
    if (tempTensor == nullptr || tempTensor->GetProducers().size() != 1 ||
        tempTensor->GetShape().size() != startTensor->GetShape().size() ||
        tempTensor->Datatype() != startTensor->Datatype() || tempTensor->Format() != startTensor->Format() ||
        !IsConcreteDynOffsetConsistent(tempTensor->GetShape(), tempTensor->GetDynValidShape())) {
        return false;
    }
    viewOp = *tempTensor->GetProducers().begin();
    return viewOp != nullptr && IsViewLikeOpcode(viewOp->GetOpcode()) &&
           IsSupportedViewAssembleCascade(viewOp->GetOpcode(), assembleOp->GetOpcode()) &&
           viewOp->GetIOperands().size() == 1 && viewOp->GetIOperands().front() == startTensor &&
           !viewOp->GetOOperands().empty() && viewOp->GetOOperands().front() == tempTensor;
}

bool CalculateRegionTranslation(const std::vector<int64_t>& sourceOffset, const std::vector<int64_t>& targetOffset,
                                std::vector<int64_t>& translation)
{
    translation.resize(sourceOffset.size());
    for (size_t idx = 0; idx < sourceOffset.size(); ++idx) {
        if (sourceOffset[idx] < targetOffset[idx]) {
            return false;
        }
        translation[idx] = sourceOffset[idx] - targetOffset[idx];
    }
    return true;
}

bool TryBuildRegionMapping(const Operation* assembleOp, const LogicalTensorPtr& startTensor,
                           const LogicalTensorPtr& endTensor, RegionMapping& mapping)
{
    LogicalTensorPtr tempTensor;
    Operation* viewOp = nullptr;
    if (!GetTranslatedMappingChain(assembleOp, startTensor, endTensor, tempTensor, viewOp)) {
        return false;
    }
    auto viewAttr = GetViewAttr(*viewOp);
    auto assembleAttr = GetAssembleAttr(*assembleOp);
    if (viewAttr == nullptr || assembleAttr == nullptr) {
        return false;
    }

    mapping.sourceOffset = viewAttr->GetFromOffset();
    mapping.targetOffset = assembleAttr->GetToOffset();
    mapping.shape = tempTensor->GetShape();
    const size_t rank = startTensor->GetShape().size();
    if (mapping.sourceOffset.size() != rank || mapping.targetOffset.size() != rank ||
        !IsConcreteDynOffsetConsistent(mapping.sourceOffset, viewAttr->GetFromDynOffset()) ||
        !IsConcreteDynOffsetConsistent(mapping.targetOffset, assembleAttr->GetToDynOffset()) ||
        !IsRegionWithinShape(mapping.sourceOffset, mapping.shape, startTensor->GetShape()) ||
        !IsRegionWithinShape(mapping.targetOffset, mapping.shape, endTensor->GetShape()) ||
        !CalculateRegionTranslation(mapping.sourceOffset, mapping.targetOffset, mapping.translation)) {
        return false;
    }
    mapping.hasDynOffset = !viewAttr->GetFromDynOffset().empty() || !assembleAttr->GetToDynOffset().empty();
    return true;
}

bool CollectRegionMappings(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor,
                           std::vector<RegionMapping>& mappings, std::vector<int64_t>& commonTranslation,
                           bool& hasDynOffset)
{
    for (auto* assembleOp : endTensor->GetProducers()) {
        RegionMapping mapping;
        if (!TryBuildRegionMapping(assembleOp, startTensor, endTensor, mapping)) {
            return false;
        }
        if (mappings.empty()) {
            commonTranslation = mapping.translation;
        } else if (mapping.translation != commonTranslation) {
            return false;
        }
        hasDynOffset = hasDynOffset || mapping.hasDynOffset;
        mappings.push_back(std::move(mapping));
    }
    return !mappings.empty();
}

bool HasCompleteRegionCoverage(const std::vector<RegionMapping>& mappings, const std::vector<int64_t>& endShape)
{
    uint64_t coveredVolume = 0;
    for (size_t lhs = 0; lhs < mappings.size(); ++lhs) {
        uint64_t regionVolume = 0;
        if (!TryCalculateVolume(mappings[lhs].shape, regionVolume) ||
            coveredVolume > std::numeric_limits<uint64_t>::max() - regionVolume) {
            return false;
        }
        coveredVolume += regionVolume;
        for (size_t rhs = lhs + 1; rhs < mappings.size(); ++rhs) {
            if (RegionsOverlap(mappings[lhs], mappings[rhs])) {
                return false;
            }
        }
    }
    uint64_t endVolume = 0;
    return TryCalculateVolume(endShape, endVolume) && coveredVolume == endVolume;
}

bool TryCalculateTranslatedViewOffset(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor,
                                      std::vector<int64_t>& newOffset, std::vector<SymbolicScalar>& newDynOffset)
{
    if (!AreTranslatedEndpointsCompatible(startTensor, endTensor)) {
        return false;
    }
    std::vector<RegionMapping> mappings;
    std::vector<int64_t> commonTranslation;
    bool hasDynOffset = false;
    if (!CollectRegionMappings(startTensor, endTensor, mappings, commonTranslation, hasDynOffset) ||
        IsZeroOffset(commonTranslation) ||
        !IsRegionWithinShape(commonTranslation, endTensor->GetShape(), startTensor->GetShape()) ||
        !HasCompleteRegionCoverage(mappings, endTensor->GetShape())) {
        return false;
    }

    newOffset = commonTranslation;
    newDynOffset = hasDynOffset ? SymbolicScalar::FromConcrete(commonTranslation) : std::vector<SymbolicScalar>{};
    return true;
}

std::shared_ptr<ViewOpAttribute> GetViewAttr(const Operation& op)
{
    return std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
}

std::shared_ptr<AssembleOpAttribute> GetAssembleAttr(const Operation& op)
{
    return std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
}

bool GetConcreteImmediateValues(const std::vector<OpImmediate>& values, std::vector<int64_t>& result)
{
    result.clear();
    result.reserve(values.size());
    for (const auto& value : values) {
        if (!value.IsSpecified()) {
            return false;
        }
        auto simplified = value.GetSpecifiedValue().Simplify();
        if (!simplified.ConcreteValid()) {
            return false;
        }
        result.push_back(simplified.Concrete());
    }
    return true;
}

bool GetSpecifiedImmediateValues(const std::vector<OpImmediate>& values, std::vector<SymbolicScalar>& result)
{
    result.clear();
    result.reserve(values.size());
    for (const auto& value : values) {
        if (!value.IsSpecified()) {
            return false;
        }
        result.push_back(value.GetSpecifiedValue().Simplify());
    }
    return true;
}

bool IsTensorZeroOffset(const LogicalTensorPtr& tensor)
{
    return tensor != nullptr && IsZeroTensorOffset(tensor->GetOffset(), tensor->GetDynOffset());
}

bool HasPossiblyNonZeroTensorOffset(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || !IsZeroOffset(tensor->GetOffset())) {
        return tensor != nullptr;
    }
    return std::any_of(tensor->GetDynOffset().begin(), tensor->GetDynOffset().end(), [](const SymbolicScalar& value) {
        auto simplified = value.Simplify();
        return !simplified.ConcreteValid() || simplified.Concrete() != 0;
    });
}

bool IsSameTensorTypeAndRank(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    return lhs != nullptr && rhs != nullptr && lhs->GetShape().size() == rhs->GetShape().size() &&
           lhs->Datatype() == rhs->Datatype() && lhs->Format() == rhs->Format();
}

bool IsOffsetEqualToSum(const std::vector<SymbolicScalar>& actual, const std::vector<SymbolicScalar>& base,
                        const std::vector<int64_t>& delta)
{
    if (actual.size() != base.size() || base.size() != delta.size()) {
        return false;
    }
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        auto difference = (actual[idx] - base[idx]).Simplify();
        if (!difference.ConcreteValid() || difference.Concrete() != delta[idx]) {
            return false;
        }
    }
    return true;
}

bool IsCopyShapeCompatible(const CopyOpAttribute& copyAttr, const LogicalTensorPtr& copiedTensor, bool isCopyOut)
{
    if (copiedTensor == nullptr || copiedTensor->GetRawTensor() == nullptr) {
        return false;
    }
    std::vector<int64_t> copyShape;
    if (!GetConcreteImmediateValues(copyAttr.GetShape(), copyShape) || copyShape != copiedTensor->GetShape() ||
        copyAttr.GetRawShape().size() != copiedTensor->GetRawTensor()->GetDynRawShape().size()) {
        return false;
    }
    const auto& validShape = isCopyOut ? copyAttr.GetFromDynValidShape() : copyAttr.GetToDynValidShape();
    return validShape.empty() || validShape.size() == copiedTensor->GetShape().size();
}

struct ViewCopyoutBranch {
    Operation* viewOp{nullptr};
    Operation* copyOutOp{nullptr};
    LogicalTensorPtr viewOutput;
    std::shared_ptr<ViewOpAttribute> viewAttr;
    std::shared_ptr<CopyOpAttribute> copyAttr;
    std::vector<int64_t> viewOffset;
    std::vector<SymbolicScalar> copyOffset;
};

bool BuildViewCopyoutBranch(const LogicalTensorPtr& viewInput, const LogicalTensorPtr& copyOutOutput, Operation& viewOp,
                            Operation& copyOutOp, ViewCopyoutBranch& branch)
{
    if (viewInput == nullptr || copyOutOutput == nullptr || viewOp.GetOpcode() != Opcode::OP_VIEW ||
        viewOp.GetIOperands().size() != 1 || viewOp.GetOOperands().size() != 1 ||
        viewOp.GetIOperands().front() != viewInput || copyOutOp.GetIOperands().size() != 1 ||
        copyOutOp.GetOOperands().size() != 1 || copyOutOp.GetOOperands().front() != copyOutOutput ||
        !OpcodeManager::Inst().IsCopyOut(copyOutOp.GetOpcode())) {
        return false;
    }
    auto viewOutput = viewOp.GetOOperands().front();
    if (viewOutput == nullptr || copyOutOp.GetIOperands().front() != viewOutput ||
        !IsSameTensorTypeAndRank(viewInput, viewOutput) || !IsSameTensorTypeAndRank(viewInput, copyOutOutput) ||
        viewInput->GetMemoryTypeOriginal() != viewOutput->GetMemoryTypeOriginal() ||
        copyOutOutput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        viewOp.GetSubgraphID() != copyOutOp.GetSubgraphID()) {
        return false;
    }
    auto viewAttr = GetViewAttr(viewOp);
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOutOp.GetOpAttribute());
    if (viewAttr == nullptr || copyAttr == nullptr || !copyAttr->IsCopyOut() ||
        copyAttr->GetCopyOutAttr().first != viewInput->GetMemoryTypeOriginal() ||
        (viewAttr->GetTo() != MemoryType::MEM_UNKNOWN && viewAttr->GetTo() != viewOutput->GetMemoryTypeOriginal()) ||
        !IsConcreteDynOffsetConsistent(viewAttr->GetFromOffset(), viewAttr->GetFromDynOffset()) ||
        !GetSpecifiedImmediateValues(copyAttr->GetToOffset(), branch.copyOffset) ||
        !IsCopyShapeCompatible(*copyAttr, viewOutput, true)) {
        return false;
    }
    branch.viewOffset = viewAttr->GetFromOffset();
    branch.viewOp = &viewOp;
    branch.copyOutOp = &copyOutOp;
    branch.viewOutput = viewOutput;
    branch.viewAttr = std::move(viewAttr);
    branch.copyAttr = std::move(copyAttr);
    return true;
}

bool CanMergeViewCopyoutGroup(const LogicalTensorPtr& viewInput, const LogicalTensorPtr& copyOutOutput,
                              const std::vector<std::pair<Operation*, Operation*>>& candidates,
                              std::vector<ViewCopyoutBranch>& branches, size_t& retainedIndex)
{
    if (!IsTensorZeroOffset(viewInput) || candidates.empty()) {
        return false;
    }
    std::vector<RegionMapping> mappings;
    int zeroOffsetCount = 0;
    int groupSubgraph = candidates.front().first->GetSubgraphID();
    Opcode copyOpcode = candidates.front().second->GetOpcode();
    for (const auto& [viewOp, copyOutOp] : candidates) {
        ViewCopyoutBranch branch;
        if (viewOp == nullptr || copyOutOp == nullptr || viewOp->GetSubgraphID() != groupSubgraph ||
            copyOutOp->GetSubgraphID() != groupSubgraph || copyOutOp->GetOpcode() != copyOpcode ||
            !BuildViewCopyoutBranch(viewInput, copyOutOutput, *viewOp, *copyOutOp, branch) ||
            !IsRegionWithinShape(branch.viewOffset, branch.viewOutput->GetShape(), viewInput->GetShape())) {
            return false;
        }
        if (IsZeroOffset(branch.viewOffset)) {
            retainedIndex = branches.size();
            ++zeroOffsetCount;
        }
        RegionMapping mapping;
        mapping.targetOffset = branch.viewOffset;
        mapping.shape = branch.viewOutput->GetShape();
        mappings.push_back(std::move(mapping));
        branches.push_back(std::move(branch));
    }
    if (zeroOffsetCount != 1 || !HasCompleteRegionCoverage(mappings, viewInput->GetShape())) {
        return false;
    }
    const auto& retainedToOffset = branches[retainedIndex].copyOffset;
    return std::all_of(branches.begin(), branches.end(), [&retainedToOffset](const ViewCopyoutBranch& branch) {
        return IsOffsetEqualToSum(branch.copyOffset, retainedToOffset, branch.viewOffset);
    });
}

void MergeViewCopyoutGroup(const LogicalTensorPtr& viewInput, std::vector<ViewCopyoutBranch>& branches,
                           size_t retainedIndex, bool& operationUpdated)
{
    auto& retained = branches[retainedIndex];
    retained.copyOutOp->ReplaceIOperand(0, viewInput);
    retained.copyAttr->SetShape(OpImmediate::Specified(viewInput->GetShape()));
    // rawShape describes the destination GM layout. Keep the retained copy-out value unchanged.
    retained.copyAttr->SetFromDynValidShape(OpImmediate::Specified(viewInput->GetDynValidShape()));
    for (size_t idx = 0; idx < branches.size(); ++idx) {
        branches[idx].viewOp->SetAsDeleted();
        if (idx != retainedIndex) {
            branches[idx].copyOutOp->SetAsDeleted();
        }
    }
    operationUpdated = true;
}

struct CopyinAssembleBranch {
    Operation* copyInOp{nullptr};
    Operation* assembleOp{nullptr};
    LogicalTensorPtr copyInOutput;
    std::shared_ptr<CopyOpAttribute> copyAttr;
    std::shared_ptr<AssembleOpAttribute> assembleAttr;
    std::vector<SymbolicScalar> copyOffset;
    std::vector<int64_t> assembleOffset;
};

bool BuildCopyinAssembleBranch(const LogicalTensorPtr& copyInInput, const LogicalTensorPtr& assembleOutput,
                               Operation& copyInOp, Operation& assembleOp, CopyinAssembleBranch& branch)
{
    if (copyInInput == nullptr || assembleOutput == nullptr || assembleOp.GetOpcode() != Opcode::OP_ASSEMBLE ||
        assembleOp.GetIOperands().size() != 1 || assembleOp.GetOOperands().size() != 1 ||
        assembleOp.GetOOperands().front() != assembleOutput || copyInOp.GetIOperands().size() != 1 ||
        copyInOp.GetOOperands().size() != 1 || copyInOp.GetIOperands().front() != copyInInput ||
        !OpcodeManager::Inst().IsCopyIn(copyInOp.GetOpcode())) {
        return false;
    }
    auto copyInOutput = copyInOp.GetOOperands().front();
    if (copyInOutput == nullptr || assembleOp.GetIOperands().front() != copyInOutput ||
        !IsSameTensorTypeAndRank(copyInInput, copyInOutput) || !IsSameTensorTypeAndRank(copyInInput, assembleOutput) ||
        copyInInput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        copyInOutput->GetMemoryTypeOriginal() != assembleOutput->GetMemoryTypeOriginal() ||
        copyInOp.GetSubgraphID() != assembleOp.GetSubgraphID()) {
        return false;
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp.GetOpAttribute());
    auto assembleAttr = GetAssembleAttr(assembleOp);
    if (copyAttr == nullptr || assembleAttr == nullptr || copyAttr->IsCopyOut() ||
        copyAttr->GetCopyInAttr().second != assembleOutput->GetMemoryTypeOriginal() ||
        (assembleAttr->GetFrom() != MemoryType::MEM_UNKNOWN &&
         assembleAttr->GetFrom() != copyInOutput->GetMemoryTypeOriginal()) ||
        !GetSpecifiedImmediateValues(copyAttr->GetFromOffset(), branch.copyOffset) ||
        !IsConcreteDynOffsetConsistent(assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset()) ||
        !IsCopyShapeCompatible(*copyAttr, copyInOutput, false)) {
        return false;
    }
    branch.assembleOffset = assembleAttr->GetToOffset();
    branch.copyInOp = &copyInOp;
    branch.assembleOp = &assembleOp;
    branch.copyInOutput = copyInOutput;
    branch.copyAttr = std::move(copyAttr);
    branch.assembleAttr = std::move(assembleAttr);
    return true;
}

bool CanMergeCopyinAssembleGroup(const LogicalTensorPtr& copyInInput, const LogicalTensorPtr& assembleOutput,
                                 const std::vector<std::pair<Operation*, Operation*>>& candidates,
                                 std::vector<CopyinAssembleBranch>& branches, size_t& retainedIndex)
{
    if (candidates.empty()) {
        return false;
    }
    std::vector<RegionMapping> mappings;
    int zeroOffsetCount = 0;
    int groupSubgraph = candidates.front().first->GetSubgraphID();
    Opcode copyOpcode = candidates.front().first->GetOpcode();
    for (const auto& [copyInOp, assembleOp] : candidates) {
        CopyinAssembleBranch branch;
        if (copyInOp == nullptr || assembleOp == nullptr || copyInOp->GetSubgraphID() != groupSubgraph ||
            assembleOp->GetSubgraphID() != groupSubgraph || copyInOp->GetOpcode() != copyOpcode ||
            !BuildCopyinAssembleBranch(copyInInput, assembleOutput, *copyInOp, *assembleOp, branch)) {
            return false;
        }
        std::vector<int64_t> copyValidShape;
        if (!GetConcreteValidShape(branch.copyInOutput, copyValidShape) ||
            !IsRegionWithinShape(branch.assembleOffset, copyValidShape, assembleOutput->GetShape())) {
            return false;
        }
        if (IsZeroOffset(branch.assembleOffset)) {
            retainedIndex = branches.size();
            ++zeroOffsetCount;
        }
        RegionMapping mapping;
        mapping.targetOffset = branch.assembleOffset;
        mapping.shape = std::move(copyValidShape);
        mappings.push_back(std::move(mapping));
        branches.push_back(std::move(branch));
    }
    if (zeroOffsetCount != 1 || !HasCompleteRegionCoverage(mappings, assembleOutput->GetShape())) {
        return false;
    }
    const auto& retainedFromOffset = branches[retainedIndex].copyOffset;
    return std::all_of(branches.begin(), branches.end(), [&retainedFromOffset](const CopyinAssembleBranch& branch) {
        return IsOffsetEqualToSum(branch.copyOffset, retainedFromOffset, branch.assembleOffset);
    });
}

void MergeCopyinAssembleGroup(const LogicalTensorPtr& copyInInput, const LogicalTensorPtr& assembleOutput,
                              std::vector<CopyinAssembleBranch>& branches, size_t retainedIndex, bool& operationUpdated)
{
    auto& retained = branches[retainedIndex];
    retained.copyInOp->ReplaceOOperand(0, assembleOutput);
    retained.copyAttr->SetShape(OpImmediate::Specified(assembleOutput->GetShape()));
    retained.copyAttr->SetRawShape(OpImmediate::Specified(copyInInput->GetRawTensor()->GetDynRawShape()));
    retained.copyAttr->SetToDynValidShape(OpImmediate::Specified(assembleOutput->GetDynValidShape()));
    for (size_t idx = 0; idx < branches.size(); ++idx) {
        branches[idx].assembleOp->SetAsDeleted();
        if (idx != retainedIndex) {
            branches[idx].copyInOp->SetAsDeleted();
        }
    }
    operationUpdated = true;
}

std::string IntVectorToString(const std::vector<int64_t>& values)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            stream << ", ";
        }
        stream << values[idx];
    }
    stream << "]";
    return stream.str();
}

std::string SymbolicVectorToString(const std::vector<SymbolicScalar>& values)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            stream << ", ";
        }
        stream << values[idx].Dump();
    }
    stream << "]";
    return stream.str();
}

void WarnUnresolvedCopyoutOffset(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.IsDeleted() || !OpcodeManager::Inst().IsCopyOut(op.GetOpcode()) || op.GetIOperands().empty()) {
            continue;
        }
        const auto& input = op.GetIOperands().front();
        if (!HasPossiblyNonZeroTensorOffset(input)) {
            continue;
        }
        const auto rawShape = input->GetRawTensor() == nullptr ? std::vector<int64_t>{} :
                                                                 input->GetRawTensor()->GetRawShape();
        APASS_LOG_WARN_F(Elements::Operation,
                         "CopyOut op[%d] still has an input tensor with offset. shape=%s, rawShape=%s, offset=%s, "
                         "dynOffset=%s. CCE may not handle this offset correctly and could cause a precision issue.",
                         op.GetOpMagic(), IntVectorToString(input->GetShape()).c_str(),
                         IntVectorToString(rawShape).c_str(), IntVectorToString(input->GetOffset()).c_str(),
                         SymbolicVectorToString(input->GetDynOffset()).c_str());
    }
}

bool IsEqualShapeWithDynShape(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr || lhs->GetShape() != rhs->GetShape()) {
        return false;
    }
    const auto& lhsDynShape = lhs->GetDynValidShape();
    const auto& rhsDynShape = rhs->GetDynValidShape();
    if (lhsDynShape.empty() && rhsDynShape.empty()) {
        return true;
    }
    if (lhsDynShape.size() != rhsDynShape.size()) {
        return false;
    }
    for (size_t idx = 0; idx < lhsDynShape.size(); ++idx) {
        if (lhsDynShape[idx].Dump() != rhsDynShape[idx].Dump()) {
            return false;
        }
    }
    return true;
}

bool IsSingleZeroOffsetSliceContract(const Operation& sliceOp, const Operation& contractOp)
{
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || contractOp.GetOpcode() != Opcode::OP_CONTRACT ||
        sliceOp.GetIOperands().size() != 1 || sliceOp.GetOOperands().size() != 1 ||
        contractOp.GetIOperands().size() != 1 || contractOp.GetOOperands().size() != 1 ||
        sliceOp.GetOOperands().front() != contractOp.GetIOperands().front()) {
        return false;
    }

    const auto& sliceOutput = sliceOp.GetOOperands().front();
    const auto& contractOutput = contractOp.GetOOperands().front();
    const auto& sliceConsumers = sliceOutput->GetConsumers();
    const auto& contractOutputProducers = contractOutput->GetProducers();
    if (sliceConsumers.size() != 1 || *sliceConsumers.begin() != &contractOp || contractOutputProducers.size() != 1 ||
        *contractOutputProducers.begin() != &contractOp || !IsEqualShapeWithDynShape(sliceOutput, contractOutput)) {
        return false;
    }

    auto contractAttr = GetAssembleAttr(contractOp);
    return contractAttr != nullptr && IsZeroTensorOffset(contractAttr->GetToOffset(), contractAttr->GetToDynOffset());
}

void GenerateSingleSliceContractView(Function& function, Operation& sliceOp, Operation& contractOp,
                                     const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor,
                                     std::vector<Operation*>& newOps, bool& operationUpdated)
{
    auto sliceAttr = GetViewAttr(sliceOp);
    if (sliceAttr == nullptr) {
        return;
    }

    auto viewAttribute = std::make_shared<ViewOpAttribute>(sliceAttr->GetFromOffset(), sliceAttr->GetFromDynOffset(),
                                                           sliceAttr->GetToDynValidShape());
    viewAttribute->SetToType(endTensor->GetMemoryTypeToBe());
    auto& newViewOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_VIEW, {startTensor}, {endTensor},
        [&viewAttribute](Operation& newOp) { newOp.SetOpAttribute(viewAttribute); });

    sliceOp.SetAsDeleted();
    contractOp.SetAsDeleted();
    newOps.push_back(&newViewOp);
    operationUpdated = true;
}

bool HasMemoryTypeTransform(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return false;
    }
    auto inputMem = inputTensor->GetMemoryTypeOriginal();
    auto outputMem = outputTensor->GetMemoryTypeOriginal();
    if (inputMem != MemoryType::MEM_UNKNOWN && outputMem != MemoryType::MEM_UNKNOWN && inputMem != outputMem) {
        return true;
    }
    auto outputToBe = outputTensor->GetMemoryTypeToBe();
    return inputMem != MemoryType::MEM_UNKNOWN && outputToBe != MemoryType::MEM_UNKNOWN && inputMem != outputToBe;
}

Opcode GetViewLikeReplacementOpcode(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor)
{
    return HasMemoryTypeTransform(inputTensor, outputTensor) ? Opcode::OP_SLICE : Opcode::OP_VIEW;
}

bool CanBypassPerfectMatch(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor)
{
    return inputTensor != nullptr && outputTensor != nullptr &&
           inputTensor->GetMemoryTypeOriginal() == outputTensor->GetMemoryTypeOriginal();
}

bool IsSliceEqualSizeMove(const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_SLICE || op.GetIOperands().empty() || op.GetOOperands().empty()) {
        return false;
    }
    if (!IsEqualShapeWithDynShape(op.GetIOperands().front(), op.GetOOperands().front())) {
        return false;
    }
    auto sliceAttr = GetViewAttr(op);
    return sliceAttr != nullptr && IsZeroTensorOffset(sliceAttr->GetFromOffset(), sliceAttr->GetFromDynOffset());
}

bool SliceRequiresL1(const Operation& sliceOp)
{
    auto sliceAttr = GetViewAttr(sliceOp);
    return sliceAttr != nullptr && sliceAttr->GetTo() == MemoryType::MEM_L1;
}

bool IsSingleContractWithMultipleL1SliceConsumers(const Operation& contractOp)
{
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.GetOOperands().empty()) {
        return false;
    }
    auto contractOutput = contractOp.GetOOperands().front();
    if (contractOutput == nullptr) {
        return false;
    }
    const auto& producers = contractOutput->GetProducers();
    if (producers.size() != 1 || *producers.begin() != &contractOp) {
        return false;
    }
    const auto& consumers = contractOutput->GetConsumers();
    return consumers.size() > 1 && std::all_of(consumers.begin(), consumers.end(), [](const Operation* consumer) {
               return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE && SliceRequiresL1(*consumer);
           });
}

bool IsInputProducedBySingleSlice(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetProducers().size() != 1) {
        return false;
    }
    auto* producer = *tensor->GetProducers().begin();
    return producer != nullptr && producer->GetOpcode() == Opcode::OP_SLICE;
}

template <typename ContractOps>
bool CollectPrecedingSlicesForL1Transfer(const Operation& sliceOp, const ContractOps& contractOps,
                                         std::set<Operation*>& precedingSlices)
{
    if (!SliceRequiresL1(sliceOp)) {
        return true;
    }
    for (auto* contractOp : contractOps) {
        if (contractOp == nullptr || contractOp->GetIOperands().size() != 1) {
            return false;
        }
        auto contractInput = contractOp->GetIOperands().front();
        if (contractInput == nullptr || contractInput->GetProducers().size() != 1) {
            return false;
        }
        auto* precedingSlice = *contractInput->GetProducers().begin();
        if (precedingSlice == nullptr || precedingSlice->GetOpcode() != Opcode::OP_SLICE ||
            precedingSlice->GetOOperands().empty() || precedingSlice->GetOOperands().front() != contractInput ||
            GetViewAttr(*precedingSlice) == nullptr) {
            return false;
        }
        precedingSlices.insert(precedingSlice);
    }
    return true;
}

void TransferL1Requirement(const std::set<Operation*>& precedingSlices)
{
    for (auto* precedingSlice : precedingSlices) {
        auto sliceAttr = GetViewAttr(*precedingSlice);
        auto newAttr = std::dynamic_pointer_cast<ViewOpAttribute>(sliceAttr->Clone());
        newAttr->SetToType(MemoryType::MEM_L1);
        precedingSlice->SetOpAttribute(newAttr);
        precedingSlice->GetOOperands().front()->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    }
}

void TransferL1CopyAttrs(const Operation& l1Slice, Operation& precedingSlice)
{
    precedingSlice.CopyAttrFrom(l1Slice, OP_ATTR_PREFIX + "copy_in_l1_");
    int64_t copyInMode = 0;
    if (l1Slice.GetAttr<int64_t>(OpAttributeKey::copyInMode, copyInMode)) {
        precedingSlice.SetAttr(OpAttributeKey::copyInMode, copyInMode);
    }
    int64_t isGemv = 0;
    if (l1Slice.GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv)) {
        precedingSlice.SetAttr(OpAttributeKey::isGemv, isGemv);
    }
}

bool IsContractInputEqualSliceOutput(const Operation& contractOp, const Operation& sliceOp)
{
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.GetIOperands().empty() ||
        sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.GetOOperands().empty()) {
        return false;
    }
    auto contractInput = contractOp.GetIOperands().front();
    auto sliceOutput = sliceOp.GetOOperands().front();
    return IsEqualShapeWithDynShape(contractInput, sliceOutput);
}

bool CalculateContractLocalSliceOffset(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor,
                                       const ViewOpAttribute& viewAttr, const AssembleOpAttribute& assembleAttr,
                                       std::vector<int64_t>& localOffset, std::vector<SymbolicScalar>& localDynOffset)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return false;
    }
    const auto& inputShape = inputTensor->GetShape();
    const auto& outputShape = outputTensor->GetShape();
    const auto& sliceOffset = viewAttr.GetFromOffset();
    const auto& contractOffset = assembleAttr.GetToOffset();
    if (inputShape.size() != outputShape.size() || inputShape.size() != sliceOffset.size() ||
        inputShape.size() != contractOffset.size() ||
        !IsConcreteDynOffsetConsistent(sliceOffset, viewAttr.GetFromDynOffset()) ||
        !IsConcreteDynOffsetConsistent(contractOffset, assembleAttr.GetToDynOffset())) {
        return false;
    }
    localOffset = TensorOffset::Sub(sliceOffset, contractOffset);
    for (size_t idx = 0; idx < inputShape.size(); ++idx) {
        if (inputShape[idx] < 0 || outputShape[idx] < 0) {
            return false;
        }
        if (localOffset[idx] < 0 || localOffset[idx] + outputShape[idx] > inputShape[idx]) {
            return false;
        }
    }
    localDynOffset = viewAttr.GetFromDynOffset().empty() && assembleAttr.GetToDynOffset().empty() ?
                         std::vector<SymbolicScalar>{} :
                         SymbolicScalar::FromConcrete(localOffset);
    return true;
}

bool IsFullSliceOfContractInput(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor,
                                const std::vector<int64_t>& localOffset,
                                const std::vector<SymbolicScalar>& localDynOffset)
{
    return inputTensor != nullptr && outputTensor != nullptr && inputTensor->GetShape() == outputTensor->GetShape() &&
           IsZeroTensorOffset(localOffset, localDynOffset);
}

Operation* GetReplaceablePrecedingSlice(Operation& contractOp, const Operation& sliceOp,
                                        const LogicalTensorPtr& contractInput, const LogicalTensorPtr& sliceOutput)
{
    if (!SliceRequiresL1(sliceOp) || contractInput == nullptr || sliceOutput == nullptr ||
        contractInput->GetShape() != sliceOutput->GetShape()) {
        return nullptr;
    }
    const auto& contractInputConsumers = contractInput->GetConsumers();
    if (contractInputConsumers.size() != 1 || *contractInputConsumers.begin() != &contractOp) {
        return nullptr;
    }
    const auto& producers = contractInput->GetProducers();
    if (producers.size() != 1) {
        return nullptr;
    }
    auto* precedingSlice = *producers.begin();
    if (precedingSlice == nullptr || precedingSlice->GetOpcode() != Opcode::OP_SLICE ||
        precedingSlice->GetOOperands().size() != 1 || precedingSlice->GetOOperands().front() != contractInput) {
        return nullptr;
    }
    return precedingSlice;
}

void ReplaceConsumers(const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor)
{
    if (oldTensor == nullptr || newTensor == nullptr) {
        return;
    }
    auto consumers = oldTensor->GetConsumers();
    for (auto* consumer : consumers) {
        if (consumer == nullptr) {
            continue;
        }
        consumer->ReplaceInput(newTensor, oldTensor);
    }
}
} // namespace

Status RemoveRedundantOpUtils::Process(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated)
{
    RemoveRedundantOpUtils utils;
    return utils.ProcessImpl(function, newOps, operationUpdated);
}

Status RemoveRedundantOpUtils::ProcessViewAssembleLike(Function& function, std::vector<Operation*>& newOps,
                                                       bool& operationUpdated)
{
    RemoveRedundantOpUtils utils;
    return utils.ProcessViewAssembleLikeImpl(function, newOps, operationUpdated);
}

Status RemoveRedundantOpUtils::ProcessContractSlice(Function& function, std::vector<Operation*>& newOps,
                                                    bool& operationUpdated)
{
    RemoveRedundantOpUtils utils;
    return utils.ProcessContractSliceImpl(function, newOps, operationUpdated);
}

Status RemoveRedundantOpUtils::ProcessViewCopyout(Function& function, bool& operationUpdated)
{
    std::unordered_set<int> visitedViewMagics;
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->IsDeleted() || op->GetOpcode() != Opcode::OP_VIEW ||
            visitedViewMagics.count(op->GetOpMagic()) != 0 || op->GetIOperands().size() != 1) {
            continue;
        }
        const auto& viewInput = op->GetIOperands().front();
        if (viewInput == nullptr) {
            continue;
        }
        std::unordered_map<LogicalTensorPtr, std::vector<std::pair<Operation*, Operation*>>> outputGroups;
        bool familyValid = true;
        for (auto* siblingView : viewInput->GetConsumers()) {
            if (siblingView == nullptr || siblingView->IsDeleted() || siblingView->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            visitedViewMagics.insert(siblingView->GetOpMagic());
            if (siblingView->GetOOperands().size() != 1 || siblingView->GetOOperands().front() == nullptr) {
                continue;
            }
            const auto& viewOutput = siblingView->GetOOperands().front();
            std::vector<Operation*> copyOutConsumers;
            for (auto* consumer : viewOutput->GetConsumers()) {
                if (consumer != nullptr && !consumer->IsDeleted() &&
                    OpcodeManager::Inst().IsCopyOut(consumer->GetOpcode())) {
                    copyOutConsumers.push_back(consumer);
                }
            }
            if (copyOutConsumers.empty()) {
                continue;
            }
            if (viewOutput->GetConsumers().size() != 1 || copyOutConsumers.size() != 1 ||
                copyOutConsumers.front()->GetOOperands().size() != 1 ||
                copyOutConsumers.front()->GetOOperands().front() == nullptr) {
                familyValid = false;
                break;
            }
            outputGroups[copyOutConsumers.front()->GetOOperands().front()].push_back(
                {siblingView, copyOutConsumers.front()});
        }
        if (!familyValid) {
            continue;
        }
        for (auto& [copyOutOutput, candidates] : outputGroups) {
            std::vector<ViewCopyoutBranch> branches;
            size_t retainedIndex = 0;
            if (!CanMergeViewCopyoutGroup(viewInput, copyOutOutput, candidates, branches, retainedIndex)) {
                continue;
            }
            MergeViewCopyoutGroup(viewInput, branches, retainedIndex, operationUpdated);
        }
    }
    WarnUnresolvedCopyoutOffset(function);
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessCopyinAssemble(Function& function, bool& operationUpdated)
{
    std::unordered_set<int> visitedAssembleMagics;
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->IsDeleted() || op->GetOpcode() != Opcode::OP_ASSEMBLE ||
            visitedAssembleMagics.count(op->GetOpMagic()) != 0 || op->GetOOperands().size() != 1) {
            continue;
        }
        const auto& assembleOutput = op->GetOOperands().front();
        if (assembleOutput == nullptr) {
            continue;
        }
        std::unordered_map<LogicalTensorPtr, std::vector<std::pair<Operation*, Operation*>>> inputGroups;
        std::unordered_set<LogicalTensorPtr> invalidInputs;
        for (auto* siblingAssemble : assembleOutput->GetProducers()) {
            if (siblingAssemble == nullptr || siblingAssemble->IsDeleted() ||
                siblingAssemble->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            visitedAssembleMagics.insert(siblingAssemble->GetOpMagic());
            if (siblingAssemble->GetIOperands().size() != 1 || siblingAssemble->GetIOperands().front() == nullptr) {
                continue;
            }
            const auto& copyInOutput = siblingAssemble->GetIOperands().front();
            if (copyInOutput->GetProducers().size() != 1) {
                continue;
            }
            auto* copyInOp = *copyInOutput->GetProducers().begin();
            if (copyInOp == nullptr || copyInOp->IsDeleted() ||
                !OpcodeManager::Inst().IsCopyIn(copyInOp->GetOpcode()) || copyInOp->GetIOperands().size() != 1 ||
                copyInOp->GetIOperands().front() == nullptr) {
                continue;
            }
            const auto& copyInInput = copyInOp->GetIOperands().front();
            if (copyInOutput->GetConsumers().size() != 1 || *copyInOutput->GetConsumers().begin() != siblingAssemble) {
                invalidInputs.insert(copyInInput);
                continue;
            }
            inputGroups[copyInInput].push_back({copyInOp, siblingAssemble});
        }
        for (auto& [copyInInput, candidates] : inputGroups) {
            if (invalidInputs.count(copyInInput) != 0) {
                continue;
            }
            std::vector<CopyinAssembleBranch> branches;
            size_t retainedIndex = 0;
            if (!CanMergeCopyinAssembleGroup(copyInInput, assembleOutput, candidates, branches, retainedIndex)) {
                continue;
            }
            MergeCopyinAssembleGroup(copyInInput, assembleOutput, branches, retainedIndex, operationUpdated);
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated)
{
    if (ProcessViewAssembleLikeImpl(function, newOps, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessViewAssembleLike failed.");
        return FAILED;
    }
    if (ProcessContractSliceImpl(function, newOps, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessContractSlice failed.");
        return FAILED;
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessViewAssembleLikeImpl(Function& function, std::vector<Operation*>& newOps,
                                                           bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    bool hasDeletedOperations = false;
    for (auto& op : opList) {
        if (op == nullptr || !IsViewLikeOpcode(op->GetOpcode()) || op->iOperand.empty() || op->oOperand.empty()) {
            continue;
        }
        auto& startTensor = op->iOperand.front();
        auto consumers = op->oOperand.front()->GetConsumers();
        for (const auto& consumer : consumers) {
            if (consumer == nullptr || !IsSupportedViewAssembleCascade(op->GetOpcode(), consumer->GetOpcode()) ||
                consumer->oOperand.empty()) {
                continue;
            }
            auto& endTensor = consumer->oOperand.front();
            if (function.IsFromInCast(startTensor) && function.IsFromOutCast(endTensor)) {
                continue;
            }
            if (function.IsFromOutCast(startTensor) && function.IsFromOutCast(endTensor)) {
                continue;
            }
            if (IsLegacyViewAssembleCascade(op->GetOpcode(), consumer->GetOpcode()) &&
                startTensor->GetMemoryTypeOriginal() != endTensor->GetMemoryTypeOriginal()) {
                continue;
            }
            if (op->GetOpcode() == Opcode::OP_SLICE && consumer->GetOpcode() == Opcode::OP_CONTRACT &&
                consumer->GetIOperands().size() == 1 &&
                IsInputProducedBySingleSlice(consumer->GetIOperands().front()) &&
                IsSingleContractWithMultipleL1SliceConsumers(*consumer)) {
                continue;
            }
            if (startTensor->shape == endTensor->shape && startTensor->offset == endTensor->offset) {
                if (!CanBypassPerfectMatch(startTensor, endTensor)) {
                    continue;
                }
                if (function.IsFromOutCast(endTensor) &&
                    HasConsumerOutsideViewAssembleCascade(startTensor, endTensor)) {
                    continue;
                }
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "CASE1: Process %s[%d]'s input and %s[%d]'s output perfectMatch.",
                    OpcodeManager::Inst().GetOpcodeStr(op->GetOpcode()).c_str(), op->opmagic,
                    OpcodeManager::Inst().GetOpcodeStr(consumer->GetOpcode()).c_str(), consumer->GetOpMagic());
                ProcessPerfectMatch(startTensor, endTensor, function.IsFromOutCast(endTensor), operationUpdated);
            } else {
                if ((function.IsFromInCast(startTensor) && CommonUtils::ContainsNegativeOne(startTensor->GetShape())) ||
                    (function.IsFromOutCast(endTensor) && CommonUtils::ContainsNegativeOne(endTensor->GetShape()))) {
                    continue;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "CASE2: Process %s[%d]'s input is a part of %s[%d]'s output.",
                                  OpcodeManager::Inst().GetOpcodeStr(op->GetOpcode()).c_str(), op->opmagic,
                                  OpcodeManager::Inst().GetOpcodeStr(consumer->GetOpcode()).c_str(),
                                  consumer->GetOpMagic());
                if (IsSingleZeroOffsetSliceContract(*op, *consumer) &&
                    !HasMemoryTypeTransform(startTensor, endTensor)) {
                    GenerateSingleSliceContractView(function, *op, *consumer, startTensor, endTensor, newOps,
                                                    operationUpdated);
                    hasDeletedOperations = true;
                    continue;
                }
                GenerateNewViewLike(function, *op, startTensor, endTensor,
                                    GetViewLikeReplacementOpcode(startTensor, endTensor), newOps, operationUpdated);
            }
        }
    }
    if (hasDeletedOperations) {
        function.EraseOperations(true, false);
    }
    return SUCCESS;
}

void RemoveRedundantOpUtils::RemoveViewAssembleForOutcast(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                                                          bool& operationUpdated)
{
    bool canRemove = false;
    std::set<Operation*, LogicalTensor::CompareOp> removeOps;
    for (auto& startConsumer : startTensor->GetConsumers()) {
        if (startConsumer == nullptr || !IsViewLikeOpcode(startConsumer->GetOpcode())) {
            continue;
        }
        canRemove = true;
        for (auto& endProducer : startConsumer->ConsumerOps()) {
            if (endProducer->GetOOperands().empty() || endProducer->GetOOperands().front() != endTensor ||
                !IsSupportedViewAssembleCascade(startConsumer->GetOpcode(), endProducer->GetOpcode())) {
                canRemove = false;
            } else {
                removeOps.insert(endProducer);
            }
        }
        if (canRemove) {
            removeOps.insert(startConsumer);
        }
    }

    if (removeOps.empty()) {
        return;
    }

    auto startProducers = startTensor->GetProducers();
    for (auto* op : removeOps) {
        for (const auto& input : op->GetIOperands()) {
            input->RemoveConsumer(*op);
        }
        for (const auto& output : op->GetOOperands()) {
            output->RemoveProducer(*op);
        }
    }

    if (startTensor != endTensor) {
        for (auto* producer : startProducers) {
            producer->ReplaceOutputOperand(startTensor, endTensor);
            endTensor->AddProducer(*producer);
        }
        startTensor->GetProducers().clear();
        if (!startTensor->GetDynValidShape().empty()) {
            endTensor->UpdateDynValidShape(startTensor->GetDynValidShape());
        }
    }
    operationUpdated = true;
}

void RemoveRedundantOpUtils::ProcessPerfectMatch(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                                                 bool endTensorIsOutcast, bool& operationUpdated)
{
    if (!IsValidViewAssemble(startTensor, endTensor)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Not valid view-assemble-like case.");
        return;
    }
    if (endTensor->GetConsumers().size() == 0) {
        if (endTensorIsOutcast) {
            RemoveViewAssembleForOutcast(startTensor, endTensor, operationUpdated);
        }
    } else {
        auto consumers = endTensor->GetConsumers();
        for (auto& assembleConsumer : consumers) {
            assembleConsumer->ReplaceInput(startTensor, endTensor);
        }
        operationUpdated = true;
    }
}

bool RemoveRedundantOpUtils::IsNotSameViewInput(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const
{
    for (auto& assembleOp : endTensor->GetProducers()) {
        if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
            assembleOp->GetIOperands().empty()) {
            return true;
        }
        auto& tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        }
        auto& producerOps = tempTensor->GetProducers();
        for (auto& producerOp : producerOps) {
            if (producerOp == nullptr || producerOp->GetIOperands().empty() ||
                !IsViewLikeOpcode(producerOp->GetOpcode())) {
                return true;
            }
            auto& inTensor = producerOp->GetIOperands().front();
            if (inTensor != startTensor) {
                return true;
            }
        }
    }
    return false;
}

bool RemoveRedundantOpUtils::IsDataReplace(LogicalTensorPtr& endTensor) const
{
    for (auto& assembleOp : endTensor->GetProducers()) {
        if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
            assembleOp->GetIOperands().empty()) {
            return true;
        }
        auto& tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        }
        auto& viewOps = tempTensor->GetProducers();
        for (auto& viewOp : viewOps) {
            if (viewOp == nullptr || viewOp->GetIOperands().empty() || !IsViewLikeOpcode(viewOp->GetOpcode())) {
                return true;
            }
            auto viewOpAttribute = GetViewAttr(*viewOp);
            auto assembleOpAttribute = GetAssembleAttr(*assembleOp);
            if (viewOpAttribute == nullptr || assembleOpAttribute == nullptr) {
                return true;
            }
            auto viewOffset = viewOpAttribute->GetFrom();
            auto assembleOffset = assembleOpAttribute->GetToOffset();
            if (viewOffset != assembleOffset) {
                return true;
            }
        }
    }
    return false;
}

bool RemoveRedundantOpUtils::IsValidViewAssemble(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const
{
    bool isNotSameViewInput = IsNotSameViewInput(startTensor, endTensor);
    if (isNotSameViewInput) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Assemble-like output endTensor[%d] has different input except startTensor[%d].",
                          endTensor->magic, startTensor->magic);
        return false;
    }
    if (endTensor->shape.size() > startTensor->shape.size()) {
        return false;
    }
    for (size_t i = 0; i < endTensor->shape.size(); ++i) {
        if (endTensor->shape[i] > startTensor->shape[i]) {
            return false;
        }
    }
    bool isDataReplace = IsDataReplace(endTensor);
    if (isDataReplace) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Assemble-like output endTensor[%d] is replaced comparing with startTensor[%d].",
                          endTensor->magic, startTensor->magic);
        return false;
    }
    return true;
}

void RemoveRedundantOpUtils::CalculateViewOffset(Operation& op, LogicalTensorPtr& startTensor,
                                                 LogicalTensorPtr& endTensor, std::vector<int64_t>& newOffset,
                                                 std::vector<SymbolicScalar>& newDynOffset)
{
    for (size_t idx = 0; idx < op.iOperand[0]->offset.size(); idx++) {
        for (auto& consumerView : startTensor->GetConsumers()) {
            if (consumerView == nullptr || !IsViewLikeOpcode(consumerView->GetOpcode()) ||
                consumerView->GetOOperands().empty()) {
                continue;
            }
            auto& tempTensor = consumerView->GetOOperands().front();
            bool leadsToCurrentEndTensor = false;
            for (auto& consumerAssemble : tempTensor->GetConsumers()) {
                if (consumerAssemble == nullptr || !IsAssembleLikeOpcode(consumerAssemble->GetOpcode())) {
                    continue;
                }
                if (!consumerAssemble->GetOOperands().empty() &&
                    consumerAssemble->GetOOperands().front() == endTensor) {
                    leadsToCurrentEndTensor = true;
                    break;
                }
            }
            if (!leadsToCurrentEndTensor) {
                continue;
            }
            auto viewOpAttribute = GetViewAttr(*consumerView);
            if (viewOpAttribute != nullptr) {
                auto viewOffset = viewOpAttribute->GetFromOffset();
                auto viewDynOffset = viewOpAttribute->GetFromDynOffset();
                newOffset[idx] = std::min(newOffset[idx], viewOffset[idx]);
                if (idx < viewDynOffset.size()) {
                    newDynOffset[idx] = std::min(newDynOffset[idx], viewDynOffset[idx]);
                }
            }
        }
    }
}

void RemoveRedundantOpUtils::GenerateNewViewLike(Function& function, Operation& op, LogicalTensorPtr& startTensor,
                                                 LogicalTensorPtr& endTensor, Opcode newOpcode,
                                                 std::vector<Operation*>& newOps, bool& operationUpdated)
{
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    bool isExistingValidCase = IsValidViewAssemble(startTensor, endTensor);
    if (!isExistingValidCase && !TryCalculateTranslatedViewOffset(startTensor, endTensor, newOffset, newDynOffset)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Not valid view-assemble-like case.");
        return;
    }
    // endTensor 是 outcast（无消费者）时不处理，避免图断裂
    if (endTensor->GetConsumers().empty()) {
        return;
    }
    if (isExistingValidCase) {
        newOffset.assign(op.iOperand[0]->offset.size(), INT_MAX);
        newDynOffset.assign(op.iOperand[0]->offset.size(), INT_MAX);
        CalculateViewOffset(op, startTensor, endTensor, newOffset, newDynOffset);
        if (IsInitialDynOffset(newDynOffset)) {
            newDynOffset.clear();
        }
    }
    LogicalTensorPtr newViewTensor;
    std::vector<int64_t> curOffset(endTensor->shape.size(), 0);
    newViewTensor = irBuilder_.CreateTensorVar(endTensor->GetRawTensor(), curOffset, endTensor->shape,
                                               std::vector<SymbolicScalar>{});
    newViewTensor->SetMemoryTypeBoth(endTensor->GetMemoryTypeOriginal());
    auto viewAttribute = std::make_shared<ViewOpAttribute>(newOffset, newDynOffset, newViewTensor->GetDynValidShape());
    viewAttribute->SetToType(endTensor->GetMemoryTypeToBe());
    auto& newViewOp = PassOperationUtils::AddOperation(
        function, newOpcode, {startTensor}, {newViewTensor},
        [&viewAttribute](Operation& newOp) { newOp.SetOpAttribute(viewAttribute); });
    auto consumers = endTensor->GetConsumers();
    for (auto* consumer : consumers) {
        consumer->ReplaceInput(newViewTensor, endTensor);
    }
    newOps.push_back(&newViewOp);
    operationUpdated = true;
}

Status RemoveRedundantOpUtils::ProcessContractSliceImpl(Function& function, std::vector<Operation*>& newOps,
                                                        bool& operationUpdated)
{
    if (ProcessMultiContractSingleSlice(function, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessMultiContractSingleSlice failed.");
        return FAILED;
    }
    if (ProcessSingleContractMultiSlice(function, newOps, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessSingleContractMultiSlice failed.");
        return FAILED;
    }
    function.EraseOperations(true, false);
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessMultiContractSingleSlice(Function& function, bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_SLICE || op->GetIOperands().empty() ||
            op->GetOOperands().empty() || !IsSliceEqualSizeMove(*op)) {
            continue;
        }
        auto sliceInput = op->GetIOperands().front();
        auto sliceOutput = op->GetOOperands().front();
        auto sliceInputConsumers = sliceInput->GetConsumers();
        if (sliceInputConsumers.size() != 1 || *sliceInputConsumers.begin() != op) {
            continue;
        }
        auto producers = sliceInput->GetProducers();
        if (producers.size() <= 1) {
            continue;
        }
        if (SliceRequiresL1(*op)) {
            continue;
        }
        bool canProcess = true;
        std::set<Operation*> precedingSlices;
        for (auto* producer : producers) {
            if (producer == nullptr || producer->GetOpcode() != Opcode::OP_CONTRACT ||
                producer->GetIOperands().size() != 1 || GetAssembleAttr(*producer) == nullptr ||
                !IsTensorOffset32BAligned(producer->GetIOperands().front())) {
                canProcess = false;
                break;
            }
            // SplitLargeFanoutTensor may materialize a matmul result as contract-slice before this fan-in.
            // Folding the outer chain would transfer its copy-out requirement across that L0C boundary.
            if (IsMatmulBackedContractInput(producer->GetIOperands().front())) {
                canProcess = false;
                break;
            }
        }
        if (canProcess) {
            canProcess = CollectPrecedingSlicesForL1Transfer(*op, producers, precedingSlices);
        }
        if (!canProcess) {
            continue;
        }
        TransferL1Requirement(precedingSlices);
        for (auto* producer : producers) {
            if (SliceRequiresL1(*op)) {
                auto contractInput = producer->GetIOperands().front();
                if (contractInput != nullptr && contractInput->GetProducers().size() == 1) {
                    TransferL1CopyAttrs(*op, **contractInput->GetProducers().begin());
                }
                auto contractAttr = GetAssembleAttr(*producer);
                auto newAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(contractAttr->Clone());
                newAttr->SetFromType(MemoryType::MEM_L1);
                producer->SetOpAttribute(newAttr);
            }
            producer->SetOpCode(Opcode::OP_ASSEMBLE);
            producer->ReplaceOOperand(0, sliceOutput);
        }
        op->SetAsDeleted();
        operationUpdated = true;
    }
    return SUCCESS;
}

void RemoveRedundantOpUtils::GenerateContractSliceView(Function& function, Operation& sliceOp,
                                                       const LogicalTensorPtr& inputTensor,
                                                       const std::vector<int64_t>& fromOffset,
                                                       const std::vector<SymbolicScalar>& fromDynOffset,
                                                       std::vector<Operation*>& newOps, bool& operationUpdated)
{
    auto sliceAttr = GetViewAttr(sliceOp);
    if (sliceAttr == nullptr || sliceOp.GetOOperands().empty()) {
        return;
    }
    auto sliceOutput = sliceOp.GetOOperands().front();
    LogicalTensorPtr newViewTensor = sliceOutput;
    if (!sliceOutput->GetConsumers().empty()) {
        std::vector<int64_t> curOffset(sliceOutput->shape.size(), 0);
        newViewTensor = irBuilder_.CreateTensorVar(sliceOutput->GetRawTensor(), curOffset, sliceOutput->shape,
                                                   std::vector<SymbolicScalar>{});
        newViewTensor->SetMemoryTypeBoth(sliceOutput->GetMemoryTypeOriginal());
        ReplaceConsumers(sliceOutput, newViewTensor);
    }
    auto viewAttribute = std::make_shared<ViewOpAttribute>(fromOffset, fromDynOffset, sliceAttr->GetToDynValidShape());
    auto targetMemory = SliceRequiresL1(sliceOp) ? MemoryType::MEM_L1 : sliceOutput->GetMemoryTypeToBe();
    if (targetMemory == MemoryType::MEM_L1) {
        newViewTensor->SetMemoryTypeBoth(targetMemory, true);
    }
    viewAttribute->SetToType(targetMemory);
    auto& newViewOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_VIEW, {inputTensor}, {newViewTensor},
        [&viewAttribute](Operation& newOp) { newOp.SetOpAttribute(viewAttribute); });
    newOps.push_back(&newViewOp);
    operationUpdated = true;
}

Status RemoveRedundantOpUtils::ProcessSingleContractMultiSlice(Function& function, std::vector<Operation*>& newOps,
                                                               bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_CONTRACT || op->GetIOperands().empty() ||
            op->GetOOperands().empty() || GetAssembleAttr(*op) == nullptr) {
            continue;
        }
        auto contractInput = op->GetIOperands().front();
        auto contractOutput = op->GetOOperands().front();
        auto consumers = contractOutput->GetConsumers();
        if (consumers.empty()) {
            continue;
        }
        // A matmul result uses the L0C layout. Folding its contract-slice chain into views would make later
        // copy-outs interpret logical slice offsets as linear L0C pointer offsets.
        auto producers = contractInput->GetProducers();
        if (consumers.size() > 1 && producers.size() == 1 && *producers.begin() != nullptr &&
            IsMatmulOpcode((*producers.begin())->GetOpcode()) &&
            std::all_of(consumers.begin(), consumers.end(), [](const Operation* consumer) {
                return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE;
            })) {
            continue;
        }
        struct ConsumerRewriteInfo {
            Operation* consumer = nullptr;
            std::vector<int64_t> localOffset;
            std::vector<SymbolicScalar> localDynOffset;
            bool isFullSlice = false;
        };
        std::vector<ConsumerRewriteInfo> rewriteInfos;
        bool canProcess = true;
        bool keepMultiL1Slices = IsSingleContractWithMultipleL1SliceConsumers(*op) &&
                                 IsInputProducedBySingleSlice(contractInput);
        if (std::any_of(consumers.begin(), consumers.end(),
                        [](const Operation* consumer) {
                            return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE &&
                                   SliceRequiresL1(*consumer);
                        }) &&
            contractInput->GetProducers().size() != 1) {
            continue;
        }
        std::set<Operation*> precedingSlices;
        for (auto* consumer : consumers) {
            if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_SLICE || consumer->GetOOperands().empty()) {
                canProcess = false;
                break;
            }
            auto sliceAttr = GetViewAttr(*consumer);
            auto contractAttr = GetAssembleAttr(*op);
            std::vector<int64_t> localOffset;
            std::vector<SymbolicScalar> localDynOffset;
            if (sliceAttr == nullptr || contractAttr == nullptr ||
                !CalculateContractLocalSliceOffset(contractInput, consumer->GetOOperands().front(), *sliceAttr,
                                                   *contractAttr, localOffset, localDynOffset)) {
                canProcess = false;
                break;
            }
            bool isFullSlice = IsFullSliceOfContractInput(contractInput, consumer->GetOOperands().front(), localOffset,
                                                          localDynOffset);
            if (isFullSlice && !IsContractInputEqualSliceOutput(*op, *consumer)) {
                canProcess = false;
                break;
            }
            if (!keepMultiL1Slices &&
                !CollectPrecedingSlicesForL1Transfer(*consumer, std::vector<Operation*>{op}, precedingSlices)) {
                canProcess = false;
                break;
            }
            rewriteInfos.push_back({consumer, localOffset, localDynOffset, isFullSlice});
        }
        if (!canProcess) {
            continue;
        }
        if (keepMultiL1Slices) {
            for (auto& rewriteInfo : rewriteInfos) {
                auto* consumer = rewriteInfo.consumer;
                auto sliceAttr = GetViewAttr(*consumer);
                auto newAttr = std::dynamic_pointer_cast<ViewOpAttribute>(sliceAttr->Clone());
                newAttr->SetFromOffset(rewriteInfo.localOffset, rewriteInfo.localDynOffset);
                newAttr->SetToType(MemoryType::MEM_L1);
                consumer->SetOpAttribute(newAttr);
                consumer->GetOOperands().front()->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
                consumer->ReplaceInput(contractInput, contractOutput);
            }
            op->SetAsDeleted();
            operationUpdated = true;
            continue;
        }
        TransferL1Requirement(precedingSlices);
        for (auto& rewriteInfo : rewriteInfos) {
            auto* consumer = rewriteInfo.consumer;
            auto sliceOutput = consumer->GetOOperands().front();
            if (SliceRequiresL1(*consumer) && contractInput->GetProducers().size() == 1) {
                TransferL1CopyAttrs(*consumer, **contractInput->GetProducers().begin());
            }
            if (rewriteInfo.isFullSlice) {
                auto* precedingSlice = GetReplaceablePrecedingSlice(*op, *consumer, contractInput, sliceOutput);
                if (precedingSlice != nullptr) {
                    sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
                    precedingSlice->ReplaceOOperand(0, sliceOutput);
                } else {
                    ReplaceConsumers(sliceOutput, contractInput);
                }
            } else {
                GenerateContractSliceView(function, *consumer, contractInput, rewriteInfo.localOffset,
                                          rewriteInfo.localDynOffset, newOps, operationUpdated);
            }
            consumer->SetAsDeleted();
            operationUpdated = true;
        }
        op->SetAsDeleted();
        operationUpdated = true;
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
