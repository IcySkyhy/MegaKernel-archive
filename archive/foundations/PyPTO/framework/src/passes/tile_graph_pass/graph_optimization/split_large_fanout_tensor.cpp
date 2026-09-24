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
 * \file split_large_fanout_tensor.cpp
 * \brief
 */

#include <algorithm>
#include <numeric>
#include "split_large_fanout_tensor.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/remove_redundant_op_utils.h"
#include "interface/configs/config_manager_ng.h"

#define MODULE_NAME "SplitLargeFanoutTensor"

namespace npu::tile_fwk {
namespace {
Operation* FindAssembleFamilyProducer(const LogicalTensorPtr& input, const LogicalTensorPtr& output)
{
    Operation* result = nullptr;
    for (const auto& consumerOp : input->GetConsumers()) {
        if (consumerOp == nullptr || !IsAssembleLike(consumerOp->GetOpcode())) {
            continue;
        }
        if (std::find(consumerOp->GetOOperands().begin(), consumerOp->GetOOperands().end(), output) ==
            consumerOp->GetOOperands().end()) {
            continue;
        }
        if (result == nullptr || result->GetOpMagic() > consumerOp->GetOpMagic()) {
            result = consumerOp;
        }
    }
    return result;
}

Operation* FindViewFamilyProducer(const LogicalTensorPtr& output, const LogicalTensorPtr& expectedInput)
{
    if (!config::EnableSlice()) {
        return output->GetProducers().empty() ? nullptr : *output->GetProducers().begin();
    }
    Operation* result = nullptr;
    for (const auto& producerOp : output->GetProducers()) {
        if (producerOp == nullptr || !IsViewLike(producerOp->GetOpcode()) || producerOp->GetIOperands().empty()) {
            continue;
        }
        auto input = producerOp->GetIOperands().front();
        const bool sameInput = input == expectedInput;
        const bool sameRaw = input != nullptr && expectedInput != nullptr && input->tensor != nullptr &&
                             expectedInput->tensor != nullptr &&
                             input->tensor->rawmagic == expectedInput->tensor->rawmagic;
        if (!sameInput && !sameRaw) {
            continue;
        }
        if (result == nullptr || result->GetOpMagic() > producerOp->GetOpMagic()) {
            result = producerOp;
        }
    }
    return result;
}

Operation* FindAssembleFamilyConsumer(const LogicalTensorPtr& input)
{
    Operation* result = nullptr;
    for (const auto& consumerOp : input->GetConsumers()) {
        if (consumerOp == nullptr || !IsAssembleLike(consumerOp->GetOpcode())) {
            continue;
        }
        if (result == nullptr || result->GetOpMagic() > consumerOp->GetOpMagic()) {
            result = consumerOp;
        }
    }
    return result;
}

} // namespace

Status SplitLargeFanoutTensor::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start SplitLargeFanoutTensor.");
    Init();
    CollectLargeTensor(function);
    SplitLargeTensor(function);
    EraseRedundantAssembleOp(function);
    if (config::EnableSlice()) {
        bool contractSliceUpdated = false;
        if (RemoveRedundantOpUtils::ProcessContractSlice(function, addedOps_, contractSliceUpdated) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessContractSlice failed.");
            return FAILED;
        }
    } else {
        EraseRedundantViewOp(function);
    }
    Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
        return status;
    }
    if (!addedOps_.empty()) {
        if (InferShapeUtils::InferShape(function, addedOps_) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape for added ops failed.");
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End SplitLargeFanoutTensor.");
    return SUCCESS;
}

void SplitLargeFanoutTensor::Init()
{
    addedOps_.clear();
    toInfoMap_.clear();
    fromInfoMap_.clear();
    toInfoIndexMap_.clear();
    fromInfoIndexMap_.clear();
    largeTensors_.clear();
    toShapes_.clear();
    fromShapes_.clear();
    mixedConsumerTensors_.clear();
}

// 求最大公约数
int64_t SplitLargeFanoutTensor::GCD(int64_t x, int64_t y)
{
    int temp = 0;
    while (y != 0) {
        temp = x;
        x = y;
        y = temp % y;
    }
    return x;
}
// 求最小公倍数
Status SplitLargeFanoutTensor::LCM(int64_t x, int64_t y, int64_t& lcm)
{
    auto gcd = GCD(x, y);
    if (gcd == 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "gcd is 0; gcd can't be 0.");
        return FAILED;
    } else {
        lcm = x * y / gcd;
        return SUCCESS;
    }
}

// 求两个shape的最小公倍数shape
Status SplitLargeFanoutTensor::CalLcmShape(const Shape& toShape, const Shape& fromShape, Shape& lcmShape)
{
    if (toShape.size() != fromShape.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Incorrect shapes dim, toShape dim is %zu, fromShape dim is %zu; "
                          "Please make sure they are the same.",
                          toShape.size(), fromShape.size());
        return FAILED;
    }
    for (size_t i = 0; i < toShape.size(); i++) {
        if (LCM(toShape[i], fromShape[i], lcmShape[i]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Shape's dim %zu, %ld and %ld cal LCM failed; "
                              "LCM is calculated to be zero, please check.",
                              i, toShape[i], fromShape[i]);
            return FAILED;
        } else {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Shape's dim %zu, shape: %ld and %ld, LCM is %ld.", i, toShape[i],
                              fromShape[i], lcmShape[i]);
        }
    }
    return SUCCESS;
}

// 求两个shape的最大公约数shape
Status SplitLargeFanoutTensor::CalGcdShape(const Shape& toShape, const Shape& fromShape, Shape& lcmShape)
{
    if (toShape.size() != fromShape.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect shapes dim, toShape dim is %zu, fromShape dim is %zu.",
                          toShape.size(), fromShape.size());
        return FAILED;
    }
    for (size_t i = 0; i < toShape.size(); i++) {
        lcmShape[i] = GCD(toShape[i], fromShape[i]);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Shape's dim is %zu, toShape is %ld, fromShape is %ld, GCD is %ld.", i,
                          toShape[i], fromShape[i], lcmShape[i]);
    }
    return SUCCESS;
}

// 递归函数, 根据maxsShape和stepsShape生成offset
void SplitLargeFanoutTensor::GenerateOffset(const Shape& maxs, const Shape& steps, Shape& current,
                                            std::vector<Shape>& result, size_t dim)
{
    if (dim == maxs.size()) {
        // 生成一个offset
        Shape offset;
        for (size_t i = 0; i < current.size(); ++i) {
            offset.push_back(current[i]);
        }
        result.push_back(offset);
        return;
    }
    for (int val = 0; val < maxs[dim]; val += steps[dim]) {
        current[dim] = val;
        GenerateOffset(maxs, steps, current, result, dim + 1);
    }
}

// 收集BE_COVERED/PERFECTLY_MATCH lcmTile 的那些tile们, 分别更新到overlaps和dualOverlaps
void SplitLargeFanoutTensor::CollectOverlaps(const Shape& lcmTileShape, const Offset& lcmTileOffset,
                                             const OverlapSearchIndex& toIndex, const OverlapSearchIndex& fromIndex,
                                             LogicalTensors& overlaps, LogicalTensors& dualOverlaps)
{
    CollectCoveredTensors(lcmTileShape, lcmTileOffset, toIndex, overlaps);
    CollectCoveredTensors(lcmTileShape, lcmTileOffset, fromIndex, dualOverlaps);
}

void SplitLargeFanoutTensor::BuildOverlapSearchIndex(
    const std::vector<std::pair<LogicalTensorPtr, Offset>>& tensorInfos, OverlapSearchIndex& index) const
{
    index.tensorInfos = &tensorInfos;
    index.orderByDim.clear();
    if (tensorInfos.empty() || tensorInfos.front().second.empty()) {
        return;
    }
    size_t ndim = tensorInfos.front().second.size();
    index.orderByDim.resize(ndim);
    for (auto& order : index.orderByDim) {
        order.resize(tensorInfos.size());
        std::iota(order.begin(), order.end(), 0);
    }
    for (size_t dim = 0; dim < ndim; ++dim) {
        std::sort(index.orderByDim[dim].begin(), index.orderByDim[dim].end(), [&](size_t lhs, size_t rhs) {
            const auto& lhsInfo = tensorInfos[lhs];
            const auto& rhsInfo = tensorInfos[rhs];
            if (lhsInfo.second[dim] != rhsInfo.second[dim]) {
                return lhsInfo.second[dim] < rhsInfo.second[dim];
            }
            return lhsInfo.first->GetMagic() < rhsInfo.first->GetMagic();
        });
    }
}

bool SplitLargeFanoutTensor::IsTensorCoveredByTile(const Offset& tensorOffset, const Shape& tensorShape,
                                                   const Offset& tileOffset, const Shape& tileShape) const
{
    if (tensorOffset.size() != tensorShape.size() || tileOffset.size() != tileShape.size() ||
        tensorOffset.size() != tileOffset.size()) {
        return false;
    }
    for (size_t dim = 0; dim < tensorOffset.size(); ++dim) {
        if (tensorOffset[dim] < tileOffset[dim] ||
            tensorOffset[dim] + tensorShape[dim] > tileOffset[dim] + tileShape[dim]) {
            return false;
        }
    }
    return true;
}

/*
 * Collect tensors that are fully covered by the given lcm tile.
 *
 * The overlap index keeps tensor indexes sorted by offset for each dimension. This function first finds the
 * dimension that produces the smallest candidate range by binary search on [tileStart, tileEnd), then verifies
 * those candidates with a full multi-dimensional containment check. Matched indexes are sorted before emitting
 * tensors so the output follows the original tensorInfos order instead of the selected dimension order.
 */
void SplitLargeFanoutTensor::CollectCoveredTensors(const Shape& lcmTileShape, const Offset& lcmTileOffset,
                                                   const OverlapSearchIndex& index, LogicalTensors& tensors) const
{
    if (index.tensorInfos == nullptr || index.tensorInfos->empty() || index.orderByDim.empty()) {
        return;
    }
    const auto& tensorInfos = *index.tensorInfos;
    size_t bestDim = 0;
    size_t bestCandidateCount = tensorInfos.size();
    for (size_t dim = 0; dim < index.orderByDim.size(); ++dim) {
        const auto& order = index.orderByDim[dim];
        int64_t tileStart = lcmTileOffset[dim];
        int64_t tileEnd = lcmTileOffset[dim] + lcmTileShape[dim];
        auto lower = std::lower_bound(order.begin(), order.end(), tileStart, [&](size_t tensorIndex, int64_t start) {
            return tensorInfos[tensorIndex].second[dim] < start;
        });
        auto upper = std::lower_bound(lower, order.end(), tileEnd, [&](size_t tensorIndex, int64_t end) {
            return tensorInfos[tensorIndex].second[dim] < end;
        });
        size_t candidateCount = static_cast<size_t>(std::distance(lower, upper));
        if (candidateCount < bestCandidateCount) {
            bestCandidateCount = candidateCount;
            bestDim = dim;
        }
    }
    const auto& bestOrder = index.orderByDim[bestDim];
    int64_t tileStart = lcmTileOffset[bestDim];
    int64_t tileEnd = lcmTileOffset[bestDim] + lcmTileShape[bestDim];
    auto lower = std::lower_bound(
        bestOrder.begin(), bestOrder.end(), tileStart,
        [&](size_t tensorIndex, int64_t start) { return tensorInfos[tensorIndex].second[bestDim] < start; });
    auto upper = std::lower_bound(lower, bestOrder.end(), tileEnd, [&](size_t tensorIndex, int64_t end) {
        return tensorInfos[tensorIndex].second[bestDim] < end;
    });
    std::vector<size_t> coveredIndexes;
    coveredIndexes.reserve(static_cast<size_t>(std::distance(lower, upper)));
    for (auto iter = lower; iter != upper; ++iter) {
        const auto& tensorInfo = tensorInfos[*iter];
        if (IsTensorCoveredByTile(tensorInfo.second, tensorInfo.first->shape, lcmTileOffset, lcmTileShape)) {
            coveredIndexes.emplace_back(*iter);
        }
    }
    std::sort(coveredIndexes.begin(), coveredIndexes.end());
    for (auto tensorIndex : coveredIndexes) {
        tensors.push_back(tensorInfos[tensorIndex].first);
    }
}

// 根据原有assembleOp增加新的assembleOp。寻找原assembleOp时，由于tensor->assemble->largeTensor中assemble可以不唯一并指向其他tensor，
// 或assemble位置为其他种类op(op_view)。所以需要找到largeTensor的生产者op来确认。
Operation* AddNewAssembleOp(Function& function, LogicalTensorPtr overlap, LogicalTensorPtr largeTensor,
                            Offset lcmTileOffset, LogicalTensorPtr& newTensor)
{
    Operation* oldAssembleOp = FindAssembleFamilyProducer(overlap, largeTensor);
    if (oldAssembleOp == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "No valid assemble-family op found between tensor[%d] and tensor[%d], skip.",
                         overlap->GetMagic(), largeTensor->GetMagic());
        return nullptr;
    }
    auto oldAssembleOpAttr = dynamic_cast<AssembleOpAttribute*>(oldAssembleOp->GetOpAttribute().get());
    if (oldAssembleOpAttr == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "%s[%d] has no valid assemble-family attribute, skip.",
                         oldAssembleOp->GetOpcodeStr().c_str(), oldAssembleOp->GetOpMagic());
        return nullptr;
    }
    Shape newAssembleOffset = oldAssembleOpAttr->GetToOffset();
    if (newAssembleOffset.size() != lcmTileOffset.size()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Assemble offset dim %zu mismatches lcm tile offset dim %zu, skip.",
                         newAssembleOffset.size(), lcmTileOffset.size());
        return nullptr;
    }
    for (size_t j = 0; j < newAssembleOffset.size(); j++) {
        newAssembleOffset[j] -= lcmTileOffset[j];
    }
    if (config::EnableSlice()) {
        auto newAssembleOp = AssembleOp{
            oldAssembleOpAttr->GetFrom(), newAssembleOffset, overlap, newTensor, oldAssembleOp,
            oldAssembleOp->GetOpcode()};
        return &GraphUtils::AddAssembleOperation(function, newAssembleOp);
    }
    auto newAssembleOp = AssembleOp{
        overlap->GetMemoryTypeOriginal(), newAssembleOffset, overlap, newTensor, nullptr, Opcode::OP_ASSEMBLE};
    return &GraphUtils::AddAssembleOperation(function, newAssembleOp);
}

// 对于一对一、一对多场景创建新的AssembleOp和Tensor
void SplitLargeFanoutTensor::CreateOpFor1toM(Function& function, const LogicalTensorPtr& largeTensor,
                                             const Shape& lcmTileShape, const Offset& lcmTileOffset,
                                             const LogicalTensors& overlaps, const LogicalTensors& dualOverlaps)
{
    if (overlaps.empty()) {
        return;
    }
    auto overlap = overlaps[0];
    if (!config::EnableSlice()) {
        for (const auto& dualOverlap : dualOverlaps) {
            auto viewOp = *dualOverlap->GetProducers().begin();
            if (viewOp->GetIOperands().empty() ||
                viewOp->GetIOperands().front()->tensor->rawmagic != largeTensor->tensor->rawmagic) {
                APASS_LOG_DEBUG_F(Elements::Tensor,
                                  "ViewOp[%d]'s input has been replaced, don't deal with this ViewOp.",
                                  viewOp->GetOpMagic());
                continue;
            }
            auto newTensor = irBuilder_.CreateTensorVar(largeTensor->Datatype(), lcmTileShape,
                                                        std::vector<SymbolicScalar>{}, largeTensor->Format());
            if (AddNewAssembleOp(function, overlap, largeTensor, lcmTileOffset, newTensor) == nullptr) {
                continue;
            }
            auto assembleOp = *newTensor->GetProducers().begin();
            addedOps_.push_back(assembleOp);
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "In one-to-multiple situation, create an AssembleOp[%d], input is a "
                              "overlap[%d], output is a newTensor[%d].",
                              assembleOp->GetOpMagic(), overlap->GetMagic(), newTensor->GetMagic());
            auto viewOpAttr = dynamic_cast<ViewOpAttribute*>(viewOp->GetOpAttribute().get());
            Shape newViewOffset = viewOpAttr->GetFromOffset();
            for (size_t j = 0; j < newViewOffset.size(); j++) {
                newViewOffset[j] -= lcmTileOffset[j];
            }
            viewOpAttr->SetFromOffset(newViewOffset);
            GraphUtils::UpdateViewAttr(function, *viewOp);
            viewOp->ReplaceInput(newTensor, largeTensor);
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "In one-to-multiple situation, "
                              "ViewOp[%d]'s input[%d] has been replaced to newTensor[%d].",
                              viewOp->GetOpMagic(), largeTensor->GetMagic(), newTensor->GetMagic());
        }
        return;
    }
    auto newTensor = irBuilder_.CreateTensorVar(largeTensor->Datatype(), lcmTileShape, std::vector<SymbolicScalar>{},
                                                largeTensor->Format());
    auto assembleOp = AddNewAssembleOp(function, overlap, largeTensor, lcmTileOffset, newTensor);
    if (assembleOp == nullptr) {
        return;
    }
    addedOps_.push_back(assembleOp);
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "In one-to-multiple situation, create a %s[%d], input is a "
                      "overlap[%d], output is a newTensor[%d].",
                      assembleOp->GetOpcodeStr().c_str(), assembleOp->GetOpMagic(), overlap->GetMagic(),
                      newTensor->GetMagic());
    for (const auto& dualOverlap : dualOverlaps) {
        auto viewOp = FindViewFamilyProducer(dualOverlap, largeTensor);
        if (viewOp == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "No view-family producer found for dualOverlap[%d], skip.",
                              dualOverlap->GetMagic());
            continue;
        }
        if (viewOp->GetIOperands().empty() ||
            viewOp->GetIOperands().front()->tensor->rawmagic != largeTensor->tensor->rawmagic) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "%s[%d]'s input has been replaced, skip.",
                              viewOp->GetOpcodeStr().c_str(), viewOp->GetOpMagic());
            continue;
        }
        auto viewOpAttr = dynamic_cast<ViewOpAttribute*>(viewOp->GetOpAttribute().get());
        if (viewOpAttr == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "%s[%d] has no valid view-family attribute, skip.",
                             viewOp->GetOpcodeStr().c_str(), viewOp->GetOpMagic());
            continue;
        }
        Shape newViewOffset = viewOpAttr->GetFromOffset();
        for (size_t j = 0; j < newViewOffset.size(); j++) {
            newViewOffset[j] -= lcmTileOffset[j];
        }
        viewOpAttr->SetFromOffset(newViewOffset);
        GraphUtils::UpdateViewAttr(function, *viewOp);
        viewOp->ReplaceInput(newTensor, largeTensor);
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "In one-to-multiple situation, "
                          "%s[%d]'s input[%d] has been replaced to newTensor[%d].",
                          viewOp->GetOpcodeStr().c_str(), viewOp->GetOpMagic(), largeTensor->GetMagic(),
                          newTensor->GetMagic());
    }
}

void SplitLargeFanoutTensor::ExtractDualOverlapTiles(Function& function, const LogicalTensorPtr& largeTensor,
                                                     const LogicalTensors& dualOverlaps,
                                                     std::vector<std::pair<Offset, Shape>>& dualOverlapTileInfos,
                                                     LogicalTensors& filteredDualOverlaps)
{
    (void)function;
    for (const auto& dualOverlap : dualOverlaps) {
        Offset dualOverlapOffset;
        for (const auto& producerOp : dualOverlap->GetProducers()) {
            if (producerOp != nullptr && IsViewLike(producerOp->GetOpcode()) && !producerOp->GetIOperands().empty() &&
                producerOp->GetIOperands().front() == largeTensor) {
                auto opAttr = dynamic_cast<ViewOpAttribute*>(producerOp->GetOpAttribute().get());
                if (opAttr != nullptr) {
                    dualOverlapOffset = opAttr->GetFromOffset();
                    break;
                }
            }
        }
        if (dualOverlapOffset.size() == 0) {
            continue;
        }
        dualOverlapTileInfos.emplace_back(dualOverlapOffset, dualOverlap->shape);
        filteredDualOverlaps.push_back(dualOverlap);
    }
}

bool SplitLargeFanoutTensor::HasIntersectionWithAnyDualOverlap(
    const Offset& overlapOffset, const Shape& overlapShape,
    const std::vector<std::pair<Offset, Shape>>& dualOverlapTileInfos)
{
    for (const auto& [dualOffset, dualShape] : dualOverlapTileInfos) {
        auto status = CalcOverlapByOffsetShape(overlapOffset, overlapShape, dualOffset, dualShape);
        if (status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH ||
            status == OverlapStatus::PARTIAL_OVERLAP || status == OverlapStatus::COVERED) {
            return true;
        }
    }
    return false;
}

void SplitLargeFanoutTensor::FilterOverlaps(Function& function, const LogicalTensorPtr& largeTensor,
                                            LogicalTensors& overlaps, const LogicalTensors& dualOverlaps)
{
    (void)function;
    LogicalTensors filteredOverlaps;
    LogicalTensors filteredDualOverlaps;
    std::vector<std::pair<Offset, Shape>> dualOverlapTileInfos;
    ExtractDualOverlapTiles(function, largeTensor, dualOverlaps, dualOverlapTileInfos, filteredDualOverlaps);
    for (const auto& overlap : overlaps) {
        Offset overlapOffset;
        for (const auto& consumerOp : overlap->GetConsumers()) {
            if (consumerOp != nullptr && IsAssembleLike(consumerOp->GetOpcode()) &&
                !consumerOp->GetOOperands().empty() && consumerOp->GetOOperands().front() == largeTensor) {
                auto opAttr = dynamic_cast<AssembleOpAttribute*>(consumerOp->GetOpAttribute().get());
                if (opAttr != nullptr) {
                    overlapOffset = opAttr->GetToOffset();
                    break;
                }
            }
        }
        if (overlapOffset.size() == 0) {
            return;
        }
        if (HasIntersectionWithAnyDualOverlap(overlapOffset, overlap->shape, dualOverlapTileInfos)) {
            filteredOverlaps.push_back(overlap);
        }
    }
    overlaps = filteredOverlaps;
}

// 对于多对一、多对多场景创建新的AssembleOp和Tensor
void SplitLargeFanoutTensor::CreateOpForMtoM(Function& function, const LogicalTensorPtr& largeTensor,
                                             const Shape& lcmTileShape, const Offset& lcmTileOffset,
                                             const LogicalTensors& overlaps, const LogicalTensors& dualOverlaps)
{
    auto newTensor = irBuilder_.CreateTensorVar(largeTensor->Datatype(), lcmTileShape, std::vector<SymbolicScalar>{},
                                                largeTensor->Format());
    for (const auto& overlap : overlaps) {
        auto assembleOp = AddNewAssembleOp(function, overlap, largeTensor, lcmTileOffset, newTensor);
        if (assembleOp == nullptr) {
            continue;
        }
        addedOps_.push_back(assembleOp);
        APASS_LOG_INFO_F(Elements::Operation,
                         "In multiple-to-multiple situation, create a %s[%d], "
                         "input is a overlap[%d], output is a newTensor[%d].",
                         assembleOp->GetOpcodeStr().c_str(), assembleOp->GetOpMagic(), overlap->GetMagic(),
                         newTensor->GetMagic());
    }
    for (const auto& dualOverlap : dualOverlaps) {
        auto viewOp = FindViewFamilyProducer(dualOverlap, largeTensor);
        if (viewOp == nullptr) {
            APASS_LOG_INFO_F(Elements::Operation, "No view-family producer found for dualOverlap[%d], skip.",
                             dualOverlap->GetMagic());
            continue;
        }
        if (viewOp->GetIOperands().front()->tensor->rawmagic != largeTensor->tensor->rawmagic) {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]'s input has been replaced, skip.",
                             viewOp->GetOpcodeStr().c_str(), viewOp->GetOpMagic());
        } else {
            auto viewOpAttr = dynamic_cast<ViewOpAttribute*>(viewOp->GetOpAttribute().get());
            if (viewOpAttr == nullptr) {
                APASS_LOG_WARN_F(Elements::Operation, "%s[%d] has no valid view-family attribute, skip.",
                                 viewOp->GetOpcodeStr().c_str(), viewOp->GetOpMagic());
                continue;
            }
            Shape newViewOffset = viewOpAttr->GetFromOffset();
            for (size_t j = 0; j < newViewOffset.size(); j++) {
                newViewOffset[j] -= lcmTileOffset[j];
            }
            viewOpAttr->SetFromOffset(newViewOffset);
            GraphUtils::UpdateViewAttr(function, *viewOp);
            viewOp->ReplaceInput(newTensor, largeTensor);
            APASS_LOG_INFO_F(Elements::Operation,
                             "In multiple-to-multiple situation, %s[%d]'s input[%d] has been "
                             "replaced to newTensor[%d].",
                             viewOp->GetOpcodeStr().c_str(), viewOp->GetOpMagic(), largeTensor->GetMagic(),
                             newTensor->GetMagic());
        }
    }
    // 进一步拆分, 未来通过旋钮的方式适时打开
    if (enableMoreSplit_) {
        MoreSplit(function, largeTensor, overlaps, dualOverlaps);
    }
}

void SplitLargeFanoutTensor::MoreSplit(Function& function, const LogicalTensorPtr& largeTensor,
                                       const LogicalTensors& overlaps, const LogicalTensors& dualOverlaps)
{
    int rawMagic = largeTensor->tensor->rawmagic;
    for (const auto& dualOverlap : dualOverlaps) {
        // 如果该dualOverlap已经被进一步拆分, 跳过(进一步拆分的特征是dualOverlap的生产者全是Assemble)
        bool isMoreSplit = true;
        for (const auto& producer : dualOverlap->GetProducers()) {
            if (producer == nullptr || producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                isMoreSplit = false;
            }
        }
        if (isMoreSplit) {
            continue;
        }
        // 否则, dualOverlap的生产者全是View, 不存在其他场景
        auto toShape = overlaps.front()->shape;
        auto fromShape = dualOverlap->shape;
        Shape gcdShape(toShape.size(), 0);
        CalGcdShape(toShape, fromShape, gcdShape);
        std::vector<Shape> gcdTileOffsets;
        Shape current(gcdShape.size());
        GenerateOffset(dualOverlap->shape, gcdShape, current, gcdTileOffsets, 0);
        auto viewOp = *dualOverlap->GetProducers().begin();
        auto opAttr = dynamic_cast<ViewOpAttribute*>(viewOp->GetOpAttribute().get());
        if (opAttr == nullptr) {
            continue;
        }
        auto viewOpOffset = opAttr->GetFromOffset();
        // 断开viewOp--> tensor: 将tensor的生产者删除viewOp, 将viewOp的输出删除tensor
        dualOverlap->RemoveProducer(viewOp);
        viewOp->GetOOperands().erase(viewOp->GetOOperands().begin(), viewOp->GetOOperands().end());
        auto fromTensorInfos = fromInfoMap_[rawMagic];
        for (const auto& fromTensorInfo : fromTensorInfos) {
            if (dualOverlap == fromTensorInfo.first) {
                viewOpOffset = fromTensorInfo.second;
            }
        }
        CreateOpForMoreSplit(function, largeTensor, overlaps, gcdShape, dualOverlap, gcdTileOffsets, viewOpOffset);
    }
}

void SplitLargeFanoutTensor::FindOverlapAndCreateViewOp(Function& function, const LogicalTensors& overlaps,
                                                        const LogicalTensorPtr& newGcdTensor,
                                                        const Shape& gcdTileOffsetForLarge, Shape& newViewOffset)
{
    for (const auto& overlap : overlaps) {
        auto oldAssembleOp = FindAssembleFamilyConsumer(overlap);
        if (oldAssembleOp == nullptr) {
            APASS_LOG_WARN_F(Elements::Tensor, "No assemble-family consumer found for overlap[%d]; Please check.",
                             overlap->GetMagic());
            continue;
        }
        auto oldAssembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(oldAssembleOp->GetOpAttribute());
        if (!oldAssembleOpAttr) {
            APASS_LOG_WARN_F(Elements::Tensor, "%s[%d] has no valid assembleOpAttribute; Please check.",
                             oldAssembleOp->GetOpcodeStr().c_str(), oldAssembleOp->GetOpMagic());
            continue;
        }
        auto oldAssembleOffset = oldAssembleOpAttr->GetToOffset();
        auto status = CalcOverlapByOffsetShape(gcdTileOffsetForLarge, newGcdTensor->shape, oldAssembleOffset,
                                               overlap->shape);
        if (status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH) {
            for (size_t j = 0; j < newViewOffset.size(); j++) {
                newViewOffset[j] -= oldAssembleOffset[j];
            }
            auto& newViewOp = PassOperationUtils::AddOperation(
                function, Opcode::OP_VIEW, {overlap}, {newGcdTensor}, [&newViewOffset, &overlap](Operation& op) {
                    op.SetOpAttribute(
                        std::make_shared<ViewOpAttribute>(newViewOffset, overlap->GetMemoryTypeOriginal()));
                });
            addedOps_.push_back(&newViewOp);
            APASS_LOG_INFO_F(Elements::Operation,
                             "For more split situation, create an ViewOp[%d], input is a "
                             "overlapGcdTile[%d], output is a newGcdTensor[%d].",
                             newViewOp.GetOpMagic(), overlap->GetMagic(), newGcdTensor->GetMagic());
        }
    }
}

void SplitLargeFanoutTensor::CreateOpForMoreSplit(Function& function, const LogicalTensorPtr& largeTensor,
                                                  const LogicalTensors& overlaps, const Shape& gcdShape,
                                                  const LogicalTensorPtr& dualOverlap,
                                                  const std::vector<Shape>& gcdTileOffsets, const Offset& viewOpOffset)
{
    for (auto& gcdTileOffset : gcdTileOffsets) {
        auto newGcdTensor = irBuilder_.CreateTensorVar(largeTensor->Datatype(), gcdShape, std::vector<SymbolicScalar>{},
                                                       largeTensor->Format());
        auto& newAssembleOp = PassOperationUtils::AddOperation(
            function, Opcode::OP_ASSEMBLE, {newGcdTensor}, {dualOverlap},
            [&largeTensor, &gcdTileOffset](Operation& op) {
                op.SetOpAttribute(
                    std::make_shared<AssembleOpAttribute>(largeTensor->GetMemoryTypeOriginal(), gcdTileOffset));
            });
        addedOps_.push_back(&newAssembleOp);
        APASS_LOG_INFO_F(Elements::Operation,
                         "For more split situation, create an AssembleOp[%d], input is a newGcdTensor[%d], "
                         "output is a dualOverlap[%d].",
                         newAssembleOp.GetOpMagic(), newGcdTensor->GetMagic(), dualOverlap->GetMagic());
        Shape newViewOffset = gcdTileOffset;
        for (size_t j = 0; j < newViewOffset.size(); j++) {
            newViewOffset[j] += viewOpOffset[j];
        }
        Shape gcdTileOffsetForLarge = gcdTileOffset;
        for (size_t j = 0; j < gcdTileOffsetForLarge.size(); j++) {
            gcdTileOffsetForLarge[j] += viewOpOffset[j];
        }
        FindOverlapAndCreateViewOp(function, overlaps, newGcdTensor, gcdTileOffsetForLarge, newViewOffset);
    }
}

void SplitLargeFanoutTensor::CollectLargeTensorToInfo(const LogicalTensorPtr& largeTensor)
{
    int rawMagic = largeTensor->tensor->rawmagic;
    for (const auto& assembleOp : largeTensor->GetProducers()) {
        if (assembleOp == nullptr || !IsAssembleLike(assembleOp->GetOpcode()) || assembleOp->GetIOperands().empty() ||
            assembleOp->GetIOperands().front() == nullptr) {
            continue;
        }
        // 收集overlaps
        auto input = assembleOp->GetIOperands().front();
        if (toInfoMap_.count(rawMagic) == 0) {
            toInfoMap_.insert({rawMagic, {}});
        }
        auto opAttr = dynamic_cast<AssembleOpAttribute*>(assembleOp->GetOpAttribute().get());
        if (opAttr != nullptr) {
            toInfoMap_[rawMagic].emplace_back(input, opAttr->GetToOffset());
        }
        // 收集overlaps的shape
        if (toShapes_.count(largeTensor) == 0) {
            toShapes_.insert({largeTensor, {}});
        }
        toShapes_[largeTensor].insert(input->shape);
    }
}

void SplitLargeFanoutTensor::CollectLargeTensorFromInfo(const LogicalTensorPtr& largeTensor)
{
    int rawMagic = largeTensor->tensor->rawmagic;
    for (const auto& viewOp : largeTensor->GetConsumers()) {
        if (viewOp == nullptr || !IsViewLike(viewOp->GetOpcode()) || viewOp->GetOOperands().empty() ||
            viewOp->GetOOperands().front() == nullptr) {
            continue;
        }
        // 收集outputs
        auto output = viewOp->GetOOperands().front();
        if (fromInfoMap_.count(rawMagic) == 0) {
            fromInfoMap_.insert({rawMagic, {}});
        }
        auto opAttr = dynamic_cast<ViewOpAttribute*>(viewOp->GetOpAttribute().get());
        if (opAttr == nullptr) { // 不可能为空，否则有问题
            continue;
        }
        if (!opAttr->GetFromDynOffset().empty()) {
            bool hasDynOffset = false;
            for (auto dynOffset : opAttr->GetFromDynOffset()) {
                if (!dynOffset.ConcreteValid()) {
                    hasDynOffset = true;
                    break;
                }
            }
            if (hasDynOffset) { // 当View存在动态offset时，无法进行split，因为不知道会用哪些Assemble
                continue;
            }
        }
        fromInfoMap_[rawMagic].emplace_back(output, opAttr->GetFromOffset());
        // 收集outputs的shape
        if (fromShapes_.count(largeTensor) == 0) {
            fromShapes_.insert({largeTensor, {}});
        }
        fromShapes_[largeTensor].insert(output->shape);
    }
}

// 遍历所有的tensor, 对前序为Assemble后序为View的大Tensor进行拆分
void SplitLargeFanoutTensor::CollectLargeTensor(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "---> CollectLargeTensor.");
    std::unordered_set<int> visited;
    auto operations = function.Operations(false);
    visited.reserve(operations.size());
    for (auto& op : operations) {
        if (!IsAssembleLike(op.GetOpcode())) {
            continue;
        }
        for (const auto& logicalTensor : op.GetOOperands()) {
            if (logicalTensor == nullptr || !visited.emplace(logicalTensor->GetMagic()).second ||
                logicalTensor->GetProducers().empty() || logicalTensor->GetConsumers().empty()) {
                continue;
            }
            bool allProducersAssemble = std::all_of(
                logicalTensor->GetProducers().begin(), logicalTensor->GetProducers().end(),
                [](Operation* producer) { return producer != nullptr && IsAssembleLike(producer->GetOpcode()); });
            bool hasAnyViewConsumer = false;
            bool allConsumersView = true;
            for (const auto& consumer : logicalTensor->GetConsumers()) {
                if (consumer == nullptr) {
                    continue;
                }
                if (IsViewLike(consumer->GetOpcode())) {
                    hasAnyViewConsumer = true;
                } else {
                    allConsumersView = false;
                }
            }
            if (!allProducersAssemble || !hasAnyViewConsumer) {
                continue;
            }
            if (!allConsumersView) {
                mixedConsumerTensors_.insert(logicalTensor->tensor->rawmagic);
            }
            largeTensors_.push_back(logicalTensor);
            CollectLargeTensorToInfo(logicalTensor);
            CollectLargeTensorFromInfo(logicalTensor);
            APASS_LOG_DEBUG_F(Elements::Tensor, "Large tensor magic is %d.", logicalTensor->GetMagic());
        }
    }
    for (const auto& largeTensor : largeTensors_) {
        int rawMagic = largeTensor->tensor->rawmagic;
        BuildOverlapSearchIndex(toInfoMap_[rawMagic], toInfoIndexMap_[rawMagic]);
        BuildOverlapSearchIndex(fromInfoMap_[rawMagic], fromInfoIndexMap_[rawMagic]);
    }
}

bool SplitLargeFanoutTensor::IsBeCovered(Function& function, const LogicalTensorPtr& largeTensor,
                                         const std::vector<std::pair<LogicalTensorPtr, Offset>>& toTensorInfos)
{
    (void)function;
    for (const auto& toTensorInfo : toTensorInfos) {
        auto status = CalcOverlapByOffsetShape(toTensorInfo.second, toTensorInfo.first->shape, largeTensor->offset,
                                               largeTensor->shape);
        if (!(status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH)) {
            return false;
        }
    }
    return true;
}

bool SplitLargeFanoutTensor::HasDuplicateToTile(const std::vector<std::pair<LogicalTensorPtr, Offset>>& toTensorInfos)
{
    std::map<Offset, int> countMap;
    for (const auto& toTensorInfo : toTensorInfos) {
        countMap[toTensorInfo.second]++;
    }
    for (const auto& pair : countMap) {
        if (pair.second > 1) {
            return true;
        }
    }
    return false;
}

void insertShapeIfNotDup(std::multiset<Shape, ShapeComparator>& set, const Shape& shape)
{
    auto range = set.equal_range(shape);
    for (auto it = range.first; it != range.second; ++it) {
        if (*it == shape) {
            return;
        }
    }
    set.insert(shape);
}

// 遍历所有的大tensor, 对前后不同的tileShape计算lcmShape, 并尝试拆分
void SplitLargeFanoutTensor::SplitLargeTensor(Function& function)
{
    for (const auto& largeTensor : largeTensors_) {
        std::multiset<Shape, ShapeComparator> lcmShapes;
        int rawMagic = largeTensor->tensor->rawmagic;
        // 验证Assemble成LargeTensor的tileTensor们需要包含于LargeTensor
        if (!IsBeCovered(function, largeTensor, toInfoMap_[rawMagic])) {
            continue;
        }
        // 验证Assemble成LargeTensor的tileTensor们(的Offset)需要彼此不同
        if (HasDuplicateToTile(toInfoMap_[rawMagic])) {
            continue;
        }
        for (const auto& toShape : toShapes_[largeTensor]) {
            for (const auto& fromShape : fromShapes_[largeTensor]) {
                Shape lcmShape(toShape.size(), 0);
                if (CalLcmShape(toShape, fromShape, lcmShape) != SUCCESS) {
                    APASS_LOG_INFO_F(Elements::Tensor, "Calculate LCM shape failed, don't cal LcmShape.");
                    continue;
                }
                // 当lcmTile的某一维度大于largeTensor时，修改为与largeTensor相等
                for (size_t i = 0; i < lcmShape.size(); i++) {
                    lcmShape[i] = std::min(lcmShape[i], largeTensor->GetShape()[i]);
                }
                // 当lcmTile的每个维度都等于largeTensor时, 仍会聚合到同样大小的Tensor
                // 因此只要有多于一个生产者就不做处理，当只有一个生产者时该生产者可能冗余，仍需处理
                if (lcmShape == largeTensor->GetShape() && largeTensor->GetProducers().size() > 1) {
                    APASS_LOG_INFO_F(Elements::Tensor,
                                     "Skip SplitLargeTensor for magic[%d] since shape to assemble (lcmShape) equals "
                                     "the largeTensor's shape and largeTensor has more than one producer assemble.",
                                     largeTensor->GetMagic());
                    continue;
                }
                insertShapeIfNotDup(lcmShapes, lcmShape);
            }
        }
        for (const auto& lcmShape : lcmShapes) {
            // 当lcmTile的shape小于largeTensor时, 开始尝试拆分
            APASS_LOG_DEBUG_F(Elements::Tensor, "Try to split with shape %s, large tensor magic is %d.",
                              CommonUtils::ContainerToStr(lcmShape).c_str(), largeTensor->GetMagic());
            TryToSplitLargeTensor(function, lcmShape, largeTensor);
        }
    }
}

/*
 * Build candidate lcm tile offsets from existing assemble/write and view/read boundaries.
 *
 * For each dimension, this function collects every tile start offset and every internal tile end offset from both
 * the assemble side and the view side. The Cartesian product of these per-dimension boundaries forms the candidate
 * offsets. The candidates are intentionally conservative: ProcessTileSplit() later checks whether each lcm tile
 * actually has matching assemble/view tensors and is fully covered by assemble inputs.
 */
void SplitLargeFanoutTensor::GetOffsets(std::set<Shape, ShapeDimComparator>& tileOffsets, const Shape& lcmShape,
                                        const LogicalTensorPtr& largeTensor)
{
    auto ndim = lcmShape.size();
    std::vector<std::set<int64_t>> boundaryPerDim(ndim); // Tile boundaries collected for each dimension.
    // Collect boundaries from both the assemble side (toInfoMap_) and the view side (fromInfoMap_).
    auto toIt = toInfoMap_.find(largeTensor->tensor->rawmagic);
    if (toIt != toInfoMap_.end()) {
        for (const auto& [tensor, offset] : toIt->second) {
            for (size_t d = 0; d < ndim; ++d) {
                boundaryPerDim[d].insert(offset[d]);
                if (offset[d] + tensor->shape[d] < largeTensor->shape[d]) {
                    boundaryPerDim[d].insert(offset[d] + tensor->shape[d]);
                }
            }
        }
    }
    auto fromIt = fromInfoMap_.find(largeTensor->tensor->rawmagic);
    if (fromIt != fromInfoMap_.end()) {
        for (const auto& [tensor, offset] : fromIt->second) {
            for (size_t d = 0; d < ndim; ++d) {
                boundaryPerDim[d].insert(offset[d]);
                if (offset[d] + tensor->shape[d] < largeTensor->shape[d]) {
                    boundaryPerDim[d].insert(offset[d] + tensor->shape[d]);
                }
            }
        }
    }
    // Generate every multidimensional offset candidate by Cartesian product.
    std::vector<Shape> results;
    results.push_back(Shape(ndim, 0));
    for (size_t d = 0; d < ndim; ++d) {
        std::vector<Shape> expanded;
        for (const auto& partial : results) {
            for (auto boundVal : boundaryPerDim[d]) {
                Shape newOffset = partial;
                newOffset[d] = boundVal;
                expanded.push_back(std::move(newOffset));
            }
        }
        results = std::move(expanded);
    }
    for (auto& offset : results) {
        tileOffsets.insert(std::move(offset));
    }
    if (!tileOffsets.empty()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Skip offset processing for large tensor [%d] due to empty offsets.",
                          largeTensor->GetMagic());
    }
}

Shape SplitLargeFanoutTensor::AdjustLcmTileShapeForTailBlock(const Shape& lcmShape, const Shape& tileOffset,
                                                             const LogicalTensorPtr& largeTensor)
{
    auto lcmTileShape = lcmShape;
    for (size_t i = 0; i < lcmShape.size(); i++) {
        if (tileOffset[i] + lcmTileShape[i] > largeTensor->shape[i]) {
            lcmTileShape[i] = largeTensor->shape[i] - tileOffset[i];
        }
    }
    return lcmTileShape;
}

bool SplitLargeFanoutTensor::CheckOverlapCoverage(const LogicalTensors& overlaps, const Shape& lcmTileShape)
{
    auto [lcmTileArea, lcmOverflow] = CommonUtils::SafeMultiplyShape(lcmTileShape);
    if (lcmOverflow || lcmTileArea == -1) {
        return false;
    }
    int64_t overlapTotalArea = 0;
    for (const auto& overlap : overlaps) {
        auto [area, overflow] = CommonUtils::SafeMultiplyShape(overlap->shape);
        if (overflow || area == -1 || overlapTotalArea > INT64_MAX - area) {
            return false;
        }
        overlapTotalArea += area;
    }
    return overlapTotalArea == lcmTileArea;
}

void SplitLargeFanoutTensor::ProcessTileSplit(Function& function, const LogicalTensorPtr& largeTensor,
                                              const Shape& lcmTileShape, const Shape& tileOffset,
                                              const OverlapSearchIndex& toIndex, const OverlapSearchIndex& fromIndex,
                                              LogicalTensors& overlaps, LogicalTensors& dualOverlaps)
{
    CollectOverlaps(lcmTileShape, tileOffset, toIndex, fromIndex, overlaps, dualOverlaps);
    if (overlaps.size() == 0 || dualOverlaps.size() == 0) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Split large tensor miss, this lcmTile does NOT have both overlaps([%zu]) "
                          "and dualOverlaps([%zu]) simultaneously.",
                          overlaps.size(), dualOverlaps.size());
        return;
    }
    if (!CheckOverlapCoverage(overlaps, lcmTileShape)) {
        APASS_LOG_DEBUG_F(
            Elements::Tensor,
            "Split large tensor miss, this lcmTile(shape %s, offset %s) of largeTensor %d is not filled up by all "
            "collected overlaps.",
            CommonUtils::ContainerToStr(lcmTileShape).c_str(), CommonUtils::ContainerToStr(tileOffset).c_str(),
            largeTensor->GetMagic());
        return;
    }
    APASS_LOG_DEBUG_F(
        Elements::Tensor,
        "Split large tensor hit, this lcmTile(shape %s, offset %s) has [%zu] overlaps and [%zu] dualOverlaps.",
        CommonUtils::ContainerToStr(lcmTileShape).c_str(), CommonUtils::ContainerToStr(tileOffset).c_str(),
        overlaps.size(), dualOverlaps.size());
    // 对于是否有[多个tensor聚合到一个Tensor]的情况进行不同处理
    bool isMixedConsumer = mixedConsumerTensors_.count(largeTensor->tensor->rawmagic) > 0;
    if (overlaps.size() == 1) {
        CreateOpFor1toM(function, largeTensor, lcmTileShape, tileOffset, overlaps, dualOverlaps);
    } else if (!isMixedConsumer) {
        FilterOverlaps(function, largeTensor, overlaps, dualOverlaps);
        CreateOpForMtoM(function, largeTensor, lcmTileShape, tileOffset, overlaps, dualOverlaps);
    }
}

void SplitLargeFanoutTensor::TryToSplitLargeTensor(Function& function, const Shape& lcmShape,
                                                   const LogicalTensorPtr& largeTensor)
{
    std::set<Shape, ShapeDimComparator> tileOffsets;
    GetOffsets(tileOffsets, lcmShape, largeTensor);
    int rawMagic = largeTensor->tensor->rawmagic;
    const auto& toIndex = toInfoIndexMap_[rawMagic];
    const auto& fromIndex = fromInfoIndexMap_[rawMagic];
    for (const auto& tileOffset : tileOffsets) {
        auto lcmTileShape = AdjustLcmTileShapeForTailBlock(lcmShape, tileOffset, largeTensor);
        LogicalTensors overlaps;
        LogicalTensors dualOverlaps;
        ProcessTileSplit(function, largeTensor, lcmTileShape, tileOffset, toIndex, fromIndex, overlaps, dualOverlaps);
    }
}

void SplitLargeFanoutTensor::RemoveOps(Function& function, std::vector<Operation*>& opList) const
{
    for (const auto& op : opList) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (const auto op : opList) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Remove %s[%d].", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        if (!op->IsDeleted()) {
            op->SetAsDeleted();
        }
    }
    // The function has no operationGroups_ at pass stage, so LIGHTWEIGHT sort is topologically
    // equivalent to GENERAL, and is faster by caching tensor deps to avoid repeated producer scans.
    function.EraseOperations(true, false);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
}

void SplitLargeFanoutTensor::UpdateForRedundantAssemble(Operation& op)
{
    auto output = op.oOperand.front();
    auto input = op.iOperand.front();
    auto consumersBackup = output->GetConsumers();
    for (const auto& childOp : consumersBackup) {
        childOp->ReplaceInput(input, output);
        if (IsViewLike(childOp->GetOpcode())) {
            auto tensorOffset = input->GetTensorOffset();
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(childOp->GetOpAttribute().get());
            if (viewOpAttribute == nullptr) {
                APASS_LOG_WARN_F(Elements::Operation, "%s[%d] has no valid view-family attribute, skip offset update.",
                                 childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic());
                continue;
            }
            auto viewOffset = viewOpAttribute->GetFromTensorOffset();
            auto newStaticOffset = TensorOffset::Add(viewOffset.offset_, tensorOffset.offset_);
            auto newDynOffset = TensorOffset::Add(viewOffset.dynOffset_, tensorOffset.dynOffset_);
            viewOpAttribute->SetFromOffset(newStaticOffset, newDynOffset);
            APASS_LOG_INFO_F(Elements::Tensor, "Update offset for %s with opmagic %d.", childOp->GetOpcodeStr().c_str(),
                             childOp->GetOpMagic());
        }
    }
}

void SplitLargeFanoutTensor::EraseRedundantAssembleOp(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "---> Remove redundant Assemble op.");
    std::vector<Operation*> redundantCopyOuts;
    std::unordered_set<int> outCastMagics;
    for (const auto& outCast : function.GetOutcast()) {
        outCastMagics.insert(outCast->GetRawMagic());
    }
    for (auto& op : function.Operations(false)) {
        if (!IsAssembleLike(op.GetOpcode())) {
            continue;
        }
        auto output = op.oOperand.front();
        auto input = op.iOperand.front();
        if ((input == nullptr) || (output == nullptr)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "%s[%d] has nullptr input/output; "
                              "Please ensure input and output are valid. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            continue;
        }
        if (outCastMagics.count(output->GetRawMagic()) == 0 && output->GetConsumers().empty()) {
            /* input --> Assemble/Contract --> output(非OCAST, 且没有consumer) */
            redundantCopyOuts.push_back(&op);
        }
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (output->GetProducers().size() != 1 || output->GetConsumers().size() != 1) {
            continue;
        }
        auto consumerOp = *(output->GetConsumers().begin());
        // Assemble输入和输出的raw tensor大小不相等，意味着要做拷贝
        bool requireCopy = (input->tensor->GetRawShapeSize() != output->tensor->GetRawShapeSize());
        if (consumerOp != nullptr && consumerOp->GetOpcode() == Opcode::OP_VIEW && !requireCopy) {
            /*
            Before: input --> Assmeble --> output --> View
            After:  input --> View
            因为input和output的raw shape相同，所以View上的offset不需要修改
            */
            redundantCopyOuts.push_back(&op);
            continue;
        }
        if (input->shape == output->shape && input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            /* 因为input和output raw shape size不同，但shape相同，因此删除前需要重新计算View的offset */
            UpdateForRedundantAssemble(op);
            redundantCopyOuts.push_back(&op);
        }
    }
    if (!redundantCopyOuts.empty()) {
        RemoveOps(function, redundantCopyOuts);
    }
}

void SplitLargeFanoutTensor::UpdateForRedundantView(Operation& op, Operation& consumer)
{
    auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    auto newOffset = viewAttr->GetFromOffset();
    auto nextViewAttr = dynamic_cast<ViewOpAttribute*>(consumer.GetOpAttribute().get());
    // Preserve consumer's explicit toDynValidShape: it may encode tighter constraints
    // (e.g. Min(cur_seq, remaining)) not recoverable from parent view's toDynValidShape alone.
    const auto consumerToDynValidShape = nextViewAttr->GetToDynValidShape();
    auto nextViewOffset = nextViewAttr->GetFromOffset();
    auto nextViewDynOffset = nextViewAttr->GetFromDynOffset();
    auto newDynOffset = viewAttr->GetFromDynOffset();
    auto ret = TensorOffset::Add(newOffset, newDynOffset, nextViewOffset, nextViewDynOffset);
    if (!ret.first.empty()) {
        newOffset = ret.first;
        newDynOffset = ret.second;
    }
    nextViewAttr->SetFromOffset(newOffset, newDynOffset);
    if (consumerToDynValidShape.empty()) {
        auto viewDynShape = GetViewValidShape(viewAttr->GetToDynValidShape(), nextViewOffset, nextViewDynOffset,
                                              consumer.oOperand.front()->GetShape());
        nextViewAttr->SetToDynValidShape(viewDynShape);
    }
    if (newDynOffset.size() == 0) {
        return;
    }
    auto consumerOOperand = consumer.oOperand.front();
    std::vector<SymbolicScalar> dynValidShape;
    if (!nextViewAttr->GetToDynValidShape().empty()) {
        dynValidShape = nextViewAttr->GetToDynValidShape();
    } else if (!consumerOOperand->GetDynValidShape().empty()) {
        dynValidShape = consumerOOperand->GetDynValidShape();
    } else {
        dynValidShape = CommonUtils::CreateConstIntVector(consumerOOperand->GetShape());
    }
    if (nextViewAttr->GetToDynValidShape().empty()) {
        nextViewAttr->SetToDynValidShape(dynValidShape);
    }
    if (consumerOOperand->GetDynValidShape().empty()) {
        consumerOOperand->UpdateDynValidShape(dynValidShape);
    }
}

/*
before:
tensor -> View1 -> tensor1 -> View2 -> tesnor2

after:
tensor -> View2_new -> tensor2
*/
void SplitLargeFanoutTensor::EraseRedundantViewOp(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "---> Remove redundant View op.");
    std::vector<Operation*> redundantView;
    for (auto& op : function.Operations(false)) {
        if (!IsViewLike(op.GetOpcode())) {
            continue;
        }
        /*
        case1. split_large_fanout_tensor 在AssignmemType 之前，view op GetOpAttribute()->GetTo() ==
        MemoryType::MEM_L1的view op一定是tile op展开时插入的，不能删； case2. 框架在tile
        展开插入的view之前还插入了一个view，目前不删除，后续优化可以考虑删除。
        */
        bool isViewToL1 = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get())->GetTo() == MemoryType::MEM_L1;
        auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
        auto consumers = op.oOperand.front()->GetConsumers();
        if (consumers.empty()) {
            continue;
        }
        bool allChildrenView = std::all_of(consumers.begin(), consumers.end(), [=](const Operation* opNext) {
            if (opNext == nullptr || !IsViewLike(opNext->GetOpcode())) {
                return false;
            }
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(opNext->GetOpAttribute().get());
            bool isL1MultiLoad = (viewOpAttribute->GetTo() == MemoryType::MEM_L1);
            // 大包搬运场景下前端插入的view输入和输出shape相同
            auto inTensor = opNext->GetIOperands().front();
            auto outTensor = opNext->GetOOperands().front();
            isL1MultiLoad &= (inTensor->GetShape() == outTensor->GetShape());
            if (isViewToL1 || isL1MultiLoad) {
                return false;
            }
            return true;
        });
        if (allChildrenView) {
            GraphUtils::UpdateViewAttr(function, op);
            for (auto& consumer : consumers) {
                UpdateForRedundantView(op, *consumer);
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Found redundant view and remove it, "
                              "opmagic: %d, to: %s. Input mem: %s, Output mem: %s.",
                              op.GetOpMagic(), BriefMemoryTypeToString(viewAttr->GetTo()).c_str(),
                              BriefMemoryTypeToString(input->GetMemoryTypeOriginal()).c_str(),
                              BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str());
            redundantView.push_back(&op);
        }
    }
    if (!redundantView.empty()) {
        RemoveOps(function, redundantView);
    }
}

void SplitLargeFanoutTensor::SetEnableMoreSplit(bool enableMoreSplit) { enableMoreSplit_ = enableMoreSplit; }

} // namespace npu::tile_fwk
