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
 * \file tensormap.cpp
 * \brief
 */

#include "interface/tensor/tensormap.h"
#include <sstream>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <algorithm> // For std::sort
#include <set>
#include <numeric>

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

std::vector<int> tensor2shape(const std::shared_ptr<LogicalTensor>& pTensor)
{
    std::vector<int> tensorShape;
    for (size_t dim = 0; dim < pTensor->offset.size(); dim++) {
        int pStart = pTensor->offset[dim];
        int pEnd = pStart + pTensor->shape[dim];
        tensorShape.push_back(pStart);
        tensorShape.push_back(pEnd);
    }
    return tensorShape;
}

bool Overlap(const std::shared_ptr<LogicalTensor>& t0, const std::shared_ptr<LogicalTensor>& t1)
{
    return CalcOverlap(t0, t1) != OverlapStatus::NO_OVER_LAP;
}

bool TensorPtrComparator::operator()(const std::shared_ptr<LogicalTensor>& lhs,
                                     const std::shared_ptr<LogicalTensor>& rhs) const
{
    return lhs->magic > rhs->magic;
}

// Insert function: Adds a tensor to the map, removing any matching tensors first
void TensorMap::Insert(std::shared_ptr<LogicalTensor> tobject, bool checkOverlap)
{
    FE_ASSERT(tobject != nullptr) << "tobject is nullptr.";
    std::vector<int> tensorShape = tensor2shape(tobject);

    if (checkOverlap) {
        auto match = Find(tobject);
        if (!match.empty()) {
            FE_LOGI("Tensor %d is full covered in function %s", tobject->magic, belongTo.GetRawName().c_str());
            return;
        }
    }

    FE_ASSERT(tobject->tensor != nullptr) << "tobject->tensor is nullptr.";
    int rawmagic = tobject->tensor->rawmagic;
    // Check if the tensor with the same rawmagic exists
    auto& tensorList = tensorMap_[rawmagic];

    // Ensure we allow new tensor to overwrite old tensor with same size and shape
    if (inverseMap_.count(tobject->magic) > 0) {
        auto& rt = inverseMap_[tobject->magic];
        tensorList.erase(rt);

        std::vector<int> rtensorShape = tensor2shape(rt);
        int rrawmagic = rt->tensor->rawmagic;
        overlapChecker_[rrawmagic].Erase(rtensorShape, rt);
    }

    // method 1
    if (checkOverlap) {
        if (belongTo.expandFunctionAccelerate) {
            auto match = Find(tobject);
            for (auto it = match.begin(); it != match.end(); it++) {
                // 完全覆盖场景，最后一次写有效，前面的写全部要失效
                if ((*it)->shape == tobject->shape && (*it)->offset == tobject->offset) {
                    tensorList.erase(*it);
                    std::vector<int> tShape = tensor2shape(*it);
                    overlapChecker_[(*it)->tensor->rawmagic].Erase(tShape, *it);
                }
            }
        } else {
            for (auto it = tensorList.begin(); it != tensorList.end();) {
                // 完全覆盖场景，最后一次写有效，前面的写全部要失效
                if ((*it)->shape == tobject->shape && (*it)->offset == tobject->offset) {
                    it = tensorList.erase(it);
                } else {
                    it++;
                }
            }
        }
    }

    tensorList.insert(tobject);
    overlapChecker_[rawmagic].Insert(tensorShape, tobject);
    // Also insert into inverseMap_ with rawmagic as key and a shared_ptr to LogicalTensor
    inverseMap_[tobject->magic] = tobject;
}

void TensorMap::Erase(const std::shared_ptr<LogicalTensor>& ttensor)
{
    if (tensorMap_.count(ttensor->tensor->rawmagic) != 0) {
        tensorMap_[ttensor->tensor->rawmagic].erase(ttensor);
    }

    if (inverseMap_.count(ttensor->magic) != 0) {
        inverseMap_.erase(ttensor->magic);
    }
    std::vector<int> tensorShape = tensor2shape(ttensor);
    int rawmagic = ttensor->tensor->rawmagic;
    overlapChecker_[rawmagic].Erase(tensorShape, ttensor);
}

void TensorMap::EraseRawMagic(int rawmagic)
{
    tensorMap_.erase(rawmagic);
    overlapChecker_.erase(rawmagic);
}

std::shared_ptr<LogicalTensor> TensorMap::GetTensorByMagic(int magic) const
{
    auto it = inverseMap_.find(magic);
    if (it != inverseMap_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<RawTensor> TensorMap::GetRawTensorByRawMagic(int rawMagic) const
{
    auto iter = tensorMap_.find(rawMagic);
    if (iter == tensorMap_.end()) {
        return nullptr;
    }
    if (iter->second.empty()) {
        return nullptr;
    }
    auto firstTensor = *(iter->second.begin());
    return firstTensor->tensor;
}

void CalcShapeAndOffsetOfGroup(const std::vector<std::shared_ptr<LogicalTensor>>& tensors,
                               std::vector<int64_t>& resultOffset, std::vector<int64_t>& resultShape)
{
    resultOffset = tensors.front()->offset;
    std::vector<int64_t> maximumOffset;
    for (size_t i = 0; i < tensors.front()->offset.size(); i++) {
        maximumOffset.emplace_back(tensors.front()->offset[i] + tensors.front()->shape[i]);
    }
    for (const auto& incast : tensors) {
        for (size_t i = 0; i < incast->offset.size(); ++i) {
            resultOffset[i] = std::min(resultOffset[i], incast->offset[i]);
            maximumOffset[i] = std::max(maximumOffset[i], incast->offset[i] + incast->shape[i]);
        }
    }
    for (size_t i = 0; i < resultOffset.size(); ++i) {
        resultShape.emplace_back(maximumOffset[i] - resultOffset[i]);
    }
};

void TensorMap::Reset()
{
    tensorMap_.clear();
    inverseMap_.clear();
    overlapChecker_.clear();
}

int CalcOverlapSize(const std::shared_ptr<LogicalTensor>& pTensor, const std::shared_ptr<LogicalTensor>& qTensor)
{
    std::vector<int> overlapEdge;
    for (size_t dim = 0; dim < pTensor->offset.size(); dim++) {
        int pStart = pTensor->offset[dim];
        int pEnd = pStart + pTensor->shape[dim];
        int qStart = qTensor->offset[dim];
        int qEnd = qStart + qTensor->shape[dim];

        std::vector<int> range = {pStart, pEnd, qStart, qEnd};
        std::sort(range.begin(), range.end());
        overlapEdge.push_back(range[2] - range[1]);
    }

    return std::accumulate(overlapEdge.begin(), overlapEdge.end(), 1, std::multiplies<>());
}

// Calculate overlap relationship between two N-D tensor regions described by (offset, shape).
// For each dimension, treat it as a closed interval [offset, offset + shape - 1]. If any
// dimension has no intersection, return NO_OVER_LAP; otherwise return PERFECTLY_MATCH,
// COVERED (P covers Q), BE_COVERED (Q covers P), or PARTIAL_OVERLAP.
OverlapStatus CalcOverlapByOffsetShape(const std::vector<int64_t>& pOffset, const std::vector<int64_t>& pShape,
                                       const std::vector<int64_t>& qOffset, const std::vector<int64_t>& qShape) noexcept
{
    // Check if tensors have same number of dimensions
    if (pOffset.size() != qOffset.size() || pShape.size() != qShape.size() || pOffset.size() != pShape.size()) {
        return OverlapStatus::NO_OVER_LAP;
    }

    bool perfectlyMatch = true;
    bool pCoverQ = true;
    bool qCoverP = true;

    // Check overlap in each dimension
    for (size_t dim = 0; dim < pOffset.size(); dim++) {
        // Two ranges [a, a+length_a] and [b, b+length_b] do NOT overlap if:
        // a + length_a <= b OR b + length_b <= a
        // Therefore, they overlap if: !(a + length_a <= b OR b + length_b <= a)
        // Which is equivalent to: a + length_a > b AND b + length_b > a
        const int64_t pStart = pOffset[dim];
        const int64_t pEnd = pStart + pShape[dim] - 1;
        const int64_t qStart = qOffset[dim];
        const int64_t qEnd = qStart + qShape[dim] - 1;

        if (pEnd < qStart || qEnd < pStart) {
            return OverlapStatus::NO_OVER_LAP;
        }

        pCoverQ &= (pStart <= qStart && qEnd <= pEnd);
        qCoverP &= (qStart <= pStart && pEnd <= qEnd);
        perfectlyMatch &= (pStart == qStart && pEnd == qEnd);
    }

    if (perfectlyMatch) {
        return OverlapStatus::PERFECTLY_MATCH;
    } else if (pCoverQ) {
        return OverlapStatus::COVERED;
    } else if (qCoverP) {
        return OverlapStatus::BE_COVERED;
    } else {
        return OverlapStatus::PARTIAL_OVERLAP;
    }
}

OverlapStatus CalcOverlap(const std::shared_ptr<LogicalTensor>& pTensor, const std::shared_ptr<LogicalTensor>& qTensor,
                          bool loose)
{
    if (!pTensor || !qTensor)
        return OverlapStatus::NO_OVER_LAP;

    // Check if tensors have same raw tensor (memory space)
    if (!loose && pTensor->tensor->rawmagic != qTensor->tensor->rawmagic) {
        return OverlapStatus::NO_OVER_LAP;
    }

    return CalcOverlapByOffsetShape(pTensor->offset, pTensor->shape, qTensor->offset, qTensor->shape);
}

OverlapStatus CalcOverlap(const std::shared_ptr<LogicalTensor>& pTensor,
                          const std::vector<std::shared_ptr<LogicalTensor>>& pGroup, bool loose)
{
    OverlapStatus status = OverlapStatus::NO_OVER_LAP;

    if (pGroup.empty()) {
        return status;
    } else if (pGroup.size() == 1) {
        status = CalcOverlap(pTensor, pGroup.front(), loose);
    } else {
        int overlapSize = 0;
        bool coveredAll = true;
        for (auto& other : pGroup) {
            auto subStatus = CalcOverlap(pTensor, other, loose);
            coveredAll &= subStatus == OverlapStatus::COVERED;
            switch (subStatus) {
                case OverlapStatus::NO_OVER_LAP:
                    return OverlapStatus::NO_OVER_LAP;
                case OverlapStatus::PERFECTLY_MATCH:
                    return OverlapStatus::BE_COVERED_BY_ALL;
                case OverlapStatus::COVERED: {
                    overlapSize += std::accumulate(other->shape.begin(), other->shape.end(), 1, std::multiplies<>());
                    break;
                }
                case OverlapStatus::PARTIAL_OVERLAP: {
                    overlapSize += CalcOverlapSize(pTensor, other);
                    break;
                }
                default:
                    break;
            }
        }
        auto pSize = std::accumulate(pTensor->shape.begin(), pTensor->shape.end(), 1, std::multiplies<>());
        if (coveredAll) {
            if (pSize == overlapSize) {
                status = OverlapStatus::PERFECTLY_MATCH_WITH_ALL;
            } else if (pSize > overlapSize) {
                status = OverlapStatus::COVERED_ALL;
            }
        } else {
            if (pSize == overlapSize) {
                status = OverlapStatus::BE_COVERED_BY_ALL;
            } else {
                status = OverlapStatus::PARTIAL_OVERLAP_WITH_ALL;
            }
        }
    }
    return status;
}

std::vector<std::shared_ptr<LogicalTensor>> TensorMap::Find(std::shared_ptr<LogicalTensor> ttensor)
{
    std::vector<std::shared_ptr<LogicalTensor>> result;

    // Lookup tensors based on rawmagic number
    auto it = tensorMap_.find(ttensor->tensor->rawmagic);
    if (it == tensorMap_.end()) {
        /* Root function does not belong to any other functions. */
        if (!belongTo.HasParent() ||
            belongTo.IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::EXECUTE_GRAPH)) {
            return {};
        }
        /* A DYNAMIC function aggregates every sibling path's out-args (same
         * loop-carry buffer, different SSA values) in its tensor map. A path/
         * hidden function must never resolve its operands against that map. */
        if (belongTo.Parent().GetFunctionType() == FunctionType::DYNAMIC) {
            return {};
        }
        return belongTo.Parent().GetTensorMap().Find(ttensor);
    }

    if (belongTo.expandFunctionAccelerate) {
        std::vector<int> tensorShape = tensor2shape(ttensor);
        int rawmagic = ttensor->tensor->rawmagic;
        result = overlapChecker_[rawmagic].Find(tensorShape);
    } else {
        for (const auto& tensorPtr : it->second) {
            bool overlap = Overlap(tensorPtr, ttensor);
            if (!overlap) {
                continue;
            }
            result.push_back(tensorPtr);
        }
    }
    return result;
}

void TensorMap::ValidCheck() const {}
} // namespace npu::tile_fwk
