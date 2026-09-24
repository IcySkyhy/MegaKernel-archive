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
 * \file tensormap.h
 * \brief
 */

#pragma once
#ifndef TENSOR_MAP_H
#define TENSOR_MAP_H

#include <iostream>
#include <vector>
#include <deque>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cassert>
#include <memory>
#include "interface/inner/pre_def.h"
#include "interface/tensor/hypercube_overlap_checker.h"

namespace npu::tile_fwk {
enum class OverlapStatus {
    PERFECTLY_MATCH_WITH_ALL, // x is equal to {y, z}
    BE_COVERED_BY_ALL,        // x is part of {y, z}
    COVERED_ALL,              // {y, z} is part of x
    PARTIAL_OVERLAP_WITH_ALL, // x only have part of overlap with {y, z}
    PERFECTLY_MATCH,          // x is equal to y
    BE_COVERED,               // x is part of y
    COVERED,                  // y is part of x
    PARTIAL_OVERLAP,          // x only have part of overlap with y
    NO_OVER_LAP               // x and y has no overlap
};

inline std::string OverlapStatusString(OverlapStatus status)
{
    switch (status) {
        case OverlapStatus::PERFECTLY_MATCH_WITH_ALL:
            return "PERFECTLY_MATCH_WITH_ALL";
        case OverlapStatus::BE_COVERED_BY_ALL:
            return "BE_COVERED_BY_ALL";
        case OverlapStatus::COVERED_ALL:
            return "COVERED_ALL";
        case OverlapStatus::PARTIAL_OVERLAP_WITH_ALL:
            return "PARTIAL_OVERLAP_WITH_ALL";
        case OverlapStatus::PERFECTLY_MATCH:
            return "PERFECTLY_MATCH";
        case OverlapStatus::BE_COVERED:
            return "BE_COVERED";
        case OverlapStatus::COVERED:
            return "COVERED";
        case OverlapStatus::PARTIAL_OVERLAP:
            return "PARTIAL_OVERLAP";
        case OverlapStatus::NO_OVER_LAP:
            return "NO_OVER_LAP";
        default:
            return "Unknown OverlapStatus";
    }
}

bool Overlap(const std::shared_ptr<LogicalTensor>& t0, const std::shared_ptr<LogicalTensor>& t1);

OverlapStatus CalcOverlapByOffsetShape(const std::vector<int64_t>& pOffset, const std::vector<int64_t>& pShape,
                                       const std::vector<int64_t>& qOffset,
                                       const std::vector<int64_t>& qShape) noexcept;

// Move the function declaration outside of the TensorMap class
OverlapStatus CalcOverlap(const std::shared_ptr<LogicalTensor>& pTensor, const std::shared_ptr<LogicalTensor>& qTensor,
                          bool loose = false);

OverlapStatus CalcOverlap(const std::shared_ptr<LogicalTensor>& pTensor,
                          const std::vector<std::shared_ptr<LogicalTensor>>& pGroup, bool loose = false);

void CalcShapeAndOffsetOfGroup(const std::vector<std::shared_ptr<LogicalTensor>>& tensors,
                               std::vector<int64_t>& resultOffset, std::vector<int64_t>& resultShape);

int CalcOverlapSize(const std::shared_ptr<LogicalTensor>& pTensor, const std::shared_ptr<LogicalTensor>& qTensor);
// Custom comparator for shared_ptr<LogicalTensor> in descending order
struct TensorPtrComparator {
    bool operator()(const std::shared_ptr<LogicalTensor>& lhs, const std::shared_ptr<LogicalTensor>& rhs) const;
};

class TensorMap {
public:
    explicit TensorMap(Function& function) : belongTo(function) {}

    TensorMap(const TensorMap& other) = delete;
    TensorMap(TensorMap&& other) = delete;
    TensorMap& operator=(const TensorMap& other) = delete;
    TensorMap& operator=(TensorMap&& other) = delete;

    // Changed to use std::set with custom comparator
    std::unordered_map<int, std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator>> tensorMap_;

    std::unordered_map<int, HypercubeOverlapChecker<std::shared_ptr<LogicalTensor>>> overlapChecker_;

    // The inverse map: magic -> shared_ptr to LogicalTensor
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> inverseMap_;

    // Find function: Returns a vector of shared_ptr to matching tensors
    std::vector<std::shared_ptr<LogicalTensor>> Find(std::shared_ptr<LogicalTensor> ttensor);

    // Insert function: Adds a tensor to the map, removing any matching tensors first
    void Insert(std::shared_ptr<LogicalTensor> tobject, bool checkOverlap = true);

    void Erase(const std::shared_ptr<LogicalTensor>& ttensor);

    void EraseRawMagic(int rawmagic);

    std::shared_ptr<LogicalTensor> GetTensorByMagic(int magic) const;

    std::shared_ptr<RawTensor> GetRawTensorByRawMagic(int rawMagic) const;

    void Reset();

    void ValidCheck() const;

private:
    Function& belongTo;
};

} // namespace npu::tile_fwk

#endif // TENSOR_MAP_H
