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
 * \file hypercube_overlap_checker.h
 * \brief
 */

#ifndef HYPERCUBE_OVERLAP_CHECKER_H
#define HYPERCUBE_OVERLAP_CHECKER_H

#include "interface/utils/common.h"

#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>

namespace npu {
namespace tile_fwk {
constexpr int elementOfDim = 2;

// 用于查找特定shape的hypercube之间的重叠
template <typename T>
class HypercubeOverlapCheckerBlock {
public:
    void Insert(const std::vector<int>& hypercube, T value);
    std::vector<T> Find(const std::vector<int>& hypercube,
                        int64_t* overlapSumPtr = nullptr); // [x_min, x_max, y_min, y_max, ...]
    std::vector<T> FindWithGuaranteeNoRedundant(const std::vector<int>& hypercube);
    void Erase(const std::vector<int>& hypercube, T value);
    void Shape2Keys(const std::vector<int>& hypercube, std::vector<uint64_t>& result, int dimIdx = 0,
                    uint64_t currValue = 0);
    bool NoOverlap(const std::vector<int>& hypercube1, const std::vector<int>& hypercube2);
    bool SamePairVal(const std::pair<std::vector<int>, T>& pairVal1, const std::pair<std::vector<int>, T>& pairVal2);
    void Clear();

    std::vector<int> wide_; // 空间网格宽度
    std::unordered_map<uint64_t, std::vector<std::pair<std::vector<int>, T>>> hashBucket_;
    std::unordered_set<T> container_;

private:
    int CalOverlap(const std::vector<int>& hypercube1, const std::vector<int>& hypercube2);
};

template <typename T>
void HypercubeOverlapCheckerBlock<T>::Clear()
{
    hashBucket_.clear();
    container_.clear();
}

// 判断两个（hypercube, value）中是否为同一个value
template <typename T>
bool HypercubeOverlapCheckerBlock<T>::SamePairVal(const std::pair<std::vector<int>, T>& pairVal1,
                                                  const std::pair<std::vector<int>, T>& pairVal2)
{
    if (pairVal1.second != pairVal2.second) {
        return false;
    }
    return true;
}

// 判断两个hypercube间是否有重叠
template <typename T>
bool HypercubeOverlapCheckerBlock<T>::NoOverlap(const std::vector<int>& hypercube1, const std::vector<int>& hypercube2)
{
    if (hypercube1.size() % elementOfDim != 0 || hypercube1.size() != hypercube2.size()) {
        return true;
    }
    int dim = hypercube1.size() / elementOfDim;
    for (int i = 0; i < dim; i++) {
        int pStart = hypercube1[i * elementOfDim];
        int pEnd = hypercube1[i * elementOfDim + 1];
        int qStart = hypercube2[i * elementOfDim];
        int qEnd = hypercube2[i * elementOfDim + 1];
        if (pEnd <= qStart || pStart >= qEnd) {
            return true;
        }
    }
    return false;
}

// 计算两个hypercube间重叠的大小
template <typename T>
int HypercubeOverlapCheckerBlock<T>::CalOverlap(const std::vector<int>& hypercube1, const std::vector<int>& hypercube2)
{
    int dim = hypercube1.size() / elementOfDim;
    int res{1};
    for (int i = 0; i < dim; i++) {
        int pStart = hypercube1[i * elementOfDim];
        int pEnd = hypercube1[i * elementOfDim + 1];
        int qStart = hypercube2[i * elementOfDim];
        int qEnd = hypercube2[i * elementOfDim + 1];
        int start = std::max(pStart, qStart);
        int end = std::min(pEnd, qEnd);
        res *= (end - start);
    }
    return res;
}

// 将hypercube转换成一个或多个哈希值
template <typename T>
void HypercubeOverlapCheckerBlock<T>::Shape2Keys(const std::vector<int>& hypercube, std::vector<uint64_t>& result,
                                                 int dimIdx, uint64_t currValue) NO_UBSAN
{
    if ((dimIdx * elementOfDim + 1) >= static_cast<int>(hypercube.size())) {
        result.push_back(currValue);
        return;
    }
    int start = hypercube[dimIdx * elementOfDim];
    int end = hypercube[dimIdx * elementOfDim + 1];
    int wide = wide_[dimIdx] > 0 ? wide_[dimIdx] : 1;
    int startGrid = start / wide;
    int endGrid = (end - 1) / wide; // 占用空间的表示为左闭右开

    constexpr uint64_t smallPrime = 131071;
    uint64_t newvalue = currValue * smallPrime;

    for (int i = startGrid; i <= endGrid; i++) {
        Shape2Keys(hypercube, result, dimIdx + 1, newvalue + static_cast<uint64_t>(i));
    }
}

// 一个hypercube可能与多个空间网格重叠，存在于多个hashBucket之中
template <typename T>
void HypercubeOverlapCheckerBlock<T>::Erase(const std::vector<int>& hypercube, T value)
{
    std::vector<uint64_t> keys;
    Shape2Keys(hypercube, keys);
    std::pair<std::vector<int>, T> pairVal{hypercube, value};
    for (auto key : keys) {
        auto newEnd = std::remove_if(hashBucket_[key].begin(), hashBucket_[key].end(),
                                     [this, &pairVal](auto& pairVal2) { return SamePairVal(pairVal, pairVal2); });
        hashBucket_[key].erase(newEnd, hashBucket_[key].end());
    }
    container_.erase(value);
}

// 在每个有重叠的hashBucket中查找有重叠的hypercube
template <typename T>
std::vector<T> HypercubeOverlapCheckerBlock<T>::Find(const std::vector<int>& hypercube, int64_t* overlapSumPtr)
{
    std::vector<T> result;
    std::unordered_set<T> alreadyChecked;
    std::vector<uint64_t> keys;
    Shape2Keys(hypercube, keys);
    for (auto key : keys) {
        for (auto& pairVal : hashBucket_[key]) {
            if (alreadyChecked.count(pairVal.second) == 0 && !NoOverlap(hypercube, pairVal.first)) {
                result.push_back(pairVal.second);
                if (overlapSumPtr != nullptr) {
                    *overlapSumPtr += CalOverlap(hypercube, pairVal.first);
                }
            }
            alreadyChecked.insert(pairVal.second);
        }
    }
    return result;
}

// 向有重叠的hashBucket中插入hypercube和value
template <typename T>
void HypercubeOverlapCheckerBlock<T>::Insert(const std::vector<int>& hypercube, T value)
{
    std::vector<uint64_t> keys;
    Shape2Keys(hypercube, keys);
    std::pair<std::vector<int>, T> pairVal{hypercube, value};
    for (auto key : keys) {
        hashBucket_[key].push_back(pairVal);
    }
    container_.insert(value);
}

// HypercubeOverlapChecker为对HypercubeOverlapCheckerBlock的封装，使每个CheckerBlock中hypercube的shape相同以加速查询
template <typename T>
class HypercubeOverlapChecker {
public:
    bool Insert(const std::vector<int>& hypercube, T value);
    std::vector<T> Find(const std::vector<int>& hypercube,
                        int64_t* overlapSumPtr = nullptr); // [x_min, x_max, y_min, y_max, ...]
    bool Erase(const std::vector<int>& hypercube, T value);
    void Clear();
    std::vector<int> Hypercube2Shape(const std::vector<int>& hypercube);

    std::map<std::vector<int>, HypercubeOverlapCheckerBlock<T>> shape2Block_;
};

// 得到hypercube的shape
template <typename T>
std::vector<int> HypercubeOverlapChecker<T>::Hypercube2Shape(const std::vector<int>& hypercube)
{
    int dim = hypercube.size() / elementOfDim;
    std::vector<int> shape;
    for (int i = 0; i < dim; i++) {
        int wide = hypercube[i * elementOfDim + 1] - hypercube[i * elementOfDim];
        wide = wide > 0 ? wide : 1;
        shape.push_back(wide);
    }
    return shape;
}

// 向特定shape的CheckerBlock中插入value，如果没有该shape则创建
template <typename T>
bool HypercubeOverlapChecker<T>::Insert(const std::vector<int>& hypercube, T value)
{
    if (hypercube.size() == 0 || hypercube.size() % elementOfDim != 0) {
        return false;
    }
    std::vector<int> shape = Hypercube2Shape(hypercube);
    if (shape2Block_.count(shape) == 0) {
        shape2Block_[shape] = HypercubeOverlapCheckerBlock<T>{};
        shape2Block_[shape].wide_ = shape;
    }
    shape2Block_[shape].Insert(hypercube, value);
    return true;
}

// 在所有CheckBlock中查询给定的hypercube
template <typename T>
std::vector<T> HypercubeOverlapChecker<T>::Find(const std::vector<int>& hypercube, int64_t* overlapSumPtr)
{
    if (hypercube.size() == 0 || hypercube.size() % elementOfDim != 0) {
        return {};
    }
    std::vector<int> shape = Hypercube2Shape(hypercube);
    std::vector<T> searchResult;

    int64_t overlapSumBlock = 0;
    int64_t* overlapSumBlockPtr = (overlapSumPtr != nullptr) ? &overlapSumBlock : nullptr;

    for (auto& pr : shape2Block_) {
        if (overlapSumBlockPtr != nullptr) {
            *overlapSumBlockPtr = 0;
        }
        std::vector<T> blockResult = pr.second.Find(hypercube, overlapSumBlockPtr);
        if (overlapSumBlockPtr != nullptr) {
            *overlapSumPtr += *overlapSumBlockPtr;
        }
        searchResult.insert(searchResult.end(), blockResult.begin(), blockResult.end());
    }
    return searchResult;
}

// 在特定shape的CheckerBlock中清除给定hypercube和value
template <typename T>
bool HypercubeOverlapChecker<T>::Erase(const std::vector<int>& hypercube, T value)
{
    if (hypercube.size() == 0 || hypercube.size() % elementOfDim != 0) {
        return false;
    }
    std::vector<int> shape = Hypercube2Shape(hypercube);
    if (shape2Block_.count(shape) == 0) {
        return true;
    }
    shape2Block_[shape].Erase(hypercube, value);
    return true;
}

template <typename T>
void HypercubeOverlapChecker<T>::Clear()
{
    shape2Block_.clear();
}

} // namespace tile_fwk
} // namespace npu
#endif
