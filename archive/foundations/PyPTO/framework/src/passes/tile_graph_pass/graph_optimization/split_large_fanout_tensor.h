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
 * \file split_large_fanout_tensor.h
 * \brief
 */

#ifndef PASS_SPLIT_LARGE_FANOUT_TENSOR_H_
#define PASS_SPLIT_LARGE_FANOUT_TENSOR_H_

#include <unordered_set>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/infer_shape_utils.h"

namespace npu::tile_fwk {
struct ShapeDimComparator {
    bool operator()(const Shape& a, const Shape& b) const
    {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) {
                return a[i] < b[i];
            }
        }
        return false;
    }
};
/*
 * SplitLargeFanoutTensor:
 本pass主要用于处理assemble被多个消费者消费时，如果每人都消费的是部分数据，因为assemble导致需要等待全部数据产生，影响并行度的问题
 * 处理原则：
    1. 仅解决OP_ASSEMBLE作为copyOut后接OP_VIEW作为CopyIn的场景
    2. 比较assemble输入与view输出之前的重合关系
    2.1 完全匹配：一到一情况
    2.2 被覆盖：多到一情况
    2.3 覆盖所有：一到多情况
*/
class SplitLargeFanoutTensor : public Pass {
public:
    SplitLargeFanoutTensor() : Pass("SplitLargeFanoutTensor") {}
    ~SplitLargeFanoutTensor() override = default;

private:
    struct OverlapSearchIndex {
        const std::vector<std::pair<LogicalTensorPtr, Offset>>* tensorInfos = nullptr;
        std::vector<std::vector<size_t>> orderByDim;
    };

    Status RunOnFunction(Function& function) override;
    void Init();
    void EraseRedundantAssembleOp(Function& function);
    void RemoveOps(Function& function, std::vector<Operation*>& opList) const;
    void UpdateForRedundantAssemble(Operation& op);
    void UpdateForRedundantView(Operation& op, Operation& consumer);
    void EraseRedundantViewOp(Function& function);
    int64_t GCD(int64_t x, int64_t y);
    Status LCM(int64_t x, int64_t y, int64_t& lcm);
    Status CalLcmShape(const Shape& toShape, const Shape& fromShape, Shape& lcmShape);
    Status CalGcdShape(const Shape& toShape, const Shape& fromShape, Shape& lcmShape);
    void GenerateOffset(const Shape& maxs, const Shape& steps, Shape& current, std::vector<Shape>& result, size_t dim);
    void CollectLargeTensorToInfo(const LogicalTensorPtr& largeTensor);
    void CollectLargeTensorFromInfo(const LogicalTensorPtr& largeTensor);
    void CollectOverlaps(const Shape& lcmTileShape, const Offset& lcmTileOffset, const OverlapSearchIndex& toIndex,
                         const OverlapSearchIndex& fromIndex, LogicalTensors& overlaps, LogicalTensors& dualOverlaps);
    void BuildOverlapSearchIndex(const std::vector<std::pair<LogicalTensorPtr, Offset>>& tensorInfos,
                                 OverlapSearchIndex& index) const;
    void CollectCoveredTensors(const Shape& lcmTileShape, const Offset& lcmTileOffset, const OverlapSearchIndex& index,
                               LogicalTensors& tensors) const;
    bool IsTensorCoveredByTile(const Offset& tensorOffset, const Shape& tensorShape, const Offset& tileOffset,
                               const Shape& tileShape) const;
    void CreateOpFor1toM(Function& function, const LogicalTensorPtr& largeTensor, const Shape& lcmTileShape,
                         const Offset& lcmTileOffset, const LogicalTensors& overlaps,
                         const LogicalTensors& dualOverlaps);
    void ExtractDualOverlapTiles(Function& function, const LogicalTensorPtr& largeTensor,
                                 const LogicalTensors& dualOverlaps,
                                 std::vector<std::pair<Offset, Shape>>& dualOverlapTileInfos,
                                 LogicalTensors& filteredDualOverlaps);
    bool HasIntersectionWithAnyDualOverlap(const Offset& overlapOffset, const Shape& overlapShape,
                                           const std::vector<std::pair<Offset, Shape>>& dualOverlapTileInfos);
    void FilterOverlaps(Function& function, const LogicalTensorPtr& largeTensor, LogicalTensors& overlaps,
                        const LogicalTensors& dualOverlaps);
    void CreateOpForMtoM(Function& function, const LogicalTensorPtr& largeTensor, const Shape& lcmTileShape,
                         const Offset& lcmTileOffset, const LogicalTensors& overlaps,
                         const LogicalTensors& dualOverlaps);
    void MoreSplit(Function& function, const LogicalTensorPtr& largeTensor, const LogicalTensors& overlaps,
                   const LogicalTensors& dualOverlaps);
    void CreateOpForMoreSplit(Function& function, const LogicalTensorPtr& largeTensor, const LogicalTensors& overlaps,
                              const Shape& gcdShape, const LogicalTensorPtr& dualOverlap,
                              const std::vector<Shape>& gcdTileOffsets, const Offset& viewOpOffset);
    void FindOverlapAndCreateViewOp(Function& function, const LogicalTensors& overlaps,
                                    const LogicalTensorPtr& newGcdTensor, const Shape& gcdTileOffsetForLarge,
                                    Shape& newViewOffset);
    void CollectLargeTensor(Function& function);
    void SplitLargeTensor(Function& function);
    bool IsBeCovered(Function& function, const LogicalTensorPtr& largeTensor,
                     const std::vector<std::pair<LogicalTensorPtr, Offset>>& toTensorInfos);
    bool HasDuplicateToTile(const std::vector<std::pair<LogicalTensorPtr, Offset>>& toTensorInfos);
    Shape AdjustLcmTileShapeForTailBlock(const Shape& lcmShape, const Shape& tileOffset,
                                         const LogicalTensorPtr& largeTensor);
    bool CheckOverlapCoverage(const LogicalTensors& overlaps, const Shape& lcmTileShape);
    void ProcessTileSplit(Function& function, const LogicalTensorPtr& largeTensor, const Shape& lcmTileShape,
                          const Shape& tileOffset, const OverlapSearchIndex& toIndex,
                          const OverlapSearchIndex& fromIndex, LogicalTensors& overlaps, LogicalTensors& dualOverlaps);
    void TryToSplitLargeTensor(Function& function, const Shape& lcmShape, const LogicalTensorPtr& largeTensor);
    void GetOffsets(std::set<Shape, ShapeDimComparator>& tileOffsets, const Shape& lcmShape,
                    const LogicalTensorPtr& largeTensor);
    void SetEnableMoreSplit(bool enableMoreSplit);
    std::unordered_map<int, std::vector<std::pair<LogicalTensorPtr, Offset>>> toInfoMap_;
    std::unordered_map<int, std::vector<std::pair<LogicalTensorPtr, Offset>>> fromInfoMap_;
    std::unordered_map<int, OverlapSearchIndex> toInfoIndexMap_;
    std::unordered_map<int, OverlapSearchIndex> fromInfoIndexMap_;
    std::vector<LogicalTensorPtr> largeTensors_;
    std::map<LogicalTensorPtr, std::set<Shape>> toShapes_;
    std::map<LogicalTensorPtr, std::set<Shape>> fromShapes_;
    bool enableMoreSplit_ = false;
    std::vector<Operation*> addedOps_;
    std::unordered_set<int> mixedConsumerTensors_;
    IRBuilder irBuilder_;

    std::string idx;
};

struct ShapeComparator {
    bool operator()(const Shape& a, const Shape& b) const { return CommonUtils::Numel(a) < CommonUtils::Numel(b); }
};
} // namespace npu::tile_fwk
#endif // PASS_SPLIT_LARGE_FANOUT_TENSOR_H_
