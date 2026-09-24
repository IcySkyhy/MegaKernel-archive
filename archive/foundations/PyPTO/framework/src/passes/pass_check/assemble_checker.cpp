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
 * \file assemble_checker.cpp
 * \brief
 */

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "assemble_checker.h"

#define MODULE_NAME "AssembleChecker"

namespace npu {
namespace tile_fwk {
namespace {
constexpr size_t kMinAreaCountForOverlap = 2;
constexpr size_t kFirstAreaIndex = 0;
constexpr size_t kSecondAreaIndex = 1;

struct AssembleArea {
    Shape shape;
    Shape offset;
    std::vector<std::pair<int64_t, int64_t>> ranges;
};

size_t SelectSweepAxis(const std::vector<AssembleArea>& areas)
{
    size_t bestAxis = 0;
    int64_t bestSpan = std::numeric_limits<int64_t>::min();
    const size_t dimSize = areas.front().ranges.size();
    for (size_t axis = 0; axis < dimSize; ++axis) {
        int64_t minStart = std::numeric_limits<int64_t>::max();
        int64_t maxStart = std::numeric_limits<int64_t>::min();
        for (const auto& area : areas) {
            minStart = std::min(minStart, area.ranges[axis].first);
            maxStart = std::max(maxStart, area.ranges[axis].first);
        }
        int64_t span = maxStart - minStart;
        if (span > bestSpan) {
            bestSpan = span;
            bestAxis = axis;
        }
    }
    return bestAxis;
}

bool RangesOverlap(const std::vector<std::pair<int64_t, int64_t>>& left,
                   const std::vector<std::pair<int64_t, int64_t>>& right)
{
    if (left.size() != right.size()) {
        return false;
    }
    return std::equal(left.begin(), left.end(), right.begin(),
                      [](const auto& a, const auto& b) { return a.second >= b.first && a.first <= b.second; });
}

bool FindOverlap(const std::vector<AssembleArea>& areas, size_t& prevIdx, size_t& curIdx)
{
    if (areas.size() < kMinAreaCountForOverlap) {
        return false;
    }
    if (areas.front().ranges.empty()) {
        prevIdx = kFirstAreaIndex;
        curIdx = kSecondAreaIndex;
        return true;
    }
    const size_t sweepAxis = SelectSweepAxis(areas);
    std::vector<size_t> order(areas.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        const auto& leftRange = areas[left].ranges[sweepAxis];
        const auto& rightRange = areas[right].ranges[sweepAxis];
        return leftRange.first == rightRange.first ? leftRange.second < rightRange.second :
                                                     leftRange.first < rightRange.first;
    });

    std::vector<size_t> active;
    active.reserve(areas.size());
    for (size_t areaIdx : order) {
        const auto& curArea = areas[areaIdx];
        const int64_t curStart = curArea.ranges[sweepAxis].first;
        size_t writeIdx = 0;
        for (size_t activeIdx : active) {
            if (areas[activeIdx].ranges[sweepAxis].second >= curStart) {
                active[writeIdx++] = activeIdx;
            }
        }
        active.resize(writeIdx);

        for (size_t activeIdx : active) {
            if (RangesOverlap(areas[activeIdx].ranges, curArea.ranges)) {
                prevIdx = activeIdx;
                curIdx = areaIdx;
                return true;
            }
        }
        active.emplace_back(areaIdx);
    }
    return false;
}

// assemble存在dynOffset和输入存在dynValidShape场景暂不判断。
Status CheckDynSkip(const LogicalTensorPtr& outputTensor, bool& needSkip)
{
    auto isConcreteValidShape = [](const Shape& shape, const std::vector<SymbolicScalar>& validShape) -> bool {
        if (validShape.size() != shape.size()) {
            return false;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (!validShape[i].ConcreteValid() || validShape[i].Concrete() != shape[i]) {
                return false;
            }
        }
        return true;
    };
    for (const auto& producerOp : outputTensor->GetProducers()) {
        if (producerOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
            needSkip = true;
            return SUCCESS;
        }
        auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (!assembleOpAttr) {
            APASS_LOG_WARN_F(Elements::Tensor, "%s[%d] has no valid assembleOpAttribute; Please check.",
                             producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
            return FAILED;
        }
        if (assembleOpAttr->GetToDynOffset().size() != 0) {
            bool isAllImmediate = true;
            for (const auto& offset : assembleOpAttr->GetToDynOffset()) {
                if (!offset.IsImmediate()) {
                    isAllImmediate = false;
                    break;
                }
            }
            if (!isAllImmediate) {
                needSkip = true;
            }
            return SUCCESS;
        }
        auto input = producerOp->iOperand.front();
        const auto& validShape = input->GetDynValidShape();
        if (!validShape.empty() && !isConcreteValidShape(input->GetShape(), validShape)) {
            needSkip = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}
} // namespace

/*
    在input->assemble->output的场景中，通过校验input之间是否每个轴都存在重叠来判断，input间是否存在覆盖output中同一数据块的情况。
    这种重叠可能由于两块数据到达时间不同，导致覆盖顺序不确定进而导致不确定的行为
*/
Status AssembleChecker::CheckAssembleOverlap(Function& function)
{
    auto needSkip = [](const Shape& vec) -> bool {
        return std::any_of(vec.begin(), vec.end(), [](int64_t val) { return val == -1; });
    };
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (const auto& outputTensor : allTensors) {
        if (outputTensor->GetProducers().size() == 0) {
            continue;
        }
        bool dynSkip = false;
        if (CheckDynSkip(outputTensor, dynSkip) == FAILED) {
            return FAILED;
        }

        if (dynSkip) {
            continue;
        }
        std::vector<AssembleArea> coveredAreas;
        coveredAreas.reserve(outputTensor->GetProducers().size());
        for (const auto& assembleOp : outputTensor->GetProducers()) {
            if (assembleOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            auto assembleOffset = dynamic_cast<AssembleOpAttribute*>(assembleOp->GetOpAttribute().get())->GetToOffset();
            auto inputTensor = assembleOp->GetIOperands().front();
            auto inputShape = inputTensor->GetShape();
            if (needSkip(inputShape) || needSkip(assembleOffset)) {
                continue;
            }
            std::vector<std::pair<int64_t, int64_t>> curInputArea;
            if (assembleOffset.size() != inputShape.size()) {
                APASS_LOG_WARN_F(
                    Elements::Tensor,
                    "Dimension of assemble op[%d]'s toOffset(%s) varies from its input[%d]'s shape(%s); Please check "
                    "the function graph.",
                    assembleOp->GetOpMagic(), CommonUtils::ContainerToStr(assembleOffset).c_str(),
                    inputTensor->GetMagic(), CommonUtils::ContainerToStr(inputShape).c_str());
                return FAILED;
            }
            curInputArea.reserve(inputShape.size());
            for (size_t i = 0; i < inputShape.size(); i++) {
                curInputArea.emplace_back(assembleOffset[i], assembleOffset[i] + inputShape[i] - 1);
            }
            coveredAreas.emplace_back(
                AssembleArea{std::move(inputShape), std::move(assembleOffset), std::move(curInputArea)});
        }
        size_t prevIdx = 0;
        size_t curIdx = 0;
        if (FindOverlap(coveredAreas, prevIdx, curIdx)) {
            const auto& prevArea = coveredAreas[prevIdx];
            const auto& curArea = coveredAreas[curIdx];
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Tensor produced by assemble has overlap inputs. Overlap input1: shape:%s offset:%s.",
                             CommonUtils::ContainerToStr(prevArea.shape).c_str(),
                             CommonUtils::ContainerToStr(prevArea.offset).c_str());
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "Overlap input2: shape:%s offset:%s; Please check the function graph; Please check Tensor[%d] and "
                "its input.",
                CommonUtils::ContainerToStr(curArea.shape).c_str(), CommonUtils::ContainerToStr(curArea.offset).c_str(),
                outputTensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
