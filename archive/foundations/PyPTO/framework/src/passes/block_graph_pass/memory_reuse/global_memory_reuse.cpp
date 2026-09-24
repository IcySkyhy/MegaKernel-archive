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
 * \file global_memory_reuse.cpp
 * \brief
 */
#include "global_memory_reuse.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>
#include <utility>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"

#define MODULE_NAME "GlobalMemoryReuse"

namespace npu {
namespace tile_fwk {
constexpr int64_t MEM_PROPORTION_COEFF = 8;
constexpr uint64_t EXTRA_SIZE_IN_BYTE = 32;
constexpr uint64_t ALIGN_SIZE_IN_BYTE = 512;
constexpr int64_t INVALID_INDEX = -1;
constexpr size_t OFFSET_INDEX = 1;
constexpr size_t RAW_SHAPE_POS = 2;
constexpr size_t TWO_CONSUMER_FAST_PATH_COUNT = 2;
inline uint64_t Align(const uint64_t n) { return (n + ALIGN_SIZE_IN_BYTE - 1) & (~(ALIGN_SIZE_IN_BYTE - 1)); }

void TensorBucket::UpdateOffset(const uint64_t offset)
{
    offset_ = offset;
    APASS_LOG_INFO_F(Elements::Tensor, "Updated bucket offset to %lu bytes with %zu tensor groups.", offset,
                     tensorGroups_.size());

    // 更新存储信息
    for (const auto& tensorGroup : tensorGroups_) {
        if (tensorGroup.empty()) {
            continue;
        }
        LogicalTensorPtr leadTensor = *tensorGroup.begin();
        leadTensor->storage_->start_ = offset;
        leadTensor->storage_->length_ = size_;
        APASS_LOG_DEBUG_F(Elements::Tensor, "Bucket Group lead tensor: magic=%d rawmagic=%d storage=[%lu, %lu].",
                          leadTensor->magic, leadTensor->tensor->rawmagic, leadTensor->storage_->start_,
                          leadTensor->storage_->length_);
    }
}

bool TensorBucket::AddTensorGroup(const TensorsDesc& tensorsDesc)
{
    if (tensorsDesc.tensors.empty()) {
        return false;
    }

    // 更新桶大小
    LogicalTensorPtr tensor = *tensorsDesc.tensors.begin();
    uint64_t tensorSize = static_cast<uint64_t>(tensor->storage_->length_);
    size_ = std::max(tensorSize, size_);
    APASS_LOG_INFO_F(Elements::Tensor, "Added tensor group - tensor: magic %d, rawmagic %d, Bucket size: %lu.",
                     tensor->magic, tensor->tensor->rawmagic, size_);

    // 存储tensor组
    tensorGroups_.emplace_back(tensorsDesc.tensors);
    consumerOpIdxs_ = tensorsDesc.consumerOpIdxs;
    return true;
}

bool Allocator::IsRawQualified(const WorkspaceInfo& outWspInfo, const WorkspaceInfo& inWspInfo)
{
    const auto& outTensor = outWspInfo.tensor->tensor;
    const auto& inTensor = inWspInfo.tensor->tensor;
    if (outTensor->GetRawDataSize() == inTensor->GetRawDataSize()) {
        return true;
    }
    if (outTensor->GetDataType() != inTensor->GetDataType()) {
        return false;
    }
    const auto& outDims = outTensor->rawshape;
    const auto& inDims = inTensor->rawshape;
    if (outDims.size() != inDims.size()) {
        return false;
    }
    // 非最高维度值校验
    const size_t dimCount = outDims.size();
    for (size_t i = 1; i < dimCount; i++) {
        if (outDims[i] != inDims[i]) {
            return false;
        }
    }
    return true;
}

const std::vector<WorkspaceInfo>& Allocator::GetLeafFuncOutputInputReuseMap(Function& leafFunc) const
{
    auto it = leafFuncOutputInputReuseMap_.find(&leafFunc);
    if (it != leafFuncOutputInputReuseMap_.end()) {
        return it->second;
    }
    static std::vector<WorkspaceInfo> empty;
    return empty;
}

std::vector<WorkspaceInfo>& Allocator::GetLeafFuncOutputInputReuseMap(Function& leafFunc)
{
    return leafFuncOutputInputReuseMap_[&leafFunc];
}

void Allocator::ScanParentOps(Function& leafFunc, const Operation& parent,
                              std::unordered_set<LogicalTensorPtr>& visited, std::unordered_set<Operation*>& operations)
{
    for (LogicalTensorPtr in : parent.GetIOperands()) {
        // 检查是否为leafFunc输入边界
        if (leafFunc.IsFromInCast(in)) {
            continue;
        }
        // 跳过已经遍历过的tensor
        if (visited.find(in) != visited.end()) {
            continue;
        }
        visited.insert(in);
        for (Operation* producer : in->GetProducers()) {
            operations.insert(producer);
        }
    }
}

bool Allocator::CheckReuseOp(const std::unordered_set<Operation*>& operations, std::deque<Operation*>& parents,
                             const WorkspaceInfo& outWspInfo,
                             std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap,
                             std::vector<WorkspaceInfo>& leafFuncReuseMap)
{
    auto& out = outWspInfo.tensor;
    for (Operation* operation : operations) {
        if (!OpcodeManager::Inst().IsCopyIn(operation->GetOpcode())) {
            parents.push_back(operation);
            continue;
        }
        LogicalTensorPtr copyInInput = operation->GetIOperands()[0];
        auto iter = inputWorkspaceInfoMap.find(copyInInput);
        if (iter == inputWorkspaceInfoMap.end()) {
            parents.push_back(operation);
            continue;
        }
        // 检查复用条件
        WorkspaceInfo& candidate = iter->second;
        bool countCheck = candidate.count == 1;
        bool sizeCheck = candidate.size >= outWspInfo.size && candidate.size < outWspInfo.size * MEM_PROPORTION_COEFF;
        bool usageCheck = !candidate.used;
        bool rawReuseCompatible = IsRawQualified(outWspInfo, candidate);
        if (countCheck && sizeCheck && usageCheck && rawReuseCompatible) {
            candidate.used = true;
            leafFuncReuseMap[outWspInfo.position] = candidate;
            // 记录复用日志
            APASS_LOG_INFO_F(Elements::Tensor,
                             "Outcast %d (rawmagic %d) can reuse incast %d (rawmagic %d) size [%zu : %zu].", out->magic,
                             out->tensor->rawmagic, copyInInput->magic, copyInInput->tensor->rawmagic, candidate.size,
                             outWspInfo.size);
            return false;
        }
    }
    return true;
}

// 在不引入额外同步的情况下，完成内存的复用，找到某一个CopyOut的前驱的CopyIn，依赖关系天然存在
// 极限的复用，可以不考虑依赖关系，只看节点之间的顺序，在后续insert sync时可以插入mte3 wait
// mte2的同步，但是可能会有性能劣化。
void Allocator::FindReusableInputForOutput(Function& leafFunc, Operation& op, const WorkspaceInfo& outWspInfo,
                                           std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap,
                                           std::vector<WorkspaceInfo>& leafFuncReuseMap)
{
    APASS_LOG_DEBUG_F(Elements::Tensor, "Searching reusable input for output tensor %d (rawmagic %d).",
                      outWspInfo.tensor->magic, outWspInfo.tensor->tensor->rawmagic);
    std::deque<Operation*> parents;
    // 已访问tensor集合, 节省BFS搜索时长，并且防止出现环路后进入死循环
    std::unordered_set<LogicalTensorPtr> visited;
    parents.push_back(&op);
    while (!parents.empty()) {
        Operation* parent = parents.front();
        parents.pop_front();
        // 存储本层的所有操作，容器可自动去除重复值
        std::unordered_set<Operation*> operations;
        ScanParentOps(leafFunc, *parent, visited, operations);
        if (!CheckReuseOp(operations, parents, outWspInfo, inputWorkspaceInfoMap, leafFuncReuseMap)) {
            APASS_LOG_INFO_F(Elements::Tensor, "CheckReuseOp for leaf function: %s hash %lu.",
                             leafFunc.GetMagicName().c_str(), leafFunc.GetFunctionHash().GetHash());
            return;
        }
    }
}

void Allocator::ProcessOutputForGlobalMemoryReuse(
    Function& leafFunc, WorkspaceInfo& wspInfo,
    std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap,
    std::vector<WorkspaceInfo>& leafFuncReuseMap)
{
    auto& out = wspInfo.tensor;
    if (wspInfo.count != 1) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor magic %d (rawmagic %d) not 1.", out->magic, out->tensor->rawmagic);
        return;
    }
    auto& producers = out->GetProducers();
    if (producers.size() > 1) {
        return;
    }
    if (producers.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Tensor %d producer is empty, function hash %lu.", out->magic,
                         leafFunc.GetFunctionHash().GetHash());
        return;
    }
    Operation* producer = *producers.begin();
    if (!OpcodeManager::Inst().IsCopyOut(producer->GetOpcode())) {
        return;
    }
    auto& producerIn = producer->GetIOperands()[0];
    if (producerIn->shape != out->shape || producerIn->shape != out->tensor->rawshape) {
        return;
    }

    // 寻找可复用的输入tensor
    FindReusableInputForOutput(leafFunc, *producer, wspInfo, inputWorkspaceInfoMap, leafFuncReuseMap);
}

bool GetCopyInSize(LogicalTensorPtr& in, Operation* copyIn, uint64_t& size)
{
    if (copyIn == nullptr || !OpcodeManager::Inst().IsCopyIn(copyIn->GetOpcode())) {
        return false;
    }
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
    if (attr == nullptr || attr->IsDynFromOffset()) {
        return false;
    }

    // 计算内存大小
    const size_t bytesPerElement = BytesOf(in->tensor->datatype);
    size = bytesPerElement;
    for (const auto shape : copyIn->GetOOperands()[0]->GetRawTensor()->GetRawShape()) {
        size *= shape;
    }

    return true;
}

/* Reshape的复用之后的Offset计算比较复杂，暂时不复用 */
bool HasReshapeConsumer(const LogicalTensorPtr& tensor)
{
    for (Operation* consumer : tensor->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

void Allocator::CollectOutputTensor(Function& leafFunc, std::unordered_map<LogicalTensorPtr, size_t>& tensorToInfo,
                                    std::vector<WorkspaceInfo>& outWspInfo,
                                    std::vector<WorkspaceInfo>& leafFuncReuseMap)
{
    for (size_t i = 0; i < leafFunc.outCasts_.size(); ++i) {
        LogicalTensorPtr out = leafFunc.outCasts_[i];
        leafFuncReuseMap.emplace_back(WorkspaceInfo());

        // 跳过根输出或特殊情况的tensor
        if (rootOutCasts_.count(out->GetRawMagic()) || HasReshapeConsumer(out) || out->tensor->actualRawmagic != -1) {
            continue;
        }

        // 记录输出tensor信息
        auto iter = tensorToInfo.find(out);
        if (iter != tensorToInfo.end()) {
            outWspInfo[iter->second].count++;
            continue;
        }
        uint64_t rawDataSize = static_cast<uint64_t>(out->tensor->GetRawDataSize());
        outWspInfo.emplace_back(WorkspaceInfo(1, i, rawDataSize, out));
        tensorToInfo[out] = outWspInfo.size() - 1;
    }
}

void Allocator::CollectInputTensor(Function& leafFunc,
                                   std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap)
{
    for (size_t i = 0; i < leafFunc.inCasts_.size(); ++i) {
        LogicalTensorPtr in = leafFunc.inCasts_[i];

        // 跳过根输入或特殊情况的tensor
        if (rootInCasts_.count(in->GetRawMagic()) || in->tensor->actualRawmagic != -1) {
            continue;
        }

        // 记录输入tensor信息
        auto iter = inputWorkspaceInfoMap.find(in);
        if (iter != inputWorkspaceInfoMap.end()) {
            iter->second.count++;
            continue;
        }
        Operation* consumer = *(in->GetConsumers().begin());
        uint64_t size = 0;
        if (GetCopyInSize(in, consumer, size)) {
            inputWorkspaceInfoMap[in] = WorkspaceInfo(1, i, size, in);
        }
    }
}

// leaf内复用注意：
// 1. leaf的Incast、Outcast如果有重复，那么是不能被复用，也不需要复用的。
// 2. reshape的相关处理
// 3. 对outcast中没有actual raw magic且数量为1，且rawshape = shape的做广度优先遍历,
// 找到距离最近的一个incast，并将他们标记为 一对可以复用的incast outcast pair。
// 4. 如果incast的size大于outcast，那么也可以复用，但是要给outcast的tensor上打上偏移量。
// 5. 如果outcast的shape不等于rawshape，那么不能复用，这种场景较为复杂，有优化空间
// 6. 如果outcast在leafFunction中存在后继的reshape，那么不需要复用
void Allocator::ProcessLeafGlobalMemoryReuse(Function& leafFunc)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Start processing the reuse of leaf function: %s (hash %lu).",
                      leafFunc.GetMagicName().c_str(), leafFunc.GetFunctionHash().GetHash());
    std::unordered_map<LogicalTensorPtr, size_t> tensorToInfo;
    std::vector<WorkspaceInfo> outWspInfo;
    std::unordered_map<LogicalTensorPtr, WorkspaceInfo> inputWorkspaceInfoMap;
    // 获取或创建该 leaf function 的 LeafFuncOutputInputReuseMap_ 数据
    std::vector<WorkspaceInfo>& leafFuncReuseMap = GetLeafFuncOutputInputReuseMap(leafFunc);
    CollectOutputTensor(leafFunc, tensorToInfo, outWspInfo, leafFuncReuseMap);
    CollectInputTensor(leafFunc, inputWorkspaceInfoMap);
    // 处理每个输出tensor的内存复用
    for (auto& wspInfo : outWspInfo) {
        ProcessOutputForGlobalMemoryReuse(leafFunc, wspInfo, inputWorkspaceInfoMap, leafFuncReuseMap);
        APASS_LOG_DEBUG_F(Elements::Tensor, "End reuse check for output: magic=%d rawmagic=%d size=%lu count=%ld.",
                          wspInfo.tensor->magic, wspInfo.tensor->tensor->rawmagic, wspInfo.size, wspInfo.count);
    }
}

bool Allocator::CheckNoOverlapForTwoConsumers(const std::vector<std::vector<int>>& allOffsets,
                                              const std::vector<std::vector<int>>& allShapes, size_t dimCount)
{
    for (size_t d = 0; d < dimCount; ++d) {
        int firstEnd = allOffsets[0][d] + allShapes[0][d] - 1;
        int secondEnd = allOffsets[1][d] + allShapes[1][d] - 1;
        if (firstEnd < allOffsets[1][d] || secondEnd < allOffsets[0][d]) {
            return true;
        }
    }
    return false;
}

size_t Allocator::SelectSweepAxis(const std::vector<std::pair<int64_t, int64_t>>& ranges, size_t consumerCount,
                                  size_t dimCount)
{
    size_t sweepAxis = 0;
    int64_t maxSpan = std::numeric_limits<int64_t>::min();
    for (size_t d = 0; d < dimCount; ++d) {
        int64_t minStart = std::numeric_limits<int64_t>::max();
        int64_t maxStart = std::numeric_limits<int64_t>::min();
        for (size_t i = 0; i < consumerCount; ++i) {
            minStart = std::min(minStart, ranges[i * dimCount + d].first);
            maxStart = std::max(maxStart, ranges[i * dimCount + d].first);
        }
        int64_t span = maxStart - minStart;
        if (span > maxSpan) {
            maxSpan = span;
            sweepAxis = d;
        }
    }
    return sweepAxis;
}

// active-set 扫描线：仅与 sweep axis 上尚未结束的消费者做全维度重叠判断。
// 活跃集自动收缩，稀疏场景下候选对数远小于 n²；
bool Allocator::CheckNoOverlapByActiveSet(const std::vector<std::pair<int64_t, int64_t>>& ranges, size_t consumerCount,
                                          size_t dimCount, size_t sweepAxis)
{
    // 按 sweep axis 的 start 排序（end 作 tiebreaker，保证同 start 时短区间在前）
    std::vector<size_t> order(consumerCount);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const auto& l = ranges[lhs * dimCount + sweepAxis];
        const auto& r = ranges[rhs * dimCount + sweepAxis];
        return l.first == r.first ? l.second < r.second : l.first < r.first;
    });

    std::vector<size_t> active;
    active.reserve(consumerCount);
    for (size_t idx : order) {
        const auto& curStart = ranges[idx * dimCount + sweepAxis].first;
        size_t writeIdx = 0;
        for (size_t actIdx : active) {
            if (ranges[actIdx * dimCount + sweepAxis].second >= curStart) {
                active[writeIdx++] = actIdx;
            }
        }
        active.resize(writeIdx);

        const auto* curRanges = &ranges[idx * dimCount];
        for (size_t actIdx : active) {
            const auto* actRanges = &ranges[actIdx * dimCount];
            bool overlap = true;
            for (size_t d = 0; d < dimCount; ++d) {
                if (actRanges[d].second < curRanges[d].first || curRanges[d].second < actRanges[d].first) {
                    overlap = false;
                    break;
                }
            }
            if (overlap) {
                return false;
            }
        }
        active.emplace_back(idx);
    }
    return true;
}

bool Allocator::CheckAllConsumerAccessNoOverlap(const std::vector<std::vector<int>>& allOffsets,
                                                const std::vector<std::vector<int>>& allShapes)
{
    size_t consumerCount = allOffsets.size();
    if (consumerCount <= 1) {
        return true;
    }
    if (allShapes.size() != consumerCount) {
        return false;
    }

    size_t dimCount = allOffsets[0].size();
    PASS_ASSERT(dimCount > 0) << "allOffsets[0] is empty";
    for (size_t i = 0; i < consumerCount; ++i) {
        PASS_ASSERT(allOffsets[i].size() == dimCount && allShapes[i].size() == dimCount)
            << "dim mismatch at consumer " << i << ": allOffsets[i].size=" << allOffsets[i].size()
            << " allShapes[i].size=" << allShapes[i].size() << " dimCount=" << dimCount;
    }

    APASS_LOG_DEBUG_F(Elements::Tensor, "CheckAllConsumerAccessNoOverlap consumerCount=%zu dimCount=%zu.",
                      consumerCount, dimCount);

    if (consumerCount == TWO_CONSUMER_FAST_PATH_COUNT) {
        return CheckNoOverlapForTwoConsumers(allOffsets, allShapes, dimCount);
    }

    std::vector<std::pair<int64_t, int64_t>> ranges(consumerCount * dimCount);
    for (size_t i = 0; i < consumerCount; ++i) {
        for (size_t d = 0; d < dimCount; ++d) {
            ranges[i * dimCount + d] = {allOffsets[i][d], allOffsets[i][d] + allShapes[i][d] - 1};
        }
    }

    size_t sweepAxis = SelectSweepAxis(ranges, consumerCount, dimCount);
    return CheckNoOverlapByActiveSet(ranges, consumerCount, dimCount, sweepAxis);
}

bool Allocator::ExtractImmediateArguments(const std::vector<SymbolicScalar>& argList, size_t startIndex, size_t count,
                                          std::vector<int>& result)
{
    for (size_t argIdx = startIndex; argIdx < startIndex + count; argIdx++) {
        if (!argList[argIdx].IsImmediate()) {
            return false;
        }
        result.push_back(argList[argIdx].Concrete());
    }
    return true;
}

void Allocator::RecordAllConsumerShapeAndOffset(LogicalTensorPtr& out, std::vector<std::vector<int>>& allOffsets,
                                                std::vector<std::vector<int>>& allShapes, bool& canReuse)
{
    const size_t outShapeSize = out->shape.size();
    const size_t rawShapeStartIdx = OFFSET_INDEX + outShapeSize;
    for (Operation* consumer : out->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_CALL) {
            continue;
        }

        // 查找当前输出tensor在消费者输入中的位置
        for (size_t inputIdx = 0; inputIdx < consumer->GetIOperands().size(); inputIdx++) {
            if (consumer->GetIOperands()[inputIdx] != out) {
                continue;
            }

            // 准备存储当前消费者的偏移和形状
            allOffsets.emplace_back();
            allShapes.emplace_back();
            auto& currentOffset = allOffsets.back();
            auto& currentShape = allShapes.back();

            auto attr = std::dynamic_pointer_cast<CallOpAttribute>(consumer->GetOpAttribute());
            if (attr == nullptr) {
                continue;
            }
            auto& argList = attr->GetArgList()[inputIdx];
            // 提取offset
            if (!ExtractImmediateArguments(argList, OFFSET_INDEX, outShapeSize, currentOffset)) {
                canReuse = false;
                return;
            }
            // 提取shape
            if (!ExtractImmediateArguments(argList, rawShapeStartIdx, outShapeSize, currentShape)) {
                canReuse = false;
                return;
            }
            if (currentOffset.size() != outShapeSize || currentShape.size() != outShapeSize) {
                canReuse = false;
                return;
            }
        }
    }
}

void Allocator::MarkNonOverlappingConsumerTensors()
{
    for (Operation& operation : function_->Operations(false)) {
        for (LogicalTensorPtr outputTensor : operation.GetOOperands()) {
            // 跳过根输出tensor
            if (rootOutCasts_.count(outputTensor->GetRawMagic()) != 0) {
                continue;
            }

            bool canReuse = true;

            // 单个消费者直接认为消费者无重叠
            if (outputTensor->GetConsumers().size() == 1) {
                tensorConsumerNoOverlap_.emplace(outputTensor->GetMagic());
                continue;
            }

            // 收集并检查所有消费者的shape和offset
            std::vector<std::vector<int>> consumerOffsets;
            std::vector<std::vector<int>> consumerShapes;
            RecordAllConsumerShapeAndOffset(outputTensor, consumerOffsets, consumerShapes, canReuse);

            if (!canReuse) {
                continue;
            }
            // 检查所有消费者的内存重叠情况
            bool noOverlap = CheckAllConsumerAccessNoOverlap(consumerOffsets, consumerShapes);
            if (noOverlap) {
                tensorConsumerNoOverlap_.emplace(outputTensor->GetMagic());
            }
        }
    }
}

void Allocator::InitializeLeafGlobalMemoryReuse()
{
    if (function_->GetFunctionType() != FunctionType::DYNAMIC_LOOP_PATH) {
        return;
    }
    if (skipReuseJudgment_) {
        APASS_LOG_EVENT_F(Elements::Tensor, "Skip reuse judgment");
        return;
    }
    for (auto& program : function_->programs_) {
        Function* leafProgram = program.second;
        ProcessLeafGlobalMemoryReuse(*leafProgram);
    }
    // 标记无重叠的消费者张量
    MarkNonOverlappingConsumerTensors();
}

/* 如果previous tensor的consumer中包含了tensor的producer，那么不能复用。
   其余场景，如果previous
   tensor的所有consumer到tensor的一个producer之间有连接，那么意味着，tensor的producer的执行，一定要
   等到preivous的所有consumer都执行完。 */
bool TensorBucket::HasTopoDependency(const LargeBitmap& producerOpsBitmap) const
{
    for (const uint64_t consumerOpIndex : consumerOpIdxs_) {
        if (!producerOpsBitmap.GetBit(consumerOpIndex)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Missing connection to consumer op %lu.", consumerOpIndex);
            return false;
        }
    }
    return true;
}

void Allocator::UpdateTensorMagicToBucketIdx(const std::set<LogicalTensorPtr>& tensors, int bucketIdx)
{
    for (auto tensor : tensors) {
        tensorMagicToBucketIdx_[tensor->GetMagic()] = bucketIdx;
    }
}

TensorBucket& Allocator::HandleNewBuckets(const TensorsDesc& tensorsDesc, int64_t rawDataSize, int magic)
{
    buckets_.emplace_back();
    UpdateTensorMagicToBucketIdx(tensorsDesc.tensors, buckets_.size() - 1);
    bucketsIdxToSize_[buckets_.size() - 1] = rawDataSize;
    APASS_LOG_DEBUG_F(Elements::Tensor, "Creating new bucket %zu for tensor magic=%d.", buckets_.size(), magic);
    return buckets_.back();
}

TensorBucket& Allocator::GetBestFitBucket(const TensorsDesc& tensorsDesc)
{
    if (tensorsDesc.isDummy) {
        return dummyPackets_;
    }
    auto& first = *(tensorsDesc.tensors.begin());
    int64_t rawDataSize = first->tensor->GetRawDataSize();
    int64_t rawDataSizeKey = rawDataSize / MEM_PROPORTION_COEFF;
    APASS_LOG_DEBUG_F(Elements::Tensor, "Searching bucket for tensor: magic=%d rawmagic=%d size=%ld.", first->magic,
                      first->tensor->rawmagic, rawDataSize);
    if (skipReuseJudgment_) {
        return HandleNewBuckets(tensorsDesc, rawDataSize, first->magic);
    }

    std::deque<LogicalTensorPtr> predecessorTensors(tensorsDesc.tensors.begin(), tensorsDesc.tensors.end());
    std::unordered_set<int> visitedTensor;
    std::unordered_set<int> visitedBucket;
    while (predecessorTensors.size() > 0) {
        LogicalTensorPtr ptr = predecessorTensors.front();
        predecessorTensors.pop_front();
        if (visitedTensor.count(ptr->GetMagic()) > 0) {
            continue;
        }
        visitedTensor.insert(ptr->GetMagic());

        if (tensorMagicToBucketIdx_.count(ptr->GetMagic()) > 0) {
            int bucketIdx = tensorMagicToBucketIdx_[ptr->GetMagic()];
            if (visitedBucket.count(bucketIdx) == 0) {
                visitedBucket.insert(bucketIdx);
                if (bucketsIdxToSize_[bucketIdx] >= rawDataSizeKey &&
                    buckets_[bucketIdx].HasTopoDependency(tensorsDesc.connectionOpsBitmap)) {
                    bucketsIdxToSize_[bucketIdx] = rawDataSize;
                    UpdateTensorMagicToBucketIdx(tensorsDesc.tensors, bucketIdx);
                    APASS_LOG_DEBUG_F(Elements::Tensor, "Reusing bucket %d for tensor magic=%d.", bucketIdx,
                                      first->magic);
                    return buckets_[bucketIdx];
                }
            }
        }

        for (const auto& op : ptr->GetProducers()) {
            for (const auto& tensor : op->GetIOperands()) {
                predecessorTensors.push_back(tensor);
            }
        }
    }

    return HandleNewBuckets(tensorsDesc, rawDataSize, first->magic);
}

bool Allocator::CalOffsetRawShape(size_t dimCount, const std::vector<SymbolicScalar>& argList,
                                  std::vector<int>& offsets, std::vector<int>& rawShapes) const
{
    const size_t offsetStartIdx = OFFSET_INDEX;
    const size_t rawShapeStartIdx = OFFSET_INDEX + 2 * dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        const size_t argPos = offsetStartIdx + i;
        if (argPos >= argList.size() || !argList[argPos].IsImmediate()) {
            return false;
        }
        offsets.emplace_back(argList[argPos].Concrete());
    }
    for (size_t i = 0; i < dimCount; i++) {
        const size_t argPos = rawShapeStartIdx + i;
        if (argPos >= argList.size() || !argList[argPos].IsImmediate()) {
            return false;
        }
        rawShapes.emplace_back(argList[argPos].Concrete());
    }
    return true;
}

void Allocator::CalStridesStorageOffset(size_t dimCount, const LogicalTensorPtr& input, std::vector<int>& offsets,
                                        std::vector<int>& rawShapes, uint64_t& storageOffset) const
{
    // 计算步长
    std::vector<int> strides(dimCount, 1);
    for (int i = static_cast<int>(dimCount) - 2; i >= 0; i--) {
        strides[i] = rawShapes[i + 1] * strides[i + 1];
    }
    // 计算存储偏移量
    for (size_t i = 0; i < dimCount; ++i) {
        storageOffset += offsets[i] * strides[i];
    }
    const size_t bytesPerElement = BytesOf(input->tensor->datatype);
    storageOffset *= bytesPerElement;
}

bool Allocator::GetStorageOffsetByCall(Operation& callOp, size_t inputIdx, uint64_t& storageOffset) const
{
    auto& input = callOp.GetIOperands()[inputIdx];
    if (input == nullptr) {
        return false;
    }
    // 以input的storageOffset_为基准，计算output的storageOffset_
    storageOffset = input->storageOffset_;

    // 获取参数列表
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(callOp.GetOpAttribute());
    if (callAttr == nullptr) {
        return false;
    }
    const size_t argIdx = inputIdx;
    if (argIdx >= callAttr->GetArgList().size()) {
        return false;
    }
    auto& argList = callAttr->GetArgList()[argIdx];

    // 提取offset和shape
    const size_t dimCount = input->shape.size();
    std::vector<int> offsets;
    std::vector<int> rawShapes;
    if (!CalOffsetRawShape(dimCount, argList, offsets, rawShapes)) {
        return false;
    }
    CalStridesStorageOffset(dimCount, input, offsets, rawShapes, storageOffset);
    return true;
}

bool Allocator::CheckAllConsumersConnectedToOp(const LogicalTensorPtr& tensor, Operation& op) const
{
    for (const Operation* consumer : tensor->GetConsumers()) {
        if (consumer == &op) {
            continue;
        }
        // 检查消费者是否通过连接矩阵连接到目标操作
        if (!connectionMatrix_.IsConnected(*consumer, op)) {
            return false;
        }
    }
    return true;
}

bool Allocator::TryReuseInputForOutput(Operation& callOp, size_t outputIdx, LogicalTensorPtr& reusedInput,
                                       uint64_t& storageOffset) const
{
    if (callOp.GetOpcode() != Opcode::OP_CALL) {
        return false;
    }
    const auto calleeHash = callOp.GetCalleeHash();
    auto cacheValue = Program::GetInstance().TryHitCahce(calleeHash);
    if (cacheValue == std::nullopt) {
        APASS_LOG_WARN_F(Elements::Operation, "Cannot find program hash %lu by op %d.",
                         callOp.GetCalleeHash().GetHash(), callOp.opmagic);
        return false;
    }
    Function* leafProgram = cacheValue->GetFunction();

    // 从 map 中获取该 program 的 leafFuncOutputInputReuseMap_ 数据
    const auto reuseInfoIt = leafFuncOutputInputReuseMap_.find(leafProgram);
    if (reuseInfoIt == leafFuncOutputInputReuseMap_.end()) {
        return false;
    }
    const auto& reuseMapping = reuseInfoIt->second;

    // 验证输出索引有效性
    if (outputIdx >= reuseMapping.size()) {
        return false;
    }

    // 获取复用配置
    const auto& incastInfo = reuseMapping[outputIdx];
    const size_t incastIdx = incastInfo.position;
    if (incastInfo.count == -1 || incastIdx >= callOp.GetIOperands().size()) {
        return false;
    }
    // 获取候选输入tensor
    LogicalTensorPtr candidateInput = callOp.GetIOperands()[incastIdx];
    // 检查输入tensor的消费者访问重叠情况，决定该候选输入tensor的内存是否可以被复用
    if (tensorConsumerNoOverlap_.count(candidateInput->GetMagic()) == 0) {
        if (!CheckAllConsumersConnectedToOp(candidateInput, callOp)) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Input %d has multiple consumers not linked to op %d.",
                              candidateInput->magic, callOp.opmagic);
            return false;
        }
    }
    // 候选输入tensor通过检查，输出结果
    reusedInput = candidateInput;
    // 计算存储偏移量
    if (!GetStorageOffsetByCall(callOp, incastIdx, storageOffset)) {
        APASS_LOG_WARN_F(Elements::Tensor, "Invalid offset for input %d.", candidateInput->magic);
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Callop %d leaf function %s (hash %lu) output %zu reuses input %zu.",
                      callOp.opmagic, leafProgram->GetMagicName().c_str(), leafProgram->GetFunctionHash().GetHash(),
                      outputIdx, incastIdx);
    return true;
}

void Allocator::UpdateStorageForActualRaw(LogicalTensorPtr& input) const
{
    if (!input || !input->tensor || input->tensor->actualRawmagic == -1) {
        return;
    }

    // 获取当前actualRawmagic的tensor集合
    int64_t actualRawMagic = input->tensor->actualRawmagic;
    auto tensorMap = GraphUtils::GetTensorsByRawMagic(input->BelongFunction(), actualRawMagic);

    // 更新所有相同actualRawmagic的tensor的存储
    for (const auto& tensor : tensorMap) {
        tensor->storage_ = input->storage_;
    }
}

void UpdateCallOpRawShape(Operation& consumer, const LogicalTensorPtr& output)
{
    IRBuilder builder;
    const std::vector<LogicalTensorPtr>& consumerInputs = consumer.GetIOperands();
    const size_t outputShapeSize = output->shape.size();
    const size_t rawShapeStartIndex = OFFSET_INDEX + RAW_SHAPE_POS * outputShapeSize;
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(consumer.GetOpAttribute());
    if (callAttr == nullptr) {
        return;
    }

    // 遍历所有输入操作tensor
    for (size_t inputIndex = 0; inputIndex < consumerInputs.size(); inputIndex++) {
        if (consumerInputs[inputIndex] != output) {
            continue;
        }

        // 获取参数列表
        auto& argList = callAttr->GetArgList()[inputIndex];

        // 更新原始形状参数
        for (size_t dimIndex = 0; dimIndex < outputShapeSize; dimIndex++) {
            size_t argPosition = rawShapeStartIndex + dimIndex;
            argList[argPosition] = builder.CreateConstInt(output->tensor->rawshape[dimIndex]);
        }
    }
}

void RefreshCallRawShape(Operation& callOp, size_t outputIndex, const LogicalTensorPtr& output,
                         const LogicalTensorPtr& reusableInput)
{
    IRBuilder builder;
    if (output->tensor->rawshape == reusableInput->tensor->rawshape) {
        return;
    }
    if (output->tensor->GetRawDataSize() == reusableInput->tensor->GetRawDataSize()) {
        return;
    }
    // 刷新producer CallOp的Attr中的output rawshape数据
    APASS_LOG_DEBUG_F(Elements::Tensor, "Updating rawshape for tensor %d: %s -> %s", output->magic,
                      IntVecToStr(output->tensor->rawshape).c_str(),
                      IntVecToStr(reusableInput->tensor->rawshape).c_str());
    output->tensor->UpdateRawShape(reusableInput->tensor->rawshape);
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(callOp.GetOpAttribute());
    if (callAttr == nullptr) {
        return;
    }
    size_t argListIndex = callOp.GetIOperands().size() + outputIndex;
    auto& argList = callAttr->GetArgList()[argListIndex];

    // 刷新producer CallOp的Attr中的output rawshape数据
    const size_t shapeSize = reusableInput->shape.size();
    const size_t rawShapeStartIdx = OFFSET_INDEX + RAW_SHAPE_POS * shapeSize;

    for (size_t dim = 0; dim < shapeSize; dim++) {
        size_t argPos = rawShapeStartIdx + dim;
        argList[argPos] = builder.CreateConstInt(output->tensor->rawshape[dim]);
    }

    // 刷新对应的consumer CallOp的Attr中的input rawshape数据
    for (Operation* consumer : output->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_CALL) {
            APASS_LOG_WARN_F(Elements::Operation, "Output magic %d consumer is %d %s.", output->magic,
                             consumer->opmagic, consumer->GetOpcodeStr().c_str());
            continue;
        }
        UpdateCallOpRawShape(*consumer, output);
    }
}

void Allocator::HandleNewTensor(Operation& callOp, size_t outputIdx, LogicalTensorPtr& outputTensor)
{
    LogicalTensorPtr reusableInput = nullptr;
    const bool canReuse = TryReuseInputForOutput(callOp, outputIdx, reusableInput, outputTensor->storageOffset_);
    if (canReuse && reusableInput != nullptr) {
        // 可以复用，复用输入存储路径
        outputTensor->storage_ = reusableInput->storage_;
        auto storageRecord = storageMap_.find(reusableInput->GetRawMagic());
        if (storageRecord == storageMap_.end()) {
            APASS_LOG_WARN_F(Elements::Tensor, "Cannot find reused input tensor: magic %d, rawmagic %d.",
                             reusableInput->magic, reusableInput->GetRawMagic());
            return;
        }
        RefreshCallRawShape(callOp, outputIdx, outputTensor, reusableInput);

        // 将新张量添加到现有存储组
        const size_t storageIndex = storageRecord->second;
        TensorsDesc& tensorsDesc = storageNeedToAllocate_.at(storageIndex);
        tensorsDesc.tensors.emplace(outputTensor);
        storageMap_.emplace(outputTensor->GetRawMagic(), storageIndex);

        APASS_LOG_INFO_F(Elements::Tensor, "Reused storage for new tensor: magic=%d via input=%d.", outputTensor->magic,
                         reusableInput->magic);
        return;
    }

    // 不可复用，创建新存储
    const uint64_t alignedSize = Align(static_cast<uint64_t>(outputTensor->tensor->GetRawDataSize()));
    outputTensor->storage_ = std::make_shared<Storage>(MemoryType::MEM_WORKSPACE, outputTensor->GetRawMagic(),
                                                       alignedSize);

    // 创建新存储组
    TensorsDesc tensorsDesc(function_);
    tensorsDesc.tensors.emplace(outputTensor);
    tensorsDesc.isDummy = outputTensor->GetProducers().empty();
    const size_t newIndex = storageNeedToAllocate_.size();
    storageNeedToAllocate_.emplace_back(tensorsDesc);
    storageMap_.emplace(outputTensor->GetRawMagic(), newIndex);

    // 处理actualRawmagic的tensor集合
    UpdateStorageForActualRaw(outputTensor);
    APASS_LOG_INFO_F(Elements::Tensor, "New storage created for tensor: magic=%d size=%lu.", outputTensor->magic,
                     outputTensor->storage_->length_);
}

void Allocator::ProcessOperations()
{
    APASS_LOG_DEBUG_F(Elements::Operation, "=== START ProcessOperations ===");
    auto allOperations = function_->Operations(false);
    for (size_t opIndex = 0; opIndex < allOperations.size(); ++opIndex) {
        Operation& currentOp = allOperations[opIndex];
        for (size_t outputIdx = 0; outputIdx < currentOp.GetOOperands().size(); ++outputIdx) {
            LogicalTensorPtr outputTensor = currentOp.GetOOperands()[outputIdx];
            // 跳过边界tensor
            if (function_->IsFromInCast(outputTensor) || function_->IsFromOutCast(outputTensor)) {
                continue;
            }

            // 跳过已有存储分配的tensor
            if (outputTensor->storage_ != nullptr) {
                // 若tensor的存储由UpdateStorageForActualRaw等路径预先分配（通过actualRawmagic别名），
                // 它可能未被加入对应storage group的tensorsDesc.tensors集合。此处补充加入，
                // 否则CollectComsuerOpDesc会漏掉该tensor的消费者，导致HasTopoDependency误判复用安全。
                auto storageIter = storageMap_.find(outputTensor->GetRawMagic());
                if (storageIter != storageMap_.end()) {
                    TensorsDesc& tensorsDesc = storageNeedToAllocate_.at(storageIter->second);
                    tensorsDesc.tensors.emplace(outputTensor);
                }
                continue;
            }

            // 检查是否已有存储分配记录
            auto storageIter = storageMap_.find(outputTensor->GetRawMagic());
            if (storageIter == storageMap_.end()) {
                // 处理新tensor存储分配
                HandleNewTensor(currentOp, outputIdx, outputTensor);
                continue;
            }

            // 处理已有存储记录的tensor
            const size_t storageIndex = storageIter->second;
            TensorsDesc& tensorsDesc = storageNeedToAllocate_.at(storageIndex);
            LogicalTensorPtr firstTensor = *(tensorsDesc.tensors.begin());

            // 共享已有存储
            outputTensor->storage_ = firstTensor->storage_;
            outputTensor->storageOffset_ = firstTensor->storageOffset_;
            tensorsDesc.tensors.emplace(outputTensor);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "=== END ProcessOperations ===");
}

void Allocator::InitializeRootCasts()
{
    // 处理所有输入边界tensor
    for (const LogicalTensorPtr& inputCast : function_->inCasts_) {
        rootInCasts_.emplace(inputCast->GetRawMagic());
    }

    // 处理所有输出边界tensor
    for (const LogicalTensorPtr& outputCast : function_->outCasts_) {
        rootOutCasts_.emplace(outputCast->GetRawMagic());
    }
}

void Allocator::Init()
{
    connectionMatrix_.Generate(function_);
    storageMap_.clear();
    InitializeRootCasts();
    InitializeLeafGlobalMemoryReuse();
    ProcessOperations();
    PreProcessStorageGroups();
}

void Allocator::CollectComsuerOpDesc(TensorsDesc& tensorsDesc)
{
    for (const LogicalTensorPtr& tensor : tensorsDesc.tensors) {
        for (Operation* consumer : tensor->GetConsumers()) {
            if (consumer->GetOpcode() != Opcode::OP_CALL) {
                continue;
            }
            uint64_t consumerIndex = connectionMatrix_.GetIndex(*consumer);
            if (consumerIndex == connectionMatrix_.INVALID_INDEX) {
                continue;
            }
            tensorsDesc.consumerOpIdxs.emplace(consumerIndex);
        }
    }
}

void Allocator::RemoveRedundantComsuerOp(TensorsDesc& tensorsDesc)
{
    bool removedConsumer = false;
    for (auto consumerIter = tensorsDesc.consumerOpIdxs.begin(); consumerIter != tensorsDesc.consumerOpIdxs.end();) {
        for (const uint64_t otherOpIndex : tensorsDesc.consumerOpIdxs) {
            if (*consumerIter == otherOpIndex) {
                continue;
            }
            if (connectionMatrix_.IsConnected(*consumerIter, otherOpIndex)) {
                consumerIter = tensorsDesc.consumerOpIdxs.erase(consumerIter);
                removedConsumer = true;
                break;
            }
        }
        if (removedConsumer) {
            removedConsumer = false;
            continue;
        }
        consumerIter++;
    }
}

void Allocator::CollectConnectionOps(TensorsDesc& tensorsDesc)
{
    tensorsDesc.connectionOpsBitmap.SetValues(0xFFFFFFFFFFFFFFFF); // And 操作前，需要将connectionOpsBitmap初始化为全1
    for (const LogicalTensorPtr& tensor : tensorsDesc.tensors) {
        for (Operation* producer : tensor->GetProducers()) {
            if (producer->GetOpcode() != Opcode::OP_CALL) {
                continue;
            }
            // And 操作保证了 clear 的 producer 自身不会再被其他 BitMap 覆写为 1
            tensorsDesc.connectionOpsBitmap.And(connectionMatrix_.GetBitMap(*producer));

            // 由于customerOp和producerOp相同时，不能进行内存复用，所以这里需要将bitmap中的指向本操作的位清零
            uint64_t producerIndex = connectionMatrix_.GetIndex(*producer);
            if (producerIndex == connectionMatrix_.INVALID_INDEX) {
                continue;
            }
            tensorsDesc.connectionOpsBitmap.ClearBit(static_cast<size_t>(producerIndex));
        }
    }
}

void Allocator::StorageNeedToAllocatePreProcess(TensorsDesc& tensorsDesc)
{
    // 收集所有消费者操作索引
    CollectComsuerOpDesc(tensorsDesc);
    // 清理冗余消费者操作索引
    RemoveRedundantComsuerOp(tensorsDesc);
    CollectConnectionOps(tensorsDesc);
}

void Allocator::PreProcessStorageGroups()
{
    for (auto& tensorsDesc : storageNeedToAllocate_) {
        StorageNeedToAllocatePreProcess(tensorsDesc);
    }
}

Status Allocator::UpdateStorageId(TensorsDesc& tensorsDesc, std::unordered_map<int64_t, int>& idMap, int& storageId)
{
    if (tensorsDesc.tensors.empty()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Storage tensors is empty.");
        return SUCCESS;
    }
    auto& tensor = *(tensorsDesc.tensors.begin());
    if (tensor->storage_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor rawMagic:%d, storage is nullptr.", tensor->GetRawMagic());
        return FAILED;
    }

    auto iter = idMap.find(tensor->storage_->start_);
    if (iter != idMap.end()) {
        tensor->storage_->id_ = iter->second;
        return SUCCESS;
    }
    idMap.emplace(std::make_pair(tensor->storage_->start_, storageId));
    tensor->storage_->id_ = storageId;
    storageId++;
    return SUCCESS;
}

Status Allocator::UpdateIncastOutCast()
{
    auto callOps = function_->Operations(false);
    for (const auto& callOp : callOps) {
        if (callOp.GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(callOp.GetOpAttribute());
        if (callAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Op %d callAttr is nullptr.%s", callOp.opmagic,
                              GetFormatBacktrace(callOp).c_str());
            return FAILED;
        }
        auto& incasts = callAttr->invokeInfo_->incastTensorParamList_;
        auto& outcasts = callAttr->invokeInfo_->outcastTensorParamList_;
        if (incasts.size() > callOp.iOperand.size()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Op incasts.size:%zu, is larger than iOperand.size:%zu, opCode:%d.%s", incasts.size(),
                              callOp.iOperand.size(), static_cast<int>(callOp.GetOpcode()),
                              GetFormatBacktrace(callOp).c_str());
            return FAILED;
        }
        if (outcasts.size() > callOp.oOperand.size()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Op outcasts.size:%zu, is larger than oOperand.size:%zu, opCode:%d.%s", outcasts.size(),
                              callOp.oOperand.size(), static_cast<int>(callOp.GetOpcode()),
                              GetFormatBacktrace(callOp).c_str());
            return FAILED;
        }
        for (size_t i = 0; i < incasts.size(); ++i) {
            auto& incast = incasts[i];
            incast.tensor = callOp.iOperand[i];
        }
        for (size_t i = 0; i < outcasts.size(); ++i) {
            auto& outcast = outcasts[i];
            outcast.tensor = callOp.oOperand[i];
        }
    }
    return SUCCESS;
}

Status Allocator::Allocate()
{
    APASS_LOG_INFO_F(Elements::Tensor, "Starting memory allocation with %zu storage entries.",
                     storageNeedToAllocate_.size());
    for (auto& tensorsDesc : storageNeedToAllocate_) {
        auto& tensor = *(tensorsDesc.tensors.begin());
        if (tensor->storage_ == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor rawMagic:%d, storage is nullptr.", tensor->GetRawMagic());
            return FAILED;
        }
        TensorBucket& bucket = GetBestFitBucket(tensorsDesc);
        if (!bucket.AddTensorGroup(tensorsDesc)) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "TensorsDesc.tensors is empty, Cannot add empty tensor group to bucket.");
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "Allocating storage for tensor: rawmagic=%d size=%lu.",
                          tensor->GetRawMagic(), static_cast<unsigned long>(tensor->storage_->length_));
    }
    for (auto& bucket : buckets_) {
        bucket.UpdateOffset(size_);
        size_ += bucket.GetSize();
    }
    dummyPackets_.UpdateOffset(size_);
    size_ += dummyPackets_.GetSize();

    APASS_LOG_DEBUG_F(Elements::Tensor, "Total memory allocated: %lu bytes across %zu buckets.", size_,
                      buckets_.size());
    // 根据storage id刷新 DDRId, start相同认为是一个storage
    std::unordered_map<int64_t, int> idMap;
    int storageId = 0;
    for (auto& tensorsDesc : storageNeedToAllocate_) {
        if (UpdateStorageId(tensorsDesc, idMap, storageId) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Tensor, "UpdateStorageId failed; Please check the UpdateStorageId method.");
            return FAILED;
        }
    }

    return UpdateIncastOutCast();
}

Status GlobalMemoryReuse::RunOnFunction(Function& function)
{
    /* 为incast、outcast类型申请storage，需要正确处理actual rawmagic */
    /* 标注每个CallOp输出Tensor生命周期，生命周期 */
    APASS_LOG_INFO_F(Elements::Operation, "===> Start GlobalMemoryReuse on function: %s.",
                     function.GetMagicName().c_str());
    if (function.rootFunc_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "rootFunc_ is nullptr.");
        return FAILED;
    }
    Allocator allocator(function.rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();
    APASS_LOG_INFO_F(Elements::Operation, "===> Completed GlobalMemoryReuse, Status: %u.", status);
    return status;
}
} // namespace tile_fwk
} // namespace npu
