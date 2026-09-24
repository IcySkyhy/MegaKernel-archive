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
 * \file global_memory_reuse.h
 * \brief
 */

#pragma once
#include <utility>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "connection_matrix.h"

namespace npu::tile_fwk {

struct WorkspaceInfo {
    int64_t count = -1;
    size_t position = 0; // 在leaf的incast/outcast中的位置
    uint64_t size = 0;
    bool used = false;
    LogicalTensorPtr tensor = nullptr;
    WorkspaceInfo() {}
    WorkspaceInfo(int64_t countIn, size_t positionIn, uint64_t sizeIn, const LogicalTensorPtr& tensorIn)
        : count(countIn), position(positionIn), size(sizeIn), tensor(tensorIn)
    {}
};

struct TensorsDesc {
    bool isDummy = false;
    LargeBitmap connectionOpsBitmap; // 作为新tensor，判断是否可以复用bucket时使用的连接bitmap
    std::set<LogicalTensorPtr> tensors;
    std::unordered_set<uint64_t> consumerOpIdxs; // 放入bucket后，作为内存桶是否可以再次复用，需要判断的consumerOp集合
    TensorsDesc() : connectionOpsBitmap(0) {}
    TensorsDesc(Function* func) : connectionOpsBitmap(func->Operations(false).size()) {}
};

class TensorBucket {
public:
    uint64_t GetSize() const { return size_; }

    void UpdateOffset(const uint64_t offset);

    bool AddTensorGroup(const TensorsDesc& tensorsDesc);

    // 检查previous的所有consumer是否有一条到tensor的producer的通路
    // 保证tensor在写的时候，previous的所有consumer都已经读取完毕
    bool HasTopoDependency(const LargeBitmap& producerOpsBitmap) const;

private:
    uint64_t offset_{0};
    uint64_t size_{0};
    std::vector<std::set<LogicalTensorPtr>> tensorGroups_; // 所有rawTensor相同的tensor构成了一个tensorGroup
    std::unordered_set<uint64_t> consumerOpIdxs_; // 新tensor能否复用本bucket，需要判断的consumerOp集合
};

class Allocator {
public:
    explicit Allocator(Function* function) : connectionMatrix_(function), function_(function) {}
    Status Allocate();
    void Init();
    static bool IsRawQualified(const WorkspaceInfo& outWspInfo, const WorkspaceInfo& inWspInfo);
    const std::vector<WorkspaceInfo>& GetLeafFuncOutputInputReuseMap(Function& leafFunc) const;
    std::vector<WorkspaceInfo>& GetLeafFuncOutputInputReuseMap(Function& leafFunc);

private:
    void InitializeRootCasts();
    void ProcessOperations();
    void HandleNewTensor(Operation& callOp, size_t outputIdx, LogicalTensorPtr& outputTensor);
    void CollectComsuerOpDesc(TensorsDesc& tensorsDesc);
    void RemoveRedundantComsuerOp(TensorsDesc& tensorsDesc);
    void CollectConnectionOps(TensorsDesc& tensorsDesc);
    void StorageNeedToAllocatePreProcess(TensorsDesc& tensorsDesc);
    void PreProcessStorageGroups();
    Status UpdateStorageId(TensorsDesc& tensorsDesc, std::unordered_map<int64_t, int>& idMap, int& storageId);
    void MarkNonOverlappingConsumerTensors();
    void RecordAllConsumerShapeAndOffset(LogicalTensorPtr& out, std::vector<std::vector<int>>& allOffsets,
                                         std::vector<std::vector<int>>& allShapes, bool& canReuse);
    bool ExtractImmediateArguments(const std::vector<SymbolicScalar>& argList, size_t startIndex, size_t count,
                                   std::vector<int>& result);
    bool CheckAllConsumerAccessNoOverlap(const std::vector<std::vector<int>>& allOffsets,
                                         const std::vector<std::vector<int>>& allShapes);
    bool CheckNoOverlapForTwoConsumers(const std::vector<std::vector<int>>& allOffsets,
                                       const std::vector<std::vector<int>>& allShapes, size_t dimCount);
    size_t SelectSweepAxis(const std::vector<std::pair<int64_t, int64_t>>& ranges, size_t consumerCount,
                           size_t dimCount);
    bool CheckNoOverlapByActiveSet(const std::vector<std::pair<int64_t, int64_t>>& ranges, size_t consumerCount,
                                   size_t dimCount, size_t sweepAxis);
    void InitializeLeafGlobalMemoryReuse();
    void CollectOutputTensor(Function& leafFunc, std::unordered_map<LogicalTensorPtr, size_t>& tensorToInfo,
                             std::vector<WorkspaceInfo>& outWspInfo, std::vector<WorkspaceInfo>& leafFuncReuseMap);
    void CollectInputTensor(Function& leafFunc,
                            std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap);
    void ProcessLeafGlobalMemoryReuse(Function& leafFunc);

    bool CheckAllConsumersConnectedToOp(const LogicalTensorPtr& tensor, Operation& op) const;
    // 检查某个CallOp的输出是否可以复用输入
    bool TryReuseInputForOutput(Operation& callOp, size_t outputIdx, LogicalTensorPtr& reusedInput,
                                uint64_t& storageOffset) const;
    bool CalOffsetRawShape(size_t dimCount, const std::vector<SymbolicScalar>& argList, std::vector<int>& offsets,
                           std::vector<int>& rawShapes) const;
    void CalStridesStorageOffset(size_t dimCount, const LogicalTensorPtr& input, std::vector<int>& offsets,
                                 std::vector<int>& rawShapes, uint64_t& storageOffset) const;
    bool GetStorageOffsetByCall(Operation& callOp, size_t inputIdx, uint64_t& storageOffset) const;
    void UpdateStorageForActualRaw(LogicalTensorPtr& input) const;
    TensorBucket& GetBestFitBucket(const TensorsDesc& tensorsDesc);
    TensorBucket& HandleNewBuckets(const TensorsDesc& tensorsDesc, int64_t rawDataSize, int magic);
    void UpdateTensorMagicToBucketIdx(const std::set<LogicalTensorPtr>& tensors, int bucketIdx);
    void ScanParentOps(Function& leafFunc, const Operation& parent, std::unordered_set<LogicalTensorPtr>& visited,
                       std::unordered_set<Operation*>& operations);
    bool CheckReuseOp(const std::unordered_set<Operation*>& operations, std::deque<Operation*>& parents,
                      const WorkspaceInfo& outWspInfo,
                      std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap,
                      std::vector<WorkspaceInfo>& leafFuncReuseMap);
    void FindReusableInputForOutput(Function& leafFunc, Operation& op, const WorkspaceInfo& outWspInfo,
                                    std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap,
                                    std::vector<WorkspaceInfo>& leafFuncReuseMap);
    void ProcessOutputForGlobalMemoryReuse(Function& leafFunc, WorkspaceInfo& wspInfo,
                                           std::unordered_map<LogicalTensorPtr, WorkspaceInfo>& inputWorkspaceInfoMap,
                                           std::vector<WorkspaceInfo>& leafFuncReuseMap);
    Status UpdateIncastOutCast();

    std::vector<TensorBucket> buckets_;

    TensorBucket dummyPackets_; // dummy tensor的bucket
    std::unordered_map<int, size_t> storageMap_;
    // 按照topo序排列
    std::vector<TensorsDesc> storageNeedToAllocate_;
    ConnectionMatrix connectionMatrix_; // 标注任意两个leafFunction之间是否有连接
    uint64_t size_{0};
    Function* function_;
    std::unordered_set<int> rootInCasts_;
    std::unordered_set<int> rootOutCasts_;
    std::unordered_set<int> tensorConsumerNoOverlap_;

    // 使用 map 存储，key 为 function 指针
    // function内，outcast可以和哪个incast复用gm内存，-1表示不能复用
    std::unordered_map<Function*, std::vector<WorkspaceInfo>> leafFuncOutputInputReuseMap_;

    std::unordered_map<int, int> tensorMagicToBucketIdx_;
    std::unordered_map<int, int64_t> bucketsIdxToSize_;

    // true：跳过内存复用判断，即不复用内存; false：正常进行内存复用判断
    bool skipReuseJudgment_{false};
};

class GlobalMemoryReuse : public Pass {
public:
    GlobalMemoryReuse() : Pass("GlobalMemoryReuse") {}
    ~GlobalMemoryReuse() override {}
    Status RunOnFunction(Function& function) override;
};
} // namespace npu::tile_fwk
