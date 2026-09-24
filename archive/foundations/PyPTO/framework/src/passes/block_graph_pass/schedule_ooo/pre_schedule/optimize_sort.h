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
 * \file optimize_sort.h
 * \brief Base class for OoO sort algorithms. Uses strategy pattern with factory registration.
 *        Each algorithm inherits OptimizeSort and implements DoSortOps().
 */

#ifndef PASS_OPTIMIZE_SORT_H
#define PASS_OPTIMIZE_SORT_H

#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <set>
#include <string>

namespace npu::tile_fwk {

class OptimizeSort {
public:
    using Factory = std::function<std::unique_ptr<OptimizeSort>(std::vector<Operation*>, Function&)>;

    OptimizeSort(std::vector<Operation*> opList, Function& function) : operations(opList), function_(function) {}
    virtual ~OptimizeSort() = default;

    Status SortOps();

    std::vector<Operation*> GetOperations() { return operations; }
    std::vector<Operation*> operations;

protected:
    void AllocAhead();
    Status ExecuteOp();

    Function& function_;
    ScheduleState state_;

    bool opFinish_{false};
    std::unordered_map<int, int> initBufRefCountCache_;
    std::map<Operation*, std::map<MemoryType, int64_t>> recordBufferAllocate_;
    std::map<Operation*, std::pair<size_t, std::shared_ptr<std::vector<Operation*>>>> recordOpList_;
    std::map<Operation*, MemoryType> recordOpBuffer_;
    std::stack<std::pair<Operation*, MemoryType>> needFreeOpStack_;
    std::map<Operation*, bool> visitedOp_;
    std::map<Operation*, std::unordered_map<int, int>> recordBufRefCount_;

    Operation* backTraceOp_{nullptr};
    std::map<Operation*, std::map<MemoryType, int64_t>> backTraceBufferAllocate_;
    std::map<Operation*, std::pair<size_t, std::shared_ptr<std::vector<Operation*>>>> backTraceOpList_;
    std::map<Operation*, std::unordered_map<int, int>> backTraceBufRefCount_;
    Operation* rollBackNodeOp_{nullptr};

    std::shared_ptr<std::vector<Operation*>> ReorderOp(std::vector<size_t>& preIdx,
                                                       std::shared_ptr<std::vector<Operation*>> curOpList,
                                                       size_t startIndex);
    void FindIndex(Operation* op, std::shared_ptr<std::vector<Operation*>> curOpList, size_t& index);
    Status FindConsumerList(size_t consumerIndex, std::vector<size_t>& preOpList,
                            std::shared_ptr<std::vector<Operation*>> curOpList);
    Status UpdateOOperandPreDependence(size_t startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                                       std::vector<Operation*> consumersGroup);
    void RecoverSymbol(size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList);
    void GetConsumerGroup(std::set<Operation*, Operation::OperationComparator>& consumers,
                          std::vector<Operation*>& consumersGroup);
    void GetStackTop(size_t& startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                     std::map<MemoryType, int64_t>& curMemoryMap);
    Status BacktraceOnMemoryExceeded(size_t& startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                                     std::map<MemoryType, int64_t>& curMemoryMap);
    bool IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size);
    Status ModifyBuffer(std::map<MemoryType, int64_t>& curMemoryMap, MemoryType memType, int64_t size, bool isAdd);
    Status RetireOpBuffer(std::map<MemoryType, int64_t>& curMemoryMap, Operation* op);
    void OpMemoryUpdate(Operation* op, size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList,
                        const std::map<MemoryType, int64_t>& curMemoryMap);
    Status ConsumeOpBuffers(Operation* op);
    Status AllocExecute(Operation* op, std::shared_ptr<std::vector<Operation*>>& curOpList,
                        std::map<MemoryType, int64_t>& curMemoryMap, size_t& startIndex, bool& isContinue);
    Status OpListExecute(std::shared_ptr<std::vector<Operation*>>& curOpList,
                         std::map<MemoryType, int64_t>& curMemoryMap, size_t& startIndex);
    std::shared_ptr<std::vector<Operation*>> ReplaceIndex(std::shared_ptr<std::vector<Operation*>> curOpList,
                                                          std::set<size_t>& advanceIndexList, size_t rollBackIndex);
    bool HasDependency(Operation* rollBackOp, Operation* backOp);
    void GetPreNode(size_t i, std::shared_ptr<std::vector<Operation*>> curOpList, size_t rollBackIndex,
                    size_t backTraceIndex, std::set<size_t>& dependencyIndexList);
    void GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
                          std::shared_ptr<std::vector<Operation*>> curOpList, std::set<size_t>& advanceIndexList);
    Status RollBack(size_t& startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                    std::map<MemoryType, int64_t>& curMemoryMap);

private:
    static std::string ResolveOooSortMode(const std::vector<Operation*>& ops, const ParamConfigs& pc);
    static std::unique_ptr<OptimizeSort> Create(std::vector<Operation*> ops, Function& func, const std::string& mode);

    virtual Status DoSortOps() { return FAILED; }

    static const std::unordered_map<std::string, Factory> SORT_ALGOS;
};
} // namespace npu::tile_fwk
#endif // PASS_OPTIMIZE_SORT_H
