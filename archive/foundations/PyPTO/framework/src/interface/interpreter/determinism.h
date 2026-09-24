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
 * \file determinism.h
 * \brief
 */
/*for flow Verify Tool */

#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>

#include "interface/schema/schema.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

class TraceMemoryRange {
public:
    TraceMemoryRange() = default;
    TraceMemoryRange(uintptr_t begin, uintptr_t end) : begin_(begin), end_(end) {}

    uintptr_t GetBegin() const { return begin_; }
    uintptr_t GetEnd() const { return end_; }

    bool operator==(const TraceMemoryRange& rhs) const { return begin_ == rhs.begin_ && end_ == rhs.end_; }

private:
    uintptr_t begin_{0};
    uintptr_t end_{0};
};

class TraceRawTensorMemory {
public:
    TraceRawTensorMemory() = default;
    TraceRawTensorMemory(const TraceMemoryRange& memoryRange, const std::vector<int64_t>& shape)
        : memoryRange_(memoryRange), shape_(shape)
    {}

    const TraceMemoryRange& GetMemoryRange() const { return memoryRange_; }
    const std::vector<int64_t>& GetShape() const { return shape_; }

    void SetMemoryRange(const TraceMemoryRange& memoryRange) { memoryRange_ = memoryRange; }
    void SetShape(const std::vector<int64_t>& shape) { shape_ = shape; }

private:
    TraceMemoryRange memoryRange_;
    std::vector<int64_t> shape_;
};

class TraceCopy {
public:
    TraceCopy(bool isCopyOut, const std::shared_ptr<TraceRawTensorMemory>& rawTensor,
              const std::vector<int64_t>& offset, const std::vector<int64_t>& shape, bool isAtomicAdd = false)
        : isCopyOut_(isCopyOut), rawTensor_(rawTensor), offset_(offset), shape_(shape), isAtomicAdd_(isAtomicAdd)
    {}

    bool IsCopyOut() const { return isCopyOut_; };
    const std::shared_ptr<TraceRawTensorMemory>& GetRawTensor() const { return rawTensor_; }
    const std::vector<int64_t>& GetOffset() const { return offset_; }
    const std::vector<int64_t>& GetShape() const { return shape_; }
    bool IsAtomicAdd() const { return isAtomicAdd_; }

    void SetOffset(const std::vector<int64_t>& offset) { offset_ = offset; }
    void SetShape(const std::vector<int64_t>& shape) { shape_ = shape; }
    void SetIsAtomicAdd(bool isAtomicAdd) { isAtomicAdd_ = isAtomicAdd; }

    static bool Overlap(const TraceCopy& src, const TraceCopy& dst);

private:
    bool isCopyOut_;
    std::shared_ptr<TraceRawTensorMemory> rawTensor_;
    std::vector<int64_t> offset_;
    std::vector<int64_t> shape_;
    bool isAtomicAdd_;
};

class TraceLeafTaskUid {
public:
    TraceLeafTaskUid() = default;
    TraceLeafTaskUid(int64_t deviceTaskIndex, int64_t dupIndex, int64_t rootIndex, int64_t operationIndex,
                     int64_t leafIndex)
        : deviceTaskIndex_(deviceTaskIndex),
          dupIndex_(dupIndex),
          rootIndex_(rootIndex),
          operationIndex_(operationIndex),
          leafIndex_(leafIndex)
    {}

    int64_t GetDeviceTaskIndex() const { return deviceTaskIndex_; }
    int64_t GetDupIndex() const { return dupIndex_; }
    int64_t GetRootIndex() const { return rootIndex_; }
    int64_t GetOperationIndex() const { return operationIndex_; }
    int64_t GetLeafIndex() const { return leafIndex_; }

    bool operator<(const TraceLeafTaskUid& uid) const;
    std::string Dump() const;

private:
    int64_t deviceTaskIndex_{-1};
    int64_t dupIndex_{-1};
    int64_t rootIndex_{-1};
    int64_t operationIndex_{-1};
    int64_t leafIndex_{-1};
};

class TraceRootTaskUid {
public:
    TraceRootTaskUid() = default;
    TraceRootTaskUid(int64_t deviceTaskIndex, int64_t dupIndex, int64_t rootIndex)
        : deviceTaskIndex_(deviceTaskIndex), dupIndex_(dupIndex), rootIndex_(rootIndex)
    {}

    int64_t GetDeviceTaskIndex() const { return deviceTaskIndex_; }
    int64_t GetDupIndex() const { return dupIndex_; }
    int64_t GetRootIndex() const { return rootIndex_; }

    bool operator<(const TraceRootTaskUid& uid) const;
    std::string Dump() const;

private:
    int64_t deviceTaskIndex_{-1};
    int64_t dupIndex_{-1};
    int64_t rootIndex_{-1};
};

class TraceDeviceTaskUid {
public:
    TraceDeviceTaskUid() = default;
    TraceDeviceTaskUid(int64_t deviceTaskIndex) : deviceTaskIndex_(deviceTaskIndex) {}

    int64_t GetDeviceTaskIndex() const { return deviceTaskIndex_; }

    bool operator<(const TraceDeviceTaskUid& uid) const;
    std::string Dump() const;

private:
    int64_t deviceTaskIndex_{-1};
};

class TraceCoa {
public:
    TraceCoa(uint64_t value, bool isExpr = false) : value_(value), isExpr_(isExpr) {}
    uint64_t GetValue() const { return value_; }
    bool IsExpr() const { return isExpr_; }

    bool operator==(const TraceCoa& coa) const { return isExpr_ == coa.isExpr_ && value_ == coa.value_; }

private:
    uint64_t value_;
    bool isExpr_;
};

class TraceLeafTask {
public:
    TraceLeafTask() = default;
    TraceLeafTask(const TraceLeafTaskUid& uid) : uid_(uid) {}

    const TraceLeafTaskUid& GetUid() const { return uid_; }
    Function* GetLeafFunc() const { return leafFunc_; }
    void SetLeafFunc(Function* func) { leafFunc_ = func; }

    const std::vector<TraceCoa>& GetCoaList() const { return coaList_; }
    std::vector<TraceCoa>& GetCoaList() { return coaList_; }

    const std::vector<TraceCopy>& GetCopyInList() const { return copyInList_; }
    std::vector<TraceCopy>& GetCopyInList() { return copyInList_; }

    const std::vector<TraceCopy>& GetCopyOutList() const { return copyOutList_; }
    std::vector<TraceCopy>& GetCopyOutList() { return copyOutList_; }

    const std::set<TraceLeafTaskUid>& GetPredSet() const { return predSet_; }
    const std::set<TraceLeafTaskUid>& GetSuccSet() const { return succSet_; }
    void AddPred(const TraceLeafTaskUid& pred) { predSet_.insert(pred); }
    void AddSucc(const TraceLeafTaskUid& succ) { succSet_.insert(succ); }

private:
    TraceLeafTaskUid uid_;
    Function* leafFunc_{nullptr};
    std::vector<TraceCoa> coaList_;
    std::vector<TraceCopy> copyInList_;
    std::vector<TraceCopy> copyOutList_;
    std::set<TraceLeafTaskUid> predSet_;
    std::set<TraceLeafTaskUid> succSet_;
};

class TraceRootTaskRawTensorDesc {
public:
    TraceRootTaskRawTensorDesc() {}
    TraceRootTaskRawTensorDesc(int64_t location, uint64_t offsetOrIndex, uint64_t size)
        : location_(location), offsetOrIndex_(offsetOrIndex), size_(size)
    {}
    int64_t GetLocation() const { return location_; }
    uint64_t GetOffsetOrIndex() const { return offsetOrIndex_; }
    uint64_t GetSize() const { return size_; }

private:
    int64_t location_{-1};
    uint64_t offsetOrIndex_{0};
    uint64_t size_{0};
};

class TraceRootTask {
public:
    TraceRootTask() = default;
    TraceRootTask(const TraceRootTaskUid& uid) : uid_(uid) {}

    const TraceRootTaskUid& GetUid() const { return uid_; }
    Function* GetTileFunc() const { return tileFunc_; }
    void SetTileFunc(Function* func) { tileFunc_ = func; }

    const std::vector<int64_t>& GetExprList() const { return exprList_; }
    std::vector<int64_t>& GetExprList() { return exprList_; }

    const std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>>& GetLeafTaskDict() const { return leafTaskDict_; };
    std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>>& GetLeafTaskDict() { return leafTaskDict_; };

    const std::vector<std::shared_ptr<TraceRawTensorMemory>>& GetIncastList() const { return incastList_; }
    std::vector<std::shared_ptr<TraceRawTensorMemory>>& GetIncastList() { return incastList_; }

    const std::vector<std::shared_ptr<TraceRawTensorMemory>>& GetOutcastList() const { return outcastList_; }
    std::vector<std::shared_ptr<TraceRawTensorMemory>>& GetOutcastList() { return outcastList_; }

    const std::vector<TraceRootTaskRawTensorDesc>& GetRawTensorDescList() const { return rawTensorDescList_; }
    std::vector<TraceRootTaskRawTensorDesc>& GetRawTensorDescList() { return rawTensorDescList_; }

    const TraceMemoryRange& GetWorkspaceMemoryRange() const { return workspaceMemoryRange_; }
    TraceMemoryRange& GetWorkspaceMemoryRange() { return workspaceMemoryRange_; }

private:
    TraceRootTaskUid uid_;
    Function* tileFunc_{nullptr};
    std::vector<int64_t> exprList_;
    std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>> leafTaskDict_;
    std::vector<std::shared_ptr<TraceRawTensorMemory>> incastList_;
    std::vector<std::shared_ptr<TraceRawTensorMemory>> outcastList_;
    std::vector<TraceRootTaskRawTensorDesc> rawTensorDescList_;
    TraceMemoryRange workspaceMemoryRange_;
};

static constexpr int INVALID_TRACE_TASK_DEPEND_INDEX = -1;
class TraceDependGraph {
public:
    TraceDependGraph(const std::vector<std::shared_ptr<TraceLeafTask>>& leafTaskList,
                     const std::map<TraceLeafTaskUid, int>& leafTaskDependIndexDict,
                     const std::vector<std::vector<int>>& reachDict)
        : leafTaskList_(leafTaskList), leafTaskDependIndexDict_(leafTaskDependIndexDict), reachDict_(reachDict)
    {}

    int GetLeafTaskSize() const { return (int)leafTaskList_.size(); }
    const std::vector<std::shared_ptr<TraceLeafTask>>& GetLeafTaskList() const { return leafTaskList_; }
    const std::map<TraceLeafTaskUid, int> GetLeafTaskDependIndexDict() const { return leafTaskDependIndexDict_; }
    const std::vector<std::vector<int>>& GetReachDict() const { return reachDict_; }
    std::vector<std::vector<int>>& GetReachDict() { return reachDict_; }

    bool Reach(int src, int dst) const { return reachDict_[src][dst] != INVALID_TRACE_TASK_DEPEND_INDEX; }

private:
    std::vector<std::shared_ptr<TraceLeafTask>> leafTaskList_;
    std::map<TraceLeafTaskUid, int> leafTaskDependIndexDict_;
    /* reach[i][j] == k, means there exists one path from i to k and k to j, and k is one successor of i.
     * If there is no such path, then k is -1. */
    std::vector<std::vector<int>> reachDict_;
};

enum class TraceRaceKind {
    RACE_READ_WRITE,
    RACE_WRITE_WRITE,
    RACE_ATOMIC_ADD,
};
struct TraceRacePart {
    TraceRacePart(const std::shared_ptr<TraceLeafTask>& leafTask, bool isCopyOut, int copyIndex)
        : leafTask_(leafTask), isCopyOut_(isCopyOut), copyIndex_(copyIndex)
    {}

    std::shared_ptr<TraceLeafTask> GetLeafTask() const { return leafTask_; }
    bool IsCopyOut() const { return isCopyOut_; }
    int GetCopyIndex() const { return copyIndex_; }

private:
    std::shared_ptr<TraceLeafTask> leafTask_;
    bool isCopyOut_;
    int copyIndex_;
};
class TraceRace {
public:
    TraceRace(TraceRaceKind kind, const TraceRacePart& src, const TraceRacePart& dst)
        : kind_(kind), src_(src), dst_(dst)
    {}

    TraceRaceKind GetKind() const { return kind_; }
    const TraceRacePart& GetSrc() const { return src_; }
    const TraceRacePart& GetDst() const { return dst_; }

private:
    TraceRaceKind kind_;
    TraceRacePart src_;
    TraceRacePart dst_;
};

class TraceDeviceTask {
public:
    TraceDeviceTask() = default;
    TraceDeviceTask(const npu::tile_fwk::TraceDeviceTaskUid& uid) : uid_(uid) {}

    const TraceDeviceTaskUid& GetUid() const { return uid_; }

    const std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>>& GetRootTaskDict() const { return rootTaskDict_; }
    std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>>& GetRootTaskDict() { return rootTaskDict_; }

    TraceDependGraph BuildDependGraph() const;
    std::vector<TraceRace> CheckRace(const TraceDependGraph& graph) const;

private:
    TraceDeviceTaskUid uid_;
    std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>> rootTaskDict_;
};

class TraceExecution {
public:
    TraceExecution() = default;

    const std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>>& GetLeafTaskDict() const { return leafTaskDict_; }
    std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>>& GetLeafTaskDict() { return leafTaskDict_; }

    const std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>>& GetRootTaskDict() const { return rootTaskDict_; }
    std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>>& GetRootTaskDict() { return rootTaskDict_; }

    const std::map<TraceDeviceTaskUid, std::shared_ptr<TraceDeviceTask>>& GetDeviceTaskDict() const
    {
        return deviceTaskDict_;
    }
    std::map<TraceDeviceTaskUid, std::shared_ptr<TraceDeviceTask>>& GetDeviceTaskDict() { return deviceTaskDict_; }

    std::shared_ptr<TraceLeafTask> GetLeafTask(const TraceLeafTaskUid& luid);
    std::shared_ptr<TraceRootTask> GetRootTask(const TraceRootTaskUid& ruid);
    std::shared_ptr<TraceDeviceTask> GetDeviceTask(const TraceDeviceTaskUid& duid);

    const TraceMemoryRange& GetWorkspaceSpillRange() const { return workspaceSpillRange_; }

    void InitRootList(OrderedSet<Function*>& devRootList) { devRootList_ = devRootList; }
    void InitLeafList(OrderedSet<Function*>& devLeafList) { devLeafList_ = devLeafList; }

    void LoadTrace(const std::string& trace);

private:
    std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>> leafTaskDict_;
    std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>> rootTaskDict_;
    std::map<TraceDeviceTaskUid, std::shared_ptr<TraceDeviceTask>> deviceTaskDict_;
    TraceMemoryRange workspaceSpillRange_;

    OrderedSet<Function*> devRootList_;
    OrderedSet<Function*> devLeafList_;
};

} // namespace npu::tile_fwk
