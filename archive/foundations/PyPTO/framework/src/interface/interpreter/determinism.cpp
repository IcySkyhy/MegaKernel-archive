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
 * \file determinism.cpp
 * \brief
 */

#include "determinism.h"
#include "interface/schema/schema.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

using schema::SchemaNode;

bool TraceCopy::Overlap(const TraceCopy& src, const TraceCopy& dst)
{
    auto srcRange = src.GetRawTensor()->GetMemoryRange();
    auto dstRange = dst.GetRawTensor()->GetMemoryRange();
    if (srcRange.GetEnd() <= dstRange.GetBegin()) {
        // not overlap
        return false;
    }
    if (dstRange.GetEnd() <= srcRange.GetBegin()) {
        // not overlap
        return false;
    }
    // memory reuse must happen for full match.
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, srcRange == dstRange);
    // memory reuse must happen for same dimension.
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, src.GetOffset().size() == dst.GetOffset().size());
    for (size_t dim = 0; dim < src.GetOffset().size(); dim++) {
        if (src.GetOffset()[dim] + src.GetShape()[dim] <= dst.GetOffset()[dim]) {
            // not overlap
            return false;
        }
        if (dst.GetOffset()[dim] + dst.GetShape()[dim] <= src.GetOffset()[dim]) {
            // not overlap
            return false;
        }
    }
    return true;
}

bool TraceLeafTaskUid::operator<(const TraceLeafTaskUid& uid) const
{
    if (deviceTaskIndex_ != uid.deviceTaskIndex_) {
        return deviceTaskIndex_ < uid.deviceTaskIndex_;
    }
    if (dupIndex_ != uid.dupIndex_) {
        return dupIndex_ < uid.dupIndex_;
    }
    if (rootIndex_ != uid.rootIndex_) {
        return rootIndex_ < uid.rootIndex_;
    }
    if (operationIndex_ != uid.operationIndex_) {
        return operationIndex_ < uid.operationIndex_;
    }
    if (leafIndex_ != uid.leafIndex_) {
        return leafIndex_ < uid.leafIndex_;
    }
    return false;
}

bool TraceRootTaskUid::operator<(const TraceRootTaskUid& uid) const
{
    if (deviceTaskIndex_ != uid.deviceTaskIndex_) {
        return deviceTaskIndex_ < uid.deviceTaskIndex_;
    }
    if (dupIndex_ != uid.dupIndex_) {
        return dupIndex_ < uid.dupIndex_;
    }
    if (rootIndex_ != uid.rootIndex_) {
        return rootIndex_ < uid.rootIndex_;
    }
    return false;
}

bool TraceDeviceTaskUid::operator<(const TraceDeviceTaskUid& uid) const
{
    if (deviceTaskIndex_ != uid.deviceTaskIndex_) {
        return deviceTaskIndex_ < uid.deviceTaskIndex_;
    }
    return false;
}

static void BuildReachDict(TraceDependGraph& graph, int dependIndex, std::vector<bool>& visitDict)
{
    if (visitDict[dependIndex]) {
        return;
    }
    auto leafTask = graph.GetLeafTaskList()[dependIndex];
    const auto& dependIndexDict = graph.GetLeafTaskDependIndexDict();
    auto& reachDict = graph.GetReachDict();
    for (auto succUid : leafTask->GetSuccSet()) {
        auto iter = dependIndexDict.find(succUid);
        ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, iter != dependIndexDict.end());
        auto succDependIndex = iter->second;
        BuildReachDict(graph, succDependIndex, visitDict);
        for (int i = 0; i < graph.GetLeafTaskSize(); i++) {
            if (reachDict[succDependIndex][i] != INVALID_TRACE_TASK_DEPEND_INDEX) {
                /* path: succDependIndex -> reachDict[succDependIndex][i] -> i */
                reachDict[dependIndex][i] = succDependIndex;
            }
        }
        reachDict[dependIndex][succDependIndex] = succDependIndex;
    }
    visitDict[dependIndex] = true;
}

TraceDependGraph TraceDeviceTask::BuildDependGraph() const
{
    std::vector<std::shared_ptr<TraceLeafTask>> leafTaskList;
    std::map<TraceLeafTaskUid, int> leafTaskDependIndexDict;
    for (auto& [ruid, rootTask] : GetRootTaskDict()) {
        (void)ruid;
        for (auto& [luid, leafTask] : rootTask->GetLeafTaskDict()) {
            (void)luid;
            leafTaskDependIndexDict[leafTask->GetUid()] = leafTaskList.size();
            leafTaskList.push_back(leafTask);
        }
    }
    std::vector<std::vector<int>> reachDict(leafTaskList.size(),
                                            std::vector<int>(leafTaskList.size(), INVALID_TRACE_TASK_DEPEND_INDEX));
    TraceDependGraph graph(leafTaskList, leafTaskDependIndexDict, reachDict);
    std::vector<bool> visitDict(leafTaskList.size(), false);
    for (int i = 0; i < graph.GetLeafTaskSize(); i++) {
        BuildReachDict(graph, i, visitDict);
    }
    return graph;
}
std::vector<TraceRace> TraceDeviceTask::CheckRace(const TraceDependGraph& graph) const
{
    std::vector<TraceRace> raceList;
    for (int src = 0; src < graph.GetLeafTaskSize(); src++) {
        for (int dst = src + 1; dst < graph.GetLeafTaskSize(); dst++) {
            // src index < dst index
            if (graph.Reach(src, dst) || graph.Reach(dst, src)) {
                continue;
            }
            auto srcLeafTask = graph.GetLeafTaskList()[src];
            auto dstLeafTask = graph.GetLeafTaskList()[dst];
            for (int i = 0; i < (int)srcLeafTask->GetCopyInList().size(); i++) {
                auto& srcCopy = srcLeafTask->GetCopyInList()[i];
                for (int j = 0; j < (int)dstLeafTask->GetCopyOutList().size(); j++) {
                    auto& dstCopy = dstLeafTask->GetCopyOutList()[j];
                    // RW-race
                    if (TraceCopy::Overlap(srcCopy, dstCopy)) {
                        raceList.emplace_back(TraceRaceKind::RACE_READ_WRITE, TraceRacePart{srcLeafTask, false, i},
                                              TraceRacePart{dstLeafTask, true, j});
                    }
                }
            }
            for (int i = 0; i < (int)srcLeafTask->GetCopyOutList().size(); i++) {
                auto& srcCopy = srcLeafTask->GetCopyOutList()[i];
                for (int j = 0; j < (int)dstLeafTask->GetCopyInList().size(); j++) {
                    auto& dstCopy = dstLeafTask->GetCopyInList()[j];
                    // RW-race
                    if (TraceCopy::Overlap(srcCopy, dstCopy)) {
                        raceList.emplace_back(TraceRaceKind::RACE_READ_WRITE, TraceRacePart{srcLeafTask, true, i},
                                              TraceRacePart{dstLeafTask, false, j});
                    }
                }
            }
            for (int i = 0; i < (int)srcLeafTask->GetCopyOutList().size(); i++) {
                auto& srcCopy = srcLeafTask->GetCopyOutList()[i];
                for (int j = 0; j < (int)dstLeafTask->GetCopyOutList().size(); j++) {
                    auto& dstCopy = dstLeafTask->GetCopyOutList()[j];
                    // RW-race
                    if (TraceCopy::Overlap(srcCopy, dstCopy)) {
                        if (srcCopy.IsAtomicAdd() && dstCopy.IsAtomicAdd()) {
                            raceList.emplace_back(TraceRaceKind::RACE_ATOMIC_ADD, TraceRacePart{srcLeafTask, true, i},
                                                  TraceRacePart{dstLeafTask, true, j});
                        } else {
                            raceList.emplace_back(TraceRaceKind::RACE_WRITE_WRITE, TraceRacePart{srcLeafTask, true, i},
                                                  TraceRacePart{dstLeafTask, true, j});
                        }
                    }
                }
            }
        }
    }
    return raceList;
}

std::shared_ptr<TraceLeafTask> TraceExecution::GetLeafTask(const TraceLeafTaskUid& luid)
{
    if (GetLeafTaskDict().count(luid)) {
        return GetLeafTaskDict()[luid];
    }

    std::shared_ptr<TraceLeafTask> ltask = std::make_shared<TraceLeafTask>(luid);
    GetLeafTaskDict()[luid] = ltask;

    TraceRootTaskUid ruid(luid.GetDeviceTaskIndex(), luid.GetDupIndex(), luid.GetRootIndex());
    std::shared_ptr<TraceRootTask> rtask = GetRootTask(ruid);
    rtask->GetLeafTaskDict()[luid] = ltask;
    return ltask;
}
std::shared_ptr<TraceRootTask> TraceExecution::GetRootTask(const TraceRootTaskUid& ruid)
{
    if (GetRootTaskDict().count(ruid)) {
        return GetRootTaskDict()[ruid];
    }
    std::shared_ptr<TraceRootTask> rtask = std::make_shared<TraceRootTask>(ruid);
    GetRootTaskDict()[ruid] = rtask;

    TraceDeviceTaskUid duid(ruid.GetDeviceTaskIndex());
    std::shared_ptr<TraceDeviceTask> dtask = GetDeviceTask(duid);
    dtask->GetRootTaskDict()[ruid] = rtask;
    return rtask;
}
std::shared_ptr<TraceDeviceTask> TraceExecution::GetDeviceTask(const TraceDeviceTaskUid& duid)
{
    if (GetDeviceTaskDict().count(duid)) {
        return GetDeviceTaskDict()[duid];
    }
    std::shared_ptr<TraceDeviceTask> dtask = std::make_shared<TraceDeviceTask>(duid);
    GetDeviceTaskDict()[duid] = dtask;
    return dtask;
}

static std::vector<int64_t> LoadTraceExprList(const std::shared_ptr<SchemaNode>& node)
{
    std::vector<int64_t> exprList;
    for (auto& elt : *node) {
        std::string name = elt->GetName();
        exprList.emplace_back(std::stoll(name));
    }
    return exprList;
}

static std::vector<TraceCoa> LoadTraceCoaList(const std::shared_ptr<SchemaNode>& node)
{
    std::vector<TraceCoa> coaList;
    for (auto& elt : *node) {
        std::string name = elt->GetName();
        if (name[0] == '?') {
            coaList.emplace_back(std::stoull(name.substr(1)), true);
        } else {
            coaList.emplace_back(std::stoull(name));
        }
    }
    return coaList;
}

static TraceMemoryRange LoadTraceMemoryRange(const std::shared_ptr<SchemaNode>& node)
{
    std::string beginStr = node->at(0)->GetName();
    std::string endStr = node->at(1)->GetName();
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, beginStr.substr(0, 0x2) == SCHEMA_ADDRESS_PREFIX);
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, endStr.substr(0, 0x2) == SCHEMA_ADDRESS_PREFIX);
    uintptr_t begin = std::stoull(beginStr, nullptr, 16);
    uintptr_t end = std::stoull(endStr, nullptr, 16);
    return TraceMemoryRange(begin, end);
}

static int64_t LoadTraceInt(const std::shared_ptr<SchemaNode>& node)
{
    int64_t value = std::stoll(node->GetName());
    return value;
}

static int64_t LoadTraceRawTensor(const std::shared_ptr<SchemaNode>& node)
{
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, node->GetName()[0] == '@');
    int64_t value = std::stoll(node->GetName().substr(1));
    return value;
}

static std::unordered_map<
    std::string, std::function<void(std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& node)>>
    rtaskLoaderDict = {
        {schema::expr::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& expr) {
             rtask->GetExprList() = LoadTraceExprList(expr->at(0));
         }},
        {schema::RActWorkspace::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& workspace) {
             rtask->GetWorkspaceMemoryRange() = LoadTraceMemoryRange(workspace->at(0));
         }},
        {schema::RActIncastCount::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& count) {
             int size = LoadTraceInt(count->at(0));
             rtask->GetIncastList().resize(size);
             for (int index = 0; index < size; index++) {
                 rtask->GetIncastList()[index] = std::make_shared<TraceRawTensorMemory>();
             }
         }},
        {schema::RActIncast::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& incast) {
             auto index = LoadTraceInt(incast->at(0)->at(0));
             auto range = LoadTraceMemoryRange(incast->at(1));
             rtask->GetIncastList()[index]->SetMemoryRange(range);
         }},
        {schema::RActOutcastCount::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& count) {
             int size = LoadTraceInt(count->at(0));
             rtask->GetOutcastList().resize(size);
             for (int index = 0; index < size; index++) {
                 rtask->GetOutcastList()[index] = std::make_shared<TraceRawTensorMemory>();
             }
         }},
        {schema::RActOutcast::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& outcast) {
             auto index = LoadTraceInt(outcast->at(0)->at(0));
             auto range = LoadTraceMemoryRange(outcast->at(1));
             rtask->GetOutcastList()[index]->SetMemoryRange(range);
         }},
        {schema::RActRawTensorCount::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& count) {
             rtask->GetRawTensorDescList().resize(LoadTraceInt(count->at(0)));
         }},
        {schema::RActRawTensor::Name(),
         [](std::shared_ptr<TraceRootTask>& rtask, const std::shared_ptr<SchemaNode>& desc) {
             auto index = LoadTraceRawTensor(desc->at(0));
             auto location = LoadTraceInt(desc->at(1)->at(0));
             auto offsetOrIndex = LoadTraceInt(desc->at(1)->at(1));
             auto size = LoadTraceInt(desc->at(1)->at(2));
             rtask->GetRawTensorDescList()[index] = TraceRootTaskRawTensorDesc(location, offsetOrIndex, size);
         }},
};

static std::unordered_map<
    std::string, std::function<void(std::shared_ptr<TraceLeafTask>& ltask, const std::shared_ptr<SchemaNode>& node)>>
    ltaskLoaderDict = {
        {schema::coa::Name(),
         [](std::shared_ptr<TraceLeafTask>& ltask, const std::shared_ptr<SchemaNode>& coa) {
             ltask->GetCoaList() = LoadTraceCoaList(coa->at(0));
         }},
};
void TraceExecution::LoadTrace(const std::string& trace)
{
    auto traceNodeList = SchemaNode::ParseSchema(trace);
    for (auto& traceNode : traceNodeList) {
        std::map<std::string, std::vector<std::shared_ptr<SchemaNode>>> dict = SchemaNode::BuildDict({traceNode});
        if (dict.count(schema::DEvent::Name())) {
        } else if (dict.count(schema::REvent::Name())) {
            auto ruidNode = dict[schema::RUid::Name()][0];
            auto deviceTaskIndex = std::stoll(ruidNode->at(0x0)->GetName());
            auto dupIndex = std::stoll(ruidNode->at(0x1)->GetName());
            auto rootIndex = std::stoll(ruidNode->at(0x2)->GetName());
            TraceRootTaskUid ruid(deviceTaskIndex, dupIndex, rootIndex);

            std::shared_ptr<TraceRootTask> rtask = GetRootTask(ruid);
            for (auto [key, loader] : rtaskLoaderDict) {
                if (dict.count(key)) {
                    auto node = dict[key][0];
                    loader(rtask, node);
                }
            }
        } else if (dict.count(schema::LEvent::Name())) {
            auto luidNode = dict[schema::LUid::Name()][0];
            auto deviceTaskIndex = std::stoll(luidNode->at(0x0)->GetName());
            auto dupIndex = std::stoll(luidNode->at(0x1)->GetName());
            auto rootIndex = std::stoll(luidNode->at(0x2)->GetName());
            auto operationIndex = std::stoll(luidNode->at(0x3)->GetName());
            auto leafIndex = std::stoll(luidNode->at(0x4)->GetName());
            TraceLeafTaskUid luid(deviceTaskIndex, dupIndex, rootIndex, operationIndex, leafIndex);

            std::shared_ptr<TraceLeafTask> ltask = GetLeafTask(luid);

            for (auto [key, loader] : ltaskLoaderDict) {
                if (dict.count(key)) {
                    auto node = dict[key][0];
                    loader(ltask, node);
                }
            }
        }
    }
}

} // namespace npu::tile_fwk
