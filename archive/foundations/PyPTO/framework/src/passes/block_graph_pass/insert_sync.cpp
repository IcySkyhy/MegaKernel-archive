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
 * \file insert_sync.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync.h"
#include <thread>
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InsertSync"

namespace npu {
namespace tile_fwk {
Status RangeSearchTree::ProcessTreeNode(const Interval& interval, IntervalTreeNode* currPtr,
                                        std::vector<IntervalTreeNode*>& intervalStack)
{
    int start = currPtr->interval.start;
    if (interval.start < start) {
        if (currPtr->left != nullptr) {
            intervalStack.push_back(currPtr->left);
            return SUCCESS;
        }
        currPtr->left = new IntervalTreeNode(interval);
        if (currPtr->left == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "New created left tree node is nullptr, ProcessTreeNode failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    if (currPtr->right != nullptr) {
        intervalStack.push_back(currPtr->right);
        return SUCCESS;
    }
    currPtr->right = new IntervalTreeNode(interval);
    if (currPtr->right == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "New created right tree node is nullptr, ProcessTreeNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status RangeSearchTree::InsertInterval(const Interval& interval)
{
    std::vector<IntervalTreeNode*> intervalStack;
    if (treeRoot == nullptr) {
        treeRoot = new IntervalTreeNode(interval);
        if (treeRoot == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "TreeRoot is nullptr, InsertInterval failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (ProcessTreeNode(interval, currPtr, intervalStack) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "InsertInterval failed at function ProcessTreeNode.");
            return FAILED;
        }
        if (currPtr->max < interval.end) {
            currPtr->max = interval.end;
        }
    }
    return SUCCESS;
}

void RangeSearchTree::OverlapSearch(const Interval& interval, std::set<int>& result)
{
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);

    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        if (interval.start <= currPtr->interval.end && interval.end >= currPtr->interval.start) {
            result.insert(currPtr->interval.idx);
        }
        if (currPtr->left != nullptr && currPtr->left->max >= interval.start) {
            intervalStack.push_back(currPtr->left);
        }
        intervalStack.push_back(currPtr->right);
    }
}

void RangeSearchTree::FreeTree()
{
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        intervalStack.push_back(currPtr->left);
        intervalStack.push_back(currPtr->right);
        delete currPtr;
    }
}

void RangeSearchTree::Insert(int left, int right, int idx)
{
    Interval interval(left, right, idx);
    InsertInterval(interval);
}

std::set<int> RangeSearchTree::GetCovered(int left, int right)
{
    Interval givenInterval(left, right, 0);
    std::set<int> overlappingIdx;
    OverlapSearch(givenInterval, overlappingIdx);
    return overlappingIdx;
}

void DataDependencySearcher::CheckWAWSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType currMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(currMemoryType) > 0) {
            TileRange rg = currMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                                  tensor->memoryrange;
            std::set<int> found = wawSearchTree_[currMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckRAWSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t inIdx = 0; inIdx < opWait->GetIOperands().size(); inIdx++) {
        auto tensor = opWait->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1 && writeDdrMemMap.count(readDDRmemId) > 0) {
            std::set<int> found = writeDdrMemMap[readDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (rawSearchTree_.count(readMemoryType) > 0) {
            TileRange rg = readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                                  tensor->memoryrange;
            std::set<int> found = rawSearchTree_[readMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckWARSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1 && readDdrMemMap.count(writeDDRmemId) > 0) {
            std::set<int> found = readDdrMemMap[writeDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (warSearchTree_.count(writeMemoryType) > 0) {
            TileRange rg = writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                                   tensor->memoryrange;
            std::set<int> found = warSearchTree_[writeMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

std::set<int> DataDependencySearcher::Find(Operation* opWait)
{
    std::set<int> res;

    std::string opStr = opWait->GetOpcodeStr();
    // check WAW
    CheckWAWSearchTree(opWait, res);
    // check RAW
    CheckRAWSearchTree(opWait, res);
    // check WAR
    CheckWARSearchTree(opWait, res);
    return res;
}

void DataDependencySearcher::InsertWAWSearchTree(const Operation* opSet, int idx)
{
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType prevMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(prevMemoryType) == 0) {
            wawSearchTree_[prevMemoryType] = RangeSearchTree();
        }
        TileRange rg = prevMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                              tensor->memoryrange;
        wawSearchTree_[prevMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertRAWSearchTree(const Operation* opSet, int idx)
{
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1) {
            if (writeDdrMemMap.count(writeDDRmemId) == 0) {
                writeDdrMemMap[writeDDRmemId] = std::set<int>{};
            }
            writeDdrMemMap[writeDDRmemId].insert(idx);
        }
        if (rawSearchTree_.count(writeMemoryType) == 0) {
            rawSearchTree_[writeMemoryType] = RangeSearchTree();
        }
        TileRange rg = writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                               tensor->memoryrange;
        rawSearchTree_[writeMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertWARSearchTree(const Operation* opSet, int idx)
{
    for (size_t inIdx = 0; inIdx < opSet->GetIOperands().size(); inIdx++) {
        auto tensor = opSet->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1) {
            if (readDdrMemMap.count(readDDRmemId) == 0) {
                readDdrMemMap[readDDRmemId] = std::set<int>{};
            }
            readDdrMemMap[readDDRmemId].insert(idx);
        }
        if (warSearchTree_.count(readMemoryType) == 0) {
            warSearchTree_[readMemoryType] = RangeSearchTree();
        }
        TileRange rg = readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                              tensor->memoryrange;
        warSearchTree_[readMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::Insert(const Operation* opSet, int idx)
{
    InsertWAWSearchTree(opSet, idx);
    InsertRAWSearchTree(opSet, idx);
    InsertWARSearchTree(opSet, idx);
}

void PipeSync::BuildTensorRangeMap(Operation* op)
{
    auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
    if (opcfg.coreType_ != CoreType::AIV) {
        return;
    }
    bool isAIV1 = op->GetAIVCore() == AIVCore::AIV1;
    auto inTensors = op->GetIOperands();
    auto outTensors = op->GetOOperands();
    LogicalTensors inOutTensors;
    inOutTensors.reserve(inTensors.size() + outTensors.size());
    inOutTensors.insert(inOutTensors.end(), inTensors.begin(), inTensors.end());
    inOutTensors.insert(inOutTensors.end(), outTensors.begin(), outTensors.end());
    for (const auto& tensor : inOutTensors) {
        if (ubTensorRangeMap.count(tensor->GetMagic()) || tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            continue;
        }
        TileRange tensorRange;
        if (!isAIV1) {
            tensorRange.start = tensor->memoryrange.start;
            tensorRange.end = tensor->memoryrange.end;
        } else {
            // 将AIV1中的tensor地址映射到的区间
            size_t curUBSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
            tensorRange.start = tensor->memoryrange.start + curUBSize;
            tensorRange.end = tensor->memoryrange.end + curUBSize;
        }
        ubTensorRangeMap.emplace(std::make_pair(tensor->GetMagic(), tensorRange));
    }
}

Status PipeSync::InsertSync(Function& function, std::vector<Operation*>& syncedOpLog)
{
    std::vector<IndexOp> synced;
    std::vector<Operation*> opLogPtr(function.Operations(false).DuplicatedOpList());
    oriOpList_ = opLogPtr;
    uint64_t idxInput = 0;
    for (const auto& op : opLogPtr) {
        BuildTensorRangeMap(op);
        APASS_LOG_DEBUG_F(Elements::Operation, "Input operation %lu %lu: %s.", static_cast<unsigned long>(idxInput),
                          static_cast<unsigned long>(op->GetOpMagic()), op->GetOpcodeStr().c_str());
        idxInput++;
    }
    if (PipeDispatch(opLogPtr, synced) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync failed at function PipeDispatch.");
        return FAILED;
    }

    if (IssueOp(function, opLogPtr, synced) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync failed at function IssueOp.");
        return FAILED;
    }

    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });

    for (auto& log : synced) {
        syncedOpLog.push_back(&log.second.get());
    }
    return SUCCESS;
}

std::string PipeSync::DepOp::DumpDepOp(const std::vector<Operation*>& opLog)
{
    std::stringstream ss;
    ss << "idx: " << idx << " opmagic: " << opLog[idx]->GetOpMagic() << ", " << opLog[idx]->GetOpcodeStr() << ", ";
    ss << "setPipe: {";
    for (auto i : setPipe) {
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}, waitPipe: {";
    for (auto i : waitPipe) {
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::IssueQueue::DumpIssueQueue(const std::vector<Operation*>& opLogPtr)
{
    std::stringstream ss;
    ss << " Op in this pipe: {";
    for (auto op : ops) {
        ss << opLogPtr[op]->GetOpMagic() << " " << opLogPtr[op]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::PipeDepInfo::DumpPipeDepInfo()
{
    std::stringstream ss;
    ss << "    wait idx: " << waitIdx << "\n";
    ss << "    setPipeCoreEx:"
       << "\n";
    for (auto pair : setPipes) {
        ss << "        pipetype: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << "  aivCore: " << static_cast<int>(pair.first.aivCore)
           << "  opidx: " << pair.second << "\n";
    }
    return ss.str();
}

std::string PipeSync::DumpLatestPipeDepMap()
{
    std::stringstream ss;
    for (auto pair : latestPipeDep_) {
        ss << "current PipeCore: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << " aivCore: " << static_cast<int>(pair.first.aivCore) << "\n";
        ss << pair.second.DumpPipeDepInfo() << "\n";
    }
    return ss.str();
}

std::string PipeSync::PipeSeqName(PipeSeq seq) const
{
    switch (seq) {
        case PipeSeq::AIC_MTE2:
            return "AIC_MTE2";
        case PipeSeq::AIC_MTE1:
            return "AIC_MTE1";
        case PipeSeq::AIC_M:
            return "AIC_M";
        case PipeSeq::AIC_FIX:
            return "AIC_FIX";
        case PipeSeq::AIV0_MTE2:
            return "AIV0_MTE2";
        case PipeSeq::AIV1_MTE2:
            return "AIV1_MTE2";
        case PipeSeq::AIV0_V:
            return "AIV0_V";
        case PipeSeq::AIV1_V:
            return "AIV1_V";
        case PipeSeq::AIV0_MTE3:
            return "AIV0_MTE3";
        case PipeSeq::AIV1_MTE3:
            return "AIV1_MTE3";
        case PipeSeq::AIC_MTE3:
            return "AIC_MTE3";
        case PipeSeq::AIV0_S:
            return "AIV0_S";
        case PipeSeq::AIV1_S:
            return "AIV1_S";
        case PipeSeq::AIC_S:
            return "AIC_S";
        case PipeSeq::PIPE_END:
            return "PIPE_END";
        default:
            return "ILLEGAL";
    }
}

std::map<PipeSync::PipeCoreRealEx, PipeSeq, PipeSync::PipeCoreRealExCompare> PipeSync::pipe2Seq = {
    {{PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE2},
    {{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE1},
    {{PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_M},
    {{PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_FIX},
    {{PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE3},
    {{PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_S},
    {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_MTE2},
    {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_MTE2},
    {{PIPE_V, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_V},
    {{PIPE_V, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_V},
    {{PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_MTE3},
    {{PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_MTE3},
    {{PIPE_S, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_S},
    {{PIPE_S, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_S},
};

std::map<PipeSeq, PipeSync::PipeCoreRealEx> PipeSync::seq2pipe = {
    {PipeSeq::AIC_MTE2, {PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_MTE1, {PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_M, {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_FIX, {PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_MTE3, {PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_S, {PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIV0_MTE2, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_MTE2, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_V, {PIPE_V, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_V, {PIPE_V, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_MTE3, {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_MTE3, {PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_S, {PIPE_S, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_S, {PIPE_S, CoreType::AIV, AIVCore::AIV1}},
};

PipeSeq PipeSync::GetPipeSeq(PipeSync::PipeCoreRealEx pipe) { return pipe2Seq.at(pipe); }

PipeSync::PipeCoreRealEx PipeSync::GetPipeFromSeq(PipeSeq seq) { return seq2pipe.at(seq); }

Status PipeSync::AdjustCopyInCfg(TileOpCfg& opcfg, const Operation& op)
{
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d COPYIN op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto dstMemType = attr->GetCopyInAttr().second;
    if (dstMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
        return SUCCESS;
    }
    if (dstMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIV;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyOutCfg(TileOpCfg& opcfg, const Operation& op)
{
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d COPYOUT op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto srcMemType = attr->GetCopyOutAttr().first;
    if (srcMemType == MemoryType::MEM_L0C) {
        opcfg.pipeIdStart_ = PipeType::PIPE_FIX;
        opcfg.pipeIdEnd_ = PipeType::PIPE_FIX;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIV;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
    }
    return SUCCESS;
}

Status PipeSync::AdjustOpCfg(TileOpCfg& opcfg, const Operation& op)
{
    opcfg.aivCore_ = op.GetAIVCore();
    if (opcfg.coreType_ == CoreType::AIC) {
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
    }
    if (opcfg.coreType_ == CoreType::AIV && opcfg.aivCore_ == AIVCore::UNSPECIFIED) {
        opcfg.aivCore_ = AIVCore::AIV0;
    }
    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (AdjustCopyInCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustCopyInCfg failed.");
            return FAILED;
        }
    }
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        if (AdjustCopyOutCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustCopyOutCfg failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::PipeDispatch(const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    DataDependencySearcher dataDependencySearcher;
    dataDependencySearcher.ubTensorRangeMap = ubTensorRangeMap;
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        if (opLogPtr[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "%d ALLOC op should not appear in InsertSync, PipeDispatch failed.%s",
                              opLogPtr[i]->GetOpMagic(), GetFormatBacktrace(opLogPtr[i]).c_str());
            return FAILED;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        if (AdjustOpCfg(opcfg, *opLogPtr[i]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PipeDispatch failed at function AdjustOpCfg.");
            return FAILED;
        }
        DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        DepOp& opRef = depOps_.emplace_back(op);
        FindDep(opRef, opLogPtr, i, dataDependencySearcher);
        EnqueueOp(opRef, opLogPtr, syncedOpLog);
    }
    return SUCCESS;
}

void PipeSync::InitIssueQueue()
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        issueState_.emplace_back(GetPipeFromSeq(static_cast<PipeSeq>(i)));
    }
}

void PipeSync::EnqueueOp(DepOp& op, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    if (opLogPtr[op.idx]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_NOP || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_RESHAPE) {
        syncedOpLog.emplace_back(std::make_pair(op.idx * SEQUENCE_IDX, std::ref(*opLogPtr[op.idx])));
        return;
    }
    PipeCoreRealEx opPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    auto& issueQ = issueState_[static_cast<int>(GetPipeSeq(opPipeCoreEx))];
    issueQ.ops.emplace_back(op.idx);
    op.idxInPipe = issueQ.ops.size() - 1;
    // 若op的pipeStart和pipeEnd不同, 进行记录
    if (op.selfPipeCore.pipeStart != op.selfPipeCore.pipeEnd) {
        PipeCoreRealEx opPipeCoreExStart(op.selfPipeCore.pipeStart, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipeCoreRealEx opPipeCoreExEnd(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipePairEx pp{opPipeCoreExStart, opPipeCoreExEnd};
        int opMagic = opLogPtr[op.idx]->GetOpMagic();
        doublePipeOp[pp].emplace_back(opMagic);
    }
    orderedOpList_.emplace(op.idx);
}

void PipeSync::RemoveOpDep(DepOp& setOp, DepOp& waitOp) const
{
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;
    std::vector<size_t> newSetDep;
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            continue;
        }
        newSetDep.emplace_back(ele);
    }
    setOp.setPipe = newSetDep;

    std::vector<size_t> newWaitDep;
    for (auto ele : waitOp.waitPipe) {
        if (ele == setOpIdx) {
            continue;
        }
        newWaitDep.emplace_back(ele);
    }
    waitOp.waitPipe = newWaitDep;
}

Status PipeSync::AddOpDep(DepOp& setOp, DepOp& waitOp, bool isMergeCvSyncBase)
{
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;

    size_t depWaitIdx = static_cast<size_t>(-1);
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            APASS_LOG_ERROR_F(Elements::Operation, "This dependency should not exist, AddOpDep failed.");
            return FAILED;
        }
        PipeCoreRealEx elePipeCoreEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                                     depOps_[ele].selfPipeCore.aivCore);
        PipeCoreRealEx waitOpPipeCoreEx(depOps_[waitOpIdx].selfPipeCore.pipeStart, depOps_[waitOpIdx].selfPipeCore.core,
                                        depOps_[waitOpIdx].selfPipeCore.aivCore);
        if (elePipeCoreEx == waitOpPipeCoreEx) {
            if (ele <= waitOpIdx && !isMergeCvSyncBase) {
                APASS_LOG_ERROR_F(Elements::Operation, "New waitidx should less than old, AddOpDep failed.");
                return FAILED;
            } else if (ele <= waitOpIdx && isMergeCvSyncBase) {
                return SUCCESS;
            }
            depWaitIdx = ele;
            break;
        }
    }
    // 同一种pipecore，不需要存记录依赖关系
    if (depWaitIdx != static_cast<size_t>(-1)) {
        RemoveOpDep(setOp, depOps_[depWaitIdx]);
    }
    setOp.setPipe.emplace_back(waitOpIdx);
    waitOp.waitPipe.emplace_back(setOpIdx);
    return SUCCESS;
}

Status PipeSync::AdjustOpDep(DepOp& op, size_t waitOpIdx, IssueQueue& issueQ, bool& failedFlag)
{
    // op为靠前的， waitOp为靠后的
    auto& waitOp = depOps_[waitOpIdx];

    if (issueQ.currOp + 1 == issueQ.ops.size()) {
        failedFlag = true;
        return SUCCESS;
    }

    auto& nextOpIdx = issueQ.ops[issueQ.currOp + 1];

    // next op的idx要小于waitop
    if (nextOpIdx > waitOpIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot AdjustOpDep because nextop idx > waitop idx.");
        failedFlag = true;
        return SUCCESS;
    }

    auto& nextOp = depOps_[nextOpIdx];
    RemoveOpDep(op, waitOp);
    if (AddOpDep(nextOp, waitOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustOpDep failed at function AddOpDep.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::ProcessSameCoreCase(const PipePairEx& pp, EventIdProcessContext& ctx)
{
    ctx.issuenum.maxIssueNum.emplace(pp, GetFreeEventIdQueue(pp).size());
    ctx.issuenum.currIssueNum.emplace(pp, 0);

    if (ctx.issuenum.currIssueNum[pp] >= ctx.issuenum.maxIssueNum[pp]) {
        if (!ctx.deadlock) {
            ctx.eventIdOk = false;
            return SUCCESS;
        }
        if (AdjustOpDep(ctx.op, ctx.eleIdx, ctx.issueQ, ctx.failedFlag) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessSameCoreCase failed at AdjustOpDep.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessEventIdElement(EventIdProcessContext& ctx)
{
    auto& op = ctx.op;
    auto ele = ctx.eleIdx;

    if (op.selfPipeCore.pipeEnd == depOps_[ele].selfPipeCore.pipeStart) {
        return SUCCESS;
    }

    PipeCoreRealEx currEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    PipeCoreRealEx eleEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                         depOps_[ele].selfPipeCore.aivCore);
    PipePairEx pp{currEx, eleEx};

    // 当前逻辑该函数不需要处理cv同步 (if mix cv core, need handle mix cv core sync)
    if (!IsMixCvCore(currEx.core, eleEx.core) && (currEx.core != eleEx.core)) {
        return SUCCESS;
    }
    return ProcessSameCoreCase(pp, ctx);
}

Status PipeSync::HandleEventID(DepOp& op, IssueQueue& issueQ, IssueNum& issuenum, bool& deadlock, bool& res,
                               std::vector<IndexOp>& syncedOpLog)
{
    bool eventIdOk = true;
    bool failedFlag = false;

    std::vector<size_t> oriSetPipe = op.setPipe;
    for (auto ele : oriSetPipe) {
        EventIdProcessContext ctx{op, ele, issueQ, issuenum, syncedOpLog, deadlock, eventIdOk, failedFlag};
        if (ProcessEventIdElement(ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "HandleEventID failed at ProcessEventIdElement.");
            return FAILED;
        }
        eventIdOk = ctx.eventIdOk;
        failedFlag = ctx.failedFlag;
        if (!eventIdOk) {
            break;
        }
    }

    if (failedFlag) {
        deadlock = true;
        res = false;
    } else {
        deadlock = false;
        res = eventIdOk;
    }
    return SUCCESS;
}

bool PipeSync::CheckIssuedOp(const DepOp& op)
{
    // current op will be issued only when all of the waitop are issued
    for (const auto& waitOp : op.waitPipe) {
        if (!depOps_[waitOp].issued) {
            return false;
        }
    }
    return true;
}

Status PipeSync::PopFromQueue(IssueQueue& issueQ, std::vector<size_t>& poped, bool& deadlock,
                              std::vector<IndexOp>& syncedOpLog)
{
    IssueNum issuenum;

    for (uint64_t i = 0; i < MAX_POP; i++) {
        if (issueQ.currOp >= issueQ.ops.size()) {
            break;
        }
        auto& op = depOps_[issueQ.ops[issueQ.currOp]];
        if (op.idx != orderedOpList_.front()) {
            break;
        }
        if (op.issued) {
            APASS_LOG_ERROR_F(Elements::Operation, "Try to issue a op which is already issued, PopFromQueue failed.");
            return FAILED;
        }
        if (!CheckIssuedOp(op)) {
            break;
        }
        bool res = false;
        // cv同步不需要在此处理eventid
        if (HandleEventID(op, issueQ, issuenum, deadlock, res, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PopFromQueue failed at function HandleEventID.");
            return FAILED;
        }
        if (!res) {
            break;
        }
        op.issued = true;
        poped.emplace_back(op.idx);
        orderedOpList_.pop();
        for (auto ele : op.setPipe) {
            PipeCoreRealEx currPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
            PipeCoreRealEx elePipeCoreEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                                         depOps_[ele].selfPipeCore.aivCore);
            auto pp = PipePairEx{currPipeCoreEx, elePipeCoreEx};
            issuenum.currIssueNum[pp] = issuenum.currIssueNum[pp] + 1;
        }
        issueQ.currOp++;
    }
    return SUCCESS;
}

// check if platform architecture is mix cv core, and coreOne and coreTwo are AIV/AIC or vice versa
bool PipeSync::IsMixCvCore(CoreType coreOne, CoreType coreTwo)
{
    return Platform::Instance().GetSoc().GetCoreWrap().IsCVMix() &&
           ((coreOne == CoreType::AIV && coreTwo == CoreType::AIC) ||
            (coreOne == CoreType::AIC && coreTwo == CoreType::AIV));
}

Status PipeSync::InjectWaitFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    // serch the waitpipe of current op
    uint64_t waitIdx = idx == 0 ? 0 : idx * SEQUENCE_IDX - HALF_SEQUENCE_IDX;
    for (const auto& ele : depOps_[idx].waitPipe) {
        PipeCore setPipe = depOps_[ele].selfPipeCore;
        PipeCoreRealEx setPipeRealEx(setPipe.pipeEnd, setPipe.core, setPipe.aivCore);
        PipeCoreRealEx currPipeRealEx(currPipe.pipeStart, currPipe.core, currPipe.aivCore);
        if ((setPipe.aivCore == AIVCore::AIV0 && currPipe.aivCore == AIVCore::AIV1) ||
            (setPipe.aivCore == AIVCore::AIV1 && currPipe.aivCore == AIVCore::AIV0)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Sync between AIV0 and AIV1 happened.");
            return FAILED;
        }
        int eventId = setWaitPairMap_[{ele, idx}];
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_SYNC_DST, input, output);
        bool res = GenSyncOp(setPipeRealEx, currPipeRealEx, eventId, false, syncOp);
        if (!res) {
            syncOp.SetAsDeleted();
            continue;
        }
        // insert wait_flag
        syncOp.SetAIVCore(currPipe.aivCore);
        syncedOpLog.emplace_back(std::make_pair(++waitIdx, std::ref(syncOp)));
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d, setAIVCore type: %d, waitAIVCore type: %d",
            syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_,
            static_cast<int>(syncOp.syncQueue_.setAivCore_), static_cast<int>(syncOp.syncQueue_.waitAivCore_));
        if (IsMixCvCore(setPipe.core, currPipe.core) || (setPipe.core == currPipe.core)) {
            GetFreeEventIdQueue({setPipeRealEx, currPipeRealEx}).push_back(eventId);
        }
        // 记录 set op 和 waitflag的对应关系
        waitOpMap.emplace(&syncOp, oriOpList_[ele]);
        for (auto it = NoWaitCVPairs_.begin(); it != NoWaitCVPairs_.end(); ++it) {
            if (it->first == std::make_pair(ele, idx)) {
                NoWaitCVPairs_.erase(it);
                break;
            }
        }
        // 在插完waitflag后，需要更新syncArriveStatus, 以waitflag的pipecore为key
        // 先创建当前eventid对应的EventResource:
        if (UpdateSyncArriveStatus(eventId, setPipe, currPipe, setPipeRealEx, currPipeRealEx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InjectWaitFlag failed at function UpdateSyncArriveStatus.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::UpdateSyncArriveStatus(int eventId, const PipeCore& setPipe, const PipeCore& currPipe,
                                        const PipeCoreRealEx& setPipeRealEx, const PipeCoreRealEx& currPipeRealEx)
{
    EventResource eventResource{eventId,
                                {setPipe.core, setPipe.aivCore},
                                {currPipe.core, currPipe.aivCore},
                                setPipe.pipeEnd,
                                currPipe.pipeStart};
    syncArriveStatus[currPipeRealEx].insert(eventResource);
    syncArriveStatus[currPipeRealEx].insert(syncArriveStatus[setPipeRealEx].begin(),
                                            syncArriveStatus[setPipeRealEx].end());
    return SUCCESS;
}

Status PipeSync::InjectSetFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    uint64_t setIdx = idx * SEQUENCE_IDX + FORCE_SYNC_OP_NUM;
    std::vector<size_t> setPipeCopy = depOps_[idx].setPipe;
    for (size_t& ele : setPipeCopy) {
        PipeCore waitPipe = depOps_[ele].selfPipeCore;
        PipeCoreRealEx waitPipeRealEx(waitPipe.pipeStart, waitPipe.core, waitPipe.aivCore);
        PipeCoreRealEx currPipeRealEx(currPipe.pipeEnd, currPipe.core, currPipe.aivCore);
        int eventId{0};
        if (GetEventId({currPipeRealEx, waitPipeRealEx}, eventId, idx, ele, syncedOpLog, function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InjectSetFlag failed at function GetEventId.");
            return FAILED;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_SYNC_SRC, input, output);
        bool res = GenSyncOp(currPipeRealEx, waitPipeRealEx, eventId, true, syncOp);
        if (res) {
            // insert set_flag
            syncOp.SetAIVCore(currPipe.aivCore);
            syncedOpLog.emplace_back(std::make_pair(++setIdx, std::ref(syncOp)));
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d, setAIVCore type: %d, waitAIVCore type: %d",
                syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_,
                static_cast<int>(syncOp.syncQueue_.setAivCore_), static_cast<int>(syncOp.syncQueue_.waitAivCore_));
            setWaitPairMap_[{idx, ele}] = eventId;
            // 记录wait op 和 setflag的对应关系
            setOpMap.emplace(&syncOp, oriOpList_[ele]);
            if (currPipe.core != waitPipe.core) {
                NoWaitCVPairs_.push_back({{idx, ele}, &syncOp});
            }
            continue;
        }
        syncOp.SetAsDeleted();
        setWaitPairMap_[{idx, ele}] = eventId;
    }
    return SUCCESS;
}

Status PipeSync::InjectSync(Function& function, const std::vector<Operation*>& opLogPtr, size_t idx,
                            std::vector<IndexOp>& syncedOpLog)
{
    // check idx range
    if (idx > std::numeric_limits<uint64_t>::max() / SEQUENCE_IDX) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation index is out of range, InjectSync failed.");
        return FAILED;
    }

    // insert wait_flag
    InjectWaitFlag(function, idx, syncedOpLog);

    // insert current operation
    syncedOpLog.emplace_back(std::make_pair(idx * SEQUENCE_IDX, std::ref(*opLogPtr[idx])));
    depOps_[idx].issued = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert %d %s", opLogPtr[idx]->GetOpMagic(),
                      opLogPtr[idx]->GetOpcodeStr().c_str());

    // insert set_flag
    if (InjectSetFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectSetFlag.");
        return FAILED;
    }

    return SUCCESS;
}

int PipeSync::GetMaxEventId(const PipePairEx& pp)
{
    PipePairEx ppReverse = {pp.second, pp.first};
    auto it1 = doublePipeOp.find(pp);
    auto it2 = doublePipeOp.find(ppReverse);
    if (it1 == doublePipeOp.end() && it2 == doublePipeOp.end()) {
        return EVENT_ID7;
    }
    return EVENT_ID7;
}

Status PipeSync::ProcessDeadLock(uint64_t& eventIdDeadlockEnterTimes, bool& eventIdDeadlock,
                                 std::vector<IndexOp>& syncedOpLog)
{
    eventIdDeadlockEnterTimes++;
    // eventID deadlock
    if (eventIdDeadlockEnterTimes > DEADLOCK_TIME_THRESHOLD) {
        eventIdDeadlock = true;
    }
    if (RelaxFakeDataDep(syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessDeadLock failed at function RelaxFakeDataDep.");
        return FAILED;
    }
    if (eventIdDeadlockEnterTimes >= EVENTID_DEADLOCK_ENTER_TIME) {
        APASS_LOG_ERROR_F(Elements::Operation, "Unbreakable deadlock detected, ProcessDeadLock failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::IssueOpPipeSeq(Function& function, const std::vector<Operation*>& opLogPtr,
                                std::vector<IndexOp>& syncedOpLog, bool& eventIdDeadlock, size_t& issued)
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        std::vector<size_t> issuedOps;
        if (PopFromQueue(issueState_[i], issuedOps, eventIdDeadlock, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function PopFromQueue.");
            return FAILED;
        }
        issued += issuedOps.size();
        for (auto idx : issuedOps) {
            if (InjectSync(function, opLogPtr, idx, syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function InjectSync.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::IssueSyncOp(Function& function, const std::vector<Operation*>& opLogPtr,
                             std::vector<IndexOp>& syncedOpLog, size_t& totalIssued, size_t& allIssued)
{
    bool eventIdDeadlock = false;
    uint64_t eventIdDeadlockEnterTimes = 0;
    InitCVEventIdQ();
    while (totalIssued < allIssued) {
        size_t issued = 0;
        if (IssueOpPipeSeq(function, opLogPtr, syncedOpLog, eventIdDeadlock, issued) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function IssueOpPipeSeq.");
            return FAILED;
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issued == 0) {
            if (ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function ProcessDeadLock.");
                return FAILED;
            }
            continue;
        }
        eventIdDeadlock = false;
        eventIdDeadlockEnterTimes = 0;
    }
    return SUCCESS;
}

Status PipeSync::IssueOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += issueState_[i].ops.size();
        APASS_LOG_DEBUG_F(Elements::Operation, "Pipe seq %d: %s %s", i, PipeSeqName(static_cast<PipeSeq>(i)).c_str(),
                          issueState_[i].DumpIssueQueue(opLogPtr).c_str());
    }
    if (IssueSyncOp(function, opLogPtr, syncedOpLog, totalIssued, allIssued) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed with IssueSyncOp.");
        return FAILED;
    }
    if (totalIssued != allIssued) {
        APASS_LOG_ERROR_F(Elements::Operation, "Issue error, IssueOp failed.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALL op issued: %zu", totalIssued);
    return SUCCESS;
}

std::vector<PipeSync::PipePair> PipeSync::dataDepPair = {
    // PIPE_MTE1只有AIC PIPE_V只有AIV PIPE_M只有AIC PIPE_FIX只有AIC
    // PIPE_MTE3->PIPE_MTE2
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE3
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE2->PIPE_V
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE2
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE3->PIPE_V
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE3
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_S->PIPE_V
    {{PIPE_S, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_S
    {{PIPE_V, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_M
    {{PIPE_S, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_S
    {{PIPE_M, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE1
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_S
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE2
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE2->PIPE_S
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_MTE3
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_MTE3->PIPE_S
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_FIX
    {{PIPE_S, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_S
    {{PIPE_FIX, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_M->PIPE_MTE1
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_M
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE2
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_M
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE3
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_M
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_FIX
    {{PIPE_M, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_M
    {{PIPE_FIX, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE2
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE1
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE3
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_MTE1
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_FIX
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE1
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE2->PIPE_FIX
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE2
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE3->PIPE_FIX
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE3
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
};

// mix aic/aiv core dependency pairs
std::vector<PipeSync::PipePair> PipeSync::mixCVDataDepPair = {
    // PIPE_FIX->PIPE_V
    {{PIPE_FIX, CoreType::AIC}, {PIPE_V, CoreType::AIV}},
    // PIPE_MTE3->PIPE_MTE1
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE3->PIPE_MTE2
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_MTE2, CoreType::AIC}},
};

bool PipeSync::ConstructDepInfo(DataDepInfo& depInfo, std::vector<IndexOp>& syncedOpLog, int i)
{
    auto& log = syncedOpLog[i].second;
    if (log.get().GetOpcodeStr() != "SYNC_SRC") {
        return false;
    }
    auto setPipe = log.get().syncQueue_.pipeId_;
    auto waitPipe = log.get().syncQueue_.trigPipeId_;
    auto setCore = log.get().syncQueue_.coreType_;
    auto waitCore = log.get().syncQueue_.trigCoreType_;
    auto eventId = log.get().syncQueue_.eventId_;
    auto setAivCore = log.get().syncQueue_.setAivCore_;
    auto waitAivCore = log.get().syncQueue_.waitAivCore_;
    if (!(setPipe == depInfo.setp && setCore == depInfo.setc && waitPipe == depInfo.waitp &&
          waitCore == depInfo.waitc && setAivCore == depInfo.setaivc && waitAivCore == depInfo.waitaivc)) {
        return false;
    }
    if (std::find(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end(), eventId) !=
        depInfo.setOpEventIdList.end()) {
        return false;
    }
    depInfo.setOpIdList.push_back(i);
    depInfo.setOpEventIdList.push_back(eventId);
    return true;
}

int PipeSync::GetSyncSrcLogIdx(const std::vector<IndexOp>& syncedOpLog, int i)
{
    int j = i - 1;
    for (; j >= 0; j--) {
        if (syncedOpLog[j].second.get().GetOpcodeStr().find("SYNC") == std::string::npos &&
            syncedOpLog[j].second.get().GetOpcodeStr().find("BAR") == std::string::npos) {
            break;
        }
    }
    return syncedOpLog[j].first;
}

bool PipeSync::FindDataDep(DataDepInfo& depInfo, std::vector<IndexOp>& syncedOpLog, int i)
{
    if (!ConstructDepInfo(depInfo, syncedOpLog, i)) {
        return false;
    }
    int syncSrcLogIdx = GetSyncSrcLogIdx(syncedOpLog, i) / SEQUENCE_IDX; // sync_src对应的非同步op的idx
    DepOp& depOpSrc = depOps_[syncSrcLogIdx];
    for (auto syncDstLogIdx : depOpSrc.setPipe) { // setpipe中的op为该op之后的，依赖于该op的op id
        DepOp& depOpDst = depOps_[syncDstLogIdx];
        if (depOpDst.selfPipeCore.core == depInfo.waitc && depOpDst.selfPipeCore.pipeStart == depInfo.waitp &&
            depOpDst.selfPipeCore.aivCore == depInfo.waitaivc) {
            depInfo.opDepList.push_back(std::make_pair(syncSrcLogIdx, syncDstLogIdx));
        }
    }
    return true;
}

bool PipeSync::FindMaxOverlap(DataDepInfo& depInfo, int& maxOverlapDepIdx)
{
    int maxOverlap = -1;
    for (int idx = 0; idx < static_cast<int>(depInfo.opDepList.size() - 1); idx++) {
        if (depInfo.opDepList[idx].second < depInfo.opDepList[idx + 1].first) { // 相邻的两个依赖之间没有重叠
            continue;
        }
        // max_overlap初始值为阈值，相邻两个依赖之间的重叠超过阈值，可以合并
        // max_overlap用来记录遍历到的最大overlap
        if ((depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first) > maxOverlap) {
            maxOverlapDepIdx = idx;
            maxOverlap = depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first;
        }
    }
    if (maxOverlapDepIdx == -1) { // 8个依赖中，前后依赖都没有重叠，或者重叠全部小于阈值。
        return false;
    }
    return true;
}

Status PipeSync::SynDependency(int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx,
                               std::vector<IndexOp>& syncedOpLog)
{
    int set1 = depInfo.opDepList[maxOverlapDepIdx].first;
    int wait1 = depInfo.opDepList[maxOverlapDepIdx].second;
    int set2 = depInfo.opDepList[maxOverlapDepIdx + 1].first;
    int wait2 = depInfo.opDepList[maxOverlapDepIdx + 1].second;
    int eventId1 = depInfo.setOpEventIdList[maxOverlapDepIdx];
    int eventId2 = depInfo.setOpEventIdList[maxOverlapDepIdx + 1];
    int syncOpIdx1 = depInfo.setOpIdList[maxOverlapDepIdx];
    if (set1 >= set2 || wait1 >= wait2) {
        APASS_LOG_ERROR_F(Elements::Operation, "Dependency error, RelaxFakeDataDep failed.");
        return FAILED;
    }
    RemoveOpDep(depOps_[set1], depOps_[wait1]);
    RemoveOpDep(depOps_[set2], depOps_[wait2]);
    if (AddOpDep(depOps_[set2], depOps_[wait1]) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RelaxFakeDataDep failed at function AddOpDep.");
        return FAILED;
    }
    // 这里不用处理核间同步 (if mix cv core, need handle mix cv core sync)
    if (!(IsMixCvCore(pipePairEx.first.core, pipePairEx.second.core)) &&
        (pipePairEx.first.core != pipePairEx.second.core)) {
        APASS_LOG_ERROR_F(Elements::Operation, "CoreType is not same, SynDependency failed.");
        return FAILED;
    }
    GetFreeEventIdQueue(pipePairEx).push_back(eventId1);
    setWaitPairMap_[{set2, wait1}] = eventId2;
    // 将靠前的一对有依赖关系op中插入的SYNC_SRC op删除
    syncedOpLog[syncOpIdx1].second.get().SetAsDeleted();
    syncedOpLog.erase(syncedOpLog.begin() + syncOpIdx1);
    return SUCCESS;
}

Status PipeSync::GetDepInfo(std::vector<IndexOp>& syncedOpLog, const PipePairEx& pipePairEx, DataDepInfo& depInfo)
{
    depInfo.setp = pipePairEx.first.pipe;
    depInfo.setc = pipePairEx.first.core;
    depInfo.waitp = pipePairEx.second.pipe;
    depInfo.waitc = pipePairEx.second.core;
    depInfo.setaivc = pipePairEx.first.aivCore;
    depInfo.waitaivc = pipePairEx.second.aivCore;
    // 8个eventid全部被占用
    // 说明该set pipe内肯定有8个set op已经发射，而wait pipe内对应的8个op一个都没发射。
    // 找到这8个setop的idx，对应的event id，以及对应的waitop idx
    auto eventNum = static_cast<std::vector<std::pair<int, int>>::size_type>(GetMaxEventId(pipePairEx));
    for (int i = syncedOpLog.size() - 1; i >= 0; i--) {
        if (!(FindDataDep(depInfo, syncedOpLog, i))) {
            continue;
        }
        if (depInfo.setOpIdList.size() == eventNum) {
            break;
        }
    }
    // 由于从后向前寻找，得到的结果列表进行反转。
    std::reverse(depInfo.opDepList.begin(), depInfo.opDepList.end());
    std::reverse(depInfo.setOpIdList.begin(), depInfo.setOpIdList.end());
    std::reverse(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end());
    if (depInfo.opDepList.size() != eventNum || depInfo.setOpIdList.size() != eventNum ||
        depInfo.setOpEventIdList.size() != eventNum) {
        APASS_LOG_ERROR_F(Elements::Operation, "dep size should be %zu, RelaxFakeDataDep failed.", eventNum);
        return FAILED;
    }
    return SUCCESS;
}

std::vector<PipeSync::PipeCoreRealEx> PipeSync::cvPipeCoreEx = {
    {PIPE_S, CoreType::AIV, AIVCore::AIV0},           {PIPE_S, CoreType::AIV, AIVCore::AIV1},
    {PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},        {PIPE_MTE2, CoreType::AIV, AIVCore::AIV1},
    {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0},        {PIPE_MTE3, CoreType::AIV, AIVCore::AIV1},
    {PIPE_V, CoreType::AIV, AIVCore::AIV0},           {PIPE_V, CoreType::AIV, AIVCore::AIV1},
    {PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED},    {PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
    {PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}, {PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED},
    {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED},    {PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED},
};

std::string PipeSync::DataDepInfo::DumpDataDepInfo(const std::vector<IndexOp>& syncedOpLog,
                                                   const std::vector<Operation*>& oriOpList)
{
    std::stringstream ss;
    ss << "    CV_SYNC_SRC magic: ";
    for (auto sync : setOpIdList) {
        ss << syncedOpLog[sync].second.get().GetOpMagic() << " ";
    }
    ss << "\n    CV EventID: ";
    for (auto id : setOpEventIdList) {
        ss << id << " ";
    }
    ss << "\n    Set Wait Pair: \n";
    for (auto [setidx, waitidx] : opDepList) {
        ss << "        " << setidx << " " << oriOpList[setidx]->GetOpMagic() << " "
           << oriOpList[setidx]->GetOpcodeStr();
        ss << "  " << waitidx << " " << oriOpList[waitidx]->GetOpMagic() << " " << oriOpList[waitidx]->GetOpcodeStr()
           << "\n";
    }
    return ss.str();
}

Status PipeSync::RelaxFakeDataDep(std::vector<IndexOp>& syncedOpLog)
{
    // 合并阈值。只有前后两个set-wait对之间的重叠（以op数量度量）超过该阈值时，才对这两个set-wait对进行合并。
    std::vector<PipePairEx> dataDepPairEx;
    for (const auto& pipePair : dataDepPair) {
        if (pipePair.first.core == CoreType::AIC) {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
            PipePairEx ppEx = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx);
        } else {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
            PipePairEx ppEx1 = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx1);
            PipeCoreRealEx pp3(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV1);
            PipeCoreRealEx pp4(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV1);
            PipePairEx ppEx2 = {pp3, pp4};
            dataDepPairEx.emplace_back(ppEx2);
        }
    }

    if (Platform::Instance().GetSoc().GetCoreWrap().IsCVMix()) {
        for (const auto& pipePair : mixCVDataDepPair) {
            if (pipePair.first.core == CoreType::AIC && pipePair.second.core == CoreType::AIV) {
                PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
                PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
                PipePairEx ppEx = {pp1, pp2};
                dataDepPairEx.emplace_back(ppEx);
            } else if (pipePair.first.core == CoreType::AIV && pipePair.second.core == CoreType::AIC) {
                PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
                PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
                PipePairEx ppEx = {pp1, pp2};
                dataDepPairEx.emplace_back(ppEx);
            }
        }
    }

    for (const auto& pipePairEx : dataDepPairEx) {
        // 合并CV同步释放cv eventid 当前逻辑不需要 (if mix cv core, need handle mix cv core sync)
        if (!(IsMixCvCore(pipePairEx.first.core, pipePairEx.second.core)) &&
            (pipePairEx.first.core != pipePairEx.second.core)) {
            continue;
        }
        if (HasFreeEventId(pipePairEx)) {
            continue;
        }
        // 找到free id exhausted的pipe pair
        DataDepInfo depInfo;
        if (GetDepInfo(syncedOpLog, pipePairEx, depInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetDepInfo failed.");
            return FAILED;
        }
        // 找到前一个依赖和后一个依赖间重叠最大的地方
        int maxOverlapDepIdx{-1};
        if (!(FindMaxOverlap(depInfo, maxOverlapDepIdx))) {
            continue;
        }
        // 合并依赖
        if (SynDependency(maxOverlapDepIdx, depInfo, pipePairEx, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SynDependency failed.");
            return FAILED;
        }
    }

    return SUCCESS;
}

bool PipeSync::GenSyncOp(PipeCoreRealEx set, PipeCoreRealEx wait, int eventId, bool isSet, Operation& op)
{
    if (set.core != wait.core) {
        if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
            op.SetOpCode(isSet ? Opcode::OP_CV_SYNC_SRC : Opcode::OP_CV_SYNC_DST);
        }
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && !isSet && wait.core == CoreType::AIV) {
            set.pipe = PipeType::PIPE_V;
        }
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
        return true;
    }
    if (set.core == CoreType::AIV && set.aivCore != wait.aivCore) {
        APASS_LOG_WARN_F(Elements::Operation, "Find dependency between AIV0 and AIV1, remove it.");
        return false;
    }
    if (set.pipe != wait.pipe) {
        op.SetOpCode(isSet ? Opcode::OP_SYNC_SRC : Opcode::OP_SYNC_DST);
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
        return true;
    }
    if (isSet || set.pipe == PipeType::PIPE_S) {
        return false;
    }
    // 同步相关的信息放在operation属性里
    op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
    if (set.core == CoreType::AIV) {
        if ((Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) ||
            (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()))) {
            return false;
        }
        op.SetOpCode(Opcode::OP_BAR_V);
        return true;
    }
    op.SetOpCode(Opcode::OP_BAR_M);
    return true;
}

std::deque<int>* PipeSync::GetCrossCoreEventIdQPtr(const PipePairEx& pp)
{
    if (pp.first.core == CoreType::AIV) {
        return &crossCoreFreeEventId_[pp.first][0];
    }
    if (pp.second.aivCore == AIVCore::AIV0) {
        return &crossCoreFreeEventId_[pp.first][0];
    }
    if (pp.second.aivCore != AIVCore::AIV1) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV op is neither AIV0 nor AIV1, GetEventId failed.");
        return nullptr;
    }
    return &crossCoreFreeEventId_[pp.first][1];
}

void PipeSync::RemoveEventIdFromCrossCoreQueues(int eventId)
{
    for (auto& [otherP, otherQArr] : crossCoreFreeEventId_) {
        (void)otherP;
        for (auto& otherQ : otherQArr) {
            otherQ.erase(std::remove(otherQ.begin(), otherQ.end(), eventId), otherQ.end());
        }
    }
}

Status PipeSync::GetEventId(const PipePairEx& pp, int& eventId, size_t setIdx, size_t& waitIdx,
                            std::vector<IndexOp>& syncedOpLog, Function& function)
{
    // reserved for future
    (void)waitIdx;
    if (pp.first.pipe == pp.second.pipe && pp.first.core == pp.second.core) {
        // Pipe Barrier
        eventId = -1;
        return SUCCESS;
    }

    // CV eventid获取
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) && pp.first.core != pp.second.core) {
        std::deque<int>* cvEventIdQPtr = GetCrossCoreEventIdQPtr(pp);
        if (cvEventIdQPtr == nullptr) {
            return FAILED;
        }
        auto& cvEventIdQ = *cvEventIdQPtr;
        if (!cvEventIdQ.empty()) {
            eventId = cvEventIdQ.front();
            cvEventIdQ.pop_front();
            RemoveEventIdFromCrossCoreQueues(eventId);
            // 更新syncArriveStatus，将其中所有该srcCore和dstcore或者反向的该eventid都删除
            RemoveEventResourceFromSyncArriveStatus(pp, eventId);
            return SUCCESS;
        }

        // insert ffts to relax all eventid
        RemoveSetIntraBlockAndOpDep(syncedOpLog);
        AddCrossCoreForceSyncOps(setIdx, syncedOpLog, function);
        InitCVEventIdQ();
        std::deque<int>* newCvEventIdQPtr = GetCrossCoreEventIdQPtr(pp);
        if (newCvEventIdQPtr == nullptr || newCvEventIdQPtr->empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "EventId still exhausted after InitCVEventIdQ, GetEventId failed.");
            return FAILED;
        }
        eventId = newCvEventIdQPtr->front();
        newCvEventIdQPtr->pop_front();
        RemoveEventIdFromCrossCoreQueues(eventId);
        // 更新syncArriveStatus，将其中所有该srcCore和dstcore或者反向的该eventid都删除
        RemoveEventResourceFromSyncArriveStatus(pp, eventId);
        return SUCCESS;
    }

    auto& eventQ = GetFreeEventIdQueue(pp);
    if (eventQ.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Eventid exhausted, GetEventId failed.");
        return FAILED;
    }

    eventId = eventQ.front();
    eventQ.pop_front();
    return SUCCESS;
}

void PipeSync::RemoveSetIntraBlockAndOpDep(std::vector<IndexOp>& syncedOpLog)
{
    for (auto& [key, op] : NoWaitCVPairs_) {
        RemoveOpDep(depOps_[key.first], depOps_[key.second]);
        op->SetAsDeleted();
        setOpMap.erase(op);
        Operation* opPtr = op;
        syncedOpLog.erase(std::remove_if(syncedOpLog.begin(), syncedOpLog.end(),
                                         [opPtr](const IndexOp& entry) { return &entry.second.get() == opPtr; }),
                          syncedOpLog.end());
    }
    NoWaitCVPairs_.clear();
    crossCoreFreeEventId_.clear();
    syncArriveStatus.clear();
}

void PipeSync::CreateForceSyncOp(Opcode opcode, PipeType pipe, CoreType core, AIVCore aivCore, uint64_t& insertIdx,
                                 Function& function, std::vector<IndexOp>& syncedOpLog)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    Operation& op = irBuilder_.CreateTensorOpStmt(function, opcode, input, output);
    op.syncQueue_ = {pipe, pipe, core, core, EVENT_ID7, aivCore, aivCore};
    syncedOpLog.emplace_back(std::make_pair(++insertIdx, std::ref(op)));
    op.SetAIVCore(aivCore);
}

void PipeSync::CreateBarAllOp(AIVCore aivCore, uint64_t& insertIdx, Function& function,
                              std::vector<IndexOp>& syncedOpLog)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    Operation& op = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_BAR_ALL, input, output);
    op.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1, aivCore, aivCore};
    syncedOpLog.emplace_back(std::make_pair(++insertIdx, std::ref(op)));
    op.SetAIVCore(aivCore);
}

void PipeSync::AddCrossCoreForceSyncOps(size_t setIdx, std::vector<IndexOp>& syncedOpLog, Function& function)
{
    PipeType prevAicPipe = PipeType::PIPE_FIX;
    PipeType prevAiv0Pipe = PipeType::PIPE_MTE3;
    PipeType prevAiv1Pipe = PipeType::PIPE_MTE3;

    uint64_t insertIdx = setIdx * SEQUENCE_IDX;

    CreateBarAllOp(AIVCore::UNSPECIFIED, insertIdx, function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_FFTS_CROSS_CORE_SYNC, prevAicPipe, CoreType::AIC, AIVCore::UNSPECIFIED, insertIdx,
                      function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED, insertIdx,
                      function, syncedOpLog);
    CreateBarAllOp(AIVCore::UNSPECIFIED, insertIdx, function, syncedOpLog);
    CreateBarAllOp(AIVCore::AIV0, insertIdx, function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV0, insertIdx, function,
                      syncedOpLog);
    CreateForceSyncOp(Opcode::OP_FFTS_CROSS_CORE_SYNC, prevAiv0Pipe, CoreType::AIV, AIVCore::AIV0, insertIdx, function,
                      syncedOpLog);
    CreateBarAllOp(AIVCore::AIV0, insertIdx, function, syncedOpLog);
    CreateBarAllOp(AIVCore::AIV1, insertIdx, function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV1, insertIdx, function,
                      syncedOpLog);
    CreateForceSyncOp(Opcode::OP_FFTS_CROSS_CORE_SYNC, prevAiv1Pipe, CoreType::AIV, AIVCore::AIV1, insertIdx, function,
                      syncedOpLog);
    CreateBarAllOp(AIVCore::AIV1, insertIdx, function, syncedOpLog);
}

bool PipeSync::HasFreeEventId(const PipePairEx& pp)
{
    std::deque<int>& eventQ = GetFreeEventIdQueue(pp);
    return !eventQ.empty();
}

bool PipeSync::BufOverlap(const TileRange& range1, const TileRange& range2) const
{
    return range1.end > range2.start && range2.end > range1.start;
}

bool PipeSync::CheckWawDependency(const Operation& opSet, const Operation& opWait)
{
    for (size_t setIdx = 0; setIdx < opSet.GetOOperands().size(); setIdx++) {
        for (size_t waitIdx = 0; waitIdx < opWait.GetOOperands().size(); waitIdx++) {
            if (opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal() !=
                opWait.GetOOperands()[waitIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetOOperands()[setIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[waitIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] :
                                                               opSet.GetOOperands()[setIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] :
                                                               opWait.GetOOperands()[waitIdx]->memoryrange;
            if (BufOverlap(range1, range2)) {
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckRawDependency(const Operation& opSet, const Operation& opWait)
{
    for (size_t outIdx = 0; outIdx < opSet.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opWait.GetIOperands().size(); inIdx++) {
            if (opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal() !=
                opSet.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opWait.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opSet.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] :
                                                               opWait.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] :
                                                               opSet.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, range2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckWarDependency(const Operation& opSet, const Operation& opWait)
{
    for (size_t outIdx = 0; outIdx < opWait.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opSet.GetIOperands().size(); inIdx++) {
            if (opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal() !=
                opWait.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] :
                                                               opSet.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] :
                                                               opWait.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, range2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::HasDataDependency(const Operation& opSet, const Operation& opWait)
{
    std::string opSetStr = opSet.GetOpcodeStr();
    std::string opWaitStr = opWait.GetOpcodeStr();

    // check WAW
    bool checkWaw = true;
    auto setCfg = OpcodeManager::Inst().GetTileOpCfg(opSet.GetOpcode());
    auto waitCfg = OpcodeManager::Inst().GetTileOpCfg(opWait.GetOpcode());
    AdjustOpCfg(setCfg, opSet);
    AdjustOpCfg(waitCfg, opWait);
    if (waitCfg.pipeIdStart_ == setCfg.pipeIdStart_ &&
        (opSetStr.find("CUBE_A_MUL") == std::string::npos || opWaitStr.find("CUBE_A_MUL") == std::string::npos)) {
        checkWaw = false;
    }
    if (checkWaw) {
        if (CheckWawDependency(opSet, opWait)) {
            return true;
        }
    }

    // check RAW
    if (CheckRawDependency(opSet, opWait)) {
        return true;
    }

    // check WAR
    if (CheckWarDependency(opSet, opWait)) {
        return true;
    }

    return false;
}

void PipeSync::UpdateDep(DepOp& currOp, DepOp& prevOp)
{
    PipeCoreRealEx currPipe(currOp.selfPipeCore.pipeStart, currOp.selfPipeCore.core, currOp.selfPipeCore.aivCore);
    PipeCoreRealEx prevPipe(prevOp.selfPipeCore.pipeEnd, prevOp.selfPipeCore.core, prevOp.selfPipeCore.aivCore);
    auto& currPipeDep = latestPipeDep_[currPipe];
    currPipeDep.waitIdx = currOp.idx;

    auto currSetPipeIter = currPipeDep.setPipes.find(prevPipe);
    if (currSetPipeIter == currPipeDep.setPipes.end() || currSetPipeIter->second < prevOp.idx) {
        // no indirect dependency exist, save current dependency
        currOp.waitPipe.emplace_back(prevOp.idx);
        prevOp.setPipe.emplace_back(currOp.idx);
        currPipeDep.setPipes[prevPipe] = prevOp.idx;
        auto prevPipeDepIter = latestPipeDep_.find(prevPipe);
        auto prevWaitPipeIdx = prevPipeDepIter->second.waitIdx;
        if (prevPipeDepIter != latestPipeDep_.end() && prevWaitPipeIdx <= prevOp.idx) {
            // merge dependency
            std::map<PipeCoreRealEx, size_t, PipeCoreRealExCompare> prevSetPipes = prevPipeDepIter->second.setPipes;
            for (auto [prevSetPipeType, prevSetPipeIdx] : prevSetPipes) {
                auto res = currPipeDep.setPipes.emplace(prevSetPipeType, prevSetPipeIdx);
                // isExist == isPrevSetPipeTypeExist
                bool isExist = !res.second;
                size_t& existIdx = res.first->second;
                if (isExist && existIdx < prevSetPipeIdx) {
                    // overwrite dependency
                    existIdx = prevSetPipeIdx;
                }
            }
        }
    }
}

bool PipeSync::IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation*>& opLogPtr)
{
    // true表示依赖关系可忽略，false表示依赖关系不可忽略
    // VIEW or ASSEMBLE data dependency can be ignored
    if (opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW_TYPE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[curr]->GetOpcode() == Opcode::OP_ASSEMBLE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_NOP || opLogPtr[curr]->GetOpcode() == Opcode::OP_NOP ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_HUB || opLogPtr[curr]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_RESHAPE || opLogPtr[curr]->GetOpcode() == Opcode::OP_RESHAPE) {
        return true;
    }
    return false;
}

// find depend op in opLog for 0 to idx
void PipeSync::FindDep(DepOp& op, const std::vector<Operation*>& opLogPtr, size_t idx,
                       DataDependencySearcher& dataDependencySearcher)
{
    const auto currOp = opLogPtr[idx];
    APASS_LOG_DEBUG_F(Elements::Operation, "=== OP: %d %zu %s ===", currOp->GetOpMagic(), idx,
                      currOp->GetOpcodeStr().c_str());
    // check dependency from latest op to oldest
    auto dataDependencySet = dataDependencySearcher.Find(currOp);
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        const Operation* prevAOp = opLogPtr[k];
        DepOp& prevOp = depOps_[k];

        if (HasDataDependency(*prevAOp, *currOp)) {
            bool ignorable = false;
            if (IgnorableIntraPipeDep(k, idx, opLogPtr)) {
                ignorable = true;
            }
            if (!ignorable) {
                UpdateDep(op, prevOp);
            }
        }
    }
    dataDependencySearcher.Insert(currOp, idx);
}

void PipeSync::InitCVEventIdQ()
{
    for (const auto& pipecore : cvPipeCoreEx) {
        int base = (pipecore.aivCore == AIVCore::AIV0) ? 0 : CROSS_CORE_EVENT_NUM;
        if (pipecore.core == CoreType::AIC) {
            for (int i = 0; i < CROSS_CORE_EVENT_NUM; i++) {
                crossCoreFreeEventId_[pipecore][0].push_back(i);
            }
            for (int i = CROSS_CORE_EVENT_NUM; i < CROSS_CORE_EVENT_NUM * NUM2; i++) {
                crossCoreFreeEventId_[pipecore][1].push_back(i);
            }
        } else {
            for (int i = base; i < base + CROSS_CORE_EVENT_NUM; i++) {
                crossCoreFreeEventId_[pipecore][0].push_back(i);
            }
        }
    }
}

void PipeSync::RemoveEventResourceFromSyncArriveStatus(const PipePairEx& pp, int eventId)
{
    CoreTypeDetail firstCore{pp.first.core, pp.first.aivCore};
    CoreTypeDetail secondCore{pp.second.core, pp.second.aivCore};

    for (auto& [key, eventSet] : syncArriveStatus) {
        (void)key;
        for (auto it = eventSet.begin(); it != eventSet.end();) {
            bool matchForward = (it->srcCore == firstCore && it->dstCore == secondCore && it->eventId == eventId);
            bool matchReverse = (it->srcCore == secondCore && it->dstCore == firstCore && it->eventId == eventId);
            if (matchForward || matchReverse) {
                it = eventSet.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void PipeSync::PushEventIdIfAbsent(std::deque<int>& queue, int eventId)
{
    if (std::find(queue.begin(), queue.end(), eventId) == queue.end()) {
        queue.push_back(eventId);
    }
}

Status PipeSync::RecycleCrossCoreEventIds(const PipeCoreRealEx& currPipeRealEx)
{
    if (crossCoreFreeEventId_.count(currPipeRealEx) == 0) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "crossCoreFreeEventId_ does not contain the given PipeCoreRealEx, RecycleCrossCoreEventIds failed.");
        return FAILED;
    }

    auto& eventSet = syncArriveStatus[currPipeRealEx];
    auto& queues = crossCoreFreeEventId_[currPipeRealEx];

    // 先按确定顺序（eventId 升序）收集再入队，保证结果跨环境一致。
    std::vector<int> recycledToQueue0;
    std::vector<int> recycledToQueue1;
    for (const auto& er : eventSet) {
        if (er.srcCore == er.dstCore) {
            continue;
        }
        if (currPipeRealEx.core == CoreType::AIC) {
            if (er.eventId < CROSS_CORE_EVENT_NUM) {
                recycledToQueue0.push_back(er.eventId);
            } else {
                recycledToQueue1.push_back(er.eventId);
            }
        } else if (currPipeRealEx.aivCore == AIVCore::AIV0) {
            if (er.eventId < CROSS_CORE_EVENT_NUM) {
                recycledToQueue0.push_back(er.eventId);
            }
        } else if (currPipeRealEx.aivCore == AIVCore::AIV1) {
            if (er.eventId >= CROSS_CORE_EVENT_NUM) {
                recycledToQueue0.push_back(er.eventId);
            }
        }
    }
    std::sort(recycledToQueue0.begin(), recycledToQueue0.end());
    std::sort(recycledToQueue1.begin(), recycledToQueue1.end());
    for (int id : recycledToQueue0) {
        PushEventIdIfAbsent(queues[0], id);
    }
    for (int id : recycledToQueue1) {
        PushEventIdIfAbsent(queues[NUM1], id);
    }
    return SUCCESS;
}

// 该函数仅用于核内同步
std::deque<int>& PipeSync::GetFreeEventIdQueue(const PipePairEx& pp)
{
    if (freeEventId_.count(pp) == 0) {
        for (int i = INTRA_CORE_START_EVENT_ID; i <= GetMaxEventId(pp); i++) {
            freeEventId_[pp].push_back(i);
        }
    }
    return freeEventId_[pp];
}

void PipeSync::AddPhaseOp1(Function& function, const std::vector<Operation*>& srcLog, std::vector<Operation*>& dstLog,
                           size_t& i, size_t& prerun)
{
    constexpr size_t prerunNum = 2;
    for (; i < srcLog.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(srcLog[i]->GetOpcode());
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_IN) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        }
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        }
        if ((opcfg.pipeIdStart_ != PIPE_S && opcfg.pipeIdStart_ != PIPE_MTE2 &&
             srcLog[i]->GetOpcode() != Opcode::OP_RESHAPE && srcLog[i]->GetOpcode() != Opcode::OP_VEC_DUP) ||
            prerun == prerunNum) {
            break;
        }
        if (opcfg.pipeIdStart_ == PIPE_MTE2) {
            if (prerun == 0) {
                std::vector<std::shared_ptr<LogicalTensor>> input;
                std::vector<std::shared_ptr<LogicalTensor>> output;
                Operation& phaseOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_PHASE1, input,
                                                                   output);
                Operation* phaseOpPtr = &phaseOp;
                dstLog.emplace_back(phaseOpPtr);
            }
            prerun++;
        }
        dstLog.emplace_back(srcLog[i]);
    }
}

void PipeSync::AddPhaseOp2(Function& function, std::vector<Operation*>& dstLog, size_t& prerun)
{
    if (prerun > 0) {
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& phaseOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_PHASE2, input, output);
        Operation* phaseOpPtr = &phaseOp;
        dstLog.emplace_back(phaseOpPtr);
    }
}

void PipeSync::PhaseKernelProcess(Function& function, const std::vector<Operation*>& srcLog,
                                  std::vector<Operation*>& dstLog)
{
    size_t prerun = 0;
    size_t i = 0;
    AddPhaseOp1(function, srcLog, dstLog, i, prerun);
    AddPhaseOp2(function, dstLog, prerun);
    for (; i < srcLog.size(); i++) {
        dstLog.emplace_back(srcLog[i]);
    }
}

void InsertSync::InsertCvSyncOps(Function* subGraphFunc, std::vector<Operation*>& newOpList)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    auto createBarAll = [&](AIVCore aivCore) {
        Operation& op = irBuilder_.CreateTensorOpStmt(*subGraphFunc, Opcode::OP_BAR_ALL, input, output);
        op.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1, aivCore, aivCore};
        op.SetAIVCore(aivCore);
        newOpList.push_back(&op);
    };
    // 核间全同步固定使用EVENT_ID7
    auto createForceSync = [&](Opcode opcode, PipeType pipe, CoreType core, AIVCore aivCore) {
        Operation& op = irBuilder_.CreateTensorOpStmt(*subGraphFunc, opcode, input, output);
        op.syncQueue_ = {pipe, pipe, core, core, PipeSync::EVENT_ID7, aivCore, aivCore};
        op.SetAIVCore(aivCore);
        newOpList.push_back(&op);
    };

    createBarAll(AIVCore::UNSPECIFIED);
    createForceSync(Opcode::OP_FFTS_CROSS_CORE_SYNC, PipeType::PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    createForceSync(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED);
    createBarAll(AIVCore::UNSPECIFIED);
    createBarAll(AIVCore::AIV0);
    createForceSync(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV0);
    createForceSync(Opcode::OP_FFTS_CROSS_CORE_SYNC, PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    createBarAll(AIVCore::AIV0);
    createBarAll(AIVCore::AIV1);
    createForceSync(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV1);
    createForceSync(Opcode::OP_FFTS_CROSS_CORE_SYNC, PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV1);
    createBarAll(AIVCore::AIV1);
}

void InsertSync::InsertCvPipeAll(Function* subGraphFunc)
{
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    std::vector<Operation*> newOpList;
    PipeSync ps;
    for (size_t i = 0; i < oriOpList.size(); i++) {
        newOpList.push_back(oriOpList[i]);
        if (i == oriOpList.size() - 1) {
            continue;
        }
        auto currOpCfg = OpcodeManager::Inst().GetTileOpCfg(oriOpList[i]->GetOpcode());
        ps.AdjustOpCfg(currOpCfg, *oriOpList[i]);
        auto nextOpCfg = OpcodeManager::Inst().GetTileOpCfg(oriOpList[i + 1]->GetOpcode());
        ps.AdjustOpCfg(nextOpCfg, *oriOpList[i + 1]);
        if (currOpCfg.coreType_ == nextOpCfg.coreType_) {
            std::vector<std::shared_ptr<LogicalTensor>> input;
            std::vector<std::shared_ptr<LogicalTensor>> output;
            Operation& syncOp = irBuilder_.CreateTensorOpStmt(*subGraphFunc, npu::tile_fwk::Opcode::OP_BAR_ALL, input,
                                                              output);
            AIVCore nextAIVCore = oriOpList[i + 1]->GetAIVCore();
            syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1,
                                 nextAIVCore,        nextAIVCore};
            syncOp.SetAIVCore(nextAIVCore);
            newOpList.push_back(&syncOp);
            continue;
        }
        // 插入固定序列1C2V全同步
        InsertCvSyncOps(subGraphFunc, newOpList);
    }
    subGraphFunc->ScheduleBy(newOpList, true);
    subGraphFunc->oriOpList = oriOpList;
}

Status InsertSync::CheckNewOpListSeq(const std::vector<Operation*>& oriOpList, const std::vector<Operation*>& opListNew)
{
    if (oriOpList.size() <= 1) {
        return SUCCESS;
    }
    size_t i = 0;
    size_t j = 0;
    while (i < oriOpList.size() && j < opListNew.size()) {
        if (oriOpList[i] == opListNew[j]) {
            ++i;
            ++j;
        } else {
            ++j;
        }
    }
    if (i != oriOpList.size()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "NewOpList sequence is not equal to OriOpList sequence, CheckNewOpListSeq failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status InsertSync::GenNewOpList(Function* subGraphFunc, std::vector<Operation*>& opListNew)
{
    PipeSync ps;
    std::vector<Operation*> syncedOpLogPtr;
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    if (ps.InsertSync(*subGraphFunc, syncedOpLogPtr) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function InsertSync.");
        return FAILED;
    }
    ps.PhaseKernelProcess(*subGraphFunc, syncedOpLogPtr, opListNew);
    subGraphFunc->EraseOperations(true, false);
    if (CheckNewOpListSeq(oriOpList, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function CheckNewOpListSeq.");
        return FAILED;
    }
    subGraphFunc->setOpMap = ps.setOpMap;
    subGraphFunc->waitOpMap = ps.waitOpMap;
    subGraphFunc->oriOpList = ps.GetOriOpList();
    return SUCCESS;
}

Status InsertSync::InsertSyncMainLoop(Function* subGraphFunc)
{
    if (enableDebug_) {
        InsertCvPipeAll(subGraphFunc);
        return SUCCESS;
    }
    std::vector<Operation*> opListNew;
    if (GenNewOpList(subGraphFunc, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSyncMainLoop failed at GenNewOpList.");
        return FAILED;
    }
    subGraphFunc->ScheduleBy(opListNew, true);
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "==========================================================================================");
    for (const auto& op : subGraphFunc->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("SYNC_SRC") != std::string::npos ||
            op->GetOpcodeStr().find("SYNC_DST") != std::string::npos || op->GetOpcode() == Opcode::OP_BAR_V ||
            op->GetOpcode() == Opcode::OP_BAR_M) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Output operation %d: %s, setpipe type: %s, setcore type: %s, waitpipe type: %s, waitcore type: %s, "
                "eventid: %d, setAIVcore type: %d, waitAIVcore type: %d",
                op->GetOpMagic(), op->GetOpcodeStr().c_str(), GetPipeTypeDict().Find(op->syncQueue_.pipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.coreType_).c_str(),
                GetPipeTypeDict().Find(op->syncQueue_.trigPipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.trigCoreType_).c_str(), op->syncQueue_.eventId_,
                static_cast<int>(op->syncQueue_.setAivCore_), static_cast<int>(op->syncQueue_.waitAivCore_));
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Output operation %d: %s, AIV core type: %d", op->GetOpMagic(),
                          op->GetOpcodeStr().c_str(), static_cast<int>(op->GetAIVCore()));
    }
    return SUCCESS;
}

// regist pass
Status InsertSync::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Start InsertSync.");
    const unsigned hardwareConcurrency = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
    uint64_t index = 0;
    std::vector<std::pair<uint64_t, Function*>> subPrograms;
    for (auto& subProgram : function.rootFunc_->programs_) {
        subPrograms.push_back(subProgram);
    }
    std::atomic<size_t> nextIdx(0);
    size_t leafFuncSize = function.rootFunc_->programs_.size();
    // The max thread number by std::thread::hardware_concurrency()

    const unsigned threadNum = std::min(static_cast<unsigned>(leafFuncSize), hardwareConcurrency);
    std::vector<std::thread> workers;
    bool status{true};
    for (unsigned i = 0; i < threadNum; ++i) {
        workers.emplace_back([&subPrograms, &nextIdx, leafFuncSize, &index, this, &status] {
            for (size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed); idx < leafFuncSize;
                 idx = nextIdx.fetch_add(1, std::memory_order_relaxed)) {
                auto program = subPrograms[idx];
                APASS_LOG_DEBUG_F(
                    Elements::Operation,
                    "====================================Program %zu ===========================================",
                    index);
                if (InsertSyncMainLoop(program.second) != SUCCESS) {
                    status = false;
                    break;
                }
                index++;
            }
        });
    }
    // Wait for all threads to finish
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (!status) {
        if (threadNum == 1) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Single Thread scenario.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Multiple Threads scenario.");
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Finish InsertSync.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
