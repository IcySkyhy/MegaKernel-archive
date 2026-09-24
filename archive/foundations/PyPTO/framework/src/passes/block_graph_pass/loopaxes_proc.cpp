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
 * \file loopaxes_proc.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_interface/pass.h"
#include "loopaxes_proc.h"
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "tilefwk/error_code.h"

#undef MODULE_NAME
#define MODULE_NAME "LoopaxesProc"

namespace npu {
namespace tile_fwk {

namespace {
constexpr size_t MIN_SHAPE_DIM = 2;

void SetOpLoopEnd(Operation* op)
{
    op->SetAttribute(OpAttributeKey::loopGroupEnd, true);
    APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup --End--", op->GetOpcodeStr().c_str(),
                     op->GetOpMagic());
}

void SetOpDynLoopEnd(Operation* op)
{
    op->SetAttribute(OpAttributeKey::dynloopGroupEnd, true);
    APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup --End--", op->GetOpcodeStr().c_str(),
                     op->GetOpMagic());
}

bool NeedClearStatus(const Operation& op)
{
    auto opCode = op.GetOpcode();
    if (SUPPORT_VF_FUSE_OPS.find(opCode) == SUPPORT_VF_FUSE_OPS.end()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%d %s doesn't support VF fuse", op.GetOpMagic(),
                          op.GetOpcodeStr().c_str());
        return true;
    }
    return false;
}

void GetOpLoopAxes(const Operation& op, std::vector<int64_t>& loopAxes, std::vector<SymbolicScalar>& dynloopAxes)
{
    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    auto dynShape = output->GetDynValidShape();

    if (op.HasAttr(OpAttributeKey::dynloopAxes)) {
        dynloopAxes = op.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes);
    } else {
        for (size_t i = 0; i < dynShape.size() - MIN_SHAPE_DIM; ++i) {
            dynloopAxes.push_back(dynShape[i]);
        }
    }

    if (op.HasAttr(OpAttributeKey::loopAxes)) {
        loopAxes = op.GetVectorIntAttribute(OpAttributeKey::loopAxes);
    } else {
        for (size_t i = 0; i < shape.size() - MIN_SHAPE_DIM; ++i) {
            loopAxes.push_back(shape[i]);
        }
    }
}

void HandleSmallShapeOp(Operation& op)
{
    op.SetAttribute(OpAttributeKey::dynloopGroup, INVALID_LOOP_GROUPID);
    op.SetAttribute(OpAttributeKey::loopGroup, INVALID_LOOP_GROUPID);
}
} // namespace

Status LoopaxesProc::RunOnFunction(Function& function)
{
    bool enableVF = config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
    bool useMarkFor = enableVF || config::GetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, false);
    if (!useMarkFor) {
        return SUCCESS;
    }

    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Start LoopaxesProc.");
    UpdateFuncLoopAxes(function);
    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Finish LoopaxesProc.");
    return SUCCESS;
}

void LoopaxesProc::ClearStatus()
{
    lastGroupIdx = INVALID_LOOP_GROUPID;
    previousOutputMagic = INVALID_LOOP_GROUPID;
    previousLoopAxes.clear();
    if (lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
        lastOpInLoop = nullptr;
    }

    dynLastGroupIdx = INVALID_LOOP_GROUPID;
    dynPreviousOutputMagic = INVALID_LOOP_GROUPID;
    dynPreviousLoopAxes.clear();
    if (dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
        dynLastOpInLoop = nullptr;
    }
}

Status LoopaxesProc::UpdateOpLoopAxes(Operation& op, Function& subFunc)
{
    if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op Code %s, Op[%d] ignore this op", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic());
        return SUCCESS;
    }

    if (NeedClearStatus(op)) {
        ClearStatus();
        return SUCCESS;
    }

    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    auto dynShape = output->GetDynValidShape();
    if (shape.size() != dynShape.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op Code %s, Op[%d] output dynShape size != shape size.",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }

    if (dynShape.size() <= MIN_SHAPE_DIM) {
        HandleSmallShapeOp(op);
        ClearStatus();
        return SUCCESS;
    }

    std::vector<int64_t> loopAxes;
    std::vector<SymbolicScalar> dynloopAxes;
    GetOpLoopAxes(op, loopAxes, dynloopAxes);

    ProcessDynLoopGroup(op, dynloopAxes, subFunc);
    ProcessStaticLoopGroup(op, loopAxes);

    APASS_LOG_INFO_F(Elements::Operation,
                     "Op Code %s, Op[%d] groupIdx=%ld, loopAxes=%s, dynGroupIdx=%ld, dynLoopAxes=%s",
                     op.GetOpcodeStr().c_str(), op.GetOpMagic(), groupIdx, IntVecToStr(loopAxes).c_str(), dynGroupIdx,
                     IntVecToStr(dynloopAxes).c_str());

    return SUCCESS;
}

void LoopaxesProc::ProcessDynLoopGroup(Operation& op, const std::vector<SymbolicScalar>& dynloopAxes,
                                       const Function& subFunc)
{
    if (!SameDynLoopAxes(dynloopAxes, subFunc)) {
        CheckAddrOverLap(false, sameDynLoopOpGroup, addrDynConflictIdx, addrDynRecordMap);
        dynLastGroupIdx = dynGroupIdx++;
        dynPreviousLoopAxes = dynloopAxes;
        op.SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        if (dynLastOpInLoop != nullptr) {
            SetOpDynLoopEnd(dynLastOpInLoop);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup ++Start++",
                         op.GetOpcodeStr().c_str(), op.GetOpMagic());
        sameDynLoopOpGroup.clear();
        addrDynConflictIdx.clear();
        addrDynRecordMap.clear();
    }
    sameDynLoopOpGroup.push_back(&op);
    op.SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
    op.SetAttribute(OpAttributeKey::dynloopAxes, dynloopAxes);
    dynLastOpInLoop = &op;
    dynPreviousOutputMagic = op.GetOOperands().front()->GetMagic();
}

namespace {
struct Interval {
    int l, r;
};

std::vector<Interval> BuildIntervals(const std::set<std::pair<int, int>>& conflicts)
{
    std::vector<Interval> intervals;
    for (const auto& p : conflicts) {
        int a = p.first, b = p.second;
        if (a == b)
            continue;
        if (a > b)
            std::swap(a, b);
        if (a <= b - 1)
            intervals.push_back({a, b - 1});
    }
    return intervals;
}

bool IsIntervalCovered(const Interval& inv, const std::vector<int>& cuts)
{
    for (int cut : cuts) {
        if (cut >= inv.l && cut <= inv.r)
            return true;
    }
    return false;
}

int CalculateMaxSegment(const std::vector<int>& sortedCuts, int groupSize)
{
    int prev = 0, maxSeg = 0;
    for (int cut : sortedCuts) {
        int segSize = cut - prev + 1;
        maxSeg = std::max(maxSeg, segSize);
        prev = cut + 1;
    }
    return std::max(maxSeg, groupSize - prev);
}

void UpdateBestSolution(const std::vector<int>& cuts, int groupSize, std::vector<int>& bestCuts, int& bestCutCount,
                        int& bestMaxSeg)
{
    int curCutCount = static_cast<int>(cuts.size());
    std::vector<int> sortedCuts = cuts;
    std::sort(sortedCuts.begin(), sortedCuts.end());
    int maxSeg = CalculateMaxSegment(sortedCuts, groupSize);
    if (curCutCount < bestCutCount || (curCutCount == bestCutCount && maxSeg > bestMaxSeg)) {
        bestCutCount = curCutCount;
        bestMaxSeg = maxSeg;
        bestCuts = cuts;
    }
}
} // namespace

std::vector<int> FindCuts(const std::set<std::pair<int, int>>& conflicts, int& groupSize)
{
    if (groupSize <= 1)
        return {};
    auto intervals = BuildIntervals(conflicts);
    if (intervals.empty())
        return {};

    int totalPos = groupSize - 1;
    int bestCutCount = INT_MAX, bestMaxSeg = -1;
    std::vector<int> bestCuts;

    for (int mask = 0; mask < (1 << totalPos); ++mask) {
        std::vector<int> cuts;
        for (int i = 0; i < totalPos; ++i) {
            if (mask & (1 << i))
                cuts.push_back(i);
        }
        bool allCovered = true;
        for (const auto& inv : intervals) {
            if (!IsIntervalCovered(inv, cuts)) {
                allCovered = false;
                break;
            }
        }
        if (allCovered)
            UpdateBestSolution(cuts, groupSize, bestCuts, bestCutCount, bestMaxSeg);
    }
    return bestCutCount == INT_MAX ? std::vector<int>{} : bestCuts;
}

void LoopaxesProc::IsOverLap(std::vector<size_t>& addrRange, bool& isAdd, int& conflictIdx,
                             std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap,
                             std::set<std::pair<int, int>>& addrConflictIdx, int& idx)
{
    for (auto& entry : addrRecordMap) {
        bool noOverlapWithInput = addrRange[0] >= entry.second[0][1] || addrRange[1] <= entry.second[0][0];
        bool noOverlapWithOutput = addrRange[0] >= entry.second[1][1] || addrRange[1] <= entry.second[1][0];
        if (noOverlapWithInput && noOverlapWithOutput) {
            isAdd = true;
            conflictIdx = INVALID_LOOP_GROUPID;
        } else {
            isAdd = true;
            conflictIdx = entry.first;
            std::pair<int, int> conflictPair(conflictIdx, idx);
            addrConflictIdx.insert(conflictPair);
        }
    }
}

void LoopaxesProc::RecordAddrOverLap(Operation* op, int& idx, std::set<std::pair<int, int>>& addrConflictIdx,
                                     std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap)
{
    std::vector<size_t> inAddrRange;
    std::vector<size_t> outAddrRange;
    inAddrRange.push_back(op->GetIOperands().front()->memoryrange.start);
    inAddrRange.push_back(op->GetIOperands().front()->memoryrange.end);
    outAddrRange.push_back(op->GetOOperands().front()->memoryrange.start);
    outAddrRange.push_back(op->GetOOperands().front()->memoryrange.end);
    if (addrRecordMap.empty()) {
        addrRecordMap[idx].push_back(inAddrRange);
        addrRecordMap[idx].push_back(outAddrRange);
        return;
    }
    bool isAdd{false};
    int conflictIdx = INVALID_LOOP_GROUPID;
    addrRecordMap[idx].push_back(inAddrRange);
    addrRecordMap[idx].push_back(outAddrRange);
    IsOverLap(inAddrRange, isAdd, conflictIdx, addrRecordMap, addrConflictIdx, idx);
    IsOverLap(outAddrRange, isAdd, conflictIdx, addrRecordMap, addrConflictIdx, idx);
    return;
}

void LoopaxesProc::CheckAddrOverLap(bool isStaticLoop, std::vector<Operation*>& sameLoopOpGroup,
                                    std::set<std::pair<int, int>>& addrConflictIdx,
                                    std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap)
{
    if (sameLoopOpGroup.size() != 1) {
        for (int idx = 0; idx < static_cast<int>(sameLoopOpGroup.size()); idx++) {
            APASS_LOG_INFO_F(Elements::Operation, "RecordAddrOverLap %s[%d].",
                             sameLoopOpGroup[idx]->GetOpcodeStr().c_str(), sameLoopOpGroup[idx]->GetOpMagic());
            RecordAddrOverLap(sameLoopOpGroup[idx], idx, addrConflictIdx, addrRecordMap);
        }
    }
    if (addrConflictIdx.empty()) {
        return;
    }
    std::vector<int> cutResult;
    int groupSize = static_cast<int>(sameLoopOpGroup.size());
    cutResult = FindCuts(addrConflictIdx, groupSize);
    if (cutResult.empty()) {
        return;
    }
    if (isStaticLoop) {
        ProcessCutStaticGroup(cutResult, sameLoopOpGroup);
    } else {
        ProcessCutDynGroup(cutResult, sameLoopOpGroup);
    }
}

void LoopaxesProc::ProcessCutStaticGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup)
{
    for (size_t i = 0; i < cutResult.size(); i++) {
        lastGroupIdx = groupIdx++;
        lastOpInLoop1 = sameLoopOpGroup[cutResult[i]];
        if (lastOpInLoop1 != nullptr) {
            SetOpLoopEnd(lastOpInLoop1);
        }
        sameLoopOpGroup[cutResult[i] + 1]->SetAttribute(OpAttributeKey::loopGroupStart, true);
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpcodeStr().c_str(),
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpMagic());
        if (i != cutResult.size() - 1) {
            for (int opIdx = cutResult[i] + 1; opIdx <= cutResult[i + 1]; opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::loopGroup, groupIdx);
            }
        } else {
            for (int opIdx = cutResult[i] + 1; opIdx < static_cast<int>(sameLoopOpGroup.size()); opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::loopGroup, groupIdx);
            }
        }
    }
}

void LoopaxesProc::ProcessCutDynGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup)
{
    for (size_t i = 0; i < cutResult.size(); i++) {
        dynLastGroupIdx = dynGroupIdx++;
        lastOpInLoop1 = sameLoopOpGroup[cutResult[i]];
        if (lastOpInLoop1 != nullptr) {
            SetOpDynLoopEnd(lastOpInLoop1);
        }
        sameLoopOpGroup[cutResult[i] + 1]->SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpcodeStr().c_str(),
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpMagic());
        if (i != cutResult.size() - 1) {
            for (int opIdx = cutResult[i] + 1; opIdx <= cutResult[i + 1]; opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
            }
        } else {
            for (int opIdx = cutResult[i] + 1; opIdx < static_cast<int>(sameLoopOpGroup.size()); opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
            }
        }
    }
}

void LoopaxesProc::ProcessStaticLoopGroup(Operation& op, const std::vector<int64_t>& loopAxes)
{
    if (!SameLoopAxes(loopAxes)) {
        CheckAddrOverLap(true, sameStaticLoopOpGroup, addrStaticConflictIdx, addrStaticRecordMap);
        lastGroupIdx = groupIdx++;
        previousLoopAxes = loopAxes;
        op.SetAttribute(OpAttributeKey::loopGroupStart, true);
        if (lastOpInLoop != nullptr) {
            SetOpLoopEnd(lastOpInLoop);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++", op.GetOpcodeStr().c_str(),
                         op.GetOpMagic());
        sameStaticLoopOpGroup.clear();
        addrStaticConflictIdx.clear();
        addrStaticRecordMap.clear();
    }
    sameStaticLoopOpGroup.push_back(&op);
    op.SetAttribute(OpAttributeKey::loopGroup, groupIdx);
    op.SetAttribute(OpAttributeKey::loopAxes, loopAxes);
    lastOpInLoop = &op;
    previousOutputMagic = op.GetOOperands().front()->GetMagic();
}

Status LoopaxesProc::UpdateFuncLoopAxes(Function& function)
{
    DynAttrToStatic dyn2Static;
    if (dyn2Static.BuildLeafToCaller(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to call BuildLeafToCaller.");
        return FAILED;
    }

    for (auto& pair : dyn2Static.leaf2Caller) {
        if (pair.first == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation, "subProgram of Function is nullptr.");
            continue;
        }

        ResetGroupState();

        for (auto& op : pair.first->Operations(false)) {
            UpdateOpLoopAxes(op, *pair.first);
        }

        FinalizeLoopGroups();
    }
    return SUCCESS;
}

void LoopaxesProc::ResetGroupState()
{
    groupIdx = INVALID_LOOP_GROUPID;
    lastGroupIdx = groupIdx;
    lastOpInLoop = nullptr;

    dynGroupIdx = INVALID_LOOP_GROUPID;
    dynLastGroupIdx = dynGroupIdx;
    dynLastOpInLoop = nullptr;
}

void LoopaxesProc::FinalizeLoopGroups()
{
    CheckAddrOverLap(false, sameDynLoopOpGroup, addrDynConflictIdx, addrDynRecordMap);
    CheckAddrOverLap(true, sameStaticLoopOpGroup, addrStaticConflictIdx, addrStaticRecordMap);

    if (lastGroupIdx != INVALID_LOOP_GROUPID && lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
    }
    if (dynLastGroupIdx != INVALID_LOOP_GROUPID && dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
    }
}

bool LoopaxesProc::SameLoopAxes(const std::vector<int64_t>& curLoopAxes)
{
    if (curLoopAxes.size() != previousLoopAxes.size()) {
        return false;
    }
    for (size_t i = 0; i < curLoopAxes.size(); i++) {
        if (curLoopAxes[i] != previousLoopAxes[i]) {
            return false;
        }
    }
    return true;
}

bool LoopaxesProc::SameDynLoopAxes(const std::vector<SymbolicScalar>& curLoopAxes, const Function& subFunc)
{
    if (curLoopAxes.size() != dynPreviousLoopAxes.size()) {
        return false;
    }

    auto dynParamTable = subFunc.GetDynParamTable();
    bool allReplacedSymbolsMatch = true;
    bool allExprsMatch = true;

    for (size_t i = 0; i < curLoopAxes.size(); ++i) {
        auto curExpr = SymbolicExpressionTable::BuildExpression(curLoopAxes[i]);
        auto prevExpr = SymbolicExpressionTable::BuildExpression(dynPreviousLoopAxes[i]);
        if (dynParamTable.find(curExpr) != dynParamTable.end() && dynParamTable.find(prevExpr) != dynParamTable.end()) {
            auto curParamInfo = dynParamTable[curExpr];
            auto preParamInfo = dynParamTable[prevExpr];
            if (curParamInfo.replacedSymbol.empty() || preParamInfo.replacedSymbol.empty() ||
                curParamInfo.replacedSymbol != preParamInfo.replacedSymbol) {
                allReplacedSymbolsMatch = false;
            }
        } else {
            allReplacedSymbolsMatch = false;
        }

        if (curExpr != prevExpr) {
            allExprsMatch = false;
        }
    }

    return allReplacedSymbolsMatch || allExprsMatch;
}

} // namespace tile_fwk
} // namespace npu
