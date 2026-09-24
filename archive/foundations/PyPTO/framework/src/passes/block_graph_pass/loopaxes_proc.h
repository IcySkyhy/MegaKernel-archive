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
 * \file loopaxes_proc.h
 * \brief
 */

#ifndef PASS_LOOPAXES_PROC_H
#define PASS_LOOPAXES_PROC_H

#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"

namespace npu {
namespace tile_fwk {
class LoopaxesProc : public Pass {
public:
    LoopaxesProc() : Pass("LoopaxesProc") { SetSupportedArches({NPUArch::DAV_3510}); }
    ~LoopaxesProc() override = default;
    Status RunOnFunction(Function& function) override;

private:
    Status UpdateFuncLoopAxes(Function& function);
    Status UpdateOpLoopAxes(Operation& op, Function& subFunc);
    bool SameDynLoopAxes(const std::vector<SymbolicScalar>& curLoopAxes, const Function& subFunc);
    bool SameLoopAxes(const std::vector<int64_t>& curLoopAxes);
    void ClearStatus();
    void ProcessDynLoopGroup(Operation& op, const std::vector<SymbolicScalar>& dynloopAxes, const Function& subFunc);
    void ProcessStaticLoopGroup(Operation& op, const std::vector<int64_t>& loopAxes);
    void ResetGroupState();
    void FinalizeLoopGroups();
    void RecordAddrOverLap(Operation* op, int& idx, std::set<std::pair<int, int>>& addrConflictIdx,
                           std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap);
    void IsOverLap(std::vector<size_t>& addrRange, bool& isAdd, int& conflictIdx,
                   std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap,
                   std::set<std::pair<int, int>>& addrConflictIdx, int& idx);
    void CheckAddrOverLap(bool isStaticLoop, std::vector<Operation*>& sameLoopOpGroup,
                          std::set<std::pair<int, int>>& addrConflictIdx,
                          std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap);
    void ProcessCutStaticGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup);
    void ProcessCutDynGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup);

    int64_t groupIdx{INVALID_LOOP_GROUPID};
    int64_t lastGroupIdx{INVALID_LOOP_GROUPID};
    int64_t previousOutputMagic{INVALID_LOOP_GROUPID};
    Operation* lastOpInLoop{nullptr};
    Operation* lastOpInLoop1{nullptr};
    std::vector<int64_t> previousLoopAxes;

    int64_t dynGroupIdx{INVALID_LOOP_GROUPID};
    int64_t dynLastGroupIdx{INVALID_LOOP_GROUPID};
    int64_t dynPreviousOutputMagic{INVALID_LOOP_GROUPID};
    Operation* dynLastOpInLoop{nullptr};
    std::vector<SymbolicScalar> dynPreviousLoopAxes;

    std::map<int, std::vector<std::vector<size_t>>> addrStaticRecordMap;
    std::vector<Operation*> sameStaticLoopOpGroup;
    std::set<std::pair<int, int>> addrStaticConflictIdx;

    std::map<int, std::vector<std::vector<size_t>>> addrDynRecordMap;
    std::vector<Operation*> sameDynLoopOpGroup;
    std::set<std::pair<int, int>> addrDynConflictIdx;
};

} // namespace tile_fwk
} // namespace npu
#endif // PASS_LOOPAXES_PROC_H
