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
 * \file schedule_ooo_checker.h
 * \brief
 */

#ifndef SCHEDULE_OOO_CHECKER_H
#define SCHEDULE_OOO_CHECKER_H

#include "checker.h"
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class OoOScheduleChecker : Checker {
public:
    void SetOriFunctions(const std::vector<Function*>& oriFunctions);
    Status DoPreCheck(Function& function) override;
    Status DoPostCheck(Function& function) override;

private:
    bool PreCheckTensorInfo(const LogicalTensorPtr tensor);
    bool PreCheckOpInfo(const Operation* op);
    bool PostCheckOpMagic(std::set<int> opSet, const Operation* op, const int programIdx);
    bool PostCheckNewOpConnection(const std::vector<Operation*> opListBeforePass,
                                  const std::vector<int> opMagicListBeforePass, const Operation* op,
                                  const int programIdx);
    bool PostCheckSpecialOp(const Operation* opBeforeIncast);
    bool PostCheckTensorMagic(std::set<int> tensorSet, const LogicalTensorPtr tensor, const int programIdx);
    bool PostCheckLocalTensor(const LogicalTensorPtr tensor, const int programIdx);
    bool PostCheckGlobalTensor(const LogicalTensorPtr tensor, const int programIdx);
    bool PostCheckDynValidShape(const LogicalTensorPtr tensor, const int programIdx);
    bool PostCheckNewTensor(std::pair<const int, Function*> program, const int programIdx);
    Status PostCheckTensor(const LogicalTensorPtr& tensor, const std::set<int>& tensorSet, int programIdx);
    Status PostCheckSubGraph(const std::pair<uint64_t, Function*>& program, int programIdx);

    std::vector<Function*> oriFunctions_;
    std::vector<std::unordered_set<LogicalTensorPtr>> tensorListBeforePass_;
    std::vector<std::unordered_set<LogicalTensorPtr>> tensorListAfterPass_;
};
} // namespace tile_fwk
} // namespace npu
#endif // SCHEDULE_OOO_CHECKER_H
