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
 * \file duplicate_op.h
 * \brief
 */

#pragma once

#include <stack>
#include <unordered_set>
#include <vector>
#include "passes/pass_interface/pass.h"
namespace npu::tile_fwk {
/*
    DuplicateOp: 对于一个View OP，如果存在多消费者的情况，则为每一个消费者创建一个新的View OP;对于一个GatherIn
   OP，如果存在多消费者的情况，则为每一个消费者创建一个新的GatherIn OP
*/
class DuplicateOp : public Pass {
public:
    DuplicateOp() : Pass("DuplicateOp") {}
    ~DuplicateOp() override = default;

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status ProcessOp(Function& function, Operation& operation, std::vector<Operation*>& newOps) const;
    Status Process(Function& function, std::vector<Operation*>& newOps) const;
    void PushProducers(const LogicalTensors& operands, std::stack<Operation*>& stack,
                       std::unordered_set<int>& visitedOps, std::unordered_set<int>& visitedTensors) const;
    Status ProcessGatherIn(Function& function, Operation& operation, std::vector<Operation*>& newOps) const;
    Status ProcessViewLike(Function& function, Operation& operation, std::vector<Operation*>& newOps) const;
};

} // namespace npu::tile_fwk
