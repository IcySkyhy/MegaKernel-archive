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
 * \file inplace_process.h
 * \brief
 */

#ifndef INPLACE_PROCESS_H
#define INPLACE_PROCESS_H
#include <vector>
#include <climits>

#include "tilefwk/data_type.h"
#include "interface/operation/opcode.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {
/*
key: Opcode类型
vaule: vector of pair, 每个pair记录了第几个输入和第几个输出存在inplace关系
*/
const std::unordered_map<Opcode, std::vector<std::pair<size_t, size_t>>> inplaceOpMap = {
    {Opcode::OP_A_MULACC_B, {std::pair<size_t, size_t>{2, 0}}},
    {Opcode::OP_INDEX_OUTCAST, {std::pair<size_t, size_t>{2, 0}}},
};

class InplaceProcess : public Pass {
public:
    InplaceProcess() : Pass("InplaceProcess") {}
    ~InplaceProcess() override = default;

private:
    /*
    补齐: Status PreCheck(Function &function) override;
    补齐: Status PostCheck(Function &function) override;
    */
    Status PreCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status ProcessOp(Function& function);
    Status InplaceProcessAssemble(Function& function, Operation& op);
    bool HasSameConsecutive(Operation& op);
    void ProcessView(Function& function, Operation& op) const;
    void ProcessAssemble(Function& function, Operation& op);
    Status AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const;
    Status AlignCopyOutProducer(std::shared_ptr<LogicalTensor> tensorGm) const;
    void ProcessReshape(Function& function, Operation& op) const;
    Status ProcessViewType(Function& function, Operation& op) const;
    Status ProcessInplaceOp(Function& function, Operation& op) const;
    Status ValidMeaninglessOp(const Operation& op) const;
    Status AdjustOffsetAndRawShape(LogicalTensorPtr& fromView, LogicalTensorPtr& toView) const;
    void ReplaceRawTensor(Function& function, std::shared_ptr<LogicalTensor> logicalTensor,
                          const std::shared_ptr<LogicalTensor> targetTensor, const Operation& op);
    void ProcessHub(Function& function, Operation& op);
    void ProcessHubAssembleChain(Function& function, Operation& hubOp, Operation& assembleOp,
                                 std::shared_ptr<LogicalTensor> hubInput, std::shared_ptr<LogicalTensor> hubOutput);
    Status RefactorViewConnectForInplace(Function& function);
    std::unordered_map<DataType, int> viewTypeTable = {{DT_INT8, 1}, {DT_BF16, 2}, {DT_FP16, 2}, {DT_FP32, 4}};
    std::vector<int> visitedAssembleOp;
    std::vector<int> hubRelatedAssembleOpMagics;
};
} // namespace tile_fwk
} // namespace npu
#endif // INPLACE_PROCESS_H
