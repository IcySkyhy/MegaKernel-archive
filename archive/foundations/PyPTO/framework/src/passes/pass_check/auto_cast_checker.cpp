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
 * \file auto_cast_checker.cpp
 * \brief
 */

#include "auto_cast_checker.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

#define MODULE_NAME "AutoCast"

namespace npu {
namespace tile_fwk {
Status AutoCastChecker::DoDefaultEnabledPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "DoDefaultEnabledPreCheck for AutoCast");
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation* op = opList[opIdx];
        if (std::find(CAST_OPS.begin(), CAST_OPS.end(), op->GetOpcode()) == CAST_OPS.end()) {
            continue;
        }

        int inputNum = static_cast<int>(op->GetIOperands().size());
        if (inputNum != 1) {
            APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                              "CAST op %d has %d input tensor, which should be 1.", op->GetOpMagic(), inputNum);
            return FAILED;
        }

        int outputNum = static_cast<int>(op->GetOOperands().size());
        if (outputNum != 1) {
            APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                              "CAST op %d has %d output tensor, which should be 1.", op->GetOpMagic(), outputNum);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AutoCastChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for AutoCast");
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation* op = opList[opIdx];
        bool supportBF16 = SupportBF16(op);
        bool supportFP16 = SupportFP16(op);
        const int opMagic = op->GetOpMagic();

        auto iOperands = op->GetIOperands();
        for (const auto& iop : iOperands) {
            if (!supportBF16 && iop->Datatype() == DataType::DT_BF16) {
                APASS_LOG_ERROR_F(Elements::Operation, "Exist unsupported BF16 compute between op %d and tensor %d",
                                  opMagic, iop->GetMagic());
                return FAILED;
            }
            if (!supportFP16 && iop->Datatype() == DataType::DT_FP16) {
                APASS_LOG_ERROR_F(Elements::Operation, "Exist unsupported FP16 compute between op %d and tensor %d",
                                  opMagic, iop->GetMagic());
                return FAILED;
            }
        }

        auto oOperands = op->GetOOperands();
        for (const auto& oop : oOperands) {
            if (!supportBF16 && oop->Datatype() == DataType::DT_BF16) {
                APASS_LOG_ERROR_F(Elements::Operation, "Exist unsupported BF16 compute between op %d and tensor %d",
                                  opMagic, oop->GetMagic());
                return FAILED;
            }
            if (!supportFP16 && oop->Datatype() == DataType::DT_FP16) {
                APASS_LOG_ERROR_F(Elements::Operation, "Exist unsupported FP16 compute between op %d and tensor %d",
                                  opMagic, oop->GetMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool AutoCastChecker::SupportBF16(Operation* op)
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        return UNSUPPORT_BF16_ARCH35_OPS.count(op->GetOpcode()) == 0;
    }
    return UNSUPPORT_BF16_OPS.count(op->GetOpcode()) == 0;
}

bool AutoCastChecker::SupportFP16(Operation* op)
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        return UNSUPPORT_FP16_ARCH35_OPS.count(op->GetOpcode()) == 0;
    }
    if (UNSUPPORT_FP16_OPS.count(op->GetOpcode()) > 0) {
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu
