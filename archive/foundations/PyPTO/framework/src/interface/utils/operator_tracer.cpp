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
 * \file operator_tracer.cpp
 * \brief
 */

#include "operator_tracer.h"
#include "interface/program/program.h"
#include "interface/utils/id_gen.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
void OperatorChecker::PreCheck()
{
    auto func = Program::GetInstance().GetCurrentFunction();
    preOpCount = func->Operations().size();
    preMagic = IdGen<IdType::LOGICAL_TENSOR>::Inst().CurId();
    preOp = func->opSeed_;
    preRawMagic = IdGen<IdType::RAW_TENSOR>::Inst().CurId();
}

void OperatorChecker::PostCheck()
{
    auto func = Program::GetInstance().GetCurrentFunction();
    auto operations = func->Operations();
    int postOpCount = func->Operations().size();
    int postMagic = IdGen<IdType::LOGICAL_TENSOR>::Inst().CurId();
    int postOp = func->opSeed_;
    int postRawMagic = IdGen<IdType::RAW_TENSOR>::Inst().CurId();

    for (int i = preOpCount; i < postOpCount; i++) {
        auto& lop = operations[i];
        ASSERT(preOp <= lop.GetOpMagic() && lop.GetOpMagic() < postOp);
        for (auto& loperand : lop.GetOOperands()) {
            ASSERT(preMagic <= loperand->GetMagic() && loperand->GetMagic() < postMagic);
            ASSERT(preRawMagic <= loperand->tensor->GetRawMagic() && loperand->tensor->GetRawMagic() < postRawMagic);
            for (int j = i + 1; j < postOpCount; j++) {
                auto& rop = operations[j];
                for (auto& roperand : rop.GetOOperands()) {
                    if (loperand->tensor->GetRawMagic() == roperand->tensor->GetRawMagic()) {
                        ASSERT(!Overlap(loperand, roperand));
                    }
                }
            }
        }
    }
}

bool OperatorTracer::IsCheckerEnabled() const
{
    static int enableChecker = -1;
    if (enableChecker == -1) {
        enableChecker = config::GetPlatformConfig(KEY_ENABLE_CHECKER, 0);
    }
    return enableChecker;
}
} // namespace npu::tile_fwk
