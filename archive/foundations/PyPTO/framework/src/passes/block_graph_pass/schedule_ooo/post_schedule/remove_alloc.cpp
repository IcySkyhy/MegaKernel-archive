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
 * \file remove_alloc.cpp
 * \brief
 */

#include "remove_alloc.h"

namespace npu::tile_fwk {
void RemoveAlloc::RemoveAllocCall(Function& function) const
{
    for (auto& program : function.rootFunc_->programs_) {
        std::vector<std::shared_ptr<Operation>>& opList = program.second->GetProgramOp();
        for (auto& op : opList) {
            if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                op->SetAsDeleted();
            }
        }
        program.second->EraseOperations(false, false);
    }
}
} // namespace npu::tile_fwk
