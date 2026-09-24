/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file last_use_mark.h
 * \brief 标记算子的LastUse属性
 */

#ifndef PASS_LAST_USE_MARK_H
#define PASS_LAST_USE_MARK_H

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include <unordered_map>

namespace npu::tile_fwk {

class LastUseMark : public Pass {
public:
    LastUseMark() : Pass("LastUseMark") {}
    ~LastUseMark() override = default;

private:
    Status RunOnFunction(Function& function) override;
    Status CollectLastUseInfo(Function& function);
    void SetLastUseAttributes();

    std::unordered_map<LogicalTensorPtr, Operation*> lastUseMap_;
};

} // namespace npu::tile_fwk

#endif // PASS_LAST_USE_MARK_H
