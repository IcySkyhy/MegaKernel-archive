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
 * \file remove_alloc.h
 * \brief
 */

#ifndef PASS_REMOVE_ALLOC_H
#define PASS_REMOVE_ALLOC_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "RemoveAlloc"

namespace npu::tile_fwk {
class RemoveAlloc : public Pass {
public:
    RemoveAlloc() : Pass("RemoveAlloc") {}
    ~RemoveAlloc() override = default;

private:
    Status RunOnFunction(Function& function) override
    {
        APASS_LOG_INFO_F(Elements::Operation, "===> Start RemoveAlloc.");
        RemoveAllocCall(function);
        APASS_LOG_INFO_F(Elements::Operation, "===> End RemoveAlloc.");
        return SUCCESS;
    }
    void RemoveAllocCall(Function& function) const;
};

} // namespace npu::tile_fwk

#endif // PASS_REMOVE_ALLOC_H
