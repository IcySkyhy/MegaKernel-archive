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

#ifndef PASS_COPY_OUT_RESOLVE_H
#define PASS_COPY_OUT_RESOLVE_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "CopyOutResolve"

namespace npu::tile_fwk {
class CopyOutResolve : public Pass {
public:
    CopyOutResolve() : Pass("CopyOutResolve") {}
    ~CopyOutResolve() override = default;

private:
    Status RunOnFunction(Function& function) override
    {
        APASS_LOG_INFO_F(Elements::Function, "===> Start CopyOutResolve.");
        CopyOutResolveCall(function);
        APASS_LOG_INFO_F(Elements::Function, "===> End CopyOutResolve.");
        return SUCCESS;
    }
    void CopyOutResolveCall(Function& function) const;

    void InsertCopyOutResolveForLeaf(int copyOutResolveCoalescing, Function* leaf) const;
    std::vector<Operation*> LookupOutcastLastCopyOut(Function* leafFunc) const;
    void CheckOutcastProducer(Function* leaf) const;
};

} // namespace npu::tile_fwk

#endif // PASS_COPY_OUT_RESOLVE_H
