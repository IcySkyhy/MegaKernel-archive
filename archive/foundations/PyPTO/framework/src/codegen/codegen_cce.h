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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_CCE_H
#define CODEGEN_CCE_H

#include <unordered_set>
#include <utility>

#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "codegen_common.h"

namespace npu::tile_fwk {

class CodeGenCCE {
public:
    explicit CodeGenCCE(CodeGenCtx cctx) : ctx(std::move(cctx))
    {
        if (ctx.IsCCEPathEmpty()) {
            PrepareDefaultOutputPath();
        }
    }
    virtual ~CodeGenCCE() = default;

    virtual void GenCode(Function& topFunc) = 0;

protected:
    CodeGenCtx ctx;

private:
    void PrepareOutputPath();
    void PrepareDefaultOutputPath();
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CCE_H
