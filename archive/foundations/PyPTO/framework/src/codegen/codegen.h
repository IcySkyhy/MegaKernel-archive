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

#ifndef CODEGEN_H
#define CODEGEN_H

#include <unordered_set>
#include <utility>

#include "interface/operation/operation.h"
#include "codegen_common.h"
namespace npu::tile_fwk {
class CodeGen {
public:
    explicit CodeGen(const CodeGenCtx& ctx) : ctx_(ctx) {};

    void GenCode(Function& topFunc);

private:
    CodeGenCtx ctx_;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_H
