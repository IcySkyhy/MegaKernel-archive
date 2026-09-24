/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <memory>
#include <vector>

#include "ir/stmt.h"

using namespace pypto;

namespace npu::tile_fwk {

class Function;

/**
 * Rebuild a TensorOpStmt after IR substitution/mutation.
 * If \p src is an Operation with a valid owning function, clone via CloneTensorOpStmt
 * (preserving pass metadata). When \p targetFunc is non-null, register the clone on that
 * function (used by RootFunctionBuilder::ProcessTensorOp). Otherwise rebuild a plain
 * ir::TensorOpStmt (IR-only path).
 */
ir::StmtPtr RebuildTensorOpStmt(const ir::TensorOpStmtPtr& src, std::vector<ir::VarPtr> results,
                                std::vector<ir::VarPtr> resultTokens, std::vector<ir::ExprPtr> args,
                                std::vector<ir::VarPtr> tokens, ir::Span span, Function* targetFunc = nullptr);

} // namespace npu::tile_fwk
