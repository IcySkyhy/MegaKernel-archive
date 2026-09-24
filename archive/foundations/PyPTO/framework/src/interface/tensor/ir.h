/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <optional>
#include <string>
#include <map>
#include <unordered_set>

#include "ir/scalar_expr.h"
#include "ir/stmt.h"

namespace pypto::ir {
std::string DumpScalarExpr(const ScalarExprPtr& op);
std::string DumpTensorVar(const VarPtr& tensor);
std::map<std::string, std::any> CollectTensorOpAttrs(const TensorOpStmtPtr& ptr);

// Returns the underlying allocation id of a tensor Var, used for alias-aware
// liveness analysis (inplace transforms such as RESHAPE alias the same allocation).
// Returns std::nullopt when the Var is not a framework tensor or has no allocation.
std::optional<int> GetVarMemoryId(const Var* var);

// Collects allocation ids of a set of tensor Vars, used for alias-aware liveness
// analysis. Skips Vars that have no allocation id.
std::unordered_set<int> CollectMemoryIds(const std::unordered_set<const Var*>& vars);

// Returns true if two tensor Vars share the same underlying raw tensor.
bool IsSameRawTensor(const VarPtr& v0, const VarPtr& v1);

// Collects all scalar variables referenced by a tensor operation.
void CollectScalarVarRefs(const TensorOpStmtPtr& op, std::unordered_set<const Var*>& var_uses);
} // namespace pypto::ir
