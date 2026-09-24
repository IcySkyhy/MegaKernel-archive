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

#include <string>
#include <vector>

#include "ir/stmt.h"

namespace pypto::ir {

/// Merge adjacent non-loop statements into the branches of IfStmts.
///
/// Each IfStmt is grown so its branch bodies become maximal runs of ops: statements surrounding an
/// if (before/after it in source order) are hoisted/sunk into both branches, and branches proven
/// unreachable under the accumulated condition path are pruned. This widens if-bodies for the
/// downstream lowering.
///
/// Algorithm (MergeStmtImpl):
///  - Process: split the sequence into barrier-free segments at Yield/Continue/For/While (loops are
///    dynamic and can't be duplicated into branches), segment-merge each, then rebuild recursively.
///  - MergeSegment: right-to-left fold over one segment. A live IfStmt absorbs the stmts to its
///    right into both branches (AppendIntoIfStmt); a SAT-proven-dead branch is spliced out
///    (SpliceSurvivor); a trailing if absorbs preceding non-For stmts (FoldLeadingStmts/Prepend).
///  - RebuildMergedStmts: the recursive half — descend into For/While bodies (re-Process under
///    strengthened conditions) and finalize each IfStmt (MergeIfStmts: prune dead branches, else
///    merge both and clone vars defined in both via ResolveDuplicateVars).
///
/// Var handling: hoisting/sinking one stmt into both branches makes its result a def in both, so
/// ResolveDuplicateVars gives the else copy a distinct identity. Cloned/spliced vars are tracked in
/// cloneMap/subst so later references and the trailing terminator are rewritten to the if's return
/// vars. Names in `extVarNames` denote parameter/external storage and are never cloned. Vars that
/// remain live after the merged if (read after it, loop-carried, or used in enclosing scopes) are
/// treated like external storage: they keep the same identity in both branches, because such
/// in-place buffers are not carried back out through the if's return vars.
SeqStmtsPtr MergeStmtsIntoIfStmt(SeqStmtsPtr seq, const std::vector<std::string>& extVarNames = {});

} // namespace pypto::ir
