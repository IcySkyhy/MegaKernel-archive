/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file expr_generator.h
 * \brief Expression batch generator for splitting large control flow functions
 */

#pragma once

#include <functional>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include "interface/tensor/symbolic_scalar.h"

namespace npu::tile_fwk {
namespace {
// Maximum expressions per batch/file
constexpr size_t EXPRS_PER_BATCH = 1000;
} // namespace
constexpr size_t TABSIZE = 2;

// Expression batch information
struct ExprBatchInfo {
    int devRootKey;
    size_t batchIndex;
    size_t startExprIndex;
    size_t endExprIndex;
    size_t totalExprs;
    std::string fileName;
    std::string functionName;
};

// GetInput* CSE for fixed GetInputShapeDim / GetInputShapeDimSize / GetInputData:
// key -> CSE_sd[i] (stack array in ControlFlowEntry, passed into SetExprBatch).
// Must NOT use namespace globals: AICPU CF blob only loads the .pypto section (no .bss/.got).
//
// Target call shapes (operands after callee must be fixed leaves only):
//   RUNTIME_GetInputShapeDim(ARG_*, <imm>)
//   RUNTIME_GetInputShapeDimSize(ARG_*, <imm>)   // same predicate as ShapeDim
//   RUNTIME_GetInputData(ARG_*, RUNTIME_*, <imm>)  // batch-local CSE only; not launch-hoisted
// Fixed leaf := Immediate | ARG_* symbol | RUNTIME_* symbol.
// Rejected: VALUE_*, sym_*, nested expressions (e.g. VALUE_i+1).
struct GetInputCse {
    std::unordered_map<std::string, std::string> keyToName;
    std::vector<std::pair<std::string, std::string>> ordered; // (CSE_sd[i], key/expr)
};

// Generator for expression batches
class ExprBatchGenerator {
public:
    ExprBatchGenerator(const std::string& outputDir, int devRootKey, size_t totalExprs);

    void HeaderFileBegin(std::ostringstream& out) const;
    void HeaderFileEnd(std::ostringstream& out) const;

    // Collect launch-hoistable fixed GetInputShapeDim across all expression tables.
    static GetInputCse CollectGetInputCse(const std::vector<SymbolicExpressionTable*>& tables);

    // Emit stack array + inits inside ControlFlowEntry. Returns true if array was emitted.
    static bool EmitGetInputCseStackInits(std::ostringstream& controlFlowOss, const GetInputCse& getInputCse,
                                          int indent);

    void GenerateBatchFile(SymbolicExpressionTable* exprTable, std::ostringstream& controlFlowOss,
                           std::ostringstream& exprHeaderOss, const std::string& expName,
                           const OrderedSet<RawSymbolicScalarPtr>& expressions, std::vector<std::string>& exprSrcFiles,
                           int indent, int devRootKey, const GetInputCse* getInputCse = nullptr);

private:
    // 同一段内的一处差异叶子，多处差异共用单循环变量 k：值轨迹 firstImm + step * k。
    struct TrackedDiff {
        std::vector<int> path;
        int64_t firstImm;
        int64_t step;
        int64_t lastImm;
    };

    static bool IsFixedGetInputCseOperandLeaf(const RawSymbolicScalarPtr& raw);
    static bool CallOperandsAreFixed(const RawSymbolicScalarPtr& call);
    static bool IsGetInputCseTargetCall(const RawSymbolicScalarPtr& raw);
    static void CollectShapeDimCalls(const RawSymbolicScalarPtr& raw,
                                     const std::function<void(const RawSymbolicScalarPtr&)>& fn);
    static void WalkGetInputCseTargetCalls(const RawSymbolicScalarPtr& raw,
                                           const std::function<void(const RawSymbolicScalarPtr&)>& fn);
    static std::unordered_map<std::string, std::string> BuildBatchStructuralGetInputCse(
        const OrderedSet<RawSymbolicScalarPtr>& expressions, size_t batchStart, size_t batchEnd,
        const GetInputCse* getInputCse, std::vector<std::pair<std::string, std::string>>& batchLocalOrdered);
    size_t DetectArithmeticRun(const OrderedSet<RawSymbolicScalarPtr>& expressions, size_t runStart, size_t batchEnd,
                               std::vector<TrackedDiff>& tracked) const;
    static bool SeedArithmeticTracked(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs,
                                      std::vector<TrackedDiff>& tracked);
    static bool AdvanceArithmeticTracked(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs,
                                         std::vector<TrackedDiff>& tracked);
    static RawSymbolicScalarPtr BuildLoopAffinePlaceholder(int64_t firstImm, int64_t step,
                                                           const RawSymbolicScalarPtr& loopVarNode);
    void EmitFoldedLoop(std::ostream& out, const RawSymbolicScalarPtr& firstExpr,
                        const std::vector<TrackedDiff>& tracked, size_t startExprIdx, size_t runLength, size_t loopId,
                        const std::unordered_map<std::string, std::string>* structuralGetInputCse) const;
    void EmitBatchBody(std::ostream& out, SymbolicExpressionTable* exprTable,
                       const OrderedSet<RawSymbolicScalarPtr>& expressions, size_t batchStart, size_t batchEnd,
                       const GetInputCse* getInputCse) const;
    void CalculateBatches();
    void GenerateLinkScript() const;

    std::string outputDir_;
    int devRootKey_;
    size_t totalExprs_;
    std::vector<ExprBatchInfo> batches_;
};

} // namespace npu::tile_fwk
