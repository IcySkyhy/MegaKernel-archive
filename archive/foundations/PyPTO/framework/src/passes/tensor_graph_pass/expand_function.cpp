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
 * \file expand_function.cpp
 * \brief
 */

#include "passes/tensor_graph_pass/expand_function.h"
#include <map>
#include "interface/function/function.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/tile_shape_resolver.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_check/expand_function_checker.h"
#include "passes/statistics/tensor_and_tile_graph_statistic.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/tile_shape.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#define MODULE_NAME "ExpandFunction"

using namespace npu::tile_fwk;

namespace {

bool SameLogicalTensor(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs);

bool IsTransparentForTileTraverse(Opcode opcode)
{
    return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_RESHAPE;
}

bool IsViewAssembleHopTransparent(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE; }

void AppendTraverseNeighbors(bool traverseConsumers, Operation* op, std::vector<Operation*>& neighbors)
{
    const auto nextOps = traverseConsumers ? op->ConsumerOps() : op->ProducerOps();
    for (auto* nextOp : nextOps) {
        neighbors.push_back(nextOp);
    }
}

bool OperandShapesHaveDynamicDim(const LogicalTensors& operands)
{
    for (const auto& tensor : operands) {
        if (tensor == nullptr) {
            continue;
        }
        for (const auto dim : tensor->GetShape()) {
            if (dim == -1) {
                return true;
            }
        }
    }
    return false;
}

// VIEW/ASSEMBLE 不参与 tile 展开的条件：
// 1) 参与展开循环的 shape 含动态维 -1（VIEW 看输出，ASSEMBLE 看输入；TiledView/TiledAssemble
//    对 -1 的 for 循环次数为 0，会断图）；
// 2) 在 consumer/producer 方向 BFS 时，若某一层（跳过 VIEW/ASSEMBLE 穿透层后）
//    首次出现的非穿透节点全部为 RESHAPE。
bool ShouldSkipViewAssembleExpand(const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_VIEW && op.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }

    // VIEW 按输出 shape 切分，ASSEMBLE 按输入 shape 切分；对应 shape 含 -1 则跳过展开。
    if (op.GetOpcode() == Opcode::OP_VIEW) {
        if (OperandShapesHaveDynamicDim(op.GetOOperands())) {
            return true;
        }
    } else if (OperandShapesHaveDynamicDim(op.GetIOperands())) {
        return true;
    }

    const bool traverseConsumers = (op.GetOpcode() == Opcode::OP_VIEW);
    std::vector<Operation*> currentLevel;
    if (traverseConsumers) {
        for (auto* consumer : op.ConsumerOps()) {
            currentLevel.push_back(consumer);
        }
    } else {
        for (auto* producer : op.ProducerOps()) {
            currentLevel.push_back(producer);
        }
    }

    std::unordered_set<Operation*> visited;
    while (!currentLevel.empty()) {
        bool levelHasReshape = false;
        bool levelHasNonReshapeCompute = false;
        std::vector<Operation*> nextLevel;
        for (auto* curOp : currentLevel) {
            if (curOp == nullptr || !visited.insert(curOp).second) {
                continue;
            }
            const auto opcode = curOp->GetOpcode();
            if (IsViewAssembleHopTransparent(opcode)) {
                AppendTraverseNeighbors(traverseConsumers, curOp, nextLevel);
            } else if (opcode == Opcode::OP_RESHAPE) {
                levelHasReshape = true;
            } else {
                levelHasNonReshapeCompute = true;
            }
        }
        if (levelHasNonReshapeCompute) {
            return false;
        }
        if (levelHasReshape) {
            return true;
        }
        if (nextLevel.empty()) {
            break;
        }
        currentLevel = std::move(nextLevel);
    }
    return false;
}

std::unordered_set<Operation*> CollectViewAssembleSkipExpandOps(const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr) {
            continue;
        }
        if (opPtr->GetOpcode() != Opcode::OP_VIEW && opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (ShouldSkipViewAssembleExpand(*opPtr)) {
            skipExpandOps.insert(opPtr.get());
        }
    }
    return skipExpandOps;
}

// ASSEMBLE_SSA 的 in-place dst 输入（INPLACE_IDX 指定的 operand）是一个 VIEW，
// 它表示"当前 dst 在指定 offset 处的视图"。ASSEMBLE_SSA 自身已经包含了
// "从 dst 读取 → 与 src 合并 → 写回 result"的完整搬运动作语义。
// 如果将该 VIEW 展开为 SLICE + CONTRACT(NeedCopy=true)，会产生一个中间 buffer，
// 下游 Pass 无法将其与 ASSEMBLE_SSA 的 in-place 语义正确合并，
// 导致残留 CONTRACT → 额外 COPY_OUT → OoOSchedule 产生 zero predicate → 部分数据为 0。
// 因此这类 VIEW 不应展开，保持视图语义即可。
std::unordered_set<Operation*> CollectAssembleSsaDstViewSkipExpandOps(const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_ASSEMBLE_SSA) {
            continue;
        }
        if (!opPtr->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        int inplaceIdx = opPtr->GetIntAttribute(OpAttributeKey::inplaceIdx);
        if (inplaceIdx < 0 || static_cast<size_t>(inplaceIdx) >= opPtr->GetInputOperandSize()) {
            continue;
        }
        auto dstTensor = opPtr->GetInputOperand(static_cast<size_t>(inplaceIdx));
        if (dstTensor == nullptr) {
            continue;
        }
        for (auto* producer : opPtr->ProducerOps()) {
            if (producer == nullptr || producer->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            const auto& oOperands = producer->GetOOperands();
            if (!oOperands.empty() && SameLogicalTensor(oOperands[0], dstTensor)) {
                skipExpandOps.insert(producer);
            }
        }
    }
    return skipExpandOps;
}

bool IsMatmulOpcode(Opcode opcode)
{
    static const std::unordered_set<Opcode> kMatmulOps = {Opcode::OP_A_MUL_B,  Opcode::OP_A_MULACC_B,
                                                          Opcode::OP_A_MUL_BT, Opcode::OP_A_MULACC_BT,
                                                          Opcode::OP_AT_MUL_B, Opcode::OP_AT_MUL_BT};
    return kMatmulOps.count(opcode) > 0;
}

bool SameLogicalTensor(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    return lhs.get() == rhs.get() || lhs->magic == rhs->magic;
}

std::optional<int> GetInputOperandIndex(const Operation& op, const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return std::nullopt;
    }
    const auto& iOperands = op.GetIOperands();
    for (size_t i = 0; i < iOperands.size(); ++i) {
        if (SameLogicalTensor(iOperands[i], tensor)) {
            return static_cast<int>(i);
        }
    }
    return std::nullopt;
}

std::vector<int64_t> AlignVecTileToViewRank(const std::vector<int64_t>& matmulVecTile,
                                            const std::vector<int64_t>& viewShape)
{
    if (viewShape.empty()) {
        return matmulVecTile;
    }
    if (viewShape.size() <= matmulVecTile.size()) {
        if (viewShape.size() == matmulVecTile.size()) {
            return matmulVecTile;
        }
        return std::vector<int64_t>(matmulVecTile.end() - viewShape.size(), matmulVecTile.end());
    }
    std::vector<int64_t> aligned;
    const size_t leadingDims = viewShape.size() - matmulVecTile.size();
    aligned.reserve(viewShape.size());
    for (size_t i = 0; i < leadingDims; ++i) {
        aligned.push_back(viewShape[i]);
    }
    aligned.insert(aligned.end(), matmulVecTile.begin(), matmulVecTile.end());
    return aligned;
}

TileShape GetEffectiveTileShapeForTraverse(const Operation& op, const LogicalTensorPtr& incomingTensor,
                                           const std::vector<int64_t>& viewShape)
{
    TileShape resolved;
    if (incomingTensor != nullptr) {
        if (auto inputIndex = GetInputOperandIndex(op, incomingTensor)) {
            resolved = TileShapeResolver::Instance().GetInputTileShape(op, *inputIndex);
        } else {
            resolved = TileShapeResolver::Instance().GetOutputTileShape(op, 0);
        }
    } else {
        resolved = TileShapeResolver::Instance().GetOutputTileShape(op, 0);
    }

    if (resolved.GetVecTile().valid()) {
        if (!viewShape.empty()) {
            resolved.SetVecTile(AlignVecTileToViewRank(resolved.GetVecTile().tile, viewShape));
        }
        if (IsMatmulOpcode(op.GetOpcode()) && op.GetTileShape().GetCubeTile().valid()) {
            resolved.GetCubeTile() = op.GetTileShape().GetCubeTile();
        }
        return resolved;
    }
    return TileShape::Current();
}

TileShape GetEffectiveTileShapeForTraverse(const Operation& op)
{
    return GetEffectiveTileShapeForTraverse(op, nullptr, {});
}

int64_t VecTileDimProduct(const VecTile& vecTile)
{
    int64_t product = 1;
    for (int64_t dim : vecTile.tile) {
        product *= dim;
    }
    return product;
}

std::optional<TileShape> SelectMaxTileShape(const std::vector<TileShape>& candidates)
{
    if (candidates.empty()) {
        return std::nullopt;
    }
    const TileShape* best = &candidates.front();
    int64_t bestProduct = VecTileDimProduct(best->GetVecTile());
    for (size_t i = 1; i < candidates.size(); ++i) {
        int64_t product = VecTileDimProduct(candidates[i].GetVecTile());
        if (product > bestProduct) {
            best = &candidates[i];
            bestProduct = product;
        }
    }
    return *best;
}

struct TileTraverseState {
    Operation* op{nullptr};
    LogicalTensorPtr tensor;
};

bool AdvanceTransparentConsumerHop(const Operation& curOp, const LogicalTensorPtr& tensor,
                                   std::vector<TileTraverseState>& nextLevel)
{
    bool tensorIsInput = false;
    for (const auto& input : curOp.GetIOperands()) {
        if (SameLogicalTensor(input, tensor)) {
            tensorIsInput = true;
            break;
        }
    }
    if (!tensorIsInput || curOp.GetOOperands().empty()) {
        return false;
    }
    const auto outTensor = curOp.GetOOperands()[0];
    for (auto* consumer : curOp.ConsumerOps()) {
        nextLevel.push_back({consumer, outTensor});
    }
    return true;
}

std::optional<TileShape> FindTileShapeFromConsumers(const Operation& op)
{
    if (op.GetOOperands().empty()) {
        return std::nullopt;
    }
    const auto viewOutput = op.GetOOperands()[0];
    const auto& viewShape = viewOutput->GetShape();

    std::vector<TileTraverseState> currentLevel;
    for (auto* consumer : op.ConsumerOps()) {
        currentLevel.push_back({consumer, viewOutput});
    }
    std::unordered_set<Operation*> visited;

    while (!currentLevel.empty()) {
        std::vector<TileShape> computeCandidates;
        std::vector<TileTraverseState> nextLevel;
        for (const auto& state : currentLevel) {
            Operation* curOp = state.op;
            if (curOp == nullptr || visited.count(curOp) > 0) {
                continue;
            }
            if (IsTransparentForTileTraverse(curOp->GetOpcode())) {
                if (AdvanceTransparentConsumerHop(*curOp, state.tensor, nextLevel)) {
                    visited.insert(curOp);
                }
                continue;
            }
            visited.insert(curOp);
            computeCandidates.push_back(GetEffectiveTileShapeForTraverse(*curOp, state.tensor, viewShape));
        }
        if (!computeCandidates.empty()) {
            return SelectMaxTileShape(computeCandidates);
        }
        currentLevel = std::move(nextLevel);
    }
    return std::nullopt;
}

std::optional<TileShape> FindTileShapeFromProducers(const Operation& op)
{
    std::vector<Operation*> currentLevel;
    for (auto* producer : op.ProducerOps()) {
        currentLevel.push_back(producer);
    }
    std::unordered_set<Operation*> visited;

    while (!currentLevel.empty()) {
        std::vector<TileShape> computeCandidates;
        std::vector<Operation*> nextLevel;
        for (auto* curOp : currentLevel) {
            if (curOp == nullptr || !visited.insert(curOp).second) {
                continue;
            }
            if (IsTransparentForTileTraverse(curOp->GetOpcode())) {
                for (auto* nextOp : curOp->ProducerOps()) {
                    nextLevel.push_back(nextOp);
                }
                continue;
            }
            computeCandidates.push_back(GetEffectiveTileShapeForTraverse(*curOp));
        }
        if (!computeCandidates.empty()) {
            return SelectMaxTileShape(computeCandidates);
        }
        currentLevel = std::move(nextLevel);
    }
    return std::nullopt;
}

} // namespace

namespace npu::tile_fwk {

// 不需要展开的操作码集合
// 这些操作在展开过程中保持原样，不进行 tile-level 展开
// OP_SLICE / OP_CONTRACT 仅在 ExpandFunction 展开过程中由 AddOperation 插入，本身已是 tile 级语义
const std::unordered_set<Opcode> ExpandFunction::kNotNeedExpandOps = {Opcode::OP_SLICE, Opcode::OP_CONTRACT,
                                                                      Opcode::OP_NOP, Opcode::OP_ATOMIC_RMW};

Status ExpandFunction::ClearIOOperand(const std::vector<OperationPtr>& tensorOperations) const
{
    for (auto& op : tensorOperations) {
        // clear consumers and producers
        for (auto& iOperand : op->GetIOperands()) {
            if (iOperand == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] input is null.%s", op->GetOpcodeStr().c_str(),
                                  op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            iOperand->GetConsumers().clear();
            iOperand->GetProducers().clear();
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (oOperand == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] output is null.%s", op->GetOpcodeStr().c_str(),
                                  op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            oOperand->GetConsumers().clear();
            oOperand->GetProducers().clear();
        }
    }
    return SUCCESS;
}

void ExpandFunction::RefreshViewAssembleTileShapes(const std::vector<OperationPtr>& tensorOperations,
                                                   const std::unordered_set<Operation*>& skipExpandOps) const
{
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || skipExpandOps.count(opPtr.get()) > 0) {
            continue;
        }
        Operation& op = *opPtr;
        std::optional<TileShape> refreshedTileShape;
        const std::vector<int64_t>* ownShape = nullptr;
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (op.GetOOperands().empty()) {
                continue;
            }
            // VIEW 按输出 shape 切分；找不到计算 op 时用输出 shape 作为 tile。
            ownShape = &op.GetOOperands()[0]->GetShape();
            refreshedTileShape = FindTileShapeFromConsumers(op);
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (op.GetIOperands().empty()) {
                continue;
            }
            // ASSEMBLE 按输入 shape 切分；找不到计算 op 时用输入 shape 作为 tile。
            ownShape = &op.GetIOperands()[0]->GetShape();
            refreshedTileShape = FindTileShapeFromProducers(op);
        } else {
            continue;
        }

        TileShape tileShape;
        if (refreshedTileShape.has_value() && refreshedTileShape->GetVecTile().valid()) {
            tileShape = *refreshedTileShape;
        } else {
            // 未找到可继承的计算 op tile，或结果无效：用自身 shape，避免回退 Current() 被污染。
            tileShape.GetVecTile().tile = *ownShape;
        }
        op.GetTileShapeForSetting() = tileShape;
    }
}

void ExpandFunction::ProcessForNotExpandOp(Function& function, Operation& op) const
{
    auto& newOp = PassOperationUtils::AddOperation(function, op.GetOpcode(), op.GetIOperands(), op.GetOOperands(),
                                                   nullptr, ir::Span::Unknown(), false);
    if (config::EnableSlice() && op.GetOpcode() == Opcode::OP_VIEW && op.HasAttribute(OpAttributeKey::isGlobalInput)) {
        newOp.SetAttribute(OpAttributeKey::isGlobalInput, op.GetBoolAttribute(OpAttributeKey::isGlobalInput));
    }
    newOp.SetOpAttribute(op.GetOpAttribute());
    newOp.SetScopeInfo(op.GetScopeInfo());
    newOp.SetOooScopeId(op.GetOooScopeId());
    newOp.CopyAttrFrom(op, OP_EMUOP_PREFIX);
    if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
        newOp.SetAttribute(OpAttributeKey::inplaceIdx, op.GetIntAttribute(OpAttributeKey::inplaceIdx));
    }
    if (op.HasAttribute(OpAttributeKey::rmwMode)) {
        newOp.SetAttribute(OpAttributeKey::rmwMode, op.GetIntAttribute(OpAttributeKey::rmwMode));
    }
    newOp.result_token_ = op.result_token_;
    newOp.tokens_ = op.tokens_;
}

Status ExpandFunction::DefaultEnabledPreCheck(Function& function)
{
    ExpandFunctionChecker checker;
    return checker.DoDefaultEnabledPreCheck(function);
}

Status ExpandFunction::PostCheck(Function& function)
{
    ExpandFunctionChecker checker;
    return checker.DoPostCheck(function);
}

Status ExpandFunction::VerifyScopeInfo(Function& function, std::ostringstream& oss) const
{
    std::unordered_map<int, Operation::ScopeInfo> scopeInfoMap;
    std::unordered_map<int, std::unordered_set<CoreType>> scopeCoreTypes;
    for (auto& op : function.Operations(false)) {
        const auto& info = op.GetScopeInfo();
        if (info.scopeId == -1 && (info.allowParallelMerge || info.allowCrossScopeMerge)) {
            oss << "Op " << op.GetOpcodeStr() << "[" << op.GetOpMagic()
                << "]: allowParallelMerge and allowCrossScopeMerge must be false when scopeId is -1.";
            return FAILED;
        }
        if (info.scopeId != -1) {
            auto it = scopeInfoMap.find(info.scopeId);
            if (it != scopeInfoMap.end()) {
                const auto& existing = it->second;
                if (existing.allowParallelMerge != info.allowParallelMerge ||
                    existing.allowCrossScopeMerge != info.allowCrossScopeMerge) {
                    oss << "Op " << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "]: scopeId=" << info.scopeId
                        << " has conflicting allowParallelMerge or allowCrossScopeMerge settings.";
                    return FAILED;
                }
            } else {
                scopeInfoMap[info.scopeId] = info;
            }
            scopeCoreTypes[info.scopeId].insert(op.GetCoreType());
        }
    }
    for (auto& [scopeId, coreTypes] : scopeCoreTypes) {
        if (coreTypes.count(CoreType::AIC) > 0 && coreTypes.count(CoreType::AIV) > 0) {
            if (!GraphUtils::IsCVMixPlatform()) {
                oss << "Cannot mix cube and vector op on a CV separate platform in function: " << function.GetRawName()
                    << ", please check your setting: sg_set_scope=" << scopeId;
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ExpandFunction::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "Start ExpandFunction function [%s].", function.GetRawName().c_str());
    std::ostringstream oss;
    if (VerifyScopeInfo(function, oss) != SUCCESS) {
        APASS_LOG_ERROR_C(OperationErr::OP_SCOPE_ERROR, Elements::Function,
                          "Function[%s] ScopeInfo verification failed: %s", function.GetRawName().c_str(),
                          oss.str().c_str());
        return FAILED;
    }
    bool verifyResult = true;
    for (auto& op : function.Operations(false)) {
        auto verifyOperationEntry = OpcodeManager::Inst().GetVerifyOperationEntry(op.GetOpcode());
        if (verifyOperationEntry) {
            verifyResult = verifyResult && verifyOperationEntry(function, op, oss);
        }
    }
    if (!verifyResult) {
        APASS_LOG_ERROR_F(Elements::Function, "FUnction[%s] ExpandFunction failed: %s", function.GetRawName().c_str(),
                          oss.str().c_str());
        return FAILED;
    }
    if (Expandfunction(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Function[%s] ExpandFunction failed.", function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "Function[%s] operation size is: %zu after expansion.",
                     function.GetMagicName().c_str(), function.Operations(false).size());
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    APASS_LOG_INFO_F(Elements::Function, "End ExpandFunction function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status ExpandFunction::Expandfunction(Function& function) const
{
    if (!function.IsGraphType(GraphType::TENSOR_GRAPH)) {
        APASS_LOG_INFO_F(Elements::Function, "Function %s is not static tensor graph, skip expanding.",
                         function.GetRawName().c_str());
        return SUCCESS;
    }
    function.expandFunctionAccelerate = true;
    function.SetGraphType(GraphType::TILE_GRAPH);

    std::vector<OperationPtr> tensorOperations;
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        tensorOperations.emplace_back(operationViewer.operations_[i]);
    }

    function.ResetOperations();
    auto skipExpandOps = CollectViewAssembleSkipExpandOps(tensorOperations);
    auto ssaDstViewSkipOps = CollectAssembleSsaDstViewSkipExpandOps(tensorOperations);
    skipExpandOps.insert(ssaDstViewSkipOps.begin(), ssaDstViewSkipOps.end());
    const bool expandViewAssemble = config::EnableSlice();
    if (!expandViewAssemble) {
        for (const auto& opPtr : tensorOperations) {
            if (opPtr == nullptr) {
                continue;
            }
            if (opPtr->GetOpcode() == Opcode::OP_VIEW || opPtr->GetOpcode() == Opcode::OP_ASSEMBLE) {
                skipExpandOps.insert(opPtr.get());
            }
        }
    }
    if (expandViewAssemble) {
        RefreshViewAssembleTileShapes(tensorOperations, skipExpandOps);
    }
    if (ClearIOOperand(tensorOperations) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ClearIOOperand failed.");
        return FAILED;
    }

    for (auto& op : tensorOperations) {
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Encountered null operation in function.");
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_PRINT) {
            continue;
        }
        if (config::EnableSlice() && op->GetOpcode() == Opcode::OP_VIEW &&
            op->HasAttribute(OpAttributeKey::isGlobalInput)) {
            ProcessForNotExpandOp(function, *op);
            continue;
        }
        if (kNotNeedExpandOps.count(op->GetOpcode())) {
            ProcessForNotExpandOp(function, *op);
            continue;
        }
        if (skipExpandOps.count(op.get()) > 0) {
            ProcessForNotExpandOp(function, *op);
            continue;
        }

        ir::Span::SetCurrent(op->GetSpan());
        config::SetSemanticLabel(op->GetSemanticLabel());
        size_t opListPreSize = function.Operations(false).size();
        if (ExpandOperation(function, *op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExpandOperation failed.");
            return FAILED;
        }
        auto opListPost = function.Operations(false);
        if (op->GetOpcode() == Opcode::OP_ADDS) {
            for (size_t i = opListPreSize; i < opListPost.size(); i++) {
                auto& newOp = opListPost[i];
                newOp.CopyAttrFrom(*op, OP_EMUOP_PREFIX);
            }
        }
        ir::Span::ClearCurrent();
    }
    function.BuildTensorMap();
    function.expandFunctionAccelerate = false;
    return SUCCESS;
}

Status ExpandFunction::ExpandOperation(Function& function, Operation& op) const
{
    const auto& info = op.GetScopeInfo();
    std::vector<int64_t> scopeVec = {static_cast<int64_t>(info.scopeId), static_cast<int64_t>(info.allowParallelMerge),
                                     static_cast<int64_t>(info.allowCrossScopeMerge)};
    config::SetPassOption(SG_SET_SCOPE, scopeVec);
    config::SetPassOption(SG_SET_OOO_SCOPE, std::vector<int64_t>{static_cast<int64_t>(op.GetOooScopeId())});
    struct ScopeConfigGuard {
        ~ScopeConfigGuard()
        {
            config::SetPassOption(SG_SET_SCOPE, std::vector<int64_t>{-1, 0, 0});
            config::SetPassOption(SG_SET_OOO_SCOPE, std::vector<int64_t>{-1});
        }
    } scopeConfigGuard;
    ExpandOperationInto(function, op.GetTileShape(), op.GetOpcode(), op.GetIOperands(), op.GetOOperands(), op);
    return SUCCESS;
}

void ExpandFunction::DoHealthCheckBefore(Function& function, const std::string& folderPath)
{
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph START");
    std::string fileName = GetDumpFilePrefix(function, true);
    HealthCheckTensorGraph(function, folderPath, fileName);
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph END");
}

} // namespace npu::tile_fwk
