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
 * \file subgraph_to_function_check.cpp
 * \brief
 */

#include "passes/pass_check/subgraph_to_function_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "tilefwk/error_code.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "SubgraphToFunction"

namespace npu {
namespace tile_fwk {
Status SubGraphToFuncChecker::NOPCheck(const Operation& op) const
{
    if (!op.IsNOP()) {
        APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation, "op[%d] is not an NOP. %s",
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetIOperands().size() > 0) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "NOP[%d] has IOperands size %zu. %s", op.GetOpMagic(), op.GetIOperands().size(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetInCtrlOperations().size() > 0) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "NOP[%d] has InCtrlOperations size %zu. %s", op.GetOpMagic(), op.GetInCtrlOperations().size(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetOOperands().size() > 0) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "NOP[%d] has OOperands size %zu. %s", op.GetOpMagic(), op.GetOOperands().size(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetOutCtrlOperations().size() > 0) {
        APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                          "NOP[%d] has OutCtrlOperations size %zu. %s", op.GetOpMagic(),
                          op.GetOutCtrlOperations().size(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SubGraphToFuncChecker::CheckSubGraphTopo(Function& function) const
{
    auto operations = function.Operations();
    int totalSubGraphNum = function.GetTotalSubGraphCount();
    if (operations.size() > 0 && totalSubGraphNum <= 0) {
        APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                          "input totalSubGraphNum %d is invalid", totalSubGraphNum);
        return FAILED;
    }
    std::vector<bool> hitSubgraph = std::vector<bool>(totalSubGraphNum, false);
    for (size_t i = 0; i < operations.size(); i++) {
        auto& op = operations[i];
        int subGraphId = op.GetSubgraphID();
        if (subGraphId < 0 && NOPCheck(op) != SUCCESS) {
            APASS_LOG_ERROR_C(GraphErr::GRAPH_SUBGRAPH_ID_INVALID, Elements::Graph,
                              "operation %zu has negative subGraphID %d and failed NOP check. %s", i, subGraphId,
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (subGraphId >= totalSubGraphNum) {
            APASS_LOG_ERROR_C(GraphErr::GRAPH_SUBGRAPH_ID_INVALID, Elements::Graph,
                              "operation %zu has subGraphID %d that exceeds totalSubGraphNum %d. %s", i, subGraphId,
                              totalSubGraphNum, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        hitSubgraph[subGraphId] = true;

        for (auto inOperand : op.GetIOperands()) {
            for (auto parentOp : inOperand->GetProducers()) {
                int parentSubGraphId = parentOp->GetSubgraphID();
                if (parentSubGraphId > subGraphId) {
                    APASS_LOG_ERROR_C(
                        GraphErr::GRAPH_TOPOLOGY_STRUCTURE, Elements::Graph,
                        "operation %zu has subGraphId %d and parent subGraphId %d, parent subGraphId should be less "
                        "than or equal to subGraphId. %s",
                        i, subGraphId, parentSubGraphId, GetFormatBacktrace(op).c_str());
                    return FAILED;
                }
            }
        }
    }

    for (int i = 0; i < totalSubGraphNum; i++) {
        if (hitSubgraph[i] == false) {
            APASS_LOG_ERROR_C(GraphErr::GRAPH_SUBGRAPH_EMPTY, Elements::Graph, "Subgraph %d is empty", i);
            return FAILED;
        }
    }

    return SUCCESS;
}

Status SubGraphToFuncChecker::EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const
{
    if (!found) {
        APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_CONNECTION, Elements::Function, "op magic not found");
        return FAILED;
    }
    if (static_cast<size_t>(newIndex) >= graphSize) {
        APASS_LOG_ERROR_C(GraphErr::GRAPH_TOPOLOGY_STRUCTURE, Elements::Graph,
                          "parent index %d is larger than operations_ size %zu", newIndex, graphSize);
        return FAILED;
    }
    return SUCCESS;
}

Status SubGraphToFuncChecker::BuildInGraph(Function& function)
{
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        inGraph_[i].clear();
        // inGraph
        for (auto& inOperand : operationViewer[i].GetIOperands()) {
            for (auto& parentOp : inOperand->GetProducers()) {
                auto [parentSeqNo, found] = operationViewer.FindOpPosition(*parentOp);
                if (EdgeIndexCheck(found, parentSeqNo, inGraph_.size()) != SUCCESS) {
                    APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_CONNECTION, Elements::Function,
                                      "error inserting op magic %d in function %d %s to inGraph. %s",
                                      parentOp->GetOpMagic(), function.GetFuncMagic(), function.GetRawName().c_str(),
                                      GetFormatBacktrace(parentOp).c_str());
                    return FAILED;
                }
                auto it = std::find(inGraph_[i].begin(), inGraph_[i].end(), parentSeqNo);
                if (it == inGraph_[i].end()) {
                    inGraph_[i].push_back(parentSeqNo);
                }
            }
        }

        for (const auto& inControlOp : operationViewer[i].GetInCtrlOperations()) {
            auto [parentSeqNo, found] = operationViewer.FindOpPosition(*inControlOp);
            if (EdgeIndexCheck(found, parentSeqNo, inGraph_.size()) != SUCCESS) {
                APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_CONNECTION, Elements::Function,
                                  "Error inserting op magic %d in function %d %s to inGraph. %s",
                                  inControlOp->GetOpMagic(), function.GetFuncMagic(), function.GetRawName().c_str(),
                                  GetFormatBacktrace(inControlOp).c_str());
                return FAILED;
            }
            auto it = std::find(inGraph_[i].begin(), inGraph_[i].end(), parentSeqNo);
            if (it == inGraph_[i].end()) {
                inGraph_[i].push_back(parentSeqNo);
            }
        }
    }
    return SUCCESS;
}

Status SubGraphToFuncChecker::BuildOutGraph(Function& function)
{
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        outGraph_[i].clear();
        for (auto& outOperand : operationViewer[i].GetOOperands()) {
            for (auto& childOp : outOperand->GetConsumers()) {
                auto [childSeqNo, found] = operationViewer.FindOpPosition(*childOp);
                if (EdgeIndexCheck(found, childSeqNo, inGraph_.size()) != SUCCESS) {
                    APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_CONNECTION, Elements::Function,
                                      "Error inserting op magic %d in function %d %s to outGraph_. %s",
                                      childOp->GetOpMagic(), function.GetFuncMagic(), function.GetRawName().c_str(),
                                      GetFormatBacktrace(childOp).c_str());
                    return FAILED;
                }
                auto it = std::find(outGraph_[i].begin(), outGraph_[i].end(), childSeqNo);
                if (it == outGraph_[i].end()) {
                    outGraph_[i].push_back(childSeqNo);
                }
            }
        }

        for (const auto& outControlOp : operationViewer[i].GetOutCtrlOperations()) {
            auto [childSeqNo, found] = operationViewer.FindOpPosition(*outControlOp);
            if (EdgeIndexCheck(found, childSeqNo, inGraph_.size()) != SUCCESS) {
                APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_CONNECTION, Elements::Function,
                                  "Error inserting op magic %d in function %d %s to outGraph_. %s",
                                  outControlOp->GetOpMagic(), function.GetFuncMagic(), function.GetRawName().c_str(),
                                  GetFormatBacktrace(outControlOp).c_str());
                return FAILED;
            }
            auto it = std::find(outGraph_[i].begin(), outGraph_[i].end(), childSeqNo);
            if (it == outGraph_[i].end()) {
                outGraph_[i].push_back(childSeqNo);
            }
        }
        std::sort(outGraph_[i].begin(), outGraph_[i].end());
    }
    return SUCCESS;
}

template <typename eType>
Status SubGraphToFuncChecker::InAndOutGraphConsistencyCheck(const std::vector<std::vector<eType>>& inEdgeGraph,
                                                            const std::vector<std::vector<eType>>& outEdgeGraph)
{
    if (inEdgeGraph.size() != outEdgeGraph.size()) {
        APASS_LOG_ERROR_F(Elements::Graph, "inEdgeGraph size %zu, outEdgeGraph size %zu", inEdgeGraph.size(),
                          outEdgeGraph.size());
        return FAILED;
    }

    std::vector<size_t> nodeColIdx = std::vector<size_t>(inEdgeGraph.size(), 0);
    for (size_t i = 0; i < inEdgeGraph.size(); i++) {
        for (size_t j = 0; j < inEdgeGraph[i].size(); j++) {
            size_t parentSeqNo = static_cast<size_t>(inEdgeGraph[i][j]);
            if (nodeColIdx[parentSeqNo] >= outEdgeGraph[parentSeqNo].size()) {
                APASS_LOG_ERROR_C(GraphErr::GRAPH_EDGE_CONSISTENCY, Elements::Graph,
                                  "node %zu, %zu th parentSeqNo %zu exceeds outgraph[%zu] size %zu", i, j, parentSeqNo,
                                  parentSeqNo, outEdgeGraph[parentSeqNo].size());
                return FAILED;
            }
            // inEdgeGraph和outEdgeGraph都是按顺序排列的
            if (static_cast<size_t>(outEdgeGraph[parentSeqNo][nodeColIdx[parentSeqNo]++]) != i) {
                APASS_LOG_ERROR_C(GraphErr::GRAPH_EDGE_CONSISTENCY, Elements::Graph,
                                  "node %zu, %zu th parentSeqNo %zu is not found in outgraph[%zu]", i, j, parentSeqNo,
                                  parentSeqNo);
                return FAILED;
            }
        }
    }

    // check outEdgeGraph has been fully traversed
    for (size_t i = 0; i < outEdgeGraph.size(); i++) {
        if (outEdgeGraph[i].size() != nodeColIdx[i]) {
            APASS_LOG_ERROR_C(GraphErr::GRAPH_EDGE_CONSISTENCY, Elements::Graph,
                              "outEdgeGraph[%zu] has size %zu, but only %zu of them have been traversed", i,
                              outEdgeGraph[i].size(), nodeColIdx[i]);
            return FAILED;
        }
    }

    return SUCCESS;
}

Status SubGraphToFuncChecker::CheckInAndOutGraphMatch(Function& function)
{
    auto operationViewer = function.Operations();
    inGraph_.resize(operationViewer.size());
    outGraph_.resize(operationViewer.size());
    for (size_t i = 0; i < operationViewer.size(); i++) {
        std::sort(inGraph_[i].begin(), inGraph_[i].end());
    }
    if (BuildInGraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Build inGraph failed");
        return FAILED;
    }
    if (BuildOutGraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Build outGraph failed");
        return FAILED;
    }
    // 2. Check inGraph_ and outGraph_
    if (InAndOutGraphConsistencyCheck(inGraph_, outGraph_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Consistency check for input inGraph_ and outGraph_ failed");
        return FAILED;
    }
    return SUCCESS;
}

bool SubGraphToFuncChecker::HasOnlyViewProducers(const std::set<Operation*, LogicalTensor::CompareOp>& producers)
{
    if (producers.empty()) {
        return false;
    }
    for (auto& producer : producers) {
        if (producer->GetOpcode() != Opcode::OP_VIEW) {
            return false;
        }
    }
    return true;
}

Status SubGraphToFuncChecker::CheckSubGraphBoundary(Function& function)
{
    auto operations = function.Operations();
    for (size_t i = 0; i < operations.size(); i++) {
        auto& op = operations[i];
        int subGraphId = op.GetSubgraphID();
        for (size_t k = 0; k < op.iOperand.size(); k++) {
            auto iOperand = op.GetInputOperand(k);
            auto producers = iOperand->GetProducers();
            // 特殊情况：如果producer有且只有view操作，在Rule 1中不需要标记为子图边界
            bool hasOnlyViewProducers = HasOnlyViewProducers(producers);
            // Rule 1: Operands from DDR memory must be marked as subgraph boundary
            // (这个规则可以在producer都是view操作时跳过)
            if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                !SubgraphUtils::IsBoundary(iOperand) && !hasOnlyViewProducers) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_SUBGRAPH_BOUNDARY, Elements::Tensor,
                    "Input operand %zu of operation %zu (opdump: %s) is from DDR but not marked as subgraph boundary! "
                    "%s",
                    k, i, op.Dump().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            // Rule 2: Operands with consumer in a different subgraph must be marked as subgraph boundary
            if (subGraphId != CommonUtils::GetTensorSubgraphID(iOperand) && !SubgraphUtils::IsBoundary(iOperand)) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_SUBGRAPH_BOUNDARY, Elements::Tensor,
                    "Input operand %zu of operation %zu (opdump: %s) has a consumer in a different subgraph but not "
                    "marked as subgraph boundary! %s",
                    k, i, op.Dump().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            // Rule 3: Input operands of special ops (e.g., OP_UB_COPY_IN) must be marked as subgraph boundary
            if (IsCopyIn(op.GetOpcode()) && !SubgraphUtils::IsBoundary(iOperand)) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_SUBGRAPH_BOUNDARY, Elements::Tensor,
                    "Input operand %zu of IsCopyIn operation %zu (opdump: %s) is not marked as subgraph boundary! %s",
                    k, i, op.Dump().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
        for (size_t k = 0; k < op.oOperand.size(); k++) {
            auto oOperand = op.GetOutputOperand(k);
            auto producers = oOperand->GetProducers();
            // 特殊情况：如果producer有且只有view操作，在Rule 1中不需要标记为子图边界
            bool hasOnlyViewProducers = HasOnlyViewProducers(producers);
            // Rule 1: Operands from DDR memory must be marked as subgraph boundary
            if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                !SubgraphUtils::IsBoundary(oOperand) && !hasOnlyViewProducers) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_SUBGRAPH_BOUNDARY, Elements::Tensor,
                    "Output operand %zu of operation %zu (opdump: %s) is from DDR but not marked as subgraph boundary! "
                    "%s",
                    k, i, op.Dump().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            // Rule 2: Operands with producer in a different subgraph must be marked as subgraph boundary
            if (subGraphId != CommonUtils::GetTensorSubgraphID(oOperand) && !SubgraphUtils::IsBoundary(oOperand)) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_SUBGRAPH_BOUNDARY, Elements::Tensor,
                    "Output operand %zu of operation %zu (opdump: %s) has a producer in a different subgraph but not "
                    "marked as subgraph boundary! %s",
                    k, i, op.Dump().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            // Rule 3: Output operands of special ops (e.g., OP_UB_COPY_OUT, OP_TRANSPOSE_DATA_MOVE, OP_INDEX_OUTCAST)
            // must be marked as subgraph boundary
            if (IsCopyOut(op.GetOpcode()) && !SubgraphUtils::IsBoundary(oOperand)) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_SUBGRAPH_BOUNDARY, Elements::Tensor,
                    "Output operand %zu of IsCopyOut operation %zu (opdump: %s) is not marked as subgraph boundary! %s",
                    k, i, op.Dump().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status SubGraphToFuncChecker::DoPreCheck(Function& function)
{
    // Check subgraph topology
    APASS_LOG_INFO_F(Elements::Operation, "Start PreCheck for SubgraphToFunction!");
    if (CheckSubGraphTopo(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckSubGraphTopo failed");
        return FAILED;
    }

    // Check subgraph boundary
    if (CheckSubGraphBoundary(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckSubGraphBoundary failed");
        return FAILED;
    }

    // Check iOperands and oOperands are matched
    if (CheckInAndOutGraphMatch(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "check input ioperands and ooperands relation failed");
        return FAILED;
    }
    return SUCCESS;
}

bool SubGraphToFuncChecker::foundNodeInNeighbor(const int dstNode, const std::vector<int>& searchGraph) const
{
    auto it = std::find(searchGraph.begin(), searchGraph.end(), dstNode);
    if (it != searchGraph.end()) {
        return true;
    }
    return false;
}

Status SubGraphToFuncChecker::VerifyRedundantEdge(const int srcNode, const int dstNode) const
{
    // dstNode需要在srcNode的三跳之内
    if (foundNodeInNeighbor(dstNode, colorOutGraph_[srcNode])) {
        return SUCCESS;
    }
    for (int firstNbr : colorOutGraph_[srcNode]) {
        if (foundNodeInNeighbor(dstNode, colorOutGraph_[firstNbr])) {
            return SUCCESS;
        }
        for (int secondNbr : colorOutGraph_[firstNbr]) {
            if (foundNodeInNeighbor(dstNode, colorOutGraph_[secondNbr])) {
                return SUCCESS;
            }
        }
    }
    APASS_LOG_ERROR_F(Elements::Graph,
                      "source node %d and destination node %d are not related in colorOutGraph_ within three jumps",
                      srcNode, dstNode);
    return FAILED;
}

Status SubGraphToFuncChecker::ColorOutGraphCheck(Function& function) const
{
    std::vector<std::vector<bool>> hitEdgeMark = std::vector<std::vector<bool>>(colorOutGraph_.size());
    for (size_t i = 0; i < colorOutGraph_.size(); i++) {
        hitEdgeMark[i] = std::vector<bool>(colorOutGraph_[i].size(), false);
    }

    auto list = function.Operations();
    for (size_t i = 0; i < list.size(); i++) {
        int iSubGraphId = list[i].GetSubgraphID();
        if (iSubGraphId < 0) {
            continue;
        }
        for (int j : outGraph_[i]) {
            int jSubGraphId = list[j].GetSubgraphID();
            if (iSubGraphId == jSubGraphId || jSubGraphId < 0) {
                continue;
            }

            auto it = std::find(colorOutGraph_[iSubGraphId].begin(), colorOutGraph_[iSubGraphId].end(), jSubGraphId);
            if (it != colorOutGraph_[iSubGraphId].end()) { // found edge
                int index = std::distance(colorOutGraph_[iSubGraphId].begin(), it);
                hitEdgeMark[iSubGraphId][index] = true;
                continue;
            }
            if (VerifyRedundantEdge(iSubGraphId, jSubGraphId) != SUCCESS) { // check whether is redundant edge
                APASS_LOG_ERROR_F(
                    Elements::Graph,
                    "edge between original operator %zu with subgraph ID %d and operator %d with subgraph ID %d is "
                    "missed in colorOutGraph_",
                    i, iSubGraphId, static_cast<int>(j), jSubGraphId);
                return FAILED;
            }
        }
    }

    // check all edges have been hit
    for (size_t i = 0; i < hitEdgeMark.size(); i++) {
        for (size_t j = 0; j < hitEdgeMark[i].size(); j++) {
            if (hitEdgeMark[i][j] == false) {
                APASS_LOG_ERROR_F(Elements::Graph,
                                  "edge between %zu and %d on colorOutGraph_ has no correspondent edge in outGraph_", i,
                                  colorOutGraph_[i][j]);
                return FAILED;
            }
        }
    }

    return SUCCESS;
}

Status SubGraphToFuncChecker::DoPostCheck(Function& function)
{
    // Check colorInGraph_ and colorOutGraph_ consistency
    APASS_LOG_INFO_F(Elements::Operation, "Start PostCheck for SubgraphToFunction!");

    // 只在静态流程中检查静态专用的图信息
    if (function.GetFunctionType() == FunctionType::STATIC) {
        if (InAndOutGraphConsistencyCheck(colorInGraph_, colorOutGraph_) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Graph, "Consistency check for input colorInGraph_ and colorOutGraph_ failed");
            return FAILED;
        }

        // Check colorOutGraph_ matches outGraph_
        if (ColorOutGraphCheck(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Graph, "Consistency check for colorOutGraph_ and input failed");
            return FAILED;
        }

        // Verify readyState matches negative predecessor count
        for (size_t i = 0; i < function.rootFunc_->topoInfo_.topology_.size(); i++) {
            if (CheckReadyStateConsistency(function, i) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Graph, "Ready state inconsistency found for topology entry %zu", i);
                return FAILED;
            }
        }

        for (size_t i = 0; i < function.rootFunc_->Operations().size(); ++i) {
            if (VerifySingleOpTopology(function, i) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Graph, "Failed to verify topology for operation %zu", i);
                return FAILED;
            }
        }
    }

    return SUCCESS;
}

Status SubGraphToFuncChecker::VerifySingleOpTopology(Function& function, size_t opIdx)
{
    const auto& callOps = function.rootFunc_->Operations();
    // 通过 subgraphId 查找对应的 currentOp
    Operation* currentOp = nullptr;
    for (const auto& op : callOps) {
        if (static_cast<size_t>(op.GetSubgraphID()) == opIdx) {
            currentOp = const_cast<Operation*>(&op);
            break;
        }
    }

    if (currentOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot find operation with subgraphId %zu", opIdx);
        return FAILED;
    }

    auto consumers = currentOp->ConsumerOps();
    auto producers = currentOp->ProducerOps();
    APASS_LOG_DEBUG_F(Elements::Operation, "=================Call ===============%zu", opIdx);
    for (auto& prod : producers) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Producer %s %d", prod->GetOpcodeStr().c_str(), prod->opmagic);
    }
    auto& topoInfo = function.rootFunc_->topoInfo_.topology_[opIdx];
    std::unordered_set<Operation*> consumersNoSelf;
    for (auto* cons : consumers) {
        if (cons->opmagic != currentOp->opmagic) {
            consumersNoSelf.insert(cons);
        }
    }
    if (consumersNoSelf.size() < topoInfo.outGraph.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Call %zu %d consumers size are %zu and %zu. %s", opIdx,
                          currentOp->opmagic, consumersNoSelf.size(), topoInfo.outGraph.size(),
                          GetFormatBacktrace(currentOp).c_str());
        return FAILED;
    }
    for (auto succ : topoInfo.outGraph) {
        const int consumerSubgraphId = static_cast<int>(succ);

        // 通过 subgraphId 查找预期的消费者op
        Operation* expectedConsumer = nullptr;
        for (const auto& op : callOps) {
            if (op.GetSubgraphID() == consumerSubgraphId) {
                expectedConsumer = const_cast<Operation*>(&op);
                break;
            }
        }

        if (expectedConsumer == nullptr) {
            APASS_LOG_ERROR_F(Elements::Graph, "Cannot find expected consumer with subgraphId %d", consumerSubgraphId);
            return FAILED;
        }
        if (consumers.count(expectedConsumer) == 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find consumer %d for call %zu. %s", succ, opIdx,
                              GetFormatBacktrace(currentOp).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SubGraphToFuncChecker::CheckReadyStateConsistency(Function& function, size_t opIdx)
{
    auto& topology = function.rootFunc_->topoInfo_.topology_;
    // Calculate actual predecessor count
    int actualPredCount = 0;

    for (size_t j = 0; j < topology.size(); j++) {
        if (opIdx == j) {
            continue;
        }
        auto& otherEntry = topology[j];
        if (std::find(otherEntry.outGraph.begin(), otherEntry.outGraph.end(), opIdx) != otherEntry.outGraph.end()) {
            actualPredCount++;
        }
    }
    // readyState should equal negative predecessor counts
    if (topology[opIdx].readyState != -actualPredCount) {
        APASS_LOG_ERROR_F(Elements::Graph, "Subgraph %zu has inconsistent readyState: actual=%d, expected=%zu", opIdx,
                          topology[opIdx].readyState, static_cast<size_t>(-actualPredCount));
        return FAILED;
    }
    return SUCCESS;
}

void SubGraphToFuncChecker::SetInOutGraph(const std::vector<std::vector<size_t>>& inGraph,
                                          const std::vector<std::vector<size_t>>& outGraph)
{
    inGraph_ = inGraph;
    outGraph_ = outGraph;
}

void SubGraphToFuncChecker::SetColorGraph(const std::vector<std::vector<int>>& colorInGraph,
                                          const std::vector<std::vector<int>>& colorOutGraph)
{
    colorInGraph_ = colorInGraph;
    colorOutGraph_ = colorOutGraph;
}

template Status SubGraphToFuncChecker::InAndOutGraphConsistencyCheck<size_t>(const std::vector<std::vector<size_t>>&,
                                                                             const std::vector<std::vector<size_t>>&);

} // namespace tile_fwk
} // namespace npu
