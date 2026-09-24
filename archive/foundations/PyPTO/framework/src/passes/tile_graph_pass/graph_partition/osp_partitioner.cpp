/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define MODULE_NAME "GraphPartition"

#include "osp_partitioner.h"
#include "passes/algorithms/osp/bsp/model/bsp_instance.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_check/iso_partitioner_checker.h"
#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"
#include "passes/algorithms/osp/bsp/model/util/set_schedule.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/grow_local_auto_cores.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/greedy_children.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/greedy_meta_scheduler.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/isomorphic_subgraph_scheduler.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/precomputed_hash_computer.h"
#include "passes/algorithms/osp/coarser/sarkar/sarkar_mul.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/comm_cost_modules/kl_hyper_total_comm_cost.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/kl_improver.h"

namespace npu::tile_fwk {

Status OspPartitioner::BuildSuperNodeGraph()
{
    std::vector<Operation*>& opList = operationInfo_->opList_;
    if (opList.size() != operationInfo_->inGraph_.size() || opList.size() != operationInfo_->outGraph_.size()) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Osp: Operation inGraph and outGraph have not been initialized.\n"
                          "Operation list size (%zu) does not match with Operation "
                          "In- and Out-Graph size (%zu, respectively %zu).",
                          opList.size(), operationInfo_->inGraph_.size(), operationInfo_->outGraph_.size());
        return FAILED;
    }
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    for (size_t i = 0; i < opList.size(); i++) {
        if (ConvertCombine(operationInfo_, opList, i, mergePair))
            continue;
        if (L1CopyInCombine(operationInfo_, opList, i, mergePair))
            continue;
        if (AssembleCombine(operationInfo_, opList, i, mergePair))
            continue;
        if (CopyOutCombine(operationInfo_, opList, i, mergePair, false))
            continue;
        if (CopyInCombine(operationInfo_, opList, i, mergePair))
            continue;
        if (MulAccCombine(operationInfo_, opList, i, mergePair))
            continue;
        if (ExpandCombine(operationInfo_, opList, i, mergePair))
            continue;
    }
    superNodeInfo_ = std::make_shared<NodeGraphInfo>();
    if (superNodeInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Osp: Create SuperNodeInfo failed.");
        return FAILED;
    }
    if (superNodeInfo_->Build(operationInfo_, mergePair, true) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Osp: Build SuperNodeInfo Failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OspPartitioner::PartitionGraph(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "Running OspPartitioner, useCVMixPartition: %d.", useCVMixPartition_);
    if (BuildOpGraph(function.Operations().DuplicatedOpList()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building operation graph.");
        return FAILED;
    }
    if (BuildSuperNodeGraph() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building SuperNode graph.");
        return FAILED;
    }
    if (ospMode_ == OspMode::MERKLEBSP) {
        BuildHashValues();
    }
    if (RunOspPartition(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate a partition.");
        return FAILED;
    }
    return SUCCESS;
}

Status OspPartitioner::RunOspPartition(Function& function)
{
    Status status = FAILED;
    std::vector<VertType> vertexContractionMap;
    CoarseGraphType coarseGraph;

    switch (ospMode_) {
        case OspMode::SARKAR:
        default: {
            GraphType graph;
            if (ConstructDag(graph) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph.");
                return FAILED;
            }
            status = RunSarkar(graph, coarseGraph, vertexContractionMap);
            break;
        }
        case OspMode::MERKLEBSP: {
            osp::BspInstance<GraphType> bspInstance;
            if (ConstructBspInstance(bspInstance) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate a bsp instance.");
                return FAILED;
            }
            status = RunMerkleBsp(bspInstance, vertexContractionMap);
            break;
        }
    }

    if (status == FAILED) {
        return FAILED;
    }

    if (UpdatePartitionResult(function, vertexContractionMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP failed to update the partition result.");
        return FAILED;
    }

    return status;
}

Status OspPartitioner::RunSarkar(const GraphType& graph, CoarseGraphType& coarseGraph,
                                 std::vector<VertType>& vertexContractionMap)
{
    constexpr uint32_t kSarkarSeed = 1729U;    // Sarkar 算法随机种子
    constexpr double kSarkarGeomDecay = 0.875; // 几何衰减系数
    constexpr double kSarkarLeniency = 0.005;  // 容忍度
    // 通信代价梯度: 逐级递增, 用于 Sarkar 算法在不同聚合层级权衡通信代价
    const WorkType kCommCostLevel1 = 1;
    const WorkType kCommCostLevel2 = 2;
    const WorkType kCommCostLevel5 = 5;
    const WorkType kCommCostLevel10 = 10;
    const WorkType kCommCostLevel20 = 20;
    const WorkType kCommCostLevel50 = 50;
    const WorkType kCommCostLevel100 = 100;
    const WorkType kCommCostLevel200 = 200;
    const WorkType kCommCostLevel500 = 500;
    const WorkType kCommCostLevel1000 = 1000;
    osp::sarkar_params::MulParameters<WorkType> params;
    params.seed_ = kSarkarSeed;
    params.geomDecay_ = kSarkarGeomDecay;
    params.leniency_ = kSarkarLeniency;
    params.commCostVec_ = std::vector<WorkType>({kCommCostLevel1, kCommCostLevel2, kCommCostLevel5, kCommCostLevel10,
                                                 kCommCostLevel20, kCommCostLevel50, kCommCostLevel100,
                                                 kCommCostLevel200, kCommCostLevel500, kCommCostLevel1000});
    params.smallWeightThreshold_ = archParameters_.partitionWorkLowerBound_;
    params.maxNumIterationWithoutChanges_ = 3U;
    params.bufferMergeMode_ = osp::sarkar_params::BufferMergeMode::FULL;

    osp::SarkarMul<GraphType, CoarseGraphType> coarser;
    coarser.SetParameters(params);

    bool coarsenStatus = coarser.CoarsenDag(graph, coarseGraph, vertexContractionMap);
    if (not coarsenStatus) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP Sarkar failed to generate a coarse graph.");
        return FAILED;
    }

    return SUCCESS;
}

Status OspPartitioner::RunMerkleBsp(const osp::BspInstance<GraphType>& bspInst,
                                    std::vector<VertType>& vertexContractionMap)
{
    constexpr int kIsoSchedulerWorkThreshold = 200;
    constexpr int kIsoSchedulerCriticalPathThreshold = 500;

    osp::GrowLocalAutoCores<ConstrGraphType> growlocal;
    osp::GreedyChildren<ConstrGraphType> children;

    osp::KlImprover<ConstrGraphType, osp::KlHyperTotalCommCostFunction<ConstrGraphType, double, 1>, 1, double> kl;
    kl.SetSuperstepRemoveStrengthParameter(1.0);
    kl.SetTimeQualityParameter(1.0);

    osp::ComboScheduler<ConstrGraphType> growlocalKl(growlocal, kl);
    osp::ComboScheduler<ConstrGraphType> childrenKl(children, kl);

    osp::GreedyMetaScheduler<ConstrGraphType> scheduler;
    if (!useCVMixPartition_) {
        scheduler.AddScheduler(growlocalKl);
    }

    scheduler.AddScheduler(childrenKl);
    scheduler.AddSerialScheduler();

    osp::MerkleHashComputer<GraphType, osp::PrecomBwdMerkleNodeHashFunc<GraphType>> hashComputer(
        bspInst.GetComputationalDag(), bspInst.GetComputationalDag(), this->superNodeInfo_->nodeHashList_);
    osp::IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> isoScheduler(scheduler, hashComputer);
    isoScheduler.SetMergeDifferentTypes(useCVMixPartition_);
    isoScheduler.SetWorkThreshold(kIsoSchedulerWorkThreshold);
    isoScheduler.SetCriticalPathThreshold(kIsoSchedulerCriticalPathThreshold);
    vertexContractionMap = isoScheduler.ComputePartition(bspInst);
    return SUCCESS;
}

Status OspPartitioner::UpdatePartitionResult(Function& function, std::vector<VertType>& vertexContractionMap)
{
    int32_t numColors = 0;
    for (size_t i = 0; i < vertexContractionMap.size(); ++i) {
        const int32_t superNode = static_cast<int32_t>(i);
        const int32_t nodeColor = vertexContractionMap[i];
        for (const int32_t idx : superNodeInfo_->node2Op_[superNode]) {
            operationInfo_->opList_[idx]->UpdateSubgraphID(nodeColor);
        }
        numColors = std::max(numColors, nodeColor + 1);
    }
    function.SetTotalSubGraphCount(numColors);
    return SUCCESS;
}

void OspPartitioner::SetVertexCommMemWeight(GraphType& graph, int32_t vertex)
{
    std::set<int32_t> operandsInChildSupernodes;
    const auto& children = superNodeInfo_->nodeOutGraphList_[vertex];
    for (const auto& child : children) {
        const auto& childOperators = superNodeInfo_->node2Op_[child];

        for (const auto& op : childOperators) {
            operandsInChildSupernodes.insert(op);
        }
    }

    const auto& operators = superNodeInfo_->node2Op_[vertex];
    WorkType commWeight = 10;
    WorkType memWeight = 10;

    for (const auto& op : operators) {
        const auto operInfo = operationInfo_->opList_[op];
        const std::string opCode = operInfo->GetOpcodeStr();
        const bool isView = opCode.find("View") != std::string::npos;
        if (isView) {
            for (auto& inputLogicalTensor : operationInfo_->opList_[op]->GetIOperands()) {
                const size_t memorySize = inputLogicalTensor->MemorySize();
                memWeight += static_cast<WorkType>(memorySize);
            }
        }

        for (auto& outputLogicalTensor : operationInfo_->opList_[op]->GetOOperands()) {
            for (auto& consumer : outputLogicalTensor->GetConsumers()) {
                if (operationInfo_->magic2Idx_.count(consumer->GetOpMagic()) == 0) {
                    continue;
                }

                int32_t operationIdx = operationInfo_->magic2Idx_[consumer->GetOpMagic()];
                if (operandsInChildSupernodes.find(operationIdx) == operandsInChildSupernodes.end()) {
                    continue;
                }

                const size_t memorySize = outputLogicalTensor->MemorySize();
                commWeight += static_cast<WorkType>(memorySize);
                break;
            }
        }
    }
    graph.SetVertexMemWeight(vertex, memWeight);
    graph.SetVertexCommWeight(vertex, static_cast<WorkType>(commWeight * archParameters_.commCorrectionFactor_));
}

OspPartitioner::VTypeType OspPartitioner::GetOspCoreTypeSplit(OpCoreType coreType)
{
    auto it = ospCoreTypeMapSplit.find(coreType);
    if (it != ospCoreTypeMapSplit.end()) {
        return it->second;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "OpCoreType (%d) not found in ospCoreTypeMapSplit.",
                      static_cast<int>(coreType));
    return 0U;
}

OspPartitioner::VTypeType OspPartitioner::GetOspCoreTypeMix(OpCoreType coreType)
{
    auto it = ospCoreTypeMapMix.find(coreType);
    if (it != ospCoreTypeMapMix.end()) {
        return it->second;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "OpCoreType (%d) not found in ospCoreTypeMapMix.",
                      static_cast<int>(coreType));
    return 0U;
}

Status OspPartitioner::ConstructDagCVSplit(GraphType& graph)
{
    if (ConstructDagCVMix(graph) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Construct CV Mixed Dag has failed.");
        return FAILED;
    }

    for (const auto& superNode : graph.Vertices()) {
        const OpCoreType vertexType = superNodeInfo_->nodeCoreType_[superNode];
        if (vertexType != OpCoreType::AIC && vertexType != OpCoreType::AIV && vertexType != OpCoreType::AICPU) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "SuperNode (%d) has core type (%d) which is neither cube"
                              " nor vector nor ai-scalar.",
                              static_cast<int>(superNode), static_cast<int>(vertexType));
            return FAILED;
        }
        graph.SetVertexType(superNode, GetOspCoreTypeSplit(vertexType));
    }

    return SUCCESS;
}

Status OspPartitioner::ConstructDagCVMix(GraphType& graph)
{
    graph = GraphType(superNodeInfo_->nodeOutGraphList_, superNodeInfo_->nodeInGraphList_);

    for (const auto& superNode : graph.Vertices()) {
        graph.SetVertexWorkWeight(superNode, superNodeInfo_->nodeCycles_[superNode]);
        SetVertexCommMemWeight(graph, superNode);

        OpCoreType vertexType = superNodeInfo_->nodeCoreType_[superNode];
        if (vertexType != OpCoreType::AIC && vertexType != OpCoreType::AIV && vertexType != OpCoreType::AICPU) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "SuperNode (%d) has core type (%d) which is neither cube"
                              " nor vector nor ai-scalar.",
                              static_cast<int>(superNode), static_cast<int>(vertexType));
            return FAILED;
        }
        graph.SetVertexType(superNode, GetOspCoreTypeMix(vertexType));
    }

    return SUCCESS;
}

void OspPartitioner::ConstructBspArchCVSplit(osp::BspArchitecture<GraphType>& bspArch)
{
    const size_t numCubeCores = Platform::Instance().GetSoc().GetAICCoreNum();
    const size_t numVectorCores = Platform::Instance().GetSoc().GetAIVCoreNum();
    const size_t numAiScalarCores = Platform::Instance().GetSoc().GetAICPUNum();

    const size_t numCores = numCubeCores + numVectorCores + numAiScalarCores;
    std::vector<VTypeType> procTypes(numCores);
    std::vector<WorkType> procMemoryBound(numCores);

    for (size_t i = 0; i < numCores; i++) {
        if (i < numCubeCores) { // Cube Cores
            procTypes[i] = GetOspCoreTypeSplit(OpCoreType::AIC);
            procMemoryBound[i] = static_cast<WorkType>(
                Platform::Instance().GetAICCore().GetMemorySize(MemoryType::MEM_L1));
        } else if (i < numCubeCores + numVectorCores) { // Vector Cores
            procTypes[i] = GetOspCoreTypeSplit(OpCoreType::AIV);
            procMemoryBound[i] = static_cast<WorkType>(
                Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB));
        } else { // AI Scalar Cores
            procTypes[i] = GetOspCoreTypeSplit(OpCoreType::AICPU);
            procMemoryBound[i] = std::numeric_limits<WorkType>::max();
        }
    }
    bspArch.SetProcessorsWithTypes(procTypes);
    bspArch.SetMemoryBound(procMemoryBound);
    bspArch.SetCommunicationCosts(archParameters_.commCost_);
    bspArch.SetSynchronisationCosts(archParameters_.synchCost_);
}

Status OspPartitioner::ConstructBspArchCVMix(osp::BspArchitecture<GraphType>& bspArch)
{
    const size_t numCubeCores = Platform::Instance().GetSoc().GetAICCoreNum();
    const size_t numVectorCores = Platform::Instance().GetSoc().GetAIVCoreNum();
    const size_t numAiScalarCores = Platform::Instance().GetSoc().GetAICPUNum();

    if (numCubeCores == 0 || ((numVectorCores % numCubeCores) != 0)) {
        APASS_LOG_ERROR_F(Elements::Config, "OSP bsp architecture does not satisfy an 1:N ratio of Cube:Vector cores.");
        return FAILED;
    }

    const size_t numCores = (numCubeCores != 0U ? numCubeCores : numVectorCores) + numAiScalarCores;
    const size_t numVecPerCube = (numCubeCores != 0U ? numVectorCores / numCubeCores : 1U);

    std::vector<VTypeType> procTypes(numCores);
    std::vector<WorkType> procMemoryBound(numCores);

    const size_t cubeVecMemoryBound = ((numCubeCores != 0U ? 1U : 0U)) *
                                          Platform::Instance().GetAICCore().GetMemorySize(MemoryType::MEM_L1) +
                                      (numVecPerCube *
                                       Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB));

    for (size_t i = 0; i < numCores; i++) {
        if (i < numCubeCores) { // Cube Vector Core Mix
            procTypes[i] = GetOspCoreTypeMix(OpCoreType::AIC);
            procMemoryBound[i] = static_cast<WorkType>(cubeVecMemoryBound);
        } else { // AI Scalar Cores
            procTypes[i] = GetOspCoreTypeMix(OpCoreType::AICPU);
            procMemoryBound[i] = std::numeric_limits<WorkType>::max();
        }
    }

    bspArch.SetProcessorsWithTypes(procTypes);
    bspArch.SetMemoryBound(procMemoryBound);
    bspArch.SetCommunicationCosts(archParameters_.commCost_);
    bspArch.SetSynchronisationCosts(archParameters_.synchCost_);

    return SUCCESS;
}

Status OspPartitioner::ConstructDag(GraphType& graph)
{
    if (useCVMixPartition_) {
        if (ConstructDagCVMix(graph) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV mix.");
            return FAILED;
        }
    } else {
        if (ConstructDagCVSplit(graph) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV split.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OspPartitioner::ConstructBspInstance(osp::BspInstance<GraphType>& bspInst)
{
    if (useCVMixPartition_) {
        if (ConstructBspArchCVMix(bspInst.GetArchitecture()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate bsp architecture with CV mix.");
            return FAILED;
        }
        if (ConstructDagCVMix(bspInst.GetComputationalDag()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV mix.");
            return FAILED;
        }
    } else {
        ConstructBspArchCVSplit(bspInst.GetArchitecture());
        if (ConstructDagCVSplit(bspInst.GetComputationalDag()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV split.");
            return FAILED;
        }
    }
    unsigned numTypes = std::max(bspInst.GetArchitecture().GetNumberOfProcessorTypes(),
                                 static_cast<unsigned>(bspInst.GetComputationalDag().NumVertexTypes()));
    bspInst.SetDiagonalCompatibilityMatrix(numTypes);
    return SUCCESS;
}

uint64_t OspPartitioner::CombineHash(const uint64_t h1, const uint64_t h2) const
{
    constexpr uint64_t magic = 0x9e3779b97f4a7c15ULL;
    constexpr uint64_t numSix = 6;
    constexpr uint64_t numTwo = 2;

    uint64_t seed = h1;
    seed ^= h2 + magic + (seed << numSix) + (seed >> numTwo);
    return seed;
}

uint64_t OspPartitioner::CombineNeighborHashes(uint64_t baseHash, const std::vector<int32_t>& neighbors,
                                               const std::vector<uint64_t>& hashSource)
{
    std::vector<uint64_t> hashes;
    for (int32_t j : neighbors) {
        hashes.push_back(hashSource[j]);
    }
    std::sort(hashes.begin(), hashes.end());
    for (const auto& h : hashes) {
        baseHash = CombineHash(baseHash, h);
    }
    return baseHash;
}

void OspPartitioner::BuildNodeHashValues(const std::vector<uint64_t>& opHashList)
{
    constexpr std::size_t numEleven = 11U;
    int32_t numNode = superNodeInfo_->node2Op_.size();
    superNodeInfo_->nodeHashList_.resize(numNode);
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->nodeHashList_[i] = CombineNeighborHashes(numEleven, superNodeInfo_->node2Op_[i], opHashList);
    }
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[i]].push_back(i);
    }
}

Status OspPartitioner::BuildHashValues()
{
    constexpr uint64_t numThree = 3U;
    const size_t numOps = operationInfo_->opList_.size();

    std::vector<uint64_t> opHashListFront(numOps, 0);
    std::vector<uint64_t> opHashListBack(numOps, numThree);

    for (size_t i = 0; i < numOps; i++) {
        const std::vector<int32_t> inNeighbors(operationInfo_->inGraph_[i].begin(), operationInfo_->inGraph_[i].end());
        opHashListFront[i] = CombineNeighborHashes(operationInfo_->opHashList_[i], inNeighbors, opHashListFront);
    }
    for (int32_t i = static_cast<int32_t>(numOps - 1); i >= 0; i--) {
        const std::vector<int32_t> outNeighbors(operationInfo_->outGraph_[i].begin(),
                                                operationInfo_->outGraph_[i].end());
        opHashListBack[i] = CombineNeighborHashes(opHashListBack[i], outNeighbors, opHashListBack);
    }

    std::vector<uint64_t> opHashList(numOps);
    for (size_t i = 0; i < numOps; i++) {
        opHashList[i] = CombineHash(opHashListFront[i], opHashListBack[i]);
    }

    BuildNodeHashValues(opHashList);
    return SUCCESS;
}

Status OspPartitioner::SetParameter(const Function& function)
{
    const auto pgLowerBound = function.paramConfigs_.sgPgLowerBound;
    if (pgLowerBound < 0) {
        APASS_LOG_ERROR_F(Elements::Config, "Illegal pgLowerBound: %d; Parameter pgLowerBound must be non-negative.",
                          pgLowerBound);
        return FAILED;
    }
    archParameters_.partitionWorkLowerBound_ = pgLowerBound;

    return SUCCESS;
}

} // namespace npu::tile_fwk
