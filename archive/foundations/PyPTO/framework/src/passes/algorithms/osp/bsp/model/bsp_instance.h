/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bsp_instance.h
 * \brief
 */

#ifndef OSP_BSP_INSTANCE_H
#define OSP_BSP_INSTANCE_H

#include <iostream>

#include "bsp_architecture.h"
#include "passes/algorithms/osp/graph_algorithms/computational_dag_construction_util.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class BspInstance
 * @brief Represents a scheduling problem instance for the Bulk Synchronous Parallel (BSP) model.
 *
 * The BspInstance class serves as a container for all the necessary information to define a
 * BSP scheduling problem. It acts as the "ground" object that holds the actual implementation
 * of the graph and architecture.
 *
 * It aggregates three main components:
 *
 * 1. **Computational DAG**: The directed acyclic graph representing the program to be executed.
 *    It defines the tasks (nodes), their dependencies (directed edges), and associated weights
 *    (work, memory, communication).
 *
 * 2. **BSP Architecture**: The hardware model description, including the number of processors,
 *    their types, memory bounds, and communication/synchronization costs.
 *    Note that processor indices are represented using `unsigned`.
 *
 * 3. **Node-Processor Compatibility**: A matrix defining which node types can be executed on which
 *    processor types. This enables the modeling of heterogeneous systems (e.g., CPU + GPU) where
 *    certain nodes are restricted to specific hardware accelerators.
 *
 * @warning Be careful when assigning an existing graph to a BspInstance. Depending on the
 * constructor or assignment operator used, this may result in a deep copy of the graph structure,
 * which can be expensive for large graphs.
 *
 * This class provides a unified interface to access and modify these components, facilitating
 * the development of scheduling algorithms that need to query problem constraints and properties.
 *
 * @tparam GraphT The type of the computational DAG, which must satisfy the `is_computational_dag` concept.
 */
template <typename GraphT>
class BspInstance {
public:
    using VertexTypeTOrDefault = VTypeT<GraphT>;
    using ProcessorTypeT = unsigned;

    /**
     * @brief Default constructor for the BspInstance class.
     */
    BspInstance() = default;

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     * Computational DAG and BSP architecture are copied!
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(const GraphT& cdag, const BspArchitecture<GraphT>& architecture,
                std::vector<std::vector<bool>> nodeProcessorCompatibility = std::vector<std::vector<bool>>({{true}}))
        : cdag_(cdag), architecture_(architecture), nodeProcessorCompatibility_(nodeProcessorCompatibility)
    {}

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     * Computational DAG and BSP architecture are moved!
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(GraphT&& cdag, BspArchitecture<GraphT>&& architecture,
                std::vector<std::vector<bool>> nodeProcessorCompatibility = std::vector<std::vector<bool>>({{true}}))
        : cdag_(std::move(cdag)),
          architecture_(std::move(architecture)),
          nodeProcessorCompatibility_(nodeProcessorCompatibility)
    {}

    template <typename GraphTOther>
    explicit BspInstance(const BspInstance<GraphTOther>& other)
        : architecture_(other.GetArchitecture()),
          nodeProcessorCompatibility_(other.GetNodeProcessorCompatibilityMatrix())
    {
        ConstructComputationalDag(other.GetComputationalDag(), cdag_);
    }

    BspInstance(const BspInstance<GraphT>& other) = default;
    BspInstance(BspInstance<GraphT>&& other) noexcept = default;

    BspInstance<GraphT>& operator=(const BspInstance<GraphT>& other) = default;
    BspInstance<GraphT>& operator=(BspInstance<GraphT>&& other) noexcept = default;

    /**
     * @brief Returns a reference to the BSP architecture of the instance.
     * Assigning the BSP architecture via the reference creates a copy of the architecture.
     * The move operator may be used to transfer ownership of the architecture.
     */
    [[nodiscard]] const BspArchitecture<GraphT>& GetArchitecture() const { return architecture_; }

    [[nodiscard]] BspArchitecture<GraphT>& GetArchitecture() { return architecture_; }

    /**
     * @brief Returns a reference to the computational DAG of the instance.
     * Assigning the computational DAG via the reference creates a copy of the DAG.
     * The move operator may be used to transfer ownership of the DAG.
     */
    [[nodiscard]] const GraphT& GetComputationalDag() const { return cdag_; }

    [[nodiscard]] GraphT& GetComputationalDag() { return cdag_; }

    /**
     * @brief Returns the number of vertices in the computational DAG.
     */
    [[nodiscard]] VertexIdxT<GraphT> NumberOfVertices() const { return cdag_.NumVertices(); }

    /**
     * @brief Returns a view over the vertex indices of the computational DAG.
     */
    [[nodiscard]] auto Vertices() const { return cdag_.Vertices(); }

    /**
     * @brief Returns a view over the processor indices of the BSP architecture.
     */
    [[nodiscard]] auto Processors() const { return architecture_.Processors(); }

    /**
     * @brief Returns the number of processors in the BSP architecture.
     */
    [[nodiscard]] unsigned NumberOfProcessors() const { return architecture_.NumberOfProcessors(); }

    /**
     * @brief Returns the communication costs between two processors. Does not perform bounds checking.
     * The communication costs are the send costs multiplied by the communication costs.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     */
    [[nodiscard]] VCommwT<GraphT> CommunicationCosts(const unsigned pSend, const unsigned pReceive) const
    {
        return architecture_.CommunicationCosts(pSend, pReceive);
    }

    /**
     * @brief Returns the send costs between two processors. Does not perform bounds checking.
     * Does not take the communication costs into account.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     */
    [[nodiscard]] VCommwT<GraphT> SendCosts(const unsigned pSend, const unsigned pReceive) const
    {
        return architecture_.SendCosts(pSend, pReceive);
    }

    /**
     * @brief Returns a copy of the send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<VCommwT<GraphT>>> SendCosts() const { return architecture_.SendCosts(); }

    /**
     * @brief Returns the flattened send costs vector.
     */
    [[nodiscard]] const std::vector<VCommwT<GraphT>>& SendCostsVector() const
    {
        return architecture_.SendCostsVector();
    }

    /**
     * @brief Returns the communication costs of the BSP architecture.
     */
    [[nodiscard]] VCommwT<GraphT> CommunicationCosts() const { return architecture_.CommunicationCosts(); }

    /**
     * @brief Returns the synchronization costs of the BSP architecture.
     */
    [[nodiscard]] VCommwT<GraphT> SynchronisationCosts() const { return architecture_.SynchronisationCosts(); }

    /**
     * @brief Returns the memory bound for a specific processor.
     * @param proc The processor index.
     */
    [[nodiscard]] VMemwT<GraphT> MemoryBound(const unsigned proc) const { return architecture_.MemoryBound(proc); }

    /**
     * @brief Sets the communication costs of the BSP architecture.
     * @param cost The communication costs to set.
     */
    void SetCommunicationCosts(const VCommwT<GraphT> cost) { architecture_.SetCommunicationCosts(cost); }

    /**
     * @brief Sets the synchronisation costs of the BSP architecture.
     * @param cost The synchronisation costs to set.
     */
    void SetSynchronisationCosts(const VCommwT<GraphT> cost) { architecture_.SetSynchronisationCosts(cost); }

    /**
     * @brief Sets the number of processors. Processor type is set to 0 for all processors.
     * Resets send costs to uniform (1) and diagonal to 0. The memory bound is set to 100 for all processors.
     * @param numberOfProcessors The number of processors. Must be greater than 0.
     */
    void SetNumberOfProcessors(const unsigned num) { architecture_.SetNumberOfProcessors(num); }

    /**
     * @brief Returns the processor type for a given processor index. Does not perform bounds checking.
     * @param proc The processor index.
     */
    [[nodiscard]] VertexTypeTOrDefault ProcessorType(const unsigned proc) const
    {
        return architecture_.ProcessorType(proc);
    }

    /**
     * @brief Checks if a node is compatible with a processor. Does not perform bounds checking.
     *
     * @param node The node index.
     * @param processor_id The processor index.
     * @return True if the node is compatible with the processor, false otherwise.
     */
    [[nodiscard]] bool IsCompatible(const VertexIdxT<GraphT>& node, const unsigned processorId) const
    {
        return IsCompatibleType(cdag_.VertexType(node), architecture_.ProcessorType(processorId));
    }

    /**
     * @brief Checks if a node type is compatible with a processor type. Does not perform bounds checking.
     *
     * @param nodeType The node type.
     * @param processorType The processor type.
     * @return True if the node type is compatible with the processor type, false otherwise.
     */
    [[nodiscard]] bool IsCompatibleType(const VertexTypeTOrDefault nodeType, const ProcessorTypeT processorType) const
    {
        return nodeProcessorCompatibility_[nodeType][processorType];
    }

    /**
     * @brief Sets the node-processor compatibility matrix. The matrix is copied. Dimensions are not checked.
     * @param compatibility_ The compatibility matrix.
     */
    void SetNodeProcessorCompatibility(const std::vector<std::vector<bool>>& compatibility)
    {
        nodeProcessorCompatibility_ = compatibility;
    }

    /**
     * @brief Returns the node-processor compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>>& GetNodeProcessorCompatibilityMatrix() const
    {
        return nodeProcessorCompatibility_;
    }

    /**
     * @brief Returns the node type - processor type compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>>& GetProcessorCompatibilityMatrix() const
    {
        return nodeProcessorCompatibility_;
    }

    /**
     * @brief Sets the compatibility matrix to be diagonal. This implies that node type `i` is only
     * compatible with processor type `i`.
     * @param number_of_types The number of types.
     */
    void SetDiagonalCompatibilityMatrix(const VertexTypeTOrDefault numberOfTypes)
    {
        nodeProcessorCompatibility_.assign(numberOfTypes, std::vector<bool>(numberOfTypes, false));
        for (VertexTypeTOrDefault i = 0; i < numberOfTypes; ++i) {
            nodeProcessorCompatibility_[i][i] = true;
        }
    }

private:
    /**
     * @brief The computational DAG representing the program structure.
     *
     * It contains the graph topology (nodes and directed edges) as well as attributes such as node types,
     * work weights, memory weights, and edge communication weights.
     */
    GraphT cdag_;
    /**
     * @brief The BSP architecture model.
     *
     * It defines the hardware characteristics including processor types, memory limits,
     * communication bandwidth/latency (send costs), and global synchronization costs.
     */
    BspArchitecture<GraphT> architecture_;

    /**
     * @brief Stores the compatibility between node types and processor types.
     *
     * The architecture defines a type for each processor, and the DAG defines a type for each node.
     * This matrix stores for each node type and processor type whether they are compatible, i.e.,
     * if a node of that type can be assigned to a processor of the given type in a schedule.
     * @note The outer vector is indexed by node type, the inner vector is indexed by processor type.
     */
    std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}});
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_BSP_INSTANCE_H
