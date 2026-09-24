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
 * \file bsp_schedule.h
 * \brief
 */

#ifndef OSP_BSP_SCHEDULE_H
#define OSP_BSP_SCHEDULE_H

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/bsp/model/cost/lazy_communication_cost.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class BspSchedule
 * @brief Represents a schedule for the Bulk Synchronous Parallel (BSP) model.
 *
 * The `BspSchedule` class manages the assignment of nodes to processors and supersteps within the BSP model.
 * It serves as a core component for scheduling algorithms, providing mechanisms to:
 * - Store and retrieve node-to-processor and node-to-superstep assignments.
 * - Validate schedules against precedence, memory, and node type constraints.
 * - Compute costs associated with the schedule.
 * - Manipulate the schedule, including updating assignments and merging supersteps.
 *
 * This class is templated on `GraphT`, which must satisfy the `computational_dag_concept`.
 * Moreover, the work and communication weights of the nodes must be of the same type in order to
 * properly compute the cost.
 *
 * It interacts closely with `BspInstance` to access problem-specific data and constraints. In fact,
 * a `BspSchedule` object is tied to a `BspInstance` object.
 *
 * @tparam GraphT The type of the computational DAG, which must satisfy `is_computational_dag_v`.
 */
template <typename GraphT>
class BspSchedule {
public:
    using VertexIdx = VertexIdxT<GraphT>;

    BspSchedule() = default;

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    BspSchedule(const BspInstance<GraphT>& inst)
        : instance_(&inst),
          numberOfSupersteps_(1),
          nodeToProcessorAssignment_(std::vector<unsigned>(inst.NumberOfVertices(), 0)),
          nodeToSuperstepAssignment_(std::vector<unsigned>(inst.NumberOfVertices(), 0))
    {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    BspSchedule(const BspInstance<GraphT>& inst, const std::vector<unsigned>& processorAssignment,
                const std::vector<unsigned>& superstepAssignment)
        : instance_(&inst),
          nodeToProcessorAssignment_(processorAssignment),
          nodeToSuperstepAssignment_(superstepAssignment)
    {
        UpdateNumberOfSupersteps();
    }

    /**
     * @brief Copy constructor.
     *
     * @param schedule The schedule to copy.
     */
    BspSchedule(const BspSchedule<GraphT>& schedule)
        : instance_(schedule.instance_),
          numberOfSupersteps_(schedule.numberOfSupersteps_),
          nodeToProcessorAssignment_(schedule.nodeToProcessorAssignment_),
          nodeToSuperstepAssignment_(schedule.nodeToSuperstepAssignment_)
    {}

    /**
     * @brief Copy assignment operator.
     *
     * @param schedule The schedule to copy.
     * @return A reference to this schedule.
     */
    BspSchedule<GraphT>& operator=(const BspSchedule<GraphT>& schedule)
    {
        if (this != &schedule) {
            instance_ = schedule.instance_;
            numberOfSupersteps_ = schedule.numberOfSupersteps_;
            nodeToProcessorAssignment_ = schedule.nodeToProcessorAssignment_;
            nodeToSuperstepAssignment_ = schedule.nodeToSuperstepAssignment_;
        }
        return *this;
    }

    /**
     * @brief Move constructor.
     *
     * @param schedule The schedule to move.
     */
    BspSchedule(BspSchedule<GraphT>&& schedule) noexcept
        : instance_(schedule.instance_),
          numberOfSupersteps_(schedule.numberOfSupersteps_),
          nodeToProcessorAssignment_(std::move(schedule.nodeToProcessorAssignment_)),
          nodeToSuperstepAssignment_(std::move(schedule.nodeToSuperstepAssignment_))
    {}

    /**
     * @brief Move assignment operator.
     *
     * @param schedule The schedule to move.
     * @return A reference to this schedule.
     */
    BspSchedule<GraphT>& operator=(BspSchedule<GraphT>&& schedule) noexcept
    {
        if (this != &schedule) {
            instance_ = schedule.instance_;
            numberOfSupersteps_ = schedule.numberOfSupersteps_;
            nodeToProcessorAssignment_ = std::move(schedule.nodeToProcessorAssignment_);
            nodeToSuperstepAssignment_ = std::move(schedule.nodeToSuperstepAssignment_);
        }
        return *this;
    }

    /**
     * @brief Constructs a BspSchedule object from another schedule with a different graph type.
     *
     * @tparam Graph_t_other The graph type of the other schedule.
     * @param instance_ The BspInstance for the new schedule.
     * @param schedule The other schedule to copy from.
     */
    template <typename GraphTOther>
    BspSchedule(const BspInstance<GraphT>& instance, const BspSchedule<GraphTOther>& schedule)
        : instance_(&instance),
          numberOfSupersteps_(schedule.NumberOfSupersteps()),
          nodeToProcessorAssignment_(schedule.AssignedProcessors()),
          nodeToSuperstepAssignment_(schedule.AssignedSupersteps())
    {}

    /**
     * @brief Destructor for the BspSchedule class.
     */
    virtual ~BspSchedule() = default;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    [[nodiscard]] const BspInstance<GraphT>& GetInstance() const { return *instance_; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    [[nodiscard]] unsigned NumberOfSupersteps() const { return numberOfSupersteps_; }

    unsigned& NumberOfSupersteps() { return numberOfSupersteps_; }

    void Clear()
    {
        nodeToProcessorAssignment_.clear();
        nodeToSuperstepAssignment_.clear();
        numberOfSupersteps_ = 0;
    }

    /**
     * @brief Updates the number of supersteps based on the current assignment.
     */
    void UpdateNumberOfSupersteps()
    {
        numberOfSupersteps_ = 0;
        for (VertexIdxT<GraphT> i = 0; i < static_cast<VertexIdxT<GraphT>>(instance_->NumberOfVertices()); ++i) {
            if (nodeToSuperstepAssignment_[i] >= numberOfSupersteps_) {
                numberOfSupersteps_ = nodeToSuperstepAssignment_[i] + 1;
            }
        }
    }

    /**
     * @brief Returns the superstep assigned to the specified node.
     *
     * @param node The node for which to return the assigned superstep.
     * @return The superstep assigned to the specified node.
     */
    [[nodiscard]] unsigned AssignedSuperstep(const VertexIdx node) const { return nodeToSuperstepAssignment_[node]; }

    /**
     * @brief Returns the processor assigned to the specified node.
     *
     * @param node The node for which to return the assigned processor.
     * @return The processor assigned to the specified node.
     */
    [[nodiscard]] unsigned AssignedProcessor(const VertexIdx node) const { return nodeToProcessorAssignment_[node]; }

    /**
     * @brief Returns the superstep assignment for the schedule.
     *
     * @return The superstep assignment for the schedule.
     */
    [[nodiscard]] const std::vector<unsigned>& AssignedSupersteps() const { return nodeToSuperstepAssignment_; }

    [[nodiscard]] std::vector<unsigned>& AssignedSupersteps() { return nodeToSuperstepAssignment_; }

    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    [[nodiscard]] const std::vector<unsigned>& AssignedProcessors() const { return nodeToProcessorAssignment_; }

    [[nodiscard]] std::vector<unsigned>& AssignedProcessors() { return nodeToProcessorAssignment_; }

    /**
     * @brief Returns the staleness of the schedule.
     * The staleness determines the minimum number of supersteps that must elapse between the
     * assignment of a node to a processor and the assignment of one of its neighbors to a different
     * processor. The staleness for the BspSchedule is always 1.
     *
     * @return The staleness of the schedule.
     */
    [[nodiscard]] virtual unsigned GetStaleness() const { return 1; }

    /**
     * @brief Sets the superstep assigned to the specified node.
     *
     * @param node The node for which to set the assigned superstep.
     * @param superstep The superstep to assign to the node.
     */
    void SetAssignedSuperstep(const VertexIdx node, const unsigned superstep)
    {
        if (node < instance_->NumberOfVertices()) {
            nodeToSuperstepAssignment_[node] = superstep;

            if (superstep >= numberOfSupersteps_) {
                numberOfSupersteps_ = superstep + 1;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Config,
                              "Invalid Argument while assigning node to superstep: index out of range.");
            throw std::invalid_argument("Invalid Argument while assigning node to superstep: index out of range.");
        }
    }

    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    void SetAssignedProcessor(const VertexIdx node, const unsigned processor)
    {
        nodeToProcessorAssignment_.at(node) = processor;
    }

    /**
     * @brief Computes the work costs of the schedule.
     * The workload of a processor in a superstep is the sum of the workloads of all nodes assigned
     * to that processor in that superstep.
     * The workload in a superstep is the maximum workload of any processor in that superstep.
     * The work cost of the schedule is the sum of the workloads of all supersteps.
     *
     * @return The work costs of the schedule.
     */
    virtual VWorkwT<GraphT> ComputeWorkCosts() const { return cost_helpers::ComputeWorkCosts(*this); }

    /**
     * @brief Computes the costs of the schedule accoring to lazy communication cost evaluation.
     *
     * @return The costs of the schedule.
     */
    virtual VWorkwT<GraphT> ComputeCosts() const { return LazyCommunicationCost<GraphT>()(*this); }

    /**
     * @brief Checks if the schedule is valid.
     *
     * A schedule is valid if it satisfies all precedence, memory, and node type constraints.
     *
     * @return True if the schedule is valid, false otherwise.
     */
    [[nodiscard]] bool IsValid() const { return SatisfiesPrecedenceConstraints() && SatisfiesNodeTypeConstraints(); }

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u
     * and v are assigned to different processors, the difference between the superstep assigned to node u and the
     * superstep assigned to node v is less than the staleness of the schedule. For the BspSchedule staleness is 1.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    [[nodiscard]] bool SatisfiesPrecedenceConstraints() const
    {
        if (static_cast<VertexIdxT<GraphT>>(nodeToProcessorAssignment_.size()) != instance_->NumberOfVertices() ||
            static_cast<VertexIdxT<GraphT>>(nodeToSuperstepAssignment_.size()) != instance_->NumberOfVertices()) {
            return false;
        }

        for (const auto& v : instance_->Vertices()) {
            if (nodeToSuperstepAssignment_[v] >= numberOfSupersteps_) {
                return false;
            }
            if (nodeToProcessorAssignment_[v] >= instance_->NumberOfProcessors()) {
                return false;
            }

            for (const auto& target : instance_->GetComputationalDag().Children(v)) {
                const unsigned differentProcessors = (nodeToProcessorAssignment_[v] ==
                                                      nodeToProcessorAssignment_[target]) ?
                                                         0u :
                                                         GetStaleness();
                if (nodeToSuperstepAssignment_[v] + differentProcessors > nodeToSuperstepAssignment_[target]) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * @brief Checks if the schedule satisfies node type constraints.
     *
     * Node type constraints are checked based on the compatibility of nodes with their assigned processors.
     *
     * @return True if node type constraints are satisfied, false otherwise.
     */
    [[nodiscard]] bool SatisfiesNodeTypeConstraints() const
    {
        if (nodeToProcessorAssignment_.size() != static_cast<std::size_t>(instance_->NumberOfVertices())) {
            return false;
        }

        for (const auto& node : instance_->Vertices()) {
            if (!instance_->IsCompatible(node, nodeToProcessorAssignment_[node])) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Sets the number of supersteps in the schedule.
     *
     * @param number_of_supersteps_ The number of supersteps.
     */
    void SetNumberOfSupersteps(const unsigned numberOfSupersteps) { numberOfSupersteps_ = numberOfSupersteps; }

protected:
    const BspInstance<GraphT>* instance_;

    unsigned numberOfSupersteps_;

    std::vector<unsigned> nodeToProcessorAssignment_;
    std::vector<unsigned> nodeToSuperstepAssignment_;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_BSP_SCHEDULE_H
