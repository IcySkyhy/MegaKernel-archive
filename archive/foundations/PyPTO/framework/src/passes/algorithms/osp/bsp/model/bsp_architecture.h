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
 * \file bsp_architecture.h
 * \brief
 */

#ifndef OSP_BSP_ARCHITECTURE_H
#define OSP_BSP_ARCHITECTURE_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "passes/algorithms/osp/concepts/graph_traits.h"
#include "passes/algorithms/osp/graph_implementations/integral_range.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @enum MemoryConstraintType
 * @brief Enumerates the different types of memory constraints.
 * Memory bounds are set per processor and apply to aggregated memory weights of nodes
 * according to the different types of memory constraints.
 */
enum class MemoryConstraintType {
    NONE,                    /** No memory constraints. */
    LOCAL,                   /** The memory bounds apply to the sum of memory weights of nodes assigned to the same
                                 processor and superstep. */
    GLOBAL,                  /** The memory bounds apply to the sum of memory weights of the nodes assigned
                                 to the same processor. */
    PERSISTENT_AND_TRANSIENT /** Memory bounds apply to the sum of memory weights of nodes
                                 assigned to the same processor plus the maximum communication
                                 weight of a node assigned to a processor. */
};

/**
 * @class BspArchitecture
 * @brief Represents the architecture of a BSP (Bulk Synchronous Parallel) system.
 *
 * The BspArchitecture class stores information about the number of processors, communication costs,
 * synchronization costs, the send costs between processors, the types of processors, and the memory
 * bounds. It provides methods to set and retrieve these values.
 *
 * **Processors:**
 * The architecture consists of p processors, indexed from 0 to p-1. Note that processor indices are
 * represented using `unsigned`.
 *
 * **Processor Types:**
 * Processors can have different types, which are represented by non-negative integers.
 * Processor types are assumed to be consecutive integers starting from 0. Note that processor types are
 * represented using `unsigned`.
 * Processor types are used to express compatabilities, which can be specified in the BspInstance, regarding node types.
 *
 * **Communication and Synchronization Costs:**
 * - Communication Cost (g): The cost of communicating a unit of data between processors, i.e., the bandwidth.
 * - Synchronization Cost (L): The cost of synchronizing all processors at the end of a superstep.
 *
 * **Send Costs (NUMA):**
 * The architecture supports Non-Uniform Memory Access (NUMA) effects via a send cost matrix.
 * The cost to send data from processor i to processor j is given by g * sendCosts[i][j].
 * By default, send costs are uniform (1 for distinct processors, 0 for self).
 *
 * **Memory Constraints:**
 * Each processor has a memory bound. The `MemoryConstraintType` determines how these bounds are applied
 * (e.g., local per superstep, global per processor).
 */
template <typename GraphT>
class BspArchitecture {
public:
    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param NumberOfProcessors The number of processors in the architecture. Must be greater than 0. Default: 2.
     * @param CommunicationCost The communication cost between processors. Default: 1.
     * @param SynchronisationCost The synchronization cost between processors. Default: 2.
     * @param MemoryBound The memory bound for each processor (default: 100).
     * @param SendCosts The matrix of send costs between processors. Needs to be a processors x
     * processors matrix. Diagonal entries are forced to zero. Default: empty (uniform costs).
     */
    BspArchitecture(const unsigned numberOfProcessors = 2U, const VCommwT<GraphT> communicationCost = 1U,
                    const VCommwT<GraphT> synchronisationCost = 2U, const VMemwT<GraphT> memoryBound = 100U,
                    const std::vector<std::vector<VCommwT<GraphT>>>& sendCosts = {})
        : numberOfProcessors_(numberOfProcessors),
          numberOfProcessorTypes_(1U),
          communicationCosts_(communicationCost),
          synchronisationCosts_(synchronisationCost),
          memoryBound_(numberOfProcessors, memoryBound),
          processorTypes_(numberOfProcessors, 0U)
    {
        if (numberOfProcessors == 0U) {
            APASS_LOG_ERROR_F(Elements::Config, "BspArchitecture: Number of processors must be greater than 0.");
            throw std::runtime_error("BspArchitecture: Number of processors must be greater than 0.");
        }

        if (sendCosts.empty()) {
            InitializeUniformSendCosts();
        } else {
            if (numberOfProcessors != sendCosts.size()) {
                APASS_LOG_ERROR_F(Elements::Config,
                                  "BspArchitecture: sendCosts_ needs to be a processors x processors matrix.\n");
                throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
            }
            if (std::any_of(sendCosts.begin(), sendCosts.end(),
                            [numberOfProcessors](const auto& thing) { return thing.size() != numberOfProcessors; })) {
                APASS_LOG_ERROR_F(Elements::Config,
                                  "BspArchitecture: sendCosts_ needs to be a processors x processors matrix.\n");
                throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
            }

            sendCosts_.reserve(numberOfProcessors * numberOfProcessors);
            for (const auto& row : sendCosts) {
                sendCosts_.insert(sendCosts_.end(), row.begin(), row.end());
            }

            SetSendCostDiagonalToZero();
        }
    }

    BspArchitecture(const BspArchitecture& other) = default;
    BspArchitecture(BspArchitecture&& other) noexcept = default;
    BspArchitecture& operator=(const BspArchitecture& other) = default;
    BspArchitecture& operator=(BspArchitecture&& other) noexcept = default;
    virtual ~BspArchitecture() = default;

    /**
     * @brief Copy constructor from a BspArchitecture with a different graph type.
     *
     * @tparam Graph_t_other The graph type of the other BspArchitecture.
     * @param other The other BspArchitecture object.
     */
    template <typename GraphTOther>
    BspArchitecture(const BspArchitecture<GraphTOther>& other)
        : numberOfProcessors_(other.NumberOfProcessors()),
          numberOfProcessorTypes_(other.GetNumberOfProcessorTypes()),
          communicationCosts_(other.CommunicationCosts()),
          synchronisationCosts_(other.SynchronisationCosts()),
          memoryBound_(other.MemoryBound()),
          processorTypes_(other.ProcessorTypes()),
          sendCosts_(other.SendCostsVector())
    {
        static_assert(std::is_same_v<VMemwT<GraphT>, VMemwT<GraphTOther>>,
                      "BspArchitecture: GraphT and Graph_t_other have the same memory weight type.");

        static_assert(std::is_same_v<VCommwT<GraphT>, VCommwT<GraphTOther>>,
                      "BspArchitecture: GraphT and Graph_t_other have the same communication weight type.");

        static_assert(std::is_same_v<VTypeT<GraphT>, VTypeT<GraphTOther>>,
                      "BspArchitecture: GraphT and Graph_t_other have the same processor type.");
    }

    /**
     * @brief Constructs a BspArchitecture object with custom send costs.
     *
     * @param NumberOfProcessors The number of processors. Must be greater than 0.
     * @param CommunicationCost The communication cost.
     * @param SynchronisationCost The synchronization cost.
     * @param SendCosts The matrix of send costs between processors. Needs to be a processors x
     * processors matrix. Diagonal entries are forced to zero.
     */
    BspArchitecture(const unsigned numberOfProcessors, const VCommwT<GraphT> communicationCost,
                    const VCommwT<GraphT> synchronisationCost,
                    const std::vector<std::vector<VCommwT<GraphT>>>& sendCosts)
        : BspArchitecture(numberOfProcessors, communicationCost, synchronisationCost, 100U, sendCosts)
    {}

    /**
     * @brief Returns a view of processor indices from 0 to numberOfProcessors_ - 1.
     * @return An integral view of processor indices.
     */
    [[nodiscard]] auto Processors() const { return IntegralRange<unsigned>(numberOfProcessors_); }

    /**
     * @brief Sets the memory bound for all processors.
     * @param MemoryBound The new memory bound for all processors.
     */
    void SetMemoryBound(const VMemwT<GraphT> memoryBound) { memoryBound_.assign(numberOfProcessors_, memoryBound); }

    /**
     * @brief Sets the memory bound for all processors using a vector.
     * @param MemoryBound The vector of memory bounds.
     */
    void SetMemoryBound(const std::vector<VMemwT<GraphT>>& memoryBound)
    {
        if (memoryBound.size() != numberOfProcessors_) {
            APASS_LOG_ERROR_F(Elements::Config,
                              "Invalid Argument: Memory bound vector size does not match number of processors.");
            throw std::invalid_argument(
                "Invalid Argument: Memory bound vector size does not match number of processors.");
        }
        memoryBound_ = memoryBound;
    }

    /**
     * @brief Sets the memory bound for a specific processor.
     * @param MemoryBound The new memory bound for the processor.
     * @param processorIndex The processor index. Must be less than numberOfProcessors_.
     */
    void SetMemoryBound(const VMemwT<GraphT> memoryBound, const unsigned processorIndex)
    {
        memoryBound_.at(processorIndex) = memoryBound;
    }

    /**
     * @brief Sets the synchronization costs.
     * @param SynchCost The new synchronization costs.
     */
    void SetSynchronisationCosts(const VCommwT<GraphT> synchCost) { synchronisationCosts_ = synchCost; }

    /**
     * @brief Sets the communication costs.
     * @param CommCost The new communication costs.
     */
    void SetCommunicationCosts(const VCommwT<GraphT> commCost) { communicationCosts_ = commCost; }

    /**
     * @brief Sets the number of processors. Processor type is set to 0 for all processors.
     * Resets send costs to uniform (1) and diagonal to 0. The memory bound is set to 100 for all processors.
     * @param numberOfProcessors The number of processors. Must be greater than 0.
     */
    void SetNumberOfProcessors(const unsigned numberOfProcessors)
    {
        if (numberOfProcessors == 0) {
            APASS_LOG_ERROR_F(Elements::Config, "Invalid Argument: Number of processors must be greater than 0.");
            throw std::invalid_argument("Invalid Argument: Number of processors must be greater than 0.");
        }
        numberOfProcessors_ = numberOfProcessors;
        numberOfProcessorTypes_ = 1U;
        processorTypes_.assign(numberOfProcessors_, 0U);

        InitializeUniformSendCosts();

        // initialize memory bound to 100 for all processors
        memoryBound_.assign(numberOfProcessors_, 100U);
    }

    /**
     * @brief Sets the number of processors and their types. Number of processors is set to the
     * size of the processor types vector. Resets send costs to uniform (1). Resets memory bound
     * to 100 for all processors.
     * @param processorTypes The types of the respective processors.
     */
    void SetProcessorsWithTypes(const std::vector<VTypeT<GraphT>>& processorTypes)
    {
        if (processorTypes.empty()) {
            APASS_LOG_ERROR_F(Elements::Config, "Invalid Argument: Processor types vector is empty.");
            throw std::invalid_argument("Invalid Argument: Processor types vector is empty.");
        }
        if (processorTypes.size() > std::numeric_limits<unsigned>::max()) {
            APASS_LOG_ERROR_F(Elements::Config, "Invalid Argument: Number of processors exceeds the limit.");
            throw std::invalid_argument("Invalid Argument: Number of processors exceeds the limit.");
        }
        numberOfProcessors_ = static_cast<unsigned>(processorTypes.size());
        processorTypes_ = processorTypes;

        InitializeUniformSendCosts();

        // initialize memory bound to 100 for all processors
        memoryBound_.assign(numberOfProcessors_, 100U);
        UpdateNumberOfProcessorTypes();
    }

    /**
     * @brief Sets processors based on counts of consecutive types.
     * The architecture will have processorTypeCount[0] processors of type 0,
     * processorTypeCount[1] processors of type 1, etc.
     * The memory bound for each processor of type i is set to processorTypeMemory[i].
     * The send costs are set to uniform (1).
     * @param processorTypeCount Vector where index is type and value is count of processors of that type.
     * @param processorTypeMemory Vector where index is type and value is memory bound for that type.
     */
    void SetProcessorsConsequTypes(const std::vector<VTypeT<GraphT>>& processorTypeCount,
                                   const std::vector<VMemwT<GraphT>>& processorTypeMemory)
    {
        if (processorTypeCount.size() != processorTypeMemory.size()) {
            APASS_LOG_ERROR_F(Elements::Config,
                              "Invalid Argument: processorTypeCount and processorTypeMemory must have the same size.");
            throw std::invalid_argument(
                "Invalid Argument: processorTypeCount and processorTypeMemory must have the same size.");
        }

        if (processorTypeCount.size() > std::numeric_limits<unsigned>::max()) {
            APASS_LOG_ERROR_F(Elements::Config, "Invalid Argument: Number of processors exceeds the limit.");
            throw std::invalid_argument("Invalid Argument: Number of processors exceeds the limit.");
        }

        numberOfProcessorTypes_ = static_cast<unsigned>(processorTypeCount.size());
        numberOfProcessors_ = std::accumulate(processorTypeCount.begin(), processorTypeCount.end(), 0U);

        // initialize processor types and memory bound
        processorTypes_.assign(numberOfProcessors_, 0U);
        memoryBound_.assign(numberOfProcessors_, 0U);

        unsigned offset = 0U;
        for (unsigned i = 0U; i < processorTypeCount.size(); i++) {
            for (unsigned j = 0U; j < processorTypeCount.at(i); j++) {
                processorTypes_.at(offset + j) = i;
                memoryBound_.at(offset + j) = processorTypeMemory.at(i);
            }
            offset += processorTypeCount.at(i);
        }

        InitializeUniformSendCosts();
    }

    /**
     * @brief Returns the memory bounds of all processors.
     * @return Vector of memory bounds.
     */
    [[nodiscard]] const std::vector<VMemwT<GraphT>>& MemoryBound() const { return memoryBound_; }

    /**
     * @brief Returns the memory bound of a specific processor.
     * @param proc The processor index.
     * @return The memory bound.
     */
    [[nodiscard]] VMemwT<GraphT> MemoryBound(const unsigned proc) const { return memoryBound_[proc]; }

    /**
     * @brief Returns the maximum memory bound over all processors.
     * @return The maximum memory bound.
     */
    [[nodiscard]] VMemwT<GraphT> MaxMemoryBound() const
    {
        return *(std::max_element(memoryBound_.begin(), memoryBound_.end()));
    }

    /**
     * @brief Returns the maximum memory bound over all processors of a specific type.
     *
     * @param procType The processor type.
     * @return The maximum memory bound.
     */
    [[nodiscard]] VMemwT<GraphT> MaxMemoryBoundProcType(const VTypeT<GraphT> procType) const
    {
        VMemwT<GraphT> maxMem = 0U;
        for (unsigned proc = 0U; proc < numberOfProcessors_; proc++) {
            if (processorTypes_[proc] == procType) {
                maxMem = std::max(maxMem, memoryBound_[proc]);
            }
        }
        return maxMem;
    }

    /**
     * @brief Returns the number of processors.
     * @return The number of processors.
     */
    [[nodiscard]] unsigned NumberOfProcessors() const { return numberOfProcessors_; }

    /**
     * @brief Returns the communication costs.
     * @return The communication costs.
     */
    [[nodiscard]] VCommwT<GraphT> CommunicationCosts() const { return communicationCosts_; }

    /**
     * @brief Returns the synchronization costs.
     * @return The synchronization costs.
     */
    [[nodiscard]] VCommwT<GraphT> SynchronisationCosts() const { return synchronisationCosts_; }

    /**
     * @brief Returns a the send costs matrix. Internally the matrix is stored as a flattened
     * matrix. The allocates, computes and returns the matrix on the fly.
     * @return The send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<VCommwT<GraphT>>> SendCost() const
    {
        std::vector<std::vector<VCommwT<GraphT>>> matrix(numberOfProcessors_,
                                                         std::vector<VCommwT<GraphT>>(numberOfProcessors_));
        for (unsigned i = 0; i < numberOfProcessors_; ++i) {
            for (unsigned j = 0; j < numberOfProcessors_; ++j) {
                matrix[i][j] = sendCosts_[FlatIndex(i, j)];
            }
        }
        return matrix;
    }

    /**
     * @brief Returns the flattened send costs vector.
     * @return The send costs vector.
     */
    [[nodiscard]] const std::vector<VCommwT<GraphT>>& SendCostsVector() const { return sendCosts_; }

    /**
     * @brief Returns the processor types.
     * @return Vector of processor types.
     */
    [[nodiscard]] const std::vector<unsigned>& ProcessorTypes() const { return processorTypes_; }

    /**
     * @brief Returns the communication costs between two processors. Does not perform bounds checking.
     * The communication costs are the send costs multiplied by the communication costs factor.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @return The communication costs between the two processors.
     */
    [[nodiscard]] VCommwT<GraphT> CommunicationCosts(const unsigned p1, const unsigned p2) const
    {
        return communicationCosts_ * sendCosts_[FlatIndex(p1, p2)];
    }

    /**
     * @brief Returns the send costs between two processors. Does not perform bounds checking.
     * Does not take the communication costs into account.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @return The send costs between the two processors.
     */
    [[nodiscard]] VCommwT<GraphT> SendCosts(const unsigned p1, const unsigned p2) const
    {
        return sendCosts_[FlatIndex(p1, p2)];
    }

    /**
     * @brief Returns the type of a specific processor. Does not perform bounds checking.
     * @param p1 The processor index.
     * @return The processor type.
     */
    [[nodiscard]] VTypeT<GraphT> ProcessorType(const unsigned p1) const { return processorTypes_[p1]; }

    /**
     * @brief Returns the count of processors for each type.
     * @return Vector where index is type and value is count.
     */
    [[nodiscard]] std::vector<unsigned> GetProcessorTypeCount() const
    {
        std::vector<unsigned> typeCount(numberOfProcessorTypes_, 0U);
        for (unsigned p = 0U; p < numberOfProcessors_; p++) {
            typeCount[processorTypes_[p]]++;
        }
        return typeCount;
    }

    [[nodiscard]] unsigned GetNumberOfProcessorTypes() const { return numberOfProcessorTypes_; }

    [[nodiscard]] MemoryConstraintType GetMemoryConstraintType() const { return memoryConstraintType_; }

    void SetMemoryConstraintType(const MemoryConstraintType memoryConstraintType)
    {
        memoryConstraintType_ = memoryConstraintType;
    }

private:
    /** @brief The number of processors in the architecture. Must be at least 1. */
    unsigned numberOfProcessors_;

    /** @brief The number of processor types in the architecture. See processorTypes_ for more details. */
    unsigned numberOfProcessorTypes_;

    /** @brief The communication costs, typically denoted 'g' for the BSP model. */
    VCommwT<GraphT> communicationCosts_;

    /** @brief The synchronisation costs, typically denoted 'L' for the BSP model. */
    VCommwT<GraphT> synchronisationCosts_;

    /** @brief The architecture allows to specify memory bounds per processor. */
    std::vector<VMemwT<GraphT>> memoryBound_;

    /** @brief The architecture allows to specify processor types. Processor types are used to express
     * compatabilities, which can be specified in the BspInstance, regarding node types. */
    std::vector<unsigned> processorTypes_;

    /** @brief A flattened p x p matrix of send costs. Access via index [i * numberOfProcessors_ + j]. */
    std::vector<VCommwT<GraphT>> sendCosts_;

    /** @brief The memory constraint type. */
    MemoryConstraintType memoryConstraintType_ = MemoryConstraintType::NONE;

    /** @brief Helper function to calculate the index of a flattened p x p matrix. */
    std::size_t FlatIndex(const unsigned row, const unsigned col) const
    {
        return static_cast<std::size_t>(row) * numberOfProcessors_ + col;
    }

    void UpdateNumberOfProcessorTypes()
    {
        numberOfProcessorTypes_ = 0U;
        for (unsigned p = 0U; p < numberOfProcessors_; p++) {
            if (processorTypes_[p] >= numberOfProcessorTypes_) {
                numberOfProcessorTypes_ = processorTypes_[p] + 1U;
            }
        }
    }

    void SetSendCostDiagonalToZero()
    {
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            sendCosts_[FlatIndex(i, i)] = 0U;
        }
    }

    void InitializeUniformSendCosts()
    {
        sendCosts_.assign(numberOfProcessors_ * numberOfProcessors_, 1U);
        SetSendCostDiagonalToZero();
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_BSP_ARCHITECTURE_H
