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
 * \file hash_computer.h
 * \brief
 */

#ifndef OSP_HASH_COMPUTER_H
#define OSP_HASH_COMPUTER_H

#include <unordered_map>
#include <vector>

namespace npu::tile_fwk {
namespace osp {
/**
 * @class HashComputer
 * @brief Abstract base class for computing and managing hash values and orbits for graph vertices.
 *
 * This class provides an interface for obtaining hash values for individual vertices,
 * the full list of vertex hashes, the number of unique orbits, and the vertices belonging to specific orbits.
 *
 * @tparam IndexType The type used for indexing vertices in the graph.
 */
template <typename IndexType>
class HashComputer {
public:
    virtual ~HashComputer() = default;

    /**
     * @brief Gets the hash value of a specific vertex.
     * @param v The vertex index.
     * @return The computed hash value of the vertex.
     */
    virtual std::size_t GetVertexHash(const IndexType& v) const = 0;

    /**
     * @brief Gets the reference to the vector of all vertex hashes.
     * @return A const reference to the vector containing hashes for all vertices.
     */
    virtual const std::vector<std::size_t>& GetVertexHashes() const = 0;

    /**
     * @brief Gets the number of unique orbits (equivalence classes) found.
     * @return The number of orbits.
     */
    virtual std::size_t NumOrbits() const = 0;

    /**
     * @brief Gets the orbit (list of equivalent vertices) that a specific vertex belongs to.
     * @param v The vertex index.
     * @return A const reference to the vector of indices in the same orbit.
     */
    virtual const std::vector<IndexType>& GetOrbit(const IndexType& v) const = 0;

    /**
     * @brief Gets the map of all orbits.
     * @return A const reference to the map where keys are hash values and values are vectors of vertex indices.
     */
    virtual const std::unordered_map<std::size_t, std::vector<IndexType>>& GetOrbits() const = 0;

    /**
     * @brief Gets the orbit corresponding to a specific hash value.
     * @param hash The hash value of the orbit.
     * @return A const reference to the vector of vertex indices in the orbit.
     */
    virtual const std::vector<IndexType>& GetOrbitFromHash(const std::size_t& hash) const = 0;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_HASH_COMPUTER_HPP
