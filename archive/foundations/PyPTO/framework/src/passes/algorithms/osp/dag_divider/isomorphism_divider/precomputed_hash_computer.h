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
 * \file precomputed_hash_computer.h
 * \brief
 */

#ifndef OSP_PRECOMPUTEDHASH_COMPUTER_H
#define OSP_PRECOMPUTEDHASH_COMPUTER_H

#include <unordered_map>
#include <vector>

#include "passes/algorithms/osp/dag_divider/isomorphism_divider/hash_computer.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class PrecomputedHashComputer
 * @brief A class to store precomputed hash values for a set of objects and provide an orbit-based interface.
 *
 * @tparam IndexType The type used for indexing the objects
 */
template <typename IndexType>
class PrecomputedHashComputer : public HashComputer<IndexType> {
    std::vector<std::size_t> vertexHashes_;
    std::unordered_map<std::size_t, std::vector<IndexType>> orbits_;

public:
    /**
     * @brief Construct a new Precomputed Hash Computer object.
     *
     * @param precomputedHashes A vector of hash values for objects 0 to n-1.
     */
    PrecomputedHashComputer(const std::vector<std::size_t>& precomputedHashes) : vertexHashes_(precomputedHashes)
    {
        for (std::size_t i = 0; i < vertexHashes_.size(); ++i) {
            const auto& hash = vertexHashes_[i];
            orbits_[hash].push_back(static_cast<IndexType>(i));
        }
    }

    ~PrecomputedHashComputer() override = default;

    std::size_t GetVertexHash(const IndexType& v) const override { return vertexHashes_[v]; }

    const std::vector<std::size_t>& GetVertexHashes() const override { return vertexHashes_; }

    std::size_t NumOrbits() const override { return orbits_.size(); }

    const std::vector<IndexType>& GetOrbit(const IndexType& v) const override
    {
        return this->GetOrbitFromHash(this->GetVertexHash(v));
    }

    const std::unordered_map<std::size_t, std::vector<IndexType>>& GetOrbits() const override { return orbits_; }

    const std::vector<IndexType>& GetOrbitFromHash(const std::size_t& hash) const override { return orbits_.at(hash); }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_PRECOMPUTEDHASH_COMPUTER_HPP
