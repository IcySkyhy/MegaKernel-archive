/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/common/soc.h"
#include "core/logging.h"
#include "tilefwk/error.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {

namespace {

template <typename CoreT>
std::optional<uint64_t> FindMemSizeInCore(const CoreT& core, ir::MemorySpace mem_type)
{
    for (const auto& mem : core.GetMems()) {
        if (mem.GetMemType() == mem_type) {
            return mem.GetMemSize();
        }
    }
    return std::nullopt;
}

template <typename ClusterT>
std::optional<uint64_t> FindMemSizeInCluster(const ClusterT& cluster, ir::MemorySpace mem_type)
{
    for (const auto& core_entry : cluster.GetCoreCounts()) {
        if (auto size = FindMemSizeInCore(core_entry.first, mem_type)) {
            return size;
        }
    }
    return std::nullopt;
}

template <typename DieT>
std::optional<uint64_t> FindMemSizeInDie(const DieT& die, ir::MemorySpace mem_type)
{
    for (const auto& cluster_entry : die.GetClusterCounts()) {
        if (auto size = FindMemSizeInCluster(cluster_entry.first, mem_type)) {
            return size;
        }
    }
    return std::nullopt;
}

} // namespace

// ========== Backend Implementation ==========

std::vector<ir::MemorySpace> Backend::FindMemPath(ir::MemorySpace from, ir::MemorySpace to) const
{
    if (from == to) {
        return {from};
    }

    const auto& mem_graph = soc_->GetMemoryGraph();

    // BFS to find shortest path
    std::queue<ir::MemorySpace> queue;
    std::unordered_map<ir::MemorySpace, ir::MemorySpace> parent;
    std::set<ir::MemorySpace> visited;

    queue.push(from);
    visited.insert(from);
    parent[from] = from; // Mark root

    bool found = false;
    while (!queue.empty() && !found) {
        auto current = queue.front();
        queue.pop();

        auto it = mem_graph.find(current);
        if (it == mem_graph.end()) {
            continue;
        }

        for (auto neighbor : it->second) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                parent[neighbor] = current;
                queue.push(neighbor);

                if (neighbor == to) {
                    found = true;
                    break;
                }
            }
        }
    }

    CHECK(found) << "No path found from " << static_cast<int>(from) << " to " << static_cast<int>(to);

    // Reconstruct path
    std::vector<ir::MemorySpace> path;
    ir::MemorySpace current = to;
    while (current != from) {
        path.push_back(current);
        current = parent[current];
    }
    path.push_back(from);
    std::reverse(path.begin(), path.end());

    return path;
}

uint64_t Backend::GetMemSize(ir::MemorySpace mem_type) const
{
    for (const auto& die_entry : soc_->GetDieCounts()) {
        if (auto size = FindMemSizeInDie(die_entry.first, mem_type)) {
            return *size;
        }
    }
    return 0;
}

// ========== Operator Registration ==========

BackendOpRegistryEntry Backend::RegisterOp(const std::string& op_name) { return BackendOpRegistryEntry(this, op_name); }

void Backend::FinalizeOpRegistration(const std::string& op_name, ir::PipeType pipe, BackendCodegenFunc func)
{
    if (backend_op_registry_.find(op_name) != backend_op_registry_.end()) {
        return;
    }
    backend_op_registry_[op_name] = BackendOpInfo{pipe, std::move(func)};
}

const Backend::BackendOpInfo* Backend::GetOpInfo(const std::string& op_name) const
{
    auto it = backend_op_registry_.find(op_name);
    if (it != backend_op_registry_.end()) {
        return &it->second;
    }
    return nullptr;
}

// ========== BackendOpRegistryEntry Implementation ==========

BackendOpRegistryEntry& BackendOpRegistryEntry::set_pipe(ir::PipeType pipe)
{
    CHECK(!pipe_.has_value()) << "Pipe type already set for op '" << op_name_ << "'";
    pipe_ = pipe;
    return *this;
}

BackendOpRegistryEntry& BackendOpRegistryEntry::f_codegen(BackendCodegenFunc func)
{
    CHECK(!codegen_func_.has_value()) << "Codegen function already set for op '" << op_name_ << "'";
    codegen_func_ = std::move(func);
    return *this;
}

BackendOpRegistryEntry::~BackendOpRegistryEntry()
{
    if (backend_ && pipe_.has_value() && codegen_func_.has_value()) {
        backend_->FinalizeOpRegistration(op_name_, *pipe_, std::move(*codegen_func_));
    }
}

} // namespace backend
} // namespace pypto
