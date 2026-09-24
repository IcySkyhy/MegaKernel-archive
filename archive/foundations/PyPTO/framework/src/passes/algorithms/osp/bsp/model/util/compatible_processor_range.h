/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OSP_COMPATIBLE_PROCESSOR_RANGE_H
#define OSP_COMPATIBLE_PROCESSOR_RANGE_H

#include <vector>

#include "passes/algorithms/osp/bsp/model/bsp_instance.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class CompatibleProcessorRange
 * @brief Helper class to efficiently iterate over compatible processors for a given node or node type.
 *
 * This class precomputes and stores the list of compatible processors for each node type.
 *
 * @tparam GraphT The type of the computational DAG.
 */
template <typename GraphT>
class CompatibleProcessorRange {
    std::vector<std::vector<unsigned>> typeProcessorIdx_;
    const BspInstance<GraphT>* instance_ = nullptr;

public:
    /**
     * @brief Default constructor.
     */
    CompatibleProcessorRange() = default;

    /**
     * @brief Constructs a CompatibleProcessorRange for the given BspInstance.
     *
     * @param inst The BspInstance.
     */
    CompatibleProcessorRange(const BspInstance<GraphT>& inst) { Initialize(inst); }

    /**
     * @brief Initializes the CompatibleProcessorRange with a BspInstance.
     *
     * @param inst The BspInstance.
     */
    void Initialize(const BspInstance<GraphT>& inst)
    {
        instance_ = &inst;

        typeProcessorIdx_.assign(inst.GetComputationalDag().NumVertexTypes(), {});
        for (VTypeT<GraphT> vType = 0; vType < inst.GetComputationalDag().NumVertexTypes(); vType++) {
            for (unsigned proc = 0; proc < inst.NumberOfProcessors(); proc++) {
                if (inst.IsCompatibleType(vType, inst.ProcessorType(proc))) {
                    typeProcessorIdx_[vType].push_back(proc);
                }
            }
        }
    }

    /**
     * @brief Returns a range of compatible processors for a given node type.
     *
     * @param type The node type.
     * @return A const reference to a vector of compatible processor indices.
     */
    [[nodiscard]] const auto& CompatibleProcessorsType(const VTypeT<GraphT> type) const
    {
        return typeProcessorIdx_[type];
    }

    /**
     * @brief Returns a range of compatible processors for a given vertex.
     *
     * @param vertex The vertex index.
     * @return A const reference to a vector of compatible processor indices.
     */
    [[nodiscard]] const auto& CompatibleProcessorsVertex(const VertexIdxT<GraphT> vertex) const
    {
        return CompatibleProcessorsType(instance_->GetComputationalDag().VertexType(vertex));
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_COMPATIBLE_PROCESSOR_RANGE_H
