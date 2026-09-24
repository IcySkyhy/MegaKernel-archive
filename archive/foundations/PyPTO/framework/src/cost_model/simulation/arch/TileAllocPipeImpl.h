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
 * \file TileAllocPipeImpl.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include <cmath>
#include "PipeMachineImpl.h"

namespace CostModel {
class TileAllocPipeImpl : public PipeMachineImpl {
public:
    // this belongs to A2A3 arch
    std::unordered_map<CorePipeType, uint64_t> bufferSize = {{CorePipeType::PIPE_VECTOR_BMU, 192 * pow(2, 10)},
                                                             {CorePipeType::PIPE_CUBE_BMU_L1, 512 * pow(2, 10)},
                                                             {CorePipeType::PIPE_CUBE_BMU_L0A, 64 * pow(2, 10)},
                                                             {CorePipeType::PIPE_CUBE_BMU_L0B, 64 * pow(2, 10)},
                                                             {CorePipeType::PIPE_CUBE_BMU_L0C, 128 * pow(2, 10)}};

    uint64_t Simulate(const TileOpPtr& tileOp) override
    {
        if (tileOp->iOperand.empty()) {
            return 1;
        }
        return 1;
    }

    uint64_t PostSimulate(const TileOpPtr& tileOp) override
    {
        if (tileOp->iOperand.empty()) {
            return 1;
        }
        return 1;
    }
};

inline UnifiedPipeMachinePtr CreateTileAllocPipeImpl() { return UnifiedPipeMachinePtr(new TileAllocPipeImpl()); }
} // namespace CostModel
