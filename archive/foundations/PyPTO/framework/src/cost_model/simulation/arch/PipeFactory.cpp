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
 * \file PipeFactory.cpp
 * \brief
 */

#include "PipeFactory.h"

#include <dlfcn.h>

#include "TileAllocPipeImpl.h"
#include "CallPipeImpl.h"
#include "PipeSimulatorFast.h"
#include "cost_model/simulation/arch/A2A3/L2CacheImplA2A3.h"
#include "cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h"
#include "cost_model/simulation/arch/A5/L2CacheImplA5.h"
#include "cost_model/simulation/arch/A5/PostSimulatorA5.h"

namespace CostModel {
UnifiedPipeMachinePtr CreateSimulator(const std::string& archType)
{
    if (archType == "A2A3") {
        return CreatePipeSimulatorFast<PostSimulatorA2A3>();
    } else if (archType == "A5") {
        return CreatePipeSimulatorFast<PostSimulatorA5>();
    } else {
        throw std::invalid_argument("unknown arch type " + archType);
    }
}

UnifiedPipeMachinePtr PipeFactory::Create(CorePipeType pipeType, std::string archType)
{
    switch (pipeType) {
        case CorePipeType::PIPE_TILE_ALLOC:
            return CreateTileAllocPipeImpl();
        case CorePipeType::PIPE_CALL:
            return CreateCallPipeImpl();
        case CorePipeType::PIPE_MTE_IN:
        case CorePipeType::PIPE_MTE1:
        case CorePipeType::PIPE_MTE_OUT:
        case CorePipeType::PIPE_FIX:
        case CorePipeType::PIPE_VECTOR_ALU:
        case CorePipeType::PIPE_S:
        case CorePipeType::PIPE_CUBE:
            return CreateSimulator(archType);
        default:
            throw std::invalid_argument("unknown pipe type " + CorePipeName(pipeType));
    }
    return nullptr;
}

std::unique_ptr<CacheMachineImpl> PipeFactory::CreateCache(CostModel::CacheType type, std::string archType)
{
    if (type == CacheType::L2CACHE) {
        if (archType == "A2A3") {
            return std::make_unique<L2CacheImplA2A3>();
        } else if (archType == "A5") {
            return std::make_unique<L2CacheImplA5>();
        } else {
            throw std::invalid_argument("unknown arch type " + archType);
        }
    } else {
        throw std::invalid_argument("unknown pipe type " + CacheName(type));
    }
}
} // namespace CostModel
