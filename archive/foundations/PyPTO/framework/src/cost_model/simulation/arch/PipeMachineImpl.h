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
 * \file PipeMachineImpl.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/common/CommonType.h"
#include "cost_model/simulation/common/ISA.h"

namespace CostModel {
class PipeMachineImpl {
public:
    virtual ~PipeMachineImpl() = default;
    virtual uint64_t Simulate(const TileOpPtr& tileOp) = 0;
    virtual uint64_t PostSimulate(const TileOpPtr& tileOp) = 0;
    virtual uint64_t SimulateForPass(const std::string& op, const std::vector<std::vector<int>>& shape, DataType dtype)
    {
        if (ValidateInput(op, shape, dtype)) {
            return 0;
        }
        return 0;
    };
    virtual uint64_t PostSimulateForPass(const std::string& op, const std::vector<std::vector<int>>& shape,
                                         DataType dtype)
    {
        if (ValidateInput(op, shape, dtype)) {
            return 0;
        }
        return 0;
    };

private:
    bool ValidateInput(const std::string& op, const std::vector<std::vector<int>>& shape, DataType dtype) const
    {
        return op.empty() || shape.empty() || !dtype;
    }
};

struct UnifiedDeleter {
    void operator()(PipeMachineImpl* ptr) const { delete ptr; }
    static void SetCustomDeleter(void (*destroy)(PipeMachineImpl*)) { customDestroy = destroy; }

private:
    static inline void (*customDestroy)(PipeMachineImpl*) = nullptr;
};

using UnifiedPipeMachinePtr = std::shared_ptr<PipeMachineImpl>;

} // namespace CostModel
