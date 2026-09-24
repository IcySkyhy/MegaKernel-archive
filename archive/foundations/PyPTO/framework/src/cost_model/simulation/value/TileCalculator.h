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
 * \file TileCalculator.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/value/TileState.h"
#include "cost_model/simulation/common/ISA.h"

namespace CostModel {
class TileCalculator {
private:
    size_t seq;

private:
    TileCalculator() : seq(0) {}
    static TileCalculator instance;

public:
    TileCalculator(const TileCalculator&) = delete;
    TileCalculator& operator=(const TileCalculator&) = delete;
    static TileCalculator& Self() { return instance; }
    void Reset();
    void CalculateInput(TilePtr tile, std::shared_ptr<TileState> global);
    void Calculate(TileOpPtr op, FunctionInvokeInfo& invoke, std::shared_ptr<TileState> local,
                   std::shared_ptr<TileState> global);
};
}; // namespace CostModel
