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
 * \file PostSimulatorA5.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/arch/PipeMachineImpl.h"

namespace CostModel {
class PostSimulatorA5 {
private:
    const std::unordered_map<std::string, std::vector<float>> opLatencyA5_{
        {"COPY_IN", {2.613, 0.00240884}},
        {"L1_COPY_IN", {2.613, 0.00240884}},
        {"UB_COPY_IN", {2.613, 0.00240884}},
        {"COPY_OUT", {3.2883, 0.00195}},
        {"L0C_COPY_OUT", {3.2883, 0.00195}},
        {"UB_COPY_OUT", {3.2883, 0.00195}},
        {"ADD", {1.4902, 0.01992}},
        {"SUB", {8, 0.0312}},
        {"A_MUL_B", {3.91153, 0.0107}},
        {"DIV", {3.82854, 0.02462}},
        {"ADDS", {0.76264, 0.0221}},
        {"EXPAND", {0.58, 0.014}},
        {"CAST", {0.91250, 0.015687}},
        {"TRANSPOSE_VNCHWCONV", {0, 0}},
        {"VIEW", {0, 0}},
        {"ASSEMBLE", {0, 0}},
        {"CALL", {0, 0}},
        {"L1_TO_L0A", {1.12719, 0.015}},
        {"L1_TO_L0At", {1.12719, 0.015}},
        {"L1_TO_L0B", {2.05893, 0.012}},
        {"L1_TO_L0Bt", {1.868830, 0.015}},
        {"A_MUL_Bt", {3.91153, 0.0110}},
        {"A_MULACC_B", {3.90376, 0.0103}},
        {"A_MULACC_Bt", {3.90376, 0.0105}},
        {"L0C_COPY_UB", {3.90376, 0.0105}},
    };

public:
    const std::unordered_map<std::string, std::vector<float>>& GetOpLatency() { return opLatencyA5_; }

    int GetFreqTrans()
    {
        const int freqTransA5_ = 1600;
        return freqTransA5_;
    }
};
} // namespace CostModel
