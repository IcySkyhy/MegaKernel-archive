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
 * \file PostSimulatorA2A3.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/arch/PipeMachineImpl.h"

namespace CostModel {
class PostSimulatorA2A3 {
private:
    const std::unordered_map<std::string, std::vector<float>> opLatencyA2A3_{
        {"COPY_IN", {2.613, 0.00240884}},
        {"L1_COPY_IN", {2.613, 0.00240884}},
        {"UB_COPY_IN", {2.613, 0.00240884}},
        {"COPY_OUT", {3.2883, 0.00195}},
        {"L0C_COPY_OUT", {3.2883, 0.00195}},
        {"UB_COPY_OUT", {3.2883, 0.00195}},
        {"ADD", {2.40457, 0.0188}},
        {"SUB", {1.64044, 0.02152}},
        {"MUL", {1.63706, 0.01419}},
        {"DIV", {3.82854, 0.02462}},
        {"ADDS", {0.76264, 0.0221}},
        {"SUBS", {0.76264, 0.0221}},
        {"RANGE", {0.76264, 0.0221}},
        {"MULS", {0.9527, 0.015}},
        {"DIVS", {0.9527, 0.015}},
        {"SQRT", {3.67358, 0.02197}},
        {"EXP", {3.49358, 0.02597}},
        {"NEG", {0.9527, 0.015}},
        {"LN", {3.49358, 0.02597}},
        {"WHERE_TT", {2.40457, 0.0188}},
        {"WHERE_TS", {2.40457, 0.0188}},
        {"WHERE_ST", {2.40457, 0.0188}},
        {"WHERE_SS", {2.40457, 0.0188}},
        {"RECIPROCAL", {0.98673, 0.00989}},
        {"ROWSUMLINE", {14.58731, 0.07288}},
        {"ROWMAX_SINGLE", {15.58731, 0.00888}},
        {"ROWMIN_SINGLE", {15.58731, 0.00888}},
        {"EXPAND", {0.58, 0.014}},
        {"CAST", {0.91250, 0.015687}},
        {"TRANSPOSE_MOVEOUT", {0.00025893, 0.03350}},
        {"L1_TO_L0B", {2.05893, 0.012}},
        {"L1_TO_L0A", {1.12719, 0.015}},
        {"L1_TO_L0Bt", {1.868830, 0.015}},
        {"A_MUL_B", {3.91153, 0.0107}},
        {"A_MUL_Bt", {3.91153, 0.0110}},
        {"A_MULACC_B", {3.90376, 0.0103}},
        {"A_MULACC_Bt", {3.90376, 0.0105}},
        {"PAIRMAX", {19.28672, 0.0}},
        {"PAIRSUM", {19.58115, 0.0}},
        {"MAXIMUM", {308.7260, 0.0}},
    };

public:
    const std::unordered_map<std::string, std::vector<float>>& GetOpLatency() { return opLatencyA2A3_; }
};
} // namespace CostModel
