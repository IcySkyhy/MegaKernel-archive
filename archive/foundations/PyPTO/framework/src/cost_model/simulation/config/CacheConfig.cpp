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
 * \file CacheConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/CacheConfig.h"

using namespace std;

namespace CostModel {
CacheConfig::CacheConfig()
{
    Config::prefix = "Cache";
    Config::dispatcher = {
        {"l2InputPortNum", [&](string v) { l2InputPortNum = ParseInteger(v); }},
        {"l2Size", [&](string v) { l2Size = ParseInteger(v); }},
        {"l2LineSize", [&](string v) { l2LineSize = ParseInteger(v); }},
        {"l2HitLatency", [&](string v) { l2HitLatency = ParseInteger(v); }},
        {"l2MissExtraLatency", [&](string v) { l2MissExtraLatency = ParseInteger(v); }},
    };

    Config::recorder = {
        {"l2InputPortNum", [&]() { return "l2InputPortNum = " + ParameterToStr(l2InputPortNum); }},
        {"l2Size", [&]() { return "l2Size = " + ParameterToStr(l2Size); }},
        {"l2LineSize", [&]() { return "l2LineSize = " + ParameterToStr(l2LineSize); }},
        {"l2HitLatency", [&]() { return "l2HitLatency = " + ParameterToStr(l2HitLatency); }},
        {"l2MissExtraLatency", [&]() { return "l2MissExtraLatency = " + ParameterToStr(l2MissExtraLatency); }},
    };
}
} // namespace CostModel
