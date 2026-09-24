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
 * \file PipeConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/PipeConfig.h"

using namespace std;

namespace CostModel {
PipeConfig::PipeConfig()
{
    Config::prefix = "Pipe";
    Config::dispatcher = {
        {"ubSizeThreshold", [&](string v) { ubSizeThreshold = ParseInteger(v); }},
        {"l1SizeThreshold", [&](string v) { l1SizeThreshold = ParseInteger(v); }},
        {"l0aSizeThreshold", [&](string v) { l0aSizeThreshold = ParseInteger(v); }},
        {"l0bSizeThreshold", [&](string v) { l0bSizeThreshold = ParseInteger(v); }},
        {"l0cSizeThreshold", [&](string v) { l0cSizeThreshold = ParseInteger(v); }},
    };

    Config::recorder = {
        {"ubSizeThreshold", [&]() { return "ubSizeThreshold = " + ParameterToStr(ubSizeThreshold); }},
        {"l1SizeThreshold", [&]() { return "l1SizeThreshold = " + ParameterToStr(l1SizeThreshold); }},
        {"l0aSizeThreshold", [&]() { return "l0aSizeThreshold = " + ParameterToStr(l0aSizeThreshold); }},
        {"l0bSizeThreshold", [&]() { return "l0bSizeThreshold = " + ParameterToStr(l0bSizeThreshold); }},
        {"l0cSizeThreshold", [&]() { return "l0cSizeThreshold = " + ParameterToStr(l0cSizeThreshold); }},
    };
}
} // namespace CostModel
