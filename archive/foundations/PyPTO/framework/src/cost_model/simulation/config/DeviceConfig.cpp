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
 * \file DeviceConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/DeviceConfig.h"

using namespace std;

namespace CostModel {
DeviceConfig::DeviceConfig()
{
    Config::prefix = "Device";
    Config::dispatcher = {
        {"stitchLatency", [&](string v) { stitchLatency = ParseInteger(v); }},
        {"submitLatency", [&](string v) { submitLatency = ParseInteger(v); }},
        {"stitchMaxSize", [&](string v) { stitchMaxSize = ParseInteger(v); }},
        {"submachineTypes", [&](string v) { ParseStrVec(v, submachineTypes); }},
        {"submitTopo", [&](string v) { submitTopo = ParseBoolean(v); }},
        {"submitTopoPath", [&](string v) { submitTopoPath = ParseString(v); }},
        {"replayEnable", [&](string v) { replayEnable = ParseBoolean(v); }},
        {"replayFile", [&](string v) { replayFile = ParseString(v); }},
        {"replayTaskTimeScaling", [&](string v) { replayTaskTimeScaling = ParseBoolean(v); }},
    };

    Config::recorder = {
        {"stitchLatency", [&]() { return "stitchLatency = " + ParameterToStr(stitchLatency); }},
        {"submitLatency", [&]() { return "submitLatency = " + ParameterToStr(submitLatency); }},
        {"stitchMaxSize", [&]() { return "stitchMaxSize = " + ParameterToStr(stitchMaxSize); }},
        {"submachineTypes", [&]() { return "submachineTypes = " + ParameterToStr(submachineTypes); }},
        {"submitTopo", [&]() { return "submitTopo = " + ParameterToStr(submitTopo); }},
        {"submitTopoPath", [&]() { return "submitTopoPath = " + submitTopoPath; }},
        {"replayEnable", [&]() { return "replayEnable = " + ParameterToStr(replayEnable); }},
        {"replayFile", [&]() { return "replayFile = " + replayFile; }},
        {"replayTaskTimeScaling", [&]() { return "replayTaskTimeScaling = " + ParameterToStr(replayTaskTimeScaling); }},
    };
}
} // namespace CostModel
