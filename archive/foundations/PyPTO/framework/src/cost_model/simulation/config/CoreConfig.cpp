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
 * \file CoreConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/CoreConfig.h"

using namespace std;

namespace CostModel {
CoreConfig::CoreConfig()
{
    Config::prefix = "Core";
    Config::dispatcher = {
        {"pipeTileAllocNum", [&](string v) { pipeTileAllocNum = ParseInteger(v); }},
        {"pipeVectorBmuNum", [&](string v) { pipeVectorBmuNum = ParseInteger(v); }},
        {"pipeCubeBmuL1NUM", [&](string v) { pipeCubeBmuL1NUM = ParseInteger(v); }},
        {"pipeCubeBmuL0ANUM", [&](string v) { pipeCubeBmuL0ANUM = ParseInteger(v); }},
        {"pipeCubeBmuL0BNUM", [&](string v) { pipeCubeBmuL0BNUM = ParseInteger(v); }},
        {"pipeCubeBmuL0CNUM", [&](string v) { pipeCubeBmuL0CNUM = ParseInteger(v); }},
        {"pipeMteInNum", [&](string v) { pipeMteInNum = ParseInteger(v); }},
        {"pipeMte1Num", [&](string v) { pipeMte1Num = ParseInteger(v); }},
        {"pipeVectorAluNum", [&](string v) { pipeVectorAluNum = ParseInteger(v); }},
        {"pipeCubeNum", [&](string v) { pipeCubeNum = ParseInteger(v); }},
        {"pipeMteOutNum", [&](string v) { pipeMteOutNum = ParseInteger(v); }},
        {"pipeSimCallNum", [&](string v) { pipeSimCallNum = ParseInteger(v); }},
        {"tileopSentToPipeThreshold", [&](string v) { tileopSentToPipeThreshold = ParseInteger(v); }},
        {"calendarSetQueueWDelay", [&](string v) { calendarSetQueueWDelay = ParseInteger(v); }},
        {"bufferBackPressure", [&](string v) { bufferBackPressure = ParseBoolean(v); }},
        {"logLabelMode", [&](string v) { logLabelMode = ParseInteger(v); }},
        {"enableTileOpFlow", [&](string v) { enableTileOpFlow = ParseBoolean(v); }},
    };

    Config::recorder = {
        {"pipeTileAllocNum", [&]() { return "pipeTileAllocNum = " + ParameterToStr(pipeTileAllocNum); }},
        {"pipeVectorBmuNum", [&]() { return "pipeVectorBmuNum = " + ParameterToStr(pipeVectorBmuNum); }},
        {"pipeCubeBmuL1NUM", [&]() { return "pipeCubeBmuL1NUM = " + ParameterToStr(pipeCubeBmuL1NUM); }},
        {"pipeCubeBmuL0ANUM", [&]() { return "pipeCubeBmuL0ANUM = " + ParameterToStr(pipeCubeBmuL0ANUM); }},
        {"pipeCubeBmuL0BNUM", [&]() { return "pipeCubeBmuL0BNUM = " + ParameterToStr(pipeCubeBmuL0BNUM); }},
        {"pipeCubeBmuL0CNUM", [&]() { return "pipeCubeBmuL0CNUM = " + ParameterToStr(pipeCubeBmuL0CNUM); }},
        {"pipeMteInNum", [&]() { return "pipeMteInNum = " + ParameterToStr(pipeMteInNum); }},
        {"pipeMte1Num", [&]() { return "pipeMte1Num = " + ParameterToStr(pipeMte1Num); }},
        {"pipeVectorAluNum", [&]() { return "pipeVectorAluNum = " + ParameterToStr(pipeVectorAluNum); }},
        {"pipeCubeNum", [&]() { return "pipeCubeNum = " + ParameterToStr(pipeCubeNum); }},
        {"pipeMteOutNum", [&]() { return "pipeMteOutNum = " + ParameterToStr(pipeMteOutNum); }},
        {"pipeSimCallNum", [&]() { return "pipeSimCallNum = " + ParameterToStr(pipeSimCallNum); }},
        {"tileopSentToPipeThreshold",
         [&]() { return "tileopSentToPipeThreshold = " + ParameterToStr(tileopSentToPipeThreshold); }},
        {"calendarSetQueueWDelay",
         [&]() { return "calendarSetQueueWDelay = " + ParameterToStr(calendarSetQueueWDelay); }},
        {"bufferBackPressure", [&]() { return "bufferBackPressure = " + ParameterToStr(bufferBackPressure); }},
        {"logLabelMode", [&]() { return "logLabelMode = " + ParameterToStr(logLabelMode); }},
        {"enableTileOpFlow", [&]() { return "enableTileOpFlow = " + ParameterToStr(enableTileOpFlow); }},
    };
}
} // namespace CostModel
