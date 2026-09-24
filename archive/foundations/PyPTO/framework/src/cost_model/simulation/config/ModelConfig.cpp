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
 * \file ModelConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/ModelConfig.h"

using namespace std;

namespace CostModel {
ModelConfig::ModelConfig()
{
    Config::prefix = "Model";
    Config::dispatcher = {
        {"statisticReportToFile", [&](string v) { statisticReportToFile = ParseBoolean(v); }},
        {"heartInterval", [&](string v) { heartInterval = ParseInteger(v); }},
        {"drawPngThresholdCycle", [&](string v) { drawPngThresholdCycle = ParseInteger(v); }},
        {"testDeadLock", [&](string v) { testDeadLock = ParseBoolean(v); }},
        {"startFunctionLabel", [&](string v) { startFunctionLabel = ParseString(v); }},
        {"useOOOPassSeq", [&](string v) { useOOOPassSeq = ParseBoolean(v); }},
        {"genCalendarScheduleCpp", [&](string v) { genCalendarScheduleCpp = ParseBoolean(v); }},
        {"deviceMachineNumber", [&](string v) { deviceMachineNumber = ParseInteger(v); }},
        {"aicpuMachineNumber", [&](string v) { aicpuMachineNumber = ParseInteger(v); }},
        {"aicpuMachineSmtNum", [&](string v) { aicpuMachineSmtNum = ParseInteger(v); }},
        {"coreMachineNumberPerAICPU", [&](string v) { coreMachineNumberPerAICPU = ParseInteger(v); }},
        {"cubeMachineNumberPerAICPU", [&](string v) { cubeMachineNumberPerAICPU = ParseInteger(v); }},
        {"vecMachineNumberPerAICPU", [&](string v) { vecMachineNumberPerAICPU = ParseInteger(v); }},
        {"coreMachineSmtNum", [&](string v) { coreMachineSmtNum = ParseInteger(v); }},
        {"cubeVecMixMode", [&](string v) { cubeVecMixMode = ParseBoolean(v); }},
        {"mteUseL2Cache", [&](string v) { mteUseL2Cache = ParseBoolean(v); }},
        {"functionCacheSize", [&](string v) { functionCacheSize = ParseInteger(v); }},
        {"deviceArch", [&](string v) { deviceArch = ParseString(v); }},
        {"simulationFixedLatencyTask", [&](string v) { simulationFixedLatencyTask = ParseBoolean(v); }},
        {"fixedLatencyTaskInfoPath", [&](string v) { fixedLatencyTaskInfoPath = ParseString(v); }},
        {"fixedLatencyTimeConvert", [&](string v) { fixedLatencyTimeConvert = ParseInteger(v); }},
        {"pipeBoardVibration", [&](string v) { pipeBoardVibration = ParseInteger(v); }},
        {"calendarMode", [&](string v) { calendarMode = ParseInteger(v); }},
        {"calendarFile", [&](string v) { calendarFile = ParseString(v); }},
    };

    Config::recorder = {
        {"statisticReportToFile", [&]() { return "statisticReportToFile = " + ParameterToStr(statisticReportToFile); }},
        {"heartInterval", [&]() { return "heartInterval = " + ParameterToStr(heartInterval); }},
        {"drawPngThresholdCycle", [&]() { return "drawPngThresholdCycle = " + ParameterToStr(drawPngThresholdCycle); }},
        {"testDeadLock", [&]() { return "testDeadLock = " + ParameterToStr(testDeadLock); }},
        {"startFunctionLabel", [&]() { return "startFunctionLabel = " + startFunctionLabel; }},
        {"useOOOPassSeq", [&]() { return "useOOOPassSeq = " + ParameterToStr(useOOOPassSeq); }},
        {"genCalendarScheduleCpp",
         [&]() { return "genCalendarScheduleCpp = " + ParameterToStr(genCalendarScheduleCpp); }},
        {"deviceMachineNumber", [&]() { return "deviceMachineNumber = " + ParameterToStr(deviceMachineNumber); }},
        {"aicpuMachineNumber", [&]() { return "aicpuMachineNumber = " + ParameterToStr(aicpuMachineNumber); }},
        {"aicpuMachineSmtNum", [&]() { return "aicpuMachineSmtNum = " + ParameterToStr(aicpuMachineSmtNum); }},
        {"coreMachineNumberPerAICPU",
         [&]() { return "coreMachineNumberPerAICPU = " + ParameterToStr(coreMachineNumberPerAICPU); }},
        {"cubeMachineNumberPerAICPU",
         [&]() { return "cubeMachineNumberPerAICPU = " + ParameterToStr(cubeMachineNumberPerAICPU); }},
        {"vecMachineNumberPerAICPU",
         [&]() { return "vecMachineNumberPerAICPU = " + ParameterToStr(vecMachineNumberPerAICPU); }},
        {"coreMachineSmtNum", [&]() { return "coreMachineSmtNum = " + ParameterToStr(coreMachineSmtNum); }},
        {"cubeVecMixMode", [&]() { return "cubeVecMixMode = " + ParameterToStr(cubeVecMixMode); }},
        {"mteUseL2Cache", [&]() { return "mteUseL2Cache = " + ParameterToStr(mteUseL2Cache); }},
        {"functionCacheSize", [&]() { return "functionCacheSize = " + ParameterToStr(functionCacheSize); }},
        {"deviceArch", [&]() { return "deviceArch = " + deviceArch; }},
        {"simulationFixedLatencyTask",
         [&]() { return "simulationFixedLatencyTask = " + ParameterToStr(simulationFixedLatencyTask); }},
        {"fixedLatencyTaskInfoPath", [&]() { return "fixedLatencyTaskInfoPath = " + fixedLatencyTaskInfoPath; }},
        {"fixedLatencyTimeConvert",
         [&]() { return "fixedLatencyTimeConvert = " + ParameterToStr(fixedLatencyTimeConvert); }},
        {"pipeBoardVibration", [&]() { return "pipeBoardVibration = " + ParameterToStr(pipeBoardVibration); }},
        {"calendarMode", [&]() { return "calendarMode = " + ParameterToStr(calendarMode); }},
        {"calendarFile", [&]() { return "calendarFile = " + calendarFile; }},
    };
}
} // namespace CostModel
