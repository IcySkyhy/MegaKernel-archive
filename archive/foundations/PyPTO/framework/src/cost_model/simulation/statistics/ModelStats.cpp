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
 * \file ModelStats.cpp
 * \brief
 */

#include "cost_model/simulation/statistics/ModelStats.h"

#include <iostream>

void ModelStats::Reset()
{
    cycles = 0;
    stepCount = 0;
    hostMachineNum = 0;
    deviceMachineNum = 0;
    aicpuMachineNum = 0;
    coreMachineNum = 0;
    cubeMachineNum = 0;
    vecMachineNum = 0;
    cvMixedCoreMachineNum = 0;
    pipeGroupNum = 0;
    totalFunctionNum = 0;
    totalFunctionCube = 0;
    totalFunctionVec = 0;
    totalFunctionMix = 0;
    totalFunctionTileOps = 0;
    coreUseCycles.clear();
}

void ModelStats::Report(std::string& name)
{
    rpt->ReportTitle(name);
    rpt->ReportVal("Total Cycles", cycles);
    rpt->ReportVal("Total Steps", stepCount);
    rpt->ReportVal("Host Machine Count", hostMachineNum);
    rpt->ReportVal("Device Machine Count", deviceMachineNum);
    rpt->ReportVal("AICPU Machine Count", aicpuMachineNum);
    rpt->ReportVal("Core Machine Count", coreMachineNum);
    rpt->ReportVal("|--Cube Count", cubeMachineNum);
    rpt->ReportVal("|--Vector Count", vecMachineNum);
    rpt->ReportVal("|--CVMIX Core Count", cvMixedCoreMachineNum);
    rpt->ReportVal("Pipe Groups Count", pipeGroupNum);

    rpt->ReportVal("Total Function Count", totalFunctionNum);
    rpt->ReportValAndPct("|--Cube Function", totalFunctionCube, totalFunctionNum);
    rpt->ReportValAndPct("|--Vector Function", totalFunctionVec, totalFunctionNum);
    rpt->ReportValAndPct("|--Mixed Function", totalFunctionMix, totalFunctionNum);

    float avgTileOpNum = ((totalFunctionNum != 0) ? (float(totalFunctionTileOps) / totalFunctionNum) : float(0.0));
    rpt->ReportVal("Average Function TileOp Num", avgTileOpNum);
}
