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
 * \file CoreStats.cpp
 * \brief
 */

#include "cost_model/simulation/statistics/CoreStats.h"
#include <utility>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

void CoreStats::Reset()
{
    completedTaskNum = 0;
    retiredTileOpNum = 0;
    retiredTileAllocNum = 0;
    totalPipeUseCycles = std::map<int, uint64_t>();
    intervalPipeUseCycles = std::map<int, uint64_t>();
}

void CoreStats::Report(std::string& name)
{
    rpt->ReportTitle("CoreMachine " + name + " Statistics");
    rpt->ReportVal("Completed Packet Count", completedTaskNum);
    rpt->ReportVal("Retire Tile Alloc Count", retiredTileAllocNum);
    rpt->ReportVal("Retire Tile Operation Count", retiredTileOpNum);
}
