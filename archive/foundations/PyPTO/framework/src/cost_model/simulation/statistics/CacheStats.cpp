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
 * \file CacheStats.cpp
 * \brief
 */

#include "cost_model/simulation/statistics/CacheStats.h"
#include <iostream>

void CacheStats::Reset()
{
    totalInsertNum = 0;
    totalEvictNum = 0;
    totalQueryNum = 0;
    totalHitNum = 0;
    totalMissNum = 0;
    totalReadNum = 0;
    totalWriteNum = 0;
    totalResponseLatency = 0;
}

void CacheStats::Report(std::string& name)
{
    rpt->ReportTitle(name + " Cache Statistics");
    rpt->ReportVal("Total Insert Count", totalInsertNum);
    rpt->ReportVal("Total Evict Count", totalEvictNum);
    rpt->ReportVal("Total Query Count", totalQueryNum);
    rpt->ReportValAndPct("Total Hit Count", totalHitNum, totalQueryNum);
    rpt->ReportValAndPct("Total Miss Count", totalMissNum, totalQueryNum);
    rpt->ReportVal("Total Read Count", totalReadNum);
    rpt->ReportVal("Total Write Count", totalWriteNum);

    float avgRespLatency = ((totalQueryNum != 0) ? (float(totalResponseLatency) / totalQueryNum) : float(0.0));
    rpt->ReportVal("Average Response Latency", avgRespLatency);
}
