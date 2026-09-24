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
 * \file Reporter.h
 * \brief
 */

#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <map>

namespace CostModel {

constexpr int TOTAL_WIDTH = 60;
constexpr int NAME_WIDTH = 50;
constexpr int VAL_WIDTH = 10;
constexpr int PCT_WIDTH = 7;
constexpr int INDENT_WIDTH = 2;

class Reporter {
    std::ofstream fout;

public:
    static const int floatPrec = 2;
    static const int basePercent = 100;
    Reporter();

    static void PrintName(std::string const& name, uint32_t len);
    static void ReportTitle(const std::string& title);
    static void ReportMap(const std::string& name, std::map<uint64_t, uint64_t>& vals);
    static void ReportMapAndPct(const std::string& name, std::map<int, uint64_t>& vals, const uint64_t& baseVal);
    static void ReportMapsAndPct(const std::string& name, std::map<int, uint64_t>& vals,
                                 std::map<int, uint64_t>& baseVals);
    static void ReportValWithLvl(const std::string& name, uint64_t val, uint32_t level);
    static void ReportValWithLvl(const std::string& name, float val, uint32_t level);
    static void ReportValWithLvl(const std::string& name, double val, uint32_t level);
    static void ReportVal(const std::string& name, uint64_t val);
    static void ReportVal(const std::string& name, float val);
    static void ReportVal(const std::string& name, double val);
    static void ReportAvg(const std::string& name, uint64_t numerator, uint64_t denominator);
    static void ReportAvg(const std::string& name, float numerator, float denominator);
    static void ReportPctWithLvl(const std::string& name, float rate, uint32_t level);
    static void ReportPct(const std::string& name, uint64_t numerator, uint64_t denominator);
    static void ReportPct(const std::string& name, float numerator, float denominator);
    static void ReportPct(const std::string& name, float rate);
    static void ReportValAndPctWithLvl(const std::string& name, uint64_t numerator, uint64_t denominator,
                                       uint32_t level);
    static void ReportValAndPctWithLvl(const std::string& name, float numerator, uint64_t denominator, uint32_t level);
    static void ReportValAndPct(const std::string& name, uint64_t numerator, uint64_t denominator);
    static void ReportValAndPct(const std::string& name, float numerator, uint64_t denominator);
    static void ReportValAndPctFlWithLvl(const std::string& name, double numerator, double denominator, uint32_t level);
    static void ReportValAndPctFl(const std::string& name, double numerator, double denominator);
    static void ReportHexCounter(const std::string& name, uint64_t pc, uint64_t counter);
    static void ReportStallLoc(const std::string& name, uint64_t localBpc, uint64_t localTpc, uint64_t peerBpc,
                               uint64_t val);
    std::streambuf* ReportSetOutStreamFile(const std::string& fileName);
    std::streambuf* ReportSetOutStreamFile(const std::string& fileName, bool isApp);

    void ReportResetOutStreamCout(std::streambuf* pOld);
};

} // namespace CostModel
