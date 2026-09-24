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
 * \file Reporter.cpp
 * \brief
 */

#include <iomanip>
#include <sstream>
#include <vector>
#include "cost_model/simulation/base/Reporter.h"

using namespace std;

namespace CostModel {

Reporter::Reporter() = default;

void Reporter::PrintName(string const& name, uint32_t len)
{
    if (len == 0) {
        cout << left << setw(NAME_WIDTH - 1) << setfill('.') << name << ':';
        return;
    }
    string output = std::string(INDENT_WIDTH * len, ' ') + "|--" + name;
    cout << left << setw(NAME_WIDTH - 1) << setfill('.') << output << ':';
}

void Reporter::ReportTitle(string const& title)
{
    int fillWidth = TOTAL_WIDTH - title.length();
    int leftWidth = fillWidth / 2;
    int rightWidth = fillWidth - leftWidth;
    cout << string(leftWidth, '=') << title << string(rightWidth, '=') << endl;
}

void Reporter::ReportMap(const std::string& name, std::map<uint64_t, uint64_t>& vals)
{
    cout << left << setw(NAME_WIDTH - 1) << setfill('.') << name << ':' << endl;

    vector<pair<string, string>> columns;
    vector<size_t> widths;
    size_t count = 0;

    // 收集数据并计算宽度
    for (const auto& val : vals) {
        string index = to_string(val.first);
        string value = to_string(val.second);
        columns.emplace_back(index, value);

        if (count >= widths.size()) {
            widths.push_back(max(index.length(), value.length()));
        } else {
            widths[count] = max(widths[count], max(index.length(), value.length()));
        }
        count++;
    }

    int displayWidth = 6; // 设置一行显示多少组数据

    for (size_t i = 0; i < columns.size(); i++) {
        if (i % displayWidth == 0) {
            if (i > 0) {
                cout << endl;
            }
            cout << " |";
        }

        // 计算居中对齐的空格
        string index = columns[i].first;
        size_t spaces = widths[i] - index.length();
        size_t leftSpaces = spaces / 2;
        size_t rightSpaces = spaces - leftSpaces;
        cout << " " << string(leftSpaces, ' ') << index << string(rightSpaces, ' ') << " |";

        // 如果是最后一组或已满6个，输出对应的值
        if ((i + 1) % displayWidth == 0 || i == columns.size() - 1) {
            cout << endl << " |";
            for (size_t j = i / displayWidth * displayWidth; j <= i; j++) {
                cout << " " << setw(int(widths[j])) << right << columns[j].second << " |";
            }
            cout << endl;
        }
    }
}

void Reporter::ReportMapAndPct(const std::string& name, std::map<int, uint64_t>& vals, const uint64_t& baseVal)
{
    cout << left << setw(NAME_WIDTH - 1) << setfill('.') << name << ':' << endl;

    vector<pair<string, string>> columns;
    vector<size_t> widths;
    size_t count = 0;
    for (const auto& val : vals) {
        string index = to_string(val.first);
        stringstream ss;
        ss << fixed << setprecision(floatPrec) << (float(val.second) / float(baseVal) * basePercent) << "%";
        string ratio = ss.str();
        columns.emplace_back(index, ratio);

        if (count >= widths.size()) {
            widths.push_back(max(index.length(), ratio.length()));
        } else {
            widths[count] = max(widths[count], max(index.length(), ratio.length()));
        }
        count++;
    }

    int displayWidth = 6; // 设置一行显示多少组数据

    for (size_t i = 0; i < columns.size(); i++) {
        if (i % displayWidth == 0) {
            if (i > 0) {
                cout << endl;
            }
            cout << " |";
        }

        string index = columns[i].first;
        size_t spaces = widths[i] - index.length();
        size_t leftSpaces = spaces / 2;
        size_t rightSpaces = spaces - leftSpaces;
        cout << " " << string(leftSpaces, ' ') << index << string(rightSpaces, ' ') << " |";

        if ((i + 1) % displayWidth == 0 || i == columns.size() - 1) {
            cout << endl << " |";
            for (size_t j = i / displayWidth * displayWidth; j <= i; j++) {
                cout << " " << setw(int(widths[j])) << right << columns[j].second << " |";
            }
            cout << endl;
        }
    }
}

void Reporter::ReportMapsAndPct(const std::string& name, std::map<int, uint64_t>& vals,
                                std::map<int, uint64_t>& baseVals)
{
    cout << left << setw(NAME_WIDTH - 1) << setfill('.') << name << ':' << endl;

    vector<pair<string, string>> columns;
    vector<size_t> widths;
    size_t count = 0;
    for (const auto& val : vals) {
        string index = to_string(val.first);
        stringstream ss;
        // 确保在base_vals中存在对应的键
        if (baseVals.find(val.first) != baseVals.end() && baseVals[val.first] != 0) {
            ss << fixed << setprecision(floatPrec) << (float(val.second) / float(baseVals[val.first]) * basePercent)
               << "%";
        } else {
            ss << "nan%";
        }
        string ratio = ss.str();
        columns.emplace_back(index, ratio);

        if (count >= widths.size()) {
            widths.push_back(max(index.length(), ratio.length()));
        } else {
            widths[count] = max(widths[count], max(index.length(), ratio.length()));
        }
        count++;
    }

    int displayWidth = 6; // 设置一行显示多少组数据

    for (size_t i = 0; i < columns.size(); i++) {
        if (i % displayWidth == 0) {
            if (i > 0) {
                cout << endl;
            }
            cout << " |";
        }

        string index = columns[i].first;
        size_t spaces = widths[i] - index.length();
        size_t leftSpaces = spaces / 2;
        size_t rightSpaces = spaces - leftSpaces;
        cout << " " << string(leftSpaces, ' ') << index << string(rightSpaces, ' ') << " |";

        if ((i + 1) % displayWidth == 0 || i == columns.size() - 1) {
            cout << endl << " |";
            for (size_t j = i / displayWidth * displayWidth; j <= i; j++) {
                cout << " " << setw(int(widths[j])) << right << columns[j].second << " |";
            }
            cout << endl;
        }
    }
}

void Reporter::ReportValWithLvl(const std::string& name, uint64_t val, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << dec << val << endl;
}

void Reporter::ReportValWithLvl(const std::string& name, float val, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << setiosflags(ios::fixed) << setprecision(floatPrec) << val
         << endl;
}

void Reporter::ReportValWithLvl(const std::string& name, double val, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << setiosflags(ios::fixed) << setprecision(floatPrec) << val
         << endl;
}

void Reporter::ReportVal(string const& name, uint64_t val) { ReportValWithLvl(name, val, 0); }

void Reporter::ReportVal(string const& name, float val) { ReportValWithLvl(name, val, 0); }

void Reporter::ReportVal(string const& name, double val) { ReportValWithLvl(name, val, 0); }

void Reporter::ReportAvg(string const& name, uint64_t numerator, uint64_t denominator)
{
    ReportAvg(name, static_cast<float>(numerator), static_cast<float>(denominator));
}

void Reporter::ReportAvg(string const& name, float numerator, float denominator)
{
    PrintName(name, 0);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << setiosflags(ios::fixed) << setprecision(floatPrec);
    if (denominator >= 0.0) {
        cout << numerator / denominator << endl;
    } else {
        cout << "nan" << endl;
    }
}

void Reporter::ReportPctWithLvl(const std::string& name, float rate, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ');
    cout << fixed << setprecision(floatPrec) << rate * basePercent << '%' << endl;
}

void Reporter::ReportPct(string const& name, uint64_t numerator, uint64_t denominator)
{
    ReportPct(name, static_cast<float>(numerator), static_cast<float>(denominator));
}

void Reporter::ReportPct(string const& name, float numerator, float denominator)
{
    PrintName(name, 0);
    cout << right << setw(VAL_WIDTH) << setfill(' ');
    if (denominator >= 0.0) {
        cout << fixed << setprecision(floatPrec) << numerator / denominator * basePercent << '%' << endl;
    } else {
        cout << "nan%" << endl;
    }
}

void Reporter::ReportPct(string const& name, float rate) { ReportPctWithLvl(name, rate, 0); }

void Reporter::ReportValAndPctWithLvl(const std::string& name, uint64_t numerator, uint64_t denominator, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << dec << numerator << ' ';
    cout << setw(PCT_WIDTH) << setfill(' ');
    if (denominator != 0) {
        cout << fixed << setprecision(floatPrec) << float(numerator) / float(denominator) * basePercent << '%' << endl;
    } else {
        cout << "nan%" << endl;
    }
}

void Reporter::ReportValAndPct(string const& name, uint64_t numerator, uint64_t denominator)
{
    ReportValAndPctWithLvl(name, numerator, denominator, 0);
}

void Reporter::ReportValAndPctWithLvl(const std::string& name, float numerator, uint64_t denominator, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << dec << numerator << ' ';
    cout << setw(PCT_WIDTH) << setfill(' ');
    if (denominator != 0) {
        cout << fixed << setprecision(floatPrec) << numerator / float(denominator) * basePercent << '%' << endl;
    } else {
        cout << "nan%" << endl;
    }
}

void Reporter::ReportValAndPct(const string& name, float numerator, uint64_t denominator)
{
    ReportValAndPctWithLvl(name, numerator, denominator, 0);
}

void Reporter::ReportValAndPctFlWithLvl(const std::string& name, double numerator, double denominator, uint32_t level)
{
    PrintName(name, level);
    cout << right << setw(VAL_WIDTH) << setfill(' ') << dec << numerator << ' ';
    cout << setw(PCT_WIDTH) << setfill(' ');
    if (denominator >= 0.0) {
        cout << fixed << setprecision(floatPrec) << float(numerator) / float(denominator) * basePercent << '%' << endl;
    } else {
        cout << "nan%" << endl;
    }
}

void Reporter::ReportValAndPctFl(string const& name, double numerator, double denominator)
{
    ReportValAndPctFlWithLvl(name, numerator, denominator, 0);
}

void Reporter::ReportHexCounter(const std::string& name, uint64_t pc, uint64_t counter)
{
    cout << left << name << setw((NAME_WIDTH - 1) - name.length()) << setfill('.') << hex << pc << ':';
    cout << right << setw(VAL_WIDTH) << setfill(' ') << dec << counter << endl;
}

void Reporter::ReportStallLoc(string const& name, uint64_t localBpc, uint64_t localTpc, uint64_t peerBpc, uint64_t val)
{
    stringstream ss;
    string newName;
    ss << name << "local_0x" << hex << localBpc << "_0x" << hex << localTpc << "---peer_0x" << hex << peerBpc;
    ss >> newName;
    cout << left << setw(NAME_WIDTH - 1) << setfill('.') << newName << ':';
    cout << right << setw(VAL_WIDTH) << setfill(' ') << setiosflags(ios::fixed) << setprecision(floatPrec) << val
         << endl;
}

std::streambuf* Reporter::ReportSetOutStreamFile(string const& fileName)
{
    if (fout.is_open()) {
        fout.close();
    }
    fout.open(fileName, ios::out | ios::trunc);
    /* return Store cout stream point */
    return cout.rdbuf(fout.rdbuf());
}

std::streambuf* Reporter::ReportSetOutStreamFile(string const& fileName, bool isApp)
{
    if (fout.is_open()) {
        fout.close();
    }

    if (isApp) {
        fout.open(fileName, ios::out | ios::app);
    } else {
        fout.open(fileName, ios::out | ios::ate);
    }

    /* return Store cout stream point */
    return cout.rdbuf(fout.rdbuf());
}

void Reporter::ReportResetOutStreamCout(std::streambuf* pOld)
{
    if (pOld == nullptr) {
        return;
    }
    /* Reset OutStream To cout */
    cout.rdbuf(pOld);
    if (fout.is_open()) {
        fout.close();
    }
}
} // namespace CostModel
