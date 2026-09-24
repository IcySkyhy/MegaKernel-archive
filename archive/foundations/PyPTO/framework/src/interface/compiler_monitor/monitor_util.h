/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdio>
#include <string>
#include <sstream>
#include <iomanip>

namespace npu::tile_fwk {

inline std::string FormatElapsed(double seconds)
{
    if (seconds < 60.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << seconds << "s";
        return oss.str();
    }

    int total_sec = static_cast<int>(seconds);
    int min = total_sec / 60;
    int sec = total_sec % 60;

    std::ostringstream oss;
    if (min >= 60) {
        int hour = min / 60;
        min = min % 60;
        oss << hour << "h " << min << "m " << sec << "s (" << total_sec << "s)";
    } else {
        oss << min << "min " << sec << "s (" << total_sec << "s)";
    }
    return oss.str();
}

const int MONITOR_LABEL_WIDTH = 21;
const int MONITOR_STAGE_NAME_WIDTH = 18;
const int MONITOR_ELAPSED_WIDTH = 8;

inline std::string PadRight(const std::string& str, int width)
{
    if (static_cast<int>(str.size()) >= width) {
        return str;
    }
    return str + std::string(width - str.size(), ' ');
}

inline std::string PadElapsed(const std::string& elapsed) { return PadRight(elapsed, MONITOR_ELAPSED_WIDTH); }

inline std::string PadLabel(const std::string& label) { return PadRight(label, MONITOR_LABEL_WIDTH); }

inline std::string PadStageName(const std::string& stageName) { return PadRight(stageName, MONITOR_STAGE_NAME_WIDTH); }

inline std::string FormatOpCount(int opCount)
{
    if (opCount < 0) {
        return "N/A";
    }
    return std::to_string(opCount);
}

} // namespace npu::tile_fwk
