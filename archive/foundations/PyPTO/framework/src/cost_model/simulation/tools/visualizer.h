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
 * \file visualizer.h
 * \brief
 */

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <stack>
#include <string>
#include <memory>
#include <set>
#include <map>
#include <unordered_map>

#include "cost_model/simulation/common/ISA.h"

namespace CostModel {
enum class Modulor {
    MODULOR_0 = 0,
    MODULOR_1 = 1,
    MODULOR_2,
    MODULOR_3,
    MODULOR_4,
    MODULOR_5,
    MODULOR_6,
    MODULOR_7,
    MODULOR_8,
    MODULOR_9,
    MODULOR_10,
    MODULOR_11,
    MODULOR_12,
    MODULOR_13,
    MODULOR_14,
    MODULOR_NUM,
};

class ModelVisualizer {
public:
    std::map<CostModel::MachineType, std::vector<std::string>> taskColorMap = {
        {MachineType::AIC, {"#1f77b4", "#87CEEB"}},
        {MachineType::AIV, {"#006400", "#2ca02c"}},
    };
    std::map<CostModel::MachineType, std::vector<std::string>> taskFontColorMap = {
        {MachineType::AIC, {"white", "black"}},
        {MachineType::AIV, {"white", "black"}},
    };

    void DrawTile(std::ofstream& os, TilePtr tensor, bool debug = false) const;
    void DrawTileOp(std::ofstream& os, TileOpPtr tileop, FunctionPtr func, bool debug = false) const;
    void DrawTask(std::ofstream& os, std::shared_ptr<Task> task, bool detail);
    void DrawFunction(FunctionPtr func, const std::string& outdir, bool debug = false) const;
    void DebugFunction(FunctionPtr func, std::unordered_map<int, TilePtr>& tiles,
                       std::unordered_map<int, TileOpPtr>& tileOps, const std::string& outdir) const;
    void DrawTasks(const TaskMap& taskMap, bool drawDetail, std::string outPath);
    std::string GetColor(uint64_t color) const;
    std::string GetReverseColor(uint64_t color) const;
    std::string GetTaskColor(MachineType type, uint64_t taskId);
    std::string GetTaskFontColor(MachineType type, uint64_t taskId);
};

} // namespace CostModel
