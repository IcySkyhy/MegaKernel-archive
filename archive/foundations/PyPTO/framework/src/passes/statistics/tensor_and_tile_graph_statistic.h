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
 * \file tensor_and_tile_graph_statistic.h
 * \brief
 */

#ifndef TILE_FWK_PROGRAM_JUDGEMENT_H
#define TILE_FWK_PROGRAM_JUDGEMENT_H

#include <map>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include <nlohmann/json.hpp>

namespace npu {
namespace tile_fwk {
constexpr int CUDE_IOPERAND_NUM2 = 2;
constexpr int CUDE_IOPERAND_NUM3 = 3;
constexpr int DUMP_WIDTH = 4;
struct MetricData {
    uint64_t maxSize = 0;
    std::vector<int> maxNodesMagic;

    void UpdateMetricData(uint64_t size, int magic)
    {
        if (size > maxSize) {
            maxSize = size;
            maxNodesMagic.clear();
            maxNodesMagic.push_back(magic);
            return;
        }
        if (size == maxSize) {
            maxNodesMagic.push_back(magic);
        }
    }

    uint64_t GetMaxSize() { return maxSize; }
    std::vector<int>* GetMaxNodes() { return &maxNodesMagic; }
};

void HealthCheckTensorGraph(Function& function, const std::string& reportPath, const std::string& fileName);
void HealthCheckTileGraph(Function& function, const std::string& reportPath, const std::string& fileName);
void CalcOperatorInfo(Function& function, nlohmann::json& report);
void CalcTensorInfo(Function& function, nlohmann::json& report);
void GetOpConnectionMap(Function& function, std::vector<std::vector<int>>& inMap, std::vector<std::vector<int>>& outMap,
                        std::vector<bool>& actualMagic);
void TraversePathUp(const int parent, const std::vector<std::vector<int>>& outMap, std::vector<int>& layerMap);
void CalcGraphMetrics(const std::vector<std::vector<int>>& inMap, const std::vector<std::vector<int>>& outMap,
                      const std::vector<bool>& actualVertex, nlohmann::json& report);
} // namespace tile_fwk
} // namespace npu

#endif
