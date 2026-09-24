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
 * \file ParseInput.h
 * \brief
 */

#ifndef PARSEINPUT_H
#define PARSEINPUT_H

#pragma once

#include <iostream>
#include <fstream>

#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "interface/function/function.h"

namespace CostModel {
class ParseInput {
public:
    static void ParseJson(std::shared_ptr<CostModel::SimSys> sim, const std::string& jsonPath);
    static bool FilterOpcode(std::string& opcode);
    static void BuildTile(npu::tile_fwk::Function& func, std::shared_ptr<npu::tile_fwk::LogicalTensor> logicalTensor,
                          TilePtr tile);
    static void GetTileAllocSeq(const std::vector<Operation*>& operationList, FunctionPtr func);
    static void BuildFunction(std::shared_ptr<CostModel::SimSys> sim, npu::tile_fwk::Function* parentFunc,
                              FunctionPtr func);
    static void BuildFunctionInvoke(FunctionPtr root, std::shared_ptr<CostModel::SimSys> sim);
    static void CheckInOutCast(FunctionPtr func);
    static void CheckTile(FunctionPtr func);
    static void CheckTileOp(FunctionPtr func);
    static void CheckFunction(npu::tile_fwk::Function* parentFunc, FunctionPtr func);
    static void ParseFunction(std::shared_ptr<CostModel::SimSys> sim, std::vector<npu::tile_fwk::Function*>& inputFuncs,
                              bool topoFromRootFunc);
    static void ParseSingleFunction(std::shared_ptr<CostModel::SimSys> sim, npu::tile_fwk::Function* func);
    static void ParseFixedLatencyTask(std::shared_ptr<CostModel::SimSys> sim, std::string const& path);
    void ParseJsonConfig(std::string const& path, std::vector<std::string>& cfg) const;
    void ParseConfig(std::string const& path, std::vector<std::string>& cfg) const;
    void ParseCalendarJson(std::shared_ptr<CostModel::SimSys> sim, const std::string& jsonPath) const;
    static void ParseTopoJson(std::string path, std::deque<TaskMap>& taskMapQueue);
    static void ParseReplayInfoJson(const std::string& path,
                                    std::unordered_map<uint64_t, std::deque<ReplayTaskEntry>>& replayTasksInfoMap);
};
} // namespace CostModel
#endif
