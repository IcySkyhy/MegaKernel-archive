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
 * \file backend.h
 * \brief
 */

#pragma once

#include <sys/types.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "cost_model/simulation/CostModelInterface.h"

namespace npu::tile_fwk {

class CostModelAgent {
public:
    bool getFunctionFromJson = false;
    std::string agentJsonPath = "";
    std::string topoJsonPath = "";
    void BuildCostModel();
    void SubmitToCostModel(Function* rootFunc);
    void SubmitSingleFuncToCostModel(Function* func);
    void SubmitLeafFunctionsToCostModel();

    uint64_t pos = 0;
    uint64_t seqPos = pos++;
    uint64_t taskIdPos = pos++;
    uint64_t rootIndexPos = pos++;
    uint64_t rootHashpos = pos++;
    uint64_t opmagicPos = pos++;
    uint64_t leafIndexPos = pos++;
    uint64_t funcHashPos = pos++;
    uint64_t coreTypePos = pos++;
    uint64_t psgIdPos = pos++;
    uint64_t wrapIdPos = pos++;
    uint64_t staticSuccCountPos = pos++;
    uint64_t succStartPos = pos++;
    uint64_t seqNumOffset = 32;
    Json ParseDynTopo(std::string& path);
    void SubmitTopo(std::string& path);
    void RunCostModel();
    void TerminateCostModel();
    void DebugSingleFunc(Function* func);
    void GetFunctionFromJson(const std::string& jsonPath);
    uint64_t GetLeafFunctionTimeCost(uint64_t hash);

private:
    std::shared_ptr<CostModel::CostModelInterface> costModel;
};

} // namespace npu::tile_fwk
