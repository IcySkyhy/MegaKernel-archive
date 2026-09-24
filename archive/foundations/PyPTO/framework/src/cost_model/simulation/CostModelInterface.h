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
 * \file CostModelInterface.h
 * \brief
 */

#pragma once
#include <memory>
#include <vector>
#include "interface/function/function.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/tools/ParseInput.h"

namespace CostModel {
class CostModelInterface {
public:
    std::shared_ptr<CostModel::SimSys> sim = nullptr;
    CostModel::ParseInput parser;

    CostModelInterface() = default;
    ~CostModelInterface() = default;

    int BuildCostModel(std::vector<std::string>& inputConfigs);
    void GetInput(std::vector<npu::tile_fwk::Function*>& inputFuncs, bool topoFromRootFunc, std::string& startFuncName);
    void Submit(std::vector<npu::tile_fwk::Function*>& inputFuncs, bool topoFromRootFunc, std::string startFuncName);
    void SubmitSingleFunction(npu::tile_fwk::Function* func);
    void Run();
    void RunPerformance();
    void RunFunctional();
    void Report();
};
} // namespace CostModel
