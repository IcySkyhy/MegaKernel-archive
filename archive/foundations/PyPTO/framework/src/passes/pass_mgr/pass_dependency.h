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
 * \file pass_dependency.h
 * \brief
 */

#pragma once

#include "interface/utils/common.h"
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"
#include "passes/pass_interface/pass_type.h"

namespace npu::tile_fwk {
class PassDependency {
public:
    static PassDependency& Instance();

    Status CheckStrategyDependency(const std::string& strategyName, const std::vector<PassName>& passes);

private:
    PassDependency();
    ~PassDependency() = default;

    PassDependency(const PassDependency&) = delete;
    PassDependency& operator=(const PassDependency&) = delete;

    void RegisterPreDependencies();
    void RegisterSequenceDependencies();
    Status CheckSequenceDependency(size_t index, const std::string& strategyName, const std::vector<PassName>& passes);

private:
    std::unordered_map<PassName, std::vector<PassName>> preDependencies_;
    std::unordered_map<PassName, std::vector<PassName>> sequenceDependencies_;
};
} // namespace npu::tile_fwk
