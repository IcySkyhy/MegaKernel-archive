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
 * \file machine_utils.h
 * \brief
 */

#ifndef MACHINE_UTILS_H
#define MACHINE_UTILS_H
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;

struct AicpuOpConfig {
    std::string functionName;
    std::string kernelSo;
    std::string opKernelLib;
    std::string computeCost = "100";
    std::string engine = "DNN_VM_AICPU";
    std::string flagAsync = "False";
    std::string flagPartial = "False";
    std::string userDefined = "False";
    std::string opType;
};

void GenAicpuOpInfoJson(Json& opConfigJson, const std::vector<AicpuOpConfig>& opConfigs);
#endif
