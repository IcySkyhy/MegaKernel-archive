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
 * \file costmodel_utils.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/common/CommonType.h"

namespace CostModel {

class ModelData {
public:
    std::map<uint64_t, uint64_t> taskTime;
    std::vector<uint64_t> functionTime;
};

class AiCoreModel {
public:
    virtual ~AiCoreModel() = default;
    virtual void InitData(int coreIdx, int64_t funcdata) = 0;
    virtual void SendTask(int coreIdx, uint64_t taskId) = 0;
};

} // namespace CostModel
