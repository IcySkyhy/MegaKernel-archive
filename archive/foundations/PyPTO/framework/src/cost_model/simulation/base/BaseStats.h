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
 * \file BaseStats.h
 * \brief
 */

#ifndef BASESTATS_H
#define BASESTATS_H

#pragma once

#include "cost_model/simulation/base/Reporter.h"

namespace CostModel {

class BaseStats {
public:
    Reporter* rpt;
    BaseStats() : rpt(nullptr) {};
    virtual ~BaseStats() = default;
    explicit BaseStats(Reporter* r) : rpt(r) {};
    virtual void Reset() = 0;
    virtual void Report(std::string& name) = 0;
};
} // namespace CostModel
#endif
