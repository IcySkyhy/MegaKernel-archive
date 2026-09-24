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
 * \file SimObj.h
 * \brief
 */

#pragma once
#include <memory>

namespace CostModel {
class SimSys;
class SimObj {
public:
    virtual ~SimObj() = default;
    /* \brief build simulation object */
    virtual void Build() = 0;
    /* \brief Logic simulation methods */
    virtual void Step() = 0;
    /* \brief shadow buffer commits*/
    virtual void Xfer() = 0;
    /* \brief reset */
    virtual void Reset() = 0;
    /* \brief reset */
    virtual std::shared_ptr<SimSys> GetSim() = 0;
};
} // namespace CostModel
