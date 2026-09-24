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
 * \file prior_scheduling.h
 * \brief
 */

#ifndef PASS_PRIOR_SCHEDULING_H_
#define PASS_PRIOR_SCHEDULING_H_

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class PriorScheduling : public Pass {
public:
    PriorScheduling() : Pass("PriorScheduling") {}
    ~PriorScheduling() override = default;
    Status RunOnFunction(Function& function) override;

private:
    void PriorSchedulingFunc(Function& function) const;
};
} // namespace npu::tile_fwk
#endif // PASS_PRIOR_SCHEDULING_H_
