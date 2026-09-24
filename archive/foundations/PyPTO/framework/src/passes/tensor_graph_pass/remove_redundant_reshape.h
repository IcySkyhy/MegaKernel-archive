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
 * \file remove_redundant_reshape.h
 * \brief
 */

#ifndef PASS_REMOVE_REDUNDEN_RESHAPE_H_
#define PASS_REMOVE_REDUNDEN_RESHAPE_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
class RemoveRedundantReshape : public Pass {
public:
    RemoveRedundantReshape() : Pass("RemoveRedundantReshape") {}
    ~RemoveRedundantReshape() override = default;

private:
    Status DefaultEnabledPreCheck(Function& function) override;
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status RemoveReshape(Function& function) const;
};
} // namespace npu::tile_fwk
#endif // PASS_REMOVE_REDUNDEN_RESHAPE_H_
