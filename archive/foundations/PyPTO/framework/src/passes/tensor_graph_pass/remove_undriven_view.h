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
 * \file remove_undriven_view.h
 * \brief
 */

#pragma once

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

/*
    为AssembleSSA删除inplaceIdx来源的undriven的View
*/
class RemoveUndrivenView : public Pass {
public:
    RemoveUndrivenView() : Pass("RemoveUndrivenView") {}
    ~RemoveUndrivenView() override = default;

private:
    Status RunOnFunction(Function& function) override;
};
} // namespace npu::tile_fwk
