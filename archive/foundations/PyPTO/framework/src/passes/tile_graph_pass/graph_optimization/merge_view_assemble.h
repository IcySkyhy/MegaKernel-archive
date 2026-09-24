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
 * \file merge_view_assemble.h
 * \brief
 */

#ifndef PASS_MERGE_VIEW_ASSEMBLE_H_
#define PASS_MERGE_VIEW_ASSEMBLE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
class MergeViewAssemble : public Pass {
public:
    MergeViewAssemble() : Pass("MergeViewAssemble") {}
    ~MergeViewAssemble() override = default;

private:
    Status PreCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
};
} // namespace npu::tile_fwk
#endif // PASS_MERGE_VIEW_ASSEMBLE_H_
