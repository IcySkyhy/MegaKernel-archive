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
 * \file pre_graph.h
 * \brief
 */

#ifndef PRE_GRAPH_PASS_H
#define PRE_GRAPH_PASS_H

#include "passes/tile_graph_pass/graph_constraint/pre_graph/color_graph.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/cube_process.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/remove_redundant_assemble.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/set_boundary.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/set_copy_attr.h"

namespace npu::tile_fwk {
class PreGraphProcess : public Pass {
public:
    PreGraphProcess() : Pass("PreGraphProcess") {}
    ~PreGraphProcess() override = default;

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    void UpdateCopyOpIsCube(Operation& op) const;
};
} // namespace npu::tile_fwk
#endif // PRE_GRAPH_PASS_H
