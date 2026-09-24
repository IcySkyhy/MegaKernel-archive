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
 * \file graph_optimization.h
 * \brief
 */

#ifndef PASS_GRAPH_OPTIM_H_
#define PASS_GRAPH_OPTIM_H_
#include "passes/tile_graph_pass/graph_optimization/merge_view_assemble.h"
#include "passes/tile_graph_pass/graph_optimization/remove_redundant_op.h"
#include "passes/tile_graph_pass/graph_optimization/split_raw.h"
#include "passes/tile_graph_pass/graph_optimization/split_large_fanout_tensor.h"
#include "passes/tile_graph_pass/graph_optimization/process_atomic.h"
#include "passes/tile_graph_pass/graph_optimization/infer_discontinuous_input.h"
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"
#include "passes/tile_graph_pass/graph_optimization/duplicate_op.h"
#include "passes/tile_graph_pass/graph_optimization/insert_op_for_viewassemble.h"
#endif
