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
 * \file graph_constraint.h
 * \brief
 */

#ifndef GRAPH_CONSTRAINT_H
#define GRAPH_CONSTRAINT_H
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "passes/tile_graph_pass/graph_constraint/replace_tensor.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/pre_graph.h"
#include "passes/tile_graph_pass/graph_constraint/remove_unaligned_reshape_op.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
#endif
