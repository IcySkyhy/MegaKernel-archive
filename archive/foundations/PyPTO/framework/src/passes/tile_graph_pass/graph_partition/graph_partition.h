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
 * \file graph_partition.h
 * \brief Contains the GraphPartition Pass, which instantiates the specified GraphPartition Pass Algorithm.
 */

#ifndef PASS_GRAPH_PARTITION_H
#define PASS_GRAPH_PARTITION_H
#include "passes/tile_graph_pass/graph_partition/iso_partitioner.h"
#include "passes/tile_graph_pass/graph_partition/osp_partitioner.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include "passes/tile_graph_pass/graph_partition/l1_copy_reuse.h"
#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"

namespace npu::tile_fwk {

class GraphPartition : public Pass {
public:
    GraphPartition() : Pass("GraphPartition") {}
    ~GraphPartition() override = default;
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
};

} // namespace npu::tile_fwk
#endif
