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
 * \file pass_type.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <map>
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
enum class PassType : int32_t {
    TYPE_INVALID = -1,
    TYPE_TENSOR_GRAPH = 0,
    TYPE_TILE_GRAPH = 1,
    TYPE_BLOCK_GRAPH = 2,
    TYPE_BOTTOM
};

enum class PassName {
    LOOP_UNROLL,
    INFER_TENSOR_FORMAT,
    AUTO_CAST,
    REMOVE_REDUNDANT_RESHAPE,
    INFER_MEMORY_CONFLICT,
    REMOVE_UNDRIVEN_VIEW,
    EXPAND_FUNCTION,
    DUPLICATE_OP,
    MERGE_VIEW_ASSEMBLE,
    SPLIT_RESHAPE,
    SPLIT_RAW_TENSOR,
    SPLIT_LARGE_FANOUT_TENSOR,
    ASSIGN_MEMORY_TYPE,
    INFER_DISCONTINUOUS_INPUT,
    REMOVE_REDUNDANT_OP,
    INSERT_OP_FOR_VIEWASSEMBLE,
    PROCESS_ATOMIC,
    GRAPH_PARTITION,
    REDUCE_COPY_MERGE,
    N_BUFFER_MERGE,
    INTRA_SUBGRAPH_ADAPTER,
    GENERATE_MOVE_OP,
    COMMON_OPERATION_ELIMINATE,
    L1_COPY_IN_REUSE_MERGE,
    AXIS_COMBINE,
    PAD_LOCAL_BUFFER,
    REMOVE_UNALIGNED_RESHAPE,
    REPLACE_TENSOR,
    PRE_GRAPH_PROCESS,
    INFER_DYN_SHAPE,
    SUBGRAPH_TO_FUNCTION,
    INFER_PARAM_INDEX,
    SRC_DST_BUFFER_MERGE,
    ADD_ALLOC,
    OOO_SCHEDULE,
    GLOBAL_MEMORY_REUSE,
    REMOVE_ALLOC,
    COPY_OUT_RESOLVE,
    INSERT_SYNC,
    MIX_SUBGRAPH_SPLIT,
    LAST_USE_MARK,
    CODEGEN_PREPROC,
    DYN_ATTR_TO_STATIC,
    LOOPAXES_PROC,
    TUNE_TILEOP_SEQ_FOR_VF,
    TUNE_SYNC_FOR_VF,
    NOT_DEFINED
};

inline const std::map<PassName, const char*> kPassNameStringMap = {
    {PassName::LOOP_UNROLL, "LoopUnroll"},
    {PassName::INFER_TENSOR_FORMAT, "InferTensorFormat"},
    {PassName::AUTO_CAST, "AutoCast"},
    {PassName::REMOVE_REDUNDANT_RESHAPE, "RemoveRedundantReshape"},
    {PassName::INFER_MEMORY_CONFLICT, "InferMemoryConflict"},
    {PassName::REMOVE_UNDRIVEN_VIEW, "RemoveUndrivenView"},
    {PassName::EXPAND_FUNCTION, "ExpandFunction"},
    {PassName::DUPLICATE_OP, "DuplicateOp"},
    {PassName::MERGE_VIEW_ASSEMBLE, "MergeViewAssemble"},
    {PassName::SPLIT_RESHAPE, "SplitReshape"},
    {PassName::SPLIT_RAW_TENSOR, "SplitRawTensor"},
    {PassName::SPLIT_LARGE_FANOUT_TENSOR, "SplitLargeFanoutTensor"},
    {PassName::ASSIGN_MEMORY_TYPE, "AssignMemoryType"},
    {PassName::INFER_DISCONTINUOUS_INPUT, "InferDiscontinuousInput"},
    {PassName::REMOVE_REDUNDANT_OP, "RemoveRedundantOp"},
    {PassName::INSERT_OP_FOR_VIEWASSEMBLE, "InsertOpForViewAssemble"},
    {PassName::PROCESS_ATOMIC, "ProcessAtomic"},
    {PassName::GRAPH_PARTITION, "GraphPartition"},
    {PassName::REDUCE_COPY_MERGE, "ReduceCopyMerge"},
    {PassName::N_BUFFER_MERGE, "NBufferMerge"},
    {PassName::INTRA_SUBGRAPH_ADAPTER, "IntraSubgraphAdapter"},
    {PassName::GENERATE_MOVE_OP, "GenerateMoveOp"},
    {PassName::COMMON_OPERATION_ELIMINATE, "CommonOperationEliminate"},
    {PassName::L1_COPY_IN_REUSE_MERGE, "L1CopyInReuseMerge"},
    {PassName::AXIS_COMBINE, "AxisCombine"},
    {PassName::PAD_LOCAL_BUFFER, "PadLocalBuffer"},
    {PassName::REMOVE_UNALIGNED_RESHAPE, "RemoveUnalignedReshape"},
    {PassName::REPLACE_TENSOR, "ReplaceTensor"},
    {PassName::PRE_GRAPH_PROCESS, "PreGraphProcess"},
    {PassName::INFER_DYN_SHAPE, "InferDynShape"},
    {PassName::SUBGRAPH_TO_FUNCTION, "SubgraphToFunction"},
    {PassName::INFER_PARAM_INDEX, "InferParamIndex"},
    {PassName::SRC_DST_BUFFER_MERGE, "SrcDstBufferMerge"},
    {PassName::ADD_ALLOC, "AddAlloc"},
    {PassName::OOO_SCHEDULE, "OoOSchedule"},
    {PassName::GLOBAL_MEMORY_REUSE, "GlobalMemoryReuse"},
    {PassName::REMOVE_ALLOC, "RemoveAlloc"},
    {PassName::COPY_OUT_RESOLVE, "CopyOutResolve"},
    {PassName::INSERT_SYNC, "InsertSync"},
    {PassName::MIX_SUBGRAPH_SPLIT, "MixSubgraphSplit"},
    {PassName::LAST_USE_MARK, "LastUseMark"},
    {PassName::CODEGEN_PREPROC, "CodegenPreproc"},
    {PassName::DYN_ATTR_TO_STATIC, "DynAttrToStatic"},
    {PassName::LOOPAXES_PROC, "LoopaxesProc"},
    {PassName::TUNE_TILEOP_SEQ_FOR_VF, "TuneTileOpSeqForVF"},
    {PassName::TUNE_SYNC_FOR_VF, "TuneSyncForVF"},
    {PassName::NOT_DEFINED, "NotDefined"},
};

inline const char* PassNameStr(PassName name)
{
    auto it = kPassNameStringMap.find(name);
    if (it != kPassNameStringMap.end()) {
        return it->second;
    }
    ASSERT(FunctionErr::FUNCTION_SPECIAL_STRUCTURE, false) << "[PassDependency][Manager][ERROR]: PassName not defined.";
    return "Invalid";
}

inline std::ostream& operator<<(std::ostream& os, PassName name) { return os << PassNameStr(name); }
} // namespace npu::tile_fwk
