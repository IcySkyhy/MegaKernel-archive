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
 * \file dev_encode_operation.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/runtime_slot.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"

namespace npu::tile_fwk::dynamic {
constexpr int INVALID_WRAP_TASK_HUB_OP_IDX = -1;
struct DevAscendOperationOperandInfo {
    int tensorIndex{0};
    int staticOffsetAttrBeginIndex{0};
    int staticShapeAttrBeginIndex{0};
    int staticRawShapeAttrBeginIndex{0};
    int staticValidShapeAttrBeginIndex{0};
    DevAscendOperationOperandInfo() {}
    DevAscendOperationOperandInfo(int tTensorIndex, int tStaticAttrBeginIndex, int tStaticDim)
        : tensorIndex(tTensorIndex),
          staticOffsetAttrBeginIndex(tStaticAttrBeginIndex),
          staticShapeAttrBeginIndex(tStaticAttrBeginIndex + tStaticDim),
          staticRawShapeAttrBeginIndex(staticShapeAttrBeginIndex + tStaticDim),
          staticValidShapeAttrBeginIndex(staticRawShapeAttrBeginIndex + tStaticDim)
    {}
    int GetDim() const { return staticShapeAttrBeginIndex - staticOffsetAttrBeginIndex; }
};

struct DevAscendOperation {
    DevLocalVector<DevAscendOperationOperandInfo> ioperandList;
    DevLocalVector<DevAscendOperationOperandInfo> ooperandList;
    DevLocalVector<SymInt> attrList; // opattr[0] -> hash
    int32_t stitchIndex;
    uint32_t depGraphPredCount;
    DevLocalVector<int> depGraphSuccList;
    DevLocalVector<int> depGraphCopyOutResolveSuccIndexList;
    uint64_t debugOpmagic; // DEBUG_ONLY
};

struct DevAscendFunctionCallOperandUse {
    int operationIdx{-1};
    int offsetAttrIdx{-1};
    SymInt* cachedAttrBase{nullptr};
    CellMatchOpType opType{CellMatchOpType::READ};
    Opcode opCode{Opcode::OP_UNKNOWN};
    int stitchOpIdx{INVALID_WRAP_TASK_HUB_OP_IDX};

    DevAscendFunctionCallOperandUse() = default;
    DevAscendFunctionCallOperandUse(int operationIdx_, int offsetAttrIdx_,
                                    CellMatchOpType opType_ = CellMatchOpType::READ,
                                    Opcode opCode_ = Opcode::OP_UNKNOWN)
        : operationIdx(operationIdx_), offsetAttrIdx(offsetAttrIdx_), opType(opType_), opCode(opCode_)
    {}
};

struct DevAscendFunctionIncast {
    int tensorIndex;
    DevLocalVector<int> fromSlotList;

    int dim;
    DevLocalVector<DevAscendFunctionCallOperandUse> consumerList;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverConsumerList;

    DevCellMatchTableDesc cellMatchTableDesc;

    DevLocalVector<uint32_t> stitchPolicyFullCoverConsumerAllOpIdxList;
};

struct DevAscendFunctionOutcast {
    int tensorIndex;
    DevLocalVector<int> toSlotList;

    int dim;
    RuntimeSlotDesc desc;

    DevLocalVector<DevAscendFunctionCallOperandUse> producerConsumerList;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevLocalVector<uint32_t> cellMatchRuntimeFullUpdateTable;

    int stitchPolicyFullCoverProducerHubOpIdx;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList;
    DevLocalVector<uint32_t> stitchPolicyFullCoverAllOpIdxList;
    int exprListIndex;
};

struct InoutOperationAttr {
    int dim;
    std::vector<DevAscendFunctionCallOperandUse> useList;
    int cellMatchSize;

    std::vector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList;
    std::vector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverConsumerList;
    int stitchPolicyFullCoverProducerHubOpIdx;

    DevCellMatchTableDesc cellMatchTableDesc;

    std::vector<uint32_t> useOpList;
    int bindTensorExprIndex{-1};
};

struct DevAscendFunctionDuppedOperation {
    uint32_t size;
    uint32_t predCountBase;
    uint32_t stitchBase;
    uint32_t stitchCount;
};

struct DevAscendFunctionDuppedVector {
    uint32_t size;
    uint32_t base;
};
} // namespace npu::tile_fwk::dynamic
