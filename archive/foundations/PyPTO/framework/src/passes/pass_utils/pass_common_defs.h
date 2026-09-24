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
 * \file pass_common_defs.h
 * \brief
 */

#ifndef TILE_FWK_PASS_PASS_COMMON_DEFS_H_
#define TILE_FWK_PASS_PASS_COMMON_DEFS_H_
#include <vector>
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
enum class Opcode;
struct AssembleOp {
    MemoryType from;
    std::vector<int64_t> toOffset;
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    const Operation* originOp = nullptr;
    Opcode opcode;
};

struct ViewOp {
    MemoryType toType;
    std::vector<int64_t> fromOffset;
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    const Operation* originOp = nullptr;
};

struct CopyInOutOp {
    MemoryType from;
    std::vector<OpImmediate> Offset;
    std::vector<OpImmediate> shape;
    std::vector<OpImmediate> rawShape;
    std::vector<OpImmediate> fromDynValidShape;
    LogicalTensorPtr input;
    LogicalTensorPtr output;
};

class ReshapeOp {
public:
    ReshapeOp(LogicalTensorPtr aInput, LogicalTensorPtr aOutput, const Operation* opPtr = nullptr)
        : input(aInput), output(aOutput), originOpPtr(opPtr)
    {}
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    const Operation* originOpPtr; // 指向被拆分之前的op_reshape，用于获取可能需要继承的属性
    std::vector<std::vector<SymbolicScalar>> dynValidShapes;
};
} // namespace tile_fwk
} // namespace npu

#endif // TILE_FWK_PASS_PASS_COMMON_DEFS_H_
