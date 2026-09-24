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
 * \file dev_encode_function_param.h
 * \brief Encode parameter structures for DevAscendFunction.
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
struct CceCodeInfo;
class Function;
class IncastOutcastLink;
class IncastOutcastSlot;
class SymbolicSymbolTable;
class SymbolicExpressionTable;
} // namespace npu::tile_fwk

namespace npu::tile_fwk::dynamic {

struct DevAscendFunctionPredInfo {
    uint64_t totalZeroPred;
    uint64_t totalZeroPredAIV;
    uint64_t totalZeroPredAIC;
    uint64_t totalZeroPredHub;
    uint64_t totalZeroPredAicpu;
};

struct EncodeDevAscendFunctionParam {
    /* The following are common parameter */
    std::unordered_map<uint64_t, int> calleeHashIndexDict;
    std::vector<CceCodeInfo> cceCodeInfoList;
    const SymbolicSymbolTable* symbolTable;
    const IncastOutcastLink* inoutLink;

    /* The following are per function parameter */
    const SymbolicExpressionTable* expressionTable;
    const IncastOutcastSlot* slot;
    Function* devRoot;
    std::vector<RuntimeSlotDesc> outcastDescList;
    std::vector<int> assembleSlotList;
};

} // namespace npu::tile_fwk::dynamic
