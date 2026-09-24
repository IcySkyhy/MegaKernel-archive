/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <functional>

#include "interface/function/function.h"

namespace npu::tile_fwk {

class PassOperationUtils {
public:
    static Operation& AddOperation(Function& function, Opcode opCode, LogicalTensors iOperands,
                                   const LogicalTensors& oOperands,
                                   std::function<void(Operation&)> beforeInferShapeHandler = nullptr,
                                   const ir::Span& span = ir::Span::Unknown(), bool inferShape = true);

private:
    static LogicalTensors PreprocessOperationInputs(Function& function, Opcode opCode, LogicalTensors iOperands);

    static LogicalTensorPtr ConnectWithOverlap(Function& function, LogicalTensorPtr iOperand);

    static std::vector<std::vector<int64_t>> ProcessOffsetAdjustment(LogicalTensors& matches,
                                                                     std::vector<int64_t>& minimumOffsets);

    static LogicalTensorPtr HandlePerfectlyMatchWithAll(Function& function, LogicalTensorPtr iOperand,
                                                        const LogicalTensors& matches,
                                                        const std::vector<std::vector<int64_t>>& offsetOfOverlaps);

    static LogicalTensorPtr HandleBeCovered(Function& function, LogicalTensorPtr iOperand,
                                            const LogicalTensors& matches);

    static LogicalTensorPtr HandleBeCoveredByAll(Function& function, LogicalTensorPtr iOperand,
                                                 const LogicalTensors& matches,
                                                 const std::vector<std::vector<int64_t>>& offsetOfOverlaps);
};

} // namespace npu::tile_fwk
