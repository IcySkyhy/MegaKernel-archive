/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file intra_subgraph_adapter_checker.h
 * \brief
 */

#ifndef INTRA_SUBGRAPH_CHECKER_H
#define INTRA_SUBGRAPH_CHECKER_H

#include "checker.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu {
namespace tile_fwk {
class IntraSubgraphAdapterChecker : Checker {
public:
    Status DoPostCheck(Function& function) override;

private:
    Status PostCheckSubgraphTensor(const std::vector<std::vector<Operation*>>& subgraphs);
};
} // namespace tile_fwk
} // namespace npu
#endif // INTRA_SUBGRAPH_CHECKER_H
