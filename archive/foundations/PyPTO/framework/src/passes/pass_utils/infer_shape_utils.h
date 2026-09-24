/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_shape_utils.h
 * \brief 公共的 InferShape 方法，支持全量推断和指定 op 推断
 */

#pragma once

#include <vector>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {
class InferShapeUtils {
public:
    /**
     * @brief Common InferShape method, supporting full-graph inference and targeted op inference.
     *
     * @param function The function on which shape inference is to be performed.
     * @param targetOps List of target operations to infer shapes for. If empty, inference is performed
     *                  on all operations in the function. If non-empty, only the specified operations
     *                  are inferred, but it is required that targetOps includes all necessary
     *                  dependent operations; otherwise, topological ordering and inference may be incorrect.
     * @return Status Operation result, SUCCESS indicates success, FAILED indicates failure.
     *
     * @note When targetOps is non-empty, the target operations are filtered, deduplicated, and ordered according
     *       to the function's LIGHTWEIGHT topological order. Only the specified operations are inferred.
     */
    static Status InferShape(Function& function, const std::vector<Operation*>& targetOps = {});
};
} // namespace tile_fwk
} // namespace npu
