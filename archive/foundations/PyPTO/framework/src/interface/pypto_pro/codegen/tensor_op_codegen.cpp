/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir; // NOLINT(build/namespaces)

// Helper function to calculate tensor size expression
[[maybe_unused]] static std::string CalculateTensorSizeExpr(const TensorTypePtr& tensor_type, CodegenBase& codegen)
{
    std::ostringstream oss;

    // Calculate total number of elements by multiplying all dimensions
    bool first = true;
    for (const auto& dim : tensor_type->shape_) {
        if (first) {
            oss << codegen.GenerateExprString(dim);
            first = false;
        } else {
            oss << " * " << codegen.GenerateExprString(dim);
        }
    }

    // If shape is empty, it's a scalar (1 element)
    if (first) {
        oss << "1";
    }

    // Multiply by element size in bytes
    size_t element_bits = tensor_type->dtype_.GetBit();
    size_t element_bytes = (element_bits + 7) / 8; // Round up to nearest byte
    oss << " * " << element_bytes;

    return oss.str();
}
} // namespace codegen
} // namespace pypto
