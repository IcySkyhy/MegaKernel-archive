/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FRAMEWORK_SRC_INTERFACE_OPERATION_CONV_CONV_UTILS_H
#define FRAMEWORK_SRC_INTERFACE_OPERATION_CONV_CONV_UTILS_H

#include "tilefwk/tensor.h"
#include "interface/operation/operation_impl.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

int64_t ConvComputeHo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam);
int64_t ConvComputeWo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam);
int64_t ConvComputeDo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam);

SymbolicScalar ConvComputeValidHo(const Tensor& inputTensor, const Tensor& weightTensor,
                                  const ConvAttrParam& attrParam);
SymbolicScalar ConvComputeValidWo(const Tensor& inputTensor, const Tensor& weightTensor,
                                  const ConvAttrParam& attrParam);
SymbolicScalar ConvComputeValidDo(const Tensor& inputTensor, const Tensor& weightTensor,
                                  const ConvAttrParam& attrParam);

void CheckConvOperands(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                       const Tensor& biasTensor, ConvAttrParam& attrParam);

void CheckTileTiling(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                     const ConvAttrParam& attrParam);

void CheckL1SizeTiling(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                       const Tensor& biasTensor, const ConvAttrParam& attrParam);

} // namespace Conv
} // namespace tile_fwk
} // namespace npu

#endif // FRAMEWORK_SRC_INTERFACE_OPERATION_CONV_CONV_UTILS_H
