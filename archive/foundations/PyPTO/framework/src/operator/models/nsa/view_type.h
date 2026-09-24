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
 * \file view_type.h
 * \brief
 */

#pragma once
#ifndef GEN_GATED_SCORE
#define GEN_GATED_SCORE

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {
void ViewTypeFunc(const Tensor& x, Tensor& result, DataType dstDtype);
std::tuple<Tensor, Tensor> MyPrologQuant(const Tensor& input);
void ViewTypeQuantTestFunc(const Tensor& x, Tensor& result);
void ViewTypeDequantTestFunc(const Tensor& x, Tensor& result);
} // namespace npu::tile_fwk

#endif
