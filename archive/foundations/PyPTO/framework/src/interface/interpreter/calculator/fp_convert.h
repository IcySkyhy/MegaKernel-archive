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
 * \file fp_convert.h
 * \brief FP8/FP4 format conversion utilities between low-precision storage and Float32.
 */

#pragma once

#include <torch/torch.h>
#include "dtype_utils.h"

namespace npu::tile_fwk {

// FP8 (stored as uint8) -> float32. actualType specifies the FP8 format.
torch::Tensor Fp8ToFloat32(const torch::Tensor& self, DataType actualType);

// float32 -> FP8 storage (uint8). actualType specifies the FP8 format.
torch::Tensor Float32ToFp8(const torch::Tensor& self, DataType actualType);

// Packed FP4 (2 nibbles per byte, high nibble first) -> float32, last dim doubled.
torch::Tensor Fp4PackedToFloat32(const torch::Tensor& packed, DataType actualType);

// float32 -> packed FP4 storage (uint8, high nibble first), last dim halved.
torch::Tensor Float32ToFp4Packed(const torch::Tensor& self, DataType actualType);

} // namespace npu::tile_fwk
