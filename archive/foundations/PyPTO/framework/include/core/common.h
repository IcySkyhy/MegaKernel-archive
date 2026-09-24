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
 * \file common.h
 * \brief Common macros, constants, and utility definitions
 *
 * This header provides commonly used macros and constants that are shared
 * across the PyPTO codebase, including:
 * - Compiler hints and attributes
 * - Utility macros for code generation
 * - Build configuration constants
 * - nanobind module configuration
 */

#pragma once
#include <cstdint>

namespace pypto {

// ============================================================================
// IR Constants
// ============================================================================

// Dynamic dimension constant for tensor/tile shapes
// Use -1 to represent dimensions that are unknown at compile time
constexpr int64_t kDynamicDim = -1;

// ============================================================================
// Compiler Hints and Attributes
// ============================================================================

#define PYPTO_ALWAYS_INLINE __attribute__((always_inline))
#define PYPTO_UNUSED __attribute__((unused))
#define PYPTO_STR_CONCAT_IMPL(__x, __y) __x##__y
#define PYPTO_STR_CONCAT(__x, __y) PYPTO_STR_CONCAT_IMPL(__x, __y)
} // namespace pypto
