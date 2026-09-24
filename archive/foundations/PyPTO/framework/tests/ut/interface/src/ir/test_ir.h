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

#define DEFINE_BINARY_EXPR_ALL()     \
    DEFINE_BINARY_EXPR(Add)          \
    DEFINE_BINARY_EXPR(Sub)          \
    DEFINE_BINARY_EXPR(Mul)          \
    DEFINE_BINARY_EXPR(FloorDiv)     \
    DEFINE_BINARY_EXPR(FloorDiv)     \
    DEFINE_BINARY_EXPR(FloorMod)     \
    DEFINE_BINARY_EXPR(FloatDiv)     \
    DEFINE_BINARY_EXPR(Min)          \
    DEFINE_BINARY_EXPR(Max)          \
    DEFINE_BINARY_EXPR(Pow)          \
    DEFINE_BINARY_EXPR(Eq)           \
    DEFINE_BINARY_EXPR(Ne)           \
    DEFINE_BINARY_EXPR(Lt)           \
    DEFINE_BINARY_EXPR(Le)           \
    DEFINE_BINARY_EXPR(Gt)           \
    DEFINE_BINARY_EXPR(Ge)           \
    DEFINE_BINARY_EXPR(And)          \
    DEFINE_BINARY_EXPR(Or)           \
    DEFINE_BINARY_EXPR(Xor)          \
    DEFINE_BINARY_EXPR(BitAnd)       \
    DEFINE_BINARY_EXPR(BitOr)        \
    DEFINE_BINARY_EXPR(BitXor)       \
    DEFINE_BINARY_EXPR(BitShiftLeft) \
    DEFINE_BINARY_EXPR(BitShiftRight)

#define DEFINE_UNARY_EXPR_ALL() \
    DEFINE_UNARY_EXPR(Abs)      \
    DEFINE_UNARY_EXPR(Neg)      \
    DEFINE_UNARY_EXPR(Not)      \
    DEFINE_UNARY_EXPR(Cast)     \
    DEFINE_UNARY_EXPR(BitNot)
