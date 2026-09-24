// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef TRITON_DIALECT_DTILE_IR_DIALECT_H_
#define TRITON_DIALECT_DTILE_IR_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

// clang-format off
#include "TritonDistributed/Dialect/DTile/IR/Dialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "TritonDistributed/Dialect/DTile/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace dtile {} // namespace dtile
} // namespace triton
} // namespace mlir

#endif
