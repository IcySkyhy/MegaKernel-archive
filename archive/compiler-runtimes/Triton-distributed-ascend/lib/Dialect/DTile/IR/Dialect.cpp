// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"

// clang-format off
#include "TritonDistributed/Dialect/DTile/IR/Dialect.h"
#include "TritonDistributed/Dialect/DTile/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::dtile;

void mlir::triton::dtile::DTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TritonDistributed/Dialect/DTile/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface>();
}
