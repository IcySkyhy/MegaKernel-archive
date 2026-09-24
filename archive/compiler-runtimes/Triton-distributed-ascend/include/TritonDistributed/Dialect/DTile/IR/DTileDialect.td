// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef DTILE_DIALECT
#define DTILE_DIALECT

include "mlir/IR/OpBase.td"

def DTile_Dialect : Dialect {
  let name = "dtile";
  let cppNamespace = "::mlir::triton::dtile";

  let description = [{
    DTile dialect for Ascend megakernel device-level ops
    (st_io, ld_io, set_status, dcci, set_cond, query_ts_condition, etc.).
  }];

  let dependentDialects = [];
}

#endif
