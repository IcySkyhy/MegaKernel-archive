// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef DTILE_OPS
#define DTILE_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "DTileDialect.td"

class DTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<DTile_Dialect, mnemonic, !listconcat(traits, [])> {
}

def DTile_WaitDepsOp : DTile_Op<"wait_deps", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Wait on dependency scoreboard.";
  let arguments = (ins TT_Ptr:$dependInfo);
  let results = (outs);
  let assemblyFormat = "$dependInfo attr-dict `:` type($dependInfo)";
}

def DTile_NotifyTileDoneOp : DTile_Op<"notify_tile_done", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Notify tile completion via scoreboard.";
  let arguments = (ins TT_Ptr:$scoreboard, I32:$tileId);
  let results = (outs);
  let assemblyFormat = "$scoreboard `,` $tileId attr-dict `:` type($scoreboard) `,` type($tileId)";
}

def DTile_QueryTsConditionOp : DTile_Op<"query_ts_condition", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Read DATA_MAIN_BASE register.";
  let arguments = (ins);
  let results = (outs I64:$value);
  let assemblyFormat = "attr-dict `->` type($value)";
}

// TODO: set_cond, dcci, st_io, ld_io will be added to AscendNPU IR.
def DTile_SetCondOp : DTile_Op<"set_cond", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Set condition register.";
  let arguments = (ins I64:$cond);
  let results = (outs);
  let assemblyFormat = "$cond attr-dict `:` type($cond)";
}

def DTile_GlobalBlockIdxMixOp : DTile_Op<"global_block_idx_mix", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Get global block index in mix mode.";
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def DTile_DcciOp : DTile_Op<"dcci", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Data cache clean/invalidate.";
  let arguments = (ins TT_Ptr:$ptr, I32:$mode, I32:$data_cache_kind);
  let results = (outs);
  let assemblyFormat = "$ptr `,` $mode `,` $data_cache_kind attr-dict `:` type($ptr) `,` type($mode) `,` type($data_cache_kind)";
}

def DTile_StIOOp : DTile_Op<"st_io", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Store scalar to GM via ST_IO.";
  let arguments = (ins I64:$addr, AnyType:$val);
  let results = (outs);
  let assemblyFormat = "$addr `,` $val attr-dict `:` type($addr) `,` type($val)";
}

def DTile_LdIOOp : DTile_Op<"ld_io", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Load scalar from GM via LD_IO.";
  let arguments = (ins I64:$addr);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result)";
}

def DTile_SetStatusOp : DTile_Op<"set_status", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Set dfxSts with core type flag.";
  let arguments = (ins I64:$addr, I32:$val);
  let results = (outs);
  let assemblyFormat = "$addr `,` $val attr-dict `:` type($addr) `,` type($val)";
}

#endif
