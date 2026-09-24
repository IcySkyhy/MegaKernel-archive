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
 * \file codegen_op_npu.cpp
 * \brief
 */

#include "codegen_op_npu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {

std::unordered_map<Opcode, std::set<int>> SKIP_PROC_PRARAM_IDX_IN_LOOP = {
    // scene: reduce for last axis
    // Parameter at 1st index (after numbering by CodeGenOp::Init) is used as temp buffer which is reused in loop body.
    {Opcode::OP_ROWSUM_SINGLE, {ID1}},
    {Opcode::OP_ROWMAX_SINGLE, {ID1}},
    {Opcode::OP_ROWMIN_SINGLE, {ID1}},
    {Opcode::OP_ROWPROD_SINGLE, {ID1}},
};

CodeGenOpNPU::CodeGenOpNPU(const CodeGenOpNPUCtx& ctx)
    : CodeGenOp(ctx),
      mteFixPipeOps_({
          // UB <-> GM
          {Opcode::OP_UB_COPY_IN, [this]() { return GenUBCopyIn(); }},
          {Opcode::OP_UB_COPY_OUT, [this]() { return GenUBCopyOut(); }},
          {Opcode::OP_RESHAPE_COPY_IN, [this]() { return GenReshapeCopyIn(); }},
          {Opcode::OP_RESHAPE_COPY_OUT, [this]() { return GenReshapeCopyOut(); }},
          {Opcode::OP_L1_TO_FIX_QUANT_PRE, [this]() { return GenMemL1ToFB(); }},
          {Opcode::OP_GATHER_IN_UB, [this]() { return GenGatherInUB(); }},
          {Opcode::OP_GATHER, [this]() { return GenGatherOp(); }},
          {Opcode::OP_PERMUTE, [this]() { return GenPermuteOp(); }},
          {Opcode::OP_PERMUTE_ELEMENT, [this]() { return GenPermuteOp(); }},
          // L1 <-> GM/BT/L1
          {Opcode::OP_L1_COPY_IN, [this]() { return GenMemL1CopyIn(); }},
          {Opcode::OP_L1_COPY_IN_A_SCALE, [this]() { return GenMemL1CopyIn(); }},
          {Opcode::OP_L1_COPY_IN_B_SCALE, [this]() { return GenMemL1CopyIn(); }},
          {Opcode::OP_L1_COPY_OUT, [this]() { return GenMemL1CopyOut(); }},
          {Opcode::OP_GATHER_IN_L1, [this]() { return GenGatherInL1(); }},
          {Opcode::OP_L1_COPY_IN_CONV, [this]() { return GenMemL1CopyInConv(); }},
          {Opcode::OP_L1_RESHAPE_COPY_IN, [this]() { return GenMemL1CopyIn(); }},

          // L0C <-> GM
          {Opcode::OP_L0C_COPY_OUT, [this]() { return GenMemL0CCopyOut(); }},
          {Opcode::OP_L0C_COPY_OUT_CONV, [this]() { return GenMemL0CCopyOutConv(); }},
          {Opcode::OP_L0C_TO_L1, [this]() { return GenMemL0CToL1(); }},
          {Opcode::OP_L0C_RESHAPE_COPY_OUT, [this]() { return GenMemL0CCopyOut(); }},

          // L1 <-> L0
          {Opcode::OP_L1_TO_L0A, [this]() { return GenMemL1ToL0(); }},
          {Opcode::OP_L1_TO_L0B, [this]() { return GenMemL1ToL0(); }},
          {Opcode::OP_L1_TO_L0_BT, [this]() { return GenMemL1ToL0(); }},
          {Opcode::OP_L1_TO_L0_AT, [this]() { return GenMemL1ToL0(); }},
          {Opcode::OP_L1_TO_L0A_SCALE, [this]() { return GenMemL1ToL0(); }},
          {Opcode::OP_L1_TO_L0B_SCALE, [this]() { return GenMemL1ToL0(); }},
          {Opcode::OP_L1_TO_BT, [this]() { return GenMemL1ToBt(); }},
          {Opcode::OP_LOAD3D_CONV, [this]() { return GenMemL1ToL0Load3D(); }},
          {Opcode::OP_LOAD2D_CONV, [this]() { return GenMemL1ToL0Load2D(); }},

          // transpose with gm
          {Opcode::OP_TRANSPOSE_MOVEOUT, [this]() { return GenTransposeDataMove(); }},
          {Opcode::OP_TRANSPOSE_MOVEIN, [this]() { return GenTransposeDataMove(); }},

          // transData
          {Opcode::OP_NCHW2NC1HWC0, [this]() { return GenTransData(); }},
          {Opcode::OP_NCHW2Fractal_Z, [this]() { return GenTransData(); }},
          {Opcode::OP_NC1HWC02NCHW, [this]() { return GenTransData(); }},
          {Opcode::OP_NCDHW2NDC1HWC0, [this]() { return GenTransData(); }},
          {Opcode::OP_NCDHW2FRACTAL_Z_3D, [this]() { return GenTransData(); }},
          {Opcode::OP_NDC1HWC02NCDHW, [this]() { return GenTransData(); }},

          // index outcast
          {Opcode::OP_INDEX_OUTCAST, [this]() { return GenIndexOutCastOp(); }},
          // lOC -> UB
          {Opcode::OP_L0C_COPY_UB, [this]() { return GenL0CToUBTileTensor(); }},
          {Opcode::OP_L0C_COPY_UB_DUAL_DST, [this]() { return GenL0CToUBTileTensor(); }},

          {Opcode::OP_UB_COPY_L1, [this]() { return GenUBToL1TileTensor(); }},
          {Opcode::OP_UB_COPY_ND2NZ, [this]() { return GenUBToUBND2NZTileTensor(); }},
      }),
      unaryOps_({
          // cast op
          {Opcode::OP_CAST, [this]() { return GenCastOp(); }},

          // unary op
          {Opcode::OP_EXP, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_NEG, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_RSQRT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_RELU, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_BITWISENOT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_SQRT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_CEIL, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_FLOOR, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_TRUNC, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_EXPAND, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ONEHOT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_RECIPROCAL, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWSUM, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWMAX, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWEXPSUM, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWEXPMAX, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_COPY_UB_TO_UB, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWMAXLINE, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWMINLINE, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWPRODLINE, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ABS, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_LN, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_BRCB, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_PACK, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_UNPACK, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_QUANT_MX, [this]() { return GenQuantMXOp(); }},

          // unary with temp buffer
          {Opcode::OP_COMPACT, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_EXP2, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_EXPM1, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROUND, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWSUMLINE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWSUM_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, [this]() { return GenArgReduceWithValue(); }},
          {Opcode::OP_ROWARGMINWITHVALUE_SINGLE, [this]() { return GenArgReduceWithValue(); }},
          {Opcode::OP_ROWARGMAXWITHVALUE_LINE, [this]() { return GenArgReduceWithValue(); }},
          {Opcode::OP_ROWARGMINWITHVALUE_LINE, [this]() { return GenArgReduceWithValue(); }},
          {Opcode::OP_ROWMAX_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWMIN_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ISFINITE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ISNAN, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ATAN, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWPROD_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_TRANSPOSE_VNCHWCONV, [this]() { return GenUnaryOpWithTmpBuff(); }},

          {Opcode::OP_SIGN, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_SIGNBIT, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_TAN, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_SIN, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_COS, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ERF, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_SINH, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_COSH, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ERFC, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_TANH, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ASIN, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ACOS, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ASINH, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ACOSH, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ATANH, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_LOG1P, [this]() { return GenUnaryOpWithTmpBuff(); }},
      }),
      binaryOps_({
          // binary op: vector operations
          {Opcode::OP_ADD, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_SUB, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_MUL, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_DIV, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_MAXIMUM, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_MINIMUM, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRSUM, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRMAX, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRMIN, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRPROD, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRARGMAX, [this]() { return GenPairArgReduce(); }},
          {Opcode::OP_PAIRARGMIN, [this]() { return GenPairArgReduce(); }},
          {Opcode::OP_BITWISEAND, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_BITWISEOR, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_EXPANDEXPDIF, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_GCD, [this]() { return GenBinaryOp(); }},

          // binary op: vector operations with tmp
          {Opcode::OP_MOD, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_POW, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_REM, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_BITWISERIGHTSHIFT, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_BITWISELEFTSHIFT, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_BITWISEXOR, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_COPYSIGN, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_PRELU, [this]() { return GenPreluOp(); }},
          {Opcode::OP_ATAN2, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_FLOORDIV, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_AXPY, [this]() { return GenAxpyOp(); }},

          // binary op: broadcast associated vector
          {Opcode::OP_ADD_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_SUB_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_MUL_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_DIV_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_MAX_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_MIN_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_GCD_BRC, [this]() { return GenBinaryWithBrc(); }},

          // binary op: vector scalar
          {Opcode::OP_ADDS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_SUBS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_MULS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_DIVS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_MAXS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_MINS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_LRELU, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISEANDS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISEORS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISERIGHTSHIFTS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISELEFTSHIFTS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_GCDS, [this]() { return GenVectorScalarOp(); }},

          // binary op: vector scalar with tmp
          {Opcode::OP_MODS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_REMRS, [this]() { return GenRemainderSOp(); }},
          {Opcode::OP_REMS, [this]() { return GenRemainderSOp(); }},
          {Opcode::OP_POWS, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_SBITWISERIGHTSHIFT, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_SBITWISELEFTSHIFT, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_BITWISEXORS, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_FLOORDIVS, [this]() { return GenVectorScalarOpWithTmp(); }},

          // binary op: vector scalar, scalar mode
          {Opcode::OP_S_ADDS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_SUBS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_MULS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_DIVS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_MAXS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_MINS, [this]() { return GenVectorScalarOpScalarMode(); }},

          // binary op: multi input, multi output
          {Opcode::OP_INTERLEAVE, [this]() { return GenInterleaveLikeOp(); }},
          {Opcode::OP_DEINTERLEAVE, [this]() { return GenInterleaveLikeOp(); }},
          {Opcode::OP_DEINTERLEAVE_SINGLE, [this]() { return GenInterleaveLikeOp(); }},
      }),
      compositeOps_({
          {Opcode::OP_ONLINE_SOFTMAX, [this]() { return GenOnlineSoftmaxOp(); }},
          {Opcode::OP_ONLINE_SOFTMAX_UPDATE, [this]() { return GenOnlineSoftmaxUpdateOp(); }},

          // range op
          {Opcode::OP_RANGE, [this]() { return GenRangeOp(); }},
          // uniform op
          {Opcode::OP_UNIFORM, [this]() { return GenUniformOp(); }},

          // logicalnot
          {Opcode::OP_LOGICALNOT, [this]() { return GenLogicalNotOp(); }},
          // logicaland
          {Opcode::OP_LOGICALAND, [this]() { return GenLogicalAndOp(); }},

          // indexadd
          {Opcode::OP_INDEX_ADD_UB, [this]() { return GenIndexAddUBOp(); }},
          {Opcode::OP_INDEX_ADD, [this]() { return GenIndexAddOp(); }},

          // indexput
          {Opcode::OP_INDEX_PUT, [this]() { return GenIndexPutOp(); }},

          // cumOperation
          {Opcode::OP_CUM_SUM, [this]() { return GenCumOperationOp(); }},
          {Opcode::OP_CUM_PROD, [this]() { return GenCumOperationOp(); }},

          // triUL
          {Opcode::OP_TRIUL, [this]() { return GenTriULOp(); }},

          // vector where
          {Opcode::OP_WHERE_SS, [this]() { return GenWhereOp(); }},
          {Opcode::OP_WHERE_TS, [this]() { return GenWhereOp(); }},
          {Opcode::OP_WHERE_ST, [this]() { return GenWhereOp(); }},
          {Opcode::OP_WHERE_TT, [this]() { return GenWhereOp(); }},

          // cmp op
          {Opcode::OP_CMP, [this]() { return GenCmpOp(); }},
          {Opcode::OP_CMPS, [this]() { return GenCmpOp(); }},

          // hypot op
          {Opcode::OP_HYPOT, [this]() { return GenHypotOp(); }},
          {Opcode::OP_PAD, [this]() { return GenPadOp(); }},
          {Opcode::OP_FILLPAD, [this]() { return GenPadOp(); }},
      }),
      sortOps_({
          // sort
          {Opcode::OP_BITSORT, [this]() { return GenBitSortOp(); }},
          {Opcode::OP_MRGSORT, [this]() { return GenMrgSortOp(); }},
          {Opcode::OP_EXTRACT, [this]() { return GenExtractOp(); }},
          {Opcode::OP_TILEDMRGSORT, [this]() { return GenTiledMrgSortOp(); }},
          {Opcode::OP_RADIX_SELECT, [this]() { return GenRadixSelectOp(); }},

          {Opcode::OP_TOPK_SORT, [this]() { return GenTopKSortOp(); }},
          {Opcode::OP_TOPK_MERGE, [this]() { return GenTopKMergeOp(); }},
          {Opcode::OP_TOPK_EXTRACT, [this]() { return GenTopKExtractOp(); }},

          // parallel sort
          {Opcode::OP_SORT, [this]() { return GenSortOp(); }},
          {Opcode::OP_COMPARE_SWAP, [this]() { return GenCompareAndSwapOp(); }},
          {Opcode::OP_MERGE, [this]() { return GenMergeOp(); }},

          {Opcode::OP_TWOTILEMRGSORT, [this]() { return GenTwoTileMrgSort(); }},
          {Opcode::OP_EXTRACT_SINGLE, [this]() { return GenExtractSingleOp(); }},
      }),
      cubeOps_({
          // matmul
          {Opcode::OP_A_MUL_B, [this]() { return GenCubeOpMatmul(); }},
          {Opcode::OP_A_MUL_BT, [this]() { return GenCubeOpMatmul(); }},
          {Opcode::OP_A_MULACC_B, [this]() { return GenCubeOpMatmulAcc(); }},
          {Opcode::OP_A_MULACC_BT, [this]() { return GenCubeOpMatmulAcc(); }},
      }),
      syncOps_({
          // sync
          {Opcode::OP_SYNC_SRC, [this]() { return GenSyncSetOp(); }},
          {Opcode::OP_SYNC_DST, [this]() { return GenSyncWaitOp(); }},
          {Opcode::OP_BAR_V, [this]() { return GenBarrier(); }},
          {Opcode::OP_BAR_M, [this]() { return GenBarrier(); }},
          {Opcode::OP_BAR_ALL, [this]() { return GenBarrier(); }},
          {Opcode::OP_CV_SYNC_SRC, [this]() { return GenCVSyncSetOp(); }},
          {Opcode::OP_CV_SYNC_DST, [this]() { return GenCVSyncWaitOp(); }},
          {Opcode::OP_FFTS_CROSS_CORE_SYNC, [this]() { return GenFFTSCrossCoreSyncOp(); }},
          {Opcode::OP_WAIT_FLAG_DEV, [this]() { return GenWaitFlagDevOp(); }},
      }),
      distributeOps_({
          // distribute op
          {Opcode::OP_SHMEM_SET, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_PUT, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_STORE, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_SIGNAL, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_GET, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_LOAD, [this]() { return GenDistOp(); }},
      }),
      gatherScatterOps_({
          // gather/scatter op
          {Opcode::OP_GATHER_FROM_UB, [this]() { return GenGatherFromUBOp(); }},
          {Opcode::OP_GATHER_ELEMENT, [this]() { return GenGatherElementOp(); }},
          {Opcode::OP_SCATTER_ELEMENT, [this]() { return GenScatterElementSOp(); }},
          {Opcode::OP_SCATTER, [this]() { return GenScatterOp(); }},
          {Opcode::OP_GATHER_MASK, [this]() { return GenGatherMaskOp(); }},
      }),
      normalVecOps_({
          // vector dup
          {Opcode::OP_VEC_DUP, [this]() { return GenDupOp(); }},
      }),
      quantOps_({
          // float -> int8/uint8
          {Opcode::OP_QUANTIZE_SYM, [this]() { return GenQuantizeOp(); }},
          {Opcode::OP_QUANTIZE_ASYM, [this]() { return GenQuantizeOp(); }},
          // int8/int16 -> float
          {Opcode::OP_DEQUANTIZE, [this]() { return GenDequantizeOp(); }},
      }),
      perfOps_({
          // for performace optimization
          {Opcode::OP_PHASE1, []() { return "SUBKERNEL_PHASE1\n"; }},
          {Opcode::OP_PHASE2, []() { return "SUBKERNEL_PHASE2\n"; }},
      }),
      aicpuOps_({
          // for aicpu call
          {Opcode::OP_AICPU_CALL_AIC, [this]() { return GenAicpuCallOp(); }},
          {Opcode::OP_AICPU_CALL_AIV, [this]() { return GenAicpuCallOp(); }},
      })
{}

void CodeGenOpNPU::InitOpsGenMap()
{
    InitScalaOpsMap();
    InitMTEOpsMap();
    InitVecOpsMap();
    InitCubeOpsMap();
    InitDistOpsMap();
    InitPerfOpsMap();
    InitAICPUOpsMap();
}

void CodeGenOpNPU::InitScalaOpsMap() { opsGenMap_.insert(syncOps_.cbegin(), syncOps_.cend()); }

void CodeGenOpNPU::InitMTEOpsMap() { opsGenMap_.insert(mteFixPipeOps_.cbegin(), mteFixPipeOps_.cend()); }

void CodeGenOpNPU::InitVecOpsMap()
{
    opsGenMap_.insert(unaryOps_.cbegin(), unaryOps_.cend());
    opsGenMap_.insert(binaryOps_.cbegin(), binaryOps_.cend());
    opsGenMap_.insert(compositeOps_.cbegin(), compositeOps_.cend());
    opsGenMap_.insert(sortOps_.cbegin(), sortOps_.cend());
    opsGenMap_.insert(gatherScatterOps_.cbegin(), gatherScatterOps_.cend());
    opsGenMap_.insert(normalVecOps_.cbegin(), normalVecOps_.cend());
    opsGenMap_.insert(quantOps_.cbegin(), quantOps_.cend());
}

void CodeGenOpNPU::InitCubeOpsMap() { opsGenMap_.insert(cubeOps_.cbegin(), cubeOps_.cend()); }

void CodeGenOpNPU::InitDistOpsMap() { opsGenMap_.insert(distributeOps_.cbegin(), distributeOps_.cend()); }

void CodeGenOpNPU::InitPerfOpsMap() { opsGenMap_.insert(perfOps_.cbegin(), perfOps_.cend()); }

void CodeGenOpNPU::InitAICPUOpsMap() { opsGenMap_.insert(aicpuOps_.cbegin(), aicpuOps_.cend()); }

void CodeGenOpNPU::AppendLocalBufferVarOffset(const std::map<unsigned, std::reference_wrapper<std::string>>& vars) const
{
    for (auto& kv : vars) {
        auto operandIdx = kv.first;
        int64_t resOffset{0};

        std::vector<int64_t> varOffset = offset[operandIdx];
        if (varOffset.empty()) {
            continue;
        }

        std::vector<int64_t> varRawShape = rawShape[operandIdx];
        ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, !varRawShape.empty())
            << "varRawShape is empty!! operandIdx: " << operandIdx;
        ASSERT(GenCodeErr::TENSOR_SHAPE_MISMATCHED, varOffset.size() == varRawShape.size())
            << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size()
            << " is not equal!! operandIdx: " << operandIdx;

        resOffset = CalcLinearOffset(varRawShape, varOffset);
        if (resOffset == 0) {
            continue;
        }

        std::string& var = kv.second.get();

        ASSERT(GenCodeErr::SYMBOL_NOT_FOUND, !var.empty()) << "operandIdx: " << operandIdx << ", var is empty !!";
        CODEGEN_LOGI("var: %s, varRawShape: %s, varOffset: %s, resOffset: %ld", var.c_str(),
                     IntVecToStr(varRawShape).c_str(), IntVecToStr(varOffset).c_str(), static_cast<long>(resOffset));

        var.append(" + ").append(std::to_string(resOffset));
    }
}

SymbolicScalar CodeGenOpNPU::GetOperandStartOffset(int operandIdx) const
{
    std::vector varOffset = offset[operandIdx];
    if (varOffset.empty()) {
        return 0;
    }

    const auto& dynOffset = dynamicOffset[operandIdx];
    if (!dynOffset.empty()) {
        std::vector varRawShape = rawShape[operandIdx]; // 内部应该不能出现dynRawShape，所以这里用立即数即可
        ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, !varRawShape.empty())
            << "varRawShape is empty!! operandIdx: " << operandIdx;
        ASSERT(GenCodeErr::TENSOR_SHAPE_MISMATCHED, dynOffset.size() == varRawShape.size())
            << "dynOffset " << SymbolicVecToStr(dynOffset) << ", size " << dynOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size()
            << " is not equal!! operandIdx: " << operandIdx;

        SymbolicScalar resOffset = 0;
        for (size_t i = 0; i < dynOffset.size(); i++) {
            resOffset = resOffset * varRawShape[i];
            resOffset = resOffset + dynOffset[i];
        }

        ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, operandIdx < operandCnt)
            << "operandIdx: " << operandIdx << ", operandCnt: " << operandCnt;
        CODEGEN_LOGD(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
        CODEGEN_LOGD(" varOffset: %s", SymbolicVecToStr(dynOffset).c_str());
        CODEGEN_LOGD(" resOffset: %s", resOffset.Dump().c_str());
        if (resOffset.ConcreteValid()) {
            return resOffset.Concrete();
        }
        return SymbolicExpressionTable::BuildExpression(resOffset);
    }

    std::vector varRawShape = rawShape[operandIdx];
    ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, !varRawShape.empty())
        << "varRawShape is empty!! operandIdx: " << operandIdx;
    ASSERT(GenCodeErr::TENSOR_SHAPE_MISMATCHED, varOffset.size() == varRawShape.size())
        << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
        << IntVecToStr(varRawShape) << ", size " << varRawShape.size() << " is not equal!! operandIdx: " << operandIdx;

    int64_t resOffset = CalcLinearOffset(varRawShape, varOffset);
    if (resOffset == 0) {
        return 0;
    }

    ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, operandIdx < operandCnt)
        << "operandIdx: " << operandIdx << ", operandCnt: " << operandCnt;
    CODEGEN_LOGD(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
    CODEGEN_LOGD(" varOffset: %s", IntVecToStr(varOffset).c_str());
    CODEGEN_LOGD(" resOffset: %ld", static_cast<long>(resOffset));
    return resOffset;
}

std::string CodeGenOpNPU::GetGmTensorAddrByAttr(unsigned gmParamIdx) const
{
    std::map<TensorAddrKey, SymbolicScalar> addrs;
    bool ret = GetTensorAttr(gmParamIdx, TensorAttributeKey::tensorAddr, addrs);
    if (!ret || addrs.empty()) {
        CODEGEN_LOGW("gmParamIdx: %u, tensorAddr is not found in attr !! op: %s", gmParamIdx,
                     originalOp.Dump().c_str());
        return "";
    }
    ASSERT(OperErr::ATTRIBUTE_INVALID, originalOp.BelongTo() != nullptr)
        << "originalOp.BelongTo() is nullptr, op: " << originalOp.Dump();
    int funcMagic = originalOp.BelongTo()->GetFuncMagic();
    TensorAddrKey key{funcMagic, originalOp.GetOpMagic()};
    auto iter = addrs.find(key);
    ASSERT(OperErr::ATTRIBUTE_INVALID, iter != addrs.end())
        << "addr is not found by TensorAddrKey{funcMagic: " << funcMagic << ", opMagic: " << originalOp.GetOpMagic()
        << "}, gmParamIdx: " << gmParamIdx << ", op: " << originalOp.Dump();
    std::string gmParamVar = SymbolicExpressionTable::BuildExpression(iter->second);
    CODEGEN_LOGI("gmParamVar from attr is : %s", gmParamVar.c_str());
    return gmParamVar;
}

std::string CodeGenOpNPU::GenGmParamVar(unsigned gmParamIdx) const
{
    std::string gmParamVar = GetGmTensorAddrByAttr(gmParamIdx);
    if (!gmParamVar.empty()) {
        return gmParamVar;
    }
    // Use CodeGen generation as the fallback
    std::ostringstream os;
    os << "GET_PARAM_ADDR(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
       << paramLocation[gmParamIdx] << ")";
    gmParamVar = os.str();
    CODEGEN_LOGI("gmParamVar by codegen: %s", gmParamVar.c_str());
    return gmParamVar;
}

// Used for parameter of GM shape and offset, e.g.
// GET_PARAM_RAWSHAPE_2(param, 19, 9), GET_PARAM_OFFSET_2(param, 19, 9)
// If dim is 2, the macro would be expanded into "shape0, shape1" which is implemented in aicore_runtime.h
std::vector<std::string> CodeGenOpNPU::GenGetParamMacroPacked(unsigned gmParamIdx, int dim,
                                                              const std::string& prefix) const
{
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    os << "GET_PARAM_" << prefix << "_" << dim << "(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc
       << ", " << paramLocation[gmParamIdx] << ")";
    paramExpr.emplace_back(os.str());
    return paramExpr;
};

std::vector<std::string> CodeGenOpNPU::GenDynRawShapePacked(unsigned paramIdx) const
{
    std::vector<std::string> paramExpr;
    FillParamWithFullInput(paramExpr, dynamicRawShape[paramIdx]);
    return paramExpr;
}

std::vector<std::string> CodeGenOpNPU::GenDynStridePacked(const std::vector<std::string>& dynRawShape) const
{
    std::ostringstream os;
    std::vector<std::string> paramExpr;
    os << "GET_PARAM_STRIDE_DIM_" << dynRawShape.size() << WrapParamByParentheses(dynRawShape);
    paramExpr.emplace_back(os.str());
    return paramExpr;
};

std::vector<std::string> CodeGenOpNPU::GenParamIdxExprByIndex(unsigned gmParamIdx, int dim,
                                                              const std::string& prefix) const
{
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    for (int index = 0; index < dim; ++index) {
        os << "GET_PARAM_" << prefix << "_BY_IDX(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
           << paramLocation[gmParamIdx] << ", " << dim << ", " << index << ")";
        paramExpr.emplace_back(os.str());
        os.str("");
    }
    return paramExpr;
}

std::vector<std::string> CodeGenOpNPU::GenSymbolicArgument(const std::vector<SymbolicScalar>& exprList) const
{
    std::vector<std::string> argList;
    FillParamWithFullInput(argList, exprList);

    CODEGEN_LOGI("argList is %s", IntVecToStr(argList).c_str());
    return argList;
}

std::vector<std::string> CodeGenOpNPU::BuildStride(const std::vector<int64_t>& input)
{
    if (input.empty()) {
        return {};
    }

    std::vector<std::string> res(input.size(), "1");
    int64_t base = 1;
    for (int i = input.size() - 2; i >= 0; --i) {
        base *= input[i + 1];
        res[i] = std::to_string(base);
    }

    return res;
}

void CodeGenOpNPU::UpdateTileTensorShapeAndStride(int paramIdx, TileTensor& tileTensor, bool isSpillToGm,
                                                  const TileTensorShape& tileTensorShape)
{
    auto newShape = tileTensorShape.shape;
    auto newRawShape = tileTensorShape.rawShape;
    auto newDynValidShape = tileTensorShape.dynamicValidShape;
    CODEGEN_LOGI("newShape is %s, newRawShape is %s, newDynValidShape is %s", IntVecToStr(newShape).c_str(),
                 IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynValidShape).c_str());

    tileTensor.rawShape = newRawShape;

    // ---- static or "main block" ----
    if (functionType == FunctionType::STATIC) {
        for (auto s : newShape) {
            tileTensor.shape.emplace_back(std::to_string(s));
        }
        tileTensor.stride = BuildStride(newRawShape);
        return;
    }

    // ---- dynamic ----
    // gm tensor use raw shape
    if (tileTensor.bufType == OperandType::BUF_DDR) {
        if (isSpillToGm) {
            for (auto s : newRawShape) {
                tileTensor.shape.emplace_back(std::to_string(s));
            }
            tileTensor.stride = BuildStride(newRawShape);
        } else {
            tileTensor.shape = dynamicRawShape[paramIdx].empty() ?
                                   GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_RAW_SHAPE) :
                                   GenDynRawShapePacked(paramIdx);
            tileTensor.stride = dynamicRawShape[paramIdx].empty() ?
                                    GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_STRIDE) :
                                    GenDynStridePacked(tileTensor.shape);
        }
        return;
    }

    // local tensor
    if (tileTensor.isConstant) {
        for (const auto& s : newShape) {
            tileTensor.shape.emplace_back(std::to_string(s));
        }
    } else {
        FillParamWithFullInput(tileTensor.shape, newDynValidShape);
    }
    tileTensor.stride = BuildStride(newRawShape);
}

TileTensor CodeGenOpNPU::BuildTileTensor(int paramIdx, const std::string& usingType,
                                         const TileTensorShape& tileTensorShape)
{
    int64_t gmOffset{0};
    bool isSpillToGm = GetTensorAttr(paramIdx, OpAttributeKey::workspaceBaseOffset, gmOffset);

    TileTensor tileTensor;
    tileTensor.isConstant = functionType == FunctionType::STATIC || isMainBlock;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.isInLoop = tileTensorShape.isInLoop;

    tileTensor.dim = tileTensor.isConstant ? tileTensorShape.shape.size() : tileTensorShape.dynamicValidShape.size();

    tileTensor.dtype = operandDtype[paramIdx];
    tileTensor.bufType = operandType[paramIdx];

    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.bufVar = isSpillToGm ? GenGMAddrExprWithOffset(paramIdx, GM_STACK_BASE) : GenGmParamVar(paramIdx);
    } else {
        tileTensor.bufVar = sm->QueryVarNameByTensorMagic(tileTensor.magic, true);
    }

    tileTensor.usingType = usingType;

    tileTensor.tensorName = sm->GenTensorName(tileTensor.bufType);
    if (tileTensorShape.isInLoop) {
        std::string tensorName = tensorNames_[paramIdx];
        if (!tensorName.empty()) {
            tensorName.append("_low").append(std::to_string(tileTensor.dim)).append("DimInLoop");
            tileTensor.tensorName = tensorName;
        }
    }
    UpdateTileTensorShapeAndStride(paramIdx, tileTensor, isSpillToGm, tileTensorShape);

    tileTensor.localBufOffset = offset[paramIdx];

    return tileTensor;
}

void CodeGenOpNPU::UpdateTileTensorInfo()
{
    if (!isSupportTileTensor) {
        return;
    }

    auto iter = SUPPORT_TILETENSOR_OPS.find(opCode);
    if (iter == SUPPORT_TILETENSOR_OPS.end()) {
        ASSERT(GenCodeErr::OP_CODE_UNSUPPORTED, iter != SUPPORT_TILETENSOR_OPS.end())
            << "opCode: " << opCodeStr << " not support tile tensor!";
        return;
    }

    tileOpName = iter->second; // update tileOpName from SUPPORT_TILETENSOR_OPS
    UpdateGemvTileOpName();

    for (int i = 0; i < operandCnt; ++i) {
        TileTensorUsing tileTensorUsing{functionType == FunctionType::STATIC || isMainBlock,
                                        operandDtype[i],
                                        operandType[i],
                                        static_cast<int>(rawShape[i].size()),
                                        shape[i],
                                        rawShape[i]};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensorShape tileTensorShape{false, shape[i], rawShape[i], dynamicValidShape[i]};
        TileTensor tileTensor = BuildTileTensor(i, usingType, tileTensorShape);
        std::string tensorName = sm->AddTileTensor(originalOp.GetOpMagic(), tileTensor);
        tensorNames_[i] = tensorName;
        CODEGEN_LOGI("AddTileTensor op idx: %d, result usingType: %s, tensorName: %s", i, usingType.c_str(),
                     tensorName.c_str());
    }
}

void CodeGenOpNPU::UpdateGemvTileOpName() {}

bool CodeGenOpNPU::ShouldSkipProcInLoop(int paramIdx)
{
    auto iter = SKIP_PROC_PRARAM_IDX_IN_LOOP.find(opCode);
    if (iter != SKIP_PROC_PRARAM_IDX_IN_LOOP.end() && iter->second.find(paramIdx) != iter->second.end()) {
        return true;
    }
    // cast with tempbuf which index is 1
    if (opCode == Opcode::OP_CAST && originalOp.oOperand.size() == NUM2 && paramIdx == 1) {
        return true;
    }
    return false;
}

std::vector<SymbolicScalar> CodeGenOpNPU::GetLoopAxes()
{
    std::vector<SymbolicScalar> dynloopAxes;
    std::vector<int64_t> loopAxes;

    if (!isMainBlock) {
        GetOpAttr(OpAttributeKey::dynloopAxes, dynloopAxes);
        CODEGEN_LOGI("dynloopAxes from attr is %s", IntVecToStr(dynloopAxes).c_str());
        return dynloopAxes;
    } else {
        GetOpAttr(OpAttributeKey::loopAxes, loopAxes);
        CODEGEN_LOGI("loopAxes from attr is %s", IntVecToStr(loopAxes).c_str());
    }

    // use dst shape as loop axes in main block
    std::vector<SymbolicScalar> newLoopAxes;
    for (size_t i = 0; i < loopAxes.size(); ++i) {
        SymbolicScalar axis = SymbolicScalar(loopAxes[i]);
        newLoopAxes.emplace_back(axis);
    }

    return newLoopAxes;
}

void CodeGenOpNPU::UpdateLoopInfo()
{
    if (SUPPORT_VF_FUSE_OPS.find(opCode) == SUPPORT_VF_FUSE_OPS.end()) {
        return;
    }

    std::vector<SymbolicScalar> loopAxes = GetLoopAxes();
    if (loopAxes.empty()) {
        return;
    }

    bool isLoopStart{false};
    if ((isMainBlock && GetOpAttr(OpAttributeKey::loopGroupStart, isLoopStart) && isLoopStart) ||
        (GetOpAttr(OpAttributeKey::dynloopGroupStart, isLoopStart) && isLoopStart)) {
        forBlkMgr_->LoopStart();
        forBlkMgr_->UpdateAxesList(loopAxes);
    }

    // Add TileTensor info in loop
    CODEGEN_LOGI("opCode %s has loopAxes: %s", opCodeStr.c_str(), IntVecToStr(loopAxes).c_str());
    for (int i = 0; i < operandCnt; ++i) {
        if (ShouldSkipProcInLoop(i)) {
            continue;
        }
        TileTensorShape tileTensorShape = BuildTileTensorShapeInLoop(i);
        CODEGEN_LOGI("tileTensorShape: isInLoop is %s newShape is %s, newRawShape is %s, newDynValidShape is %s",
                     tileTensorShape.isInLoop ? "true" : "false", IntVecToStr(tileTensorShape.shape).c_str(),
                     IntVecToStr(tileTensorShape.rawShape).c_str(),
                     IntVecToStr(tileTensorShape.dynamicValidShape).c_str());
        TileTensorUsing tileTensorUsing{
            functionType == FunctionType::STATIC || isMainBlock, operandDtype[i],       operandType[i],
            static_cast<int>(tileTensorShape.rawShape.size()),   tileTensorShape.shape, tileTensorShape.rawShape};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensor tileTensor = BuildTileTensor(i, usingType, tileTensorShape);
        forBlkMgr_->AddTensorInLoopBody(tensorNames_[i], tileTensor, originalOp, opCode);
    }
}

// Get last 2 dim of shape
TileTensorShape CodeGenOpNPU::BuildTileTensorShapeInLoop(int paramIdx)
{
    auto newShape = GetShapeInLoop(shape[paramIdx]);
    auto newRawShape = GetShapeInLoop(rawShape[paramIdx]);
    auto newDynValidShape = GetShapeInLoop<SymbolicScalar>(dynamicValidShape[paramIdx]);
    return {true, newShape, newRawShape, newDynValidShape};
}

std::string CodeGenOpNPU::PrintCoord(size_t dim, const std::string& coord) const
{
    std::string ret = COORD;
    ret.append(std::to_string(dim)).append(DIM).append(coord);
    return ret;
}

std::pair<std::string, std::string> CodeGenOpNPU::PrintDstSrcCoordFromAttr(int dstIdx, int srcIdx) const
{
    std::vector<std::string> dstOffset;
    FillParamWithFullInput(dstOffset, GetOffsetFromAttr(dstIdx));
    std::vector<std::string> srcOffset;
    FillParamWithFullInput(srcOffset, GetOffsetFromAttr(srcIdx));
    std::string coordCpDst = WrapParamByParentheses(dstOffset);
    std::string coordDst = PrintCoord(rawShape[dstIdx].size(), coordCpDst);
    std::string coordCpSrc = WrapParamByParentheses(srcOffset);
    std::string coordSrc = PrintCoord(rawShape[srcIdx].size(), coordCpSrc);
    return {coordDst, coordSrc};
}

TileTensor CodeGenOpNPU::QueryTileTensorByIdx(int paramIdx) const
{
    const int tensorMagic = operandWithMagic[paramIdx];
    const int opMagic = originalOp.GetOpMagic();
    const TileTensor* tileTensor = nullptr;
    bool isInLoop = forBlkMgr_ != nullptr && forBlkMgr_->IsInLoop();
    if (isInLoop) {
        tileTensor = sm->QueryTileTensorInLoopByMagic(tensorMagic, opMagic);
        // some tensor in loop is reused same tensor out of loop
        if (tileTensor == nullptr) {
            tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
        }
    } else {
        tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
    }

    if (tileTensor != nullptr) {
        CODEGEN_LOGI("QueryTileTensorByIdx found: %s", tileTensor->ToString().c_str());
        return *tileTensor;
    }

    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false)
        << "TileTensor: paramIdx " << paramIdx << ", tensor magic " << tensorMagic << ", op magic " << opMagic
        << ", isInLoop " << isInLoop << " is not found !!!";
    static TileTensor emptyTileTensor;
    return emptyTileTensor;
}

std::vector<SymbolicScalar> CodeGenOpNPU::GetOffsetFromAttr(int idx) const { return offsetFromAttr[idx]; }

std::string CodeGenOpNPU::InsertOpComment(std::string& tileOpSourceCode) const
{
    std::ostringstream os;

    if (config::GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE) == CFG_DEBUG_ALL) {
        tileOpSourceCode.erase(tileOpSourceCode.find_last_not_of(" \n\r\t") + 1);
        // Add comment after op. e.g. [opmagic:10016]
        os << " // [opMagic:" << originalOp.GetOpMagic() << "]\n";
        tileOpSourceCode.append(os.str());
        os.str("");
    }

    // Add comment before op
    for (auto& c : originalOp.GetCommentList()) {
        os << "/*" << c << "*/\n";
    }
    os << tileOpSourceCode;
    return os.str();
}

std::string CodeGenOpNPU::QueryTileTensorNameByIdx(int paramIdx) const
{
    const TileTensor& tileTensor = QueryTileTensorByIdx(paramIdx);
    return tileTensor.tensorName;
}

std::string CodeGenOpNPU::QueryTileTensorTypeByIdx(int paramIdx) const
{
    const TileTensor& tileTensor = QueryTileTensorByIdx(paramIdx);
    return tileTensor.usingType;
}

std::string CodeGenOpNPU::GenGmCheck() const
{
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) != CFG_RUNTIME_DEBUG_GM_OUT_OF_BOUNDS) {
        return "";
    }

    std::shared_ptr<CopyOpAttribute> attr = std::dynamic_pointer_cast<CopyOpAttribute>(originalOp.GetOpAttribute());
    if (attr == nullptr) {
        return "";
    }

    auto checkInfo = attr->GetGmOutOfRangeCheck();
    if (checkInfo == nullptr) {
        CODEGEN_LOGI_FULL("checkInfo is null, op is %s", originalOp.Dump().c_str());
        return "";
    }

    std::string gmOffset = SymbolicExpressionTable::BuildExpression(checkInfo->oneDimOffset.GetSpecifiedValue());
    std::string gmExtent = SymbolicExpressionTable::BuildExpression(checkInfo->oneDimExtent.GetSpecifiedValue());
    std::string gmTotalSize = SymbolicExpressionTable::BuildExpression(checkInfo->totalSize.GetSpecifiedValue());
    std::string rw = checkInfo->accessType == GmOutOfRangeCheckInfo::AccessType::READ_GM ? "1" : "0";

    std::ostringstream os;
    os << "CheckInvalidAccessOfDDR" << WrapParamByParentheses({gmTotalSize, gmOffset, gmExtent, rw}) << STMT_END;
    return os.str();
}

std::string CodeGenOpNPU::GenOpCode() const
{
    std::string tileOpSourceCode;
    auto iter = opsGenMap_.find(opCode);
    if (iter != opsGenMap_.end()) {
        tileOpSourceCode = iter->second();
    } else {
        // To aid in testing, do not use ASSERT.
        return std::string{"CAN NOT HANDLE OP: " + opCodeStr};
    }

    std::string ret = GenGmCheck();
    ret += InsertOpComment(tileOpSourceCode);

    if (forBlkMgr_ == nullptr || !forBlkMgr_->IsInLoop()) {
        CODEGEN_LOGI_FULL("op codegen result: \n, %s", ret.c_str());
        return ret;
    }

    forBlkMgr_->AddOpInLoopBody(ret);

    bool isLoopEnd{false};
    if (isMainBlock) {
        GetOpAttr(OpAttributeKey::loopGroupEnd, isLoopEnd);
    } else {
        GetOpAttr(OpAttributeKey::dynloopGroupEnd, isLoopEnd);
    }
    if (!isLoopEnd) {
        return "";
    }

    ret = forBlkMgr_->Print();
    forBlkMgr_->OutLoop();
    CODEGEN_LOGI_FULL("op codegen result: \n, %s", ret.c_str());
    return ret;
}

std::string CodeGenOpNPU::GetLastUse() const
{
    if (!opAttrs.count(OpAttributeKey::lastUse)) {
        return "";
    }
    std::vector<int64_t> val;
    GetOpAttr(OpAttributeKey::lastUse, val);
    int valSize = val.size();
    ASSERT(OperErr::ATTRIBUTE_INVALID, valSize != 0) << "GetLastUse error!!!";
    std::ostringstream oss;
    oss << "LastUse" << valSize << "Dim";
    oss << WrapParamByAngleBrackets(val);
    return oss.str();
}

std::vector<std::string> CodeGenOpNPU::GetTileOpParamsByOrder(int paramCnt) const
{
    int tileOpParamCnt = paramCnt == 0 ? operandCnt : paramCnt;
    ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, tileOpParamCnt > 0 && tileOpParamCnt <= operandCnt)
        << "paramCnt: " << paramCnt << " should be greater than 0 and not exceed operandCnt";

    std::vector<std::string> params;
    for (int i = 0; i < tileOpParamCnt; ++i) {
        params.emplace_back(QueryTileTensorNameByIdx(i));
    }

    CODEGEN_LOGI("TileOp params is %s", IntVecToStr(params).c_str());
    return params;
}

std::vector<std::string> CodeGenOpNPU::GetTileOpParamsWithTmpBuf(const std::vector<unsigned>& tmpBufIdx) const
{
    std::vector<std::string> params;
    for (int i = 0; i < operandCnt; ++i) {
        if (std::find(tmpBufIdx.begin(), tmpBufIdx.end(), i) == tmpBufIdx.end()) {
            params.emplace_back(QueryTileTensorNameByIdx(i));
        }
    }

    for (auto& ti : tmpBufIdx) {
        params.emplace_back(QueryTileTensorNameByIdx(ti));
    }

    CODEGEN_LOGI("TileOp params is %s", IntVecToStr(params).c_str());
    return params;
}

std::string CodeGenOpNPU::PrintTileOpWithFullParamsInOrder() const
{
    std::vector<std::string> params = GetTileOpParamsByOrder();
    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(params) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintTileOpWithFullParamsTmpBuf(const std::vector<unsigned>& tmpBufIdx) const
{
    std::vector<std::string> params = GetTileOpParamsWithTmpBuf(tmpBufIdx);
    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(params) << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
