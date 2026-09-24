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
 * \file codegen_op_npu.h
 * \brief
 */

#ifndef CODEGEN_OP_NPU_H
#define CODEGEN_OP_NPU_H

#include <utility>
#include <map>
#include <functional>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/npu/op_print_param_def.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/stmt_mgr/codegen_for_block.h"
#include "codegen/codegen_op.h"

namespace npu::tile_fwk {

struct CodeGenOpNPUCtx : public CodeGenOpCtx {
    std::shared_ptr<ForBlockManager> forBlockManager{nullptr};

    CodeGenOpNPUCtx(std::shared_ptr<SymbolManager> sm, Function& tf, Function& sf, const Operation& op,
                    bool isMainBlk = false, bool isDynAligned = false, std::shared_ptr<ForBlockManager> fbm = nullptr)
        : CodeGenOpCtx(std::move(sm), tf, sf, op, isMainBlk, isDynAligned), forBlockManager(std::move(fbm))
    {}
};

class CodeGenOpNPU : public CodeGenOp {
public:
    explicit CodeGenOpNPU(const CodeGenOpNPUCtx& ctx);
    ~CodeGenOpNPU() override = default;

    std::string GenBarrier() const;
    std::string GenSyncSetOp() const;
    std::string GenSyncWaitOp() const;
    std::string GenCVSyncSetOp() const;
    std::string GenCVSyncWaitOp() const;
    std::string GenFFTSCrossCoreSyncOp() const;
    std::string GenWaitFlagDevOp() const;

    std::string GenMemL1ToBt() const;
    std::string GenMemL1CopyIn() const;
    std::string GenMemL1CopyOut() const;
    std::string GenMemL1CopyInConv() const;
    std::string GenMemL0CCopyOutConv() const;
    std::string GenMemL1ToFB() const;
    std::string GenMemL0CCopyOut() const;
    std::string GenMemL0CToL1() const;
    std::string GenMemL1ToL0Load3D() const;
    std::string GenMemL1ToL0Load2D() const;
    std::string GenAxpyOp() const;

    std::string GenMemL1ToL0() const;

    std::string GenUBCopyIn() const;
    std::string GenUBCopyOut() const;
    std::string GenUBToL1TileTensor() const;
    std::string GenUBToUBND2NZTileTensor() const;
    std::string GenReshapeCopyIn() const;
    std::string GenReshapeCopyOut() const;

    std::string PrintGatherInL1TileTensor() const;
    std::string GenGatherInL1() const;
    std::string GenGatherInUB() const;
    std::string PrintGatherInUBDynamicUnaligned() const;
    std::string PrintGatherInUBLayout() const;

    std::string GenUnaryOp() const;
    std::string GenUnaryOpWithTmpBuff() const;
    std::string GenOnlineSoftmaxOp() const;
    std::string GenOnlineSoftmaxUpdateOp() const;
    std::string GenArgReduceWithValue() const;
    std::string GenQuantMXOp() const;

    std::string GenLogicalNotOp() const;
    std::string GenLogicalAndOp() const;

    std::string GenBinaryOp() const;
    std::string GenVectorScalarOp() const;
    std::string GenBinaryOpWithTmp() const;
    std::string GenVectorScalarOpWithTmp() const;
    std::string GenPairArgReduce() const;

    std::string GenInterleaveLikeOp() const;

    std::string GenCubeOpMatmul() const;
    std::string GenCubeOpMatmulAcc() const;

    std::string GenCastOp() const;

    std::string GenDupOp() const;

    std::string GenTransposeDataMove() const;
    std::string GenTransData() const;

    std::string GenGatherElementOp() const;
    std::string GenGatherMaskOp() const;

    std::string GenRangeOp() const;
    std::string GenUniformOp() const;
    std::string PrintRangeTileTensor(const std::string& startVal, const std::string& stepVal,
                                     const std::string& tileIdxExpr) const;
    std::string GenL0CToUBTileTensor() const;

    std::string GenScatterElementSOp() const;
    std::string GenScatterOp() const;

    std::string GenIndexAddUBOp() const;
    std::string GenIndexAddOp() const;

    std::string GenIndexPutOp() const;

    std::string GenIndexOutCastOp() const;
    std::string PrintIndexOutCastTileTensor() const;

    std::string GenCumOperationOp() const;
    std::string GenTriULOp() const;
    std::string PrintGatherDynamicUnaligned() const;
    std::string PrintGatherLayout() const;
    std::string GenGatherOp() const;
    std::string GenPermuteOp() const;
    std::string GenGatherFromUBOp() const;

    std::string GenMemCopyCube(bool isLocalToGM, unsigned uf = 0) const;
    std::string GenMemL1SpillToGM(bool isLocalToGM, unsigned uf) const;

    std::string GenBinaryWithBrc() const;

    std::string GenBitSortOp() const;
    std::string GenMrgSortOp() const;
    std::string GenExtractOp() const;
    std::string GenTiledMrgSortOp() const;
    std::string GenRadixSelectOp() const;
    std::string GenSortOpWithParams(const std::set<int>& idx) const;
    std::string GenSortOp() const;
    std::string GenCompareAndSwapOp() const;
    std::string GenMergeOp() const;

    std::string GenTopKSortOp() const;
    std::string GenTopKMergeOp() const;
    std::string GenTopKExtractOp() const;

    std::string GenTwoTileMrgSort() const;
    std::string GenExtractSingleOp() const;

    std::string GenParamsStr(const std::unordered_set<int32_t>& skipOperands = {}) const;

    std::string GenDistOp() const;
    std::string GetTemplateDType() const;
    std::string GenTemplateParams() const;
    std::string GenOffsets(int32_t operandIndex) const;
    std::string GenShapes(int32_t operandIndex) const;
    std::string GenRawShapes(int32_t operandIndex) const;
    std::string GenExtraParamsStr() const;
    std::string GenOffsetsAndRawShapes(int32_t operandIndex) const;
    std::string GenDynOffCoord(int32_t operandIndex) const;
    std::string GenOffCoord(int32_t operandIndex) const;
    std::string GenDynValidShape(int32_t operandIndex) const;
    std::string GenAicpuCallOp() const;

    std::string GenQuantizeOp() const;
    std::string GenDequantizeOp() const;
    std::string PrintQuantizeTileTensor() const;
    std::string PrintDequantizeTileTensor() const;

    std::string GenWhereOp() const;

    std::string GenOpCode() const override;

    std::string QueryTileTensorNameByIdx(int paramIdx) const;
    std::string QueryTileTensorTypeByIdx(int paramIdx) const;

    // check whether the GM address is out of bounds
    std::string GenGmCheck() const;

protected:
    virtual TileTensor QueryTileTensorByIdx(int paramIdx) const;
    virtual std::vector<SymbolicScalar> GetOffsetFromAttr(int idx) const;
    std::vector<std::string> GetTileOpParamsByOrder(int paramCnt = 0) const;
    std::vector<std::string> GetTileOpParamsWithTmpBuf(const std::vector<unsigned>& tmpBufIdx) const;
    std::string PrintTileOpWithFullParamsInOrder() const;
    std::string PrintTileOpWithFullParamsTmpBuf(const std::vector<unsigned>& tmpBufIdx) const;

    std::string InsertOpComment(std::string& tileOpSourceCode) const;

    int64_t GetConvCopyInMode() const;
    std::string GetConvCopyOutMode() const;
    std::string GenMemL1CopyInConvNZ2NZ(const std::string& dstTensor, const std::string& srcTensor,
                                        const std::string& copyInModeStr) const;
    void GetDynamicOffsetExpr(const std::vector<SymbolicScalar>& dynOffset, bool isConv3D,
                              std::vector<std::string>& gmOffsetExpr, std::vector<int64_t>& staticOffsets) const;
    void GetNZ2NZDynamicOffsetExpr(const std::vector<SymbolicScalar>& dynOffset, bool isConv3D, bool isFmap,
                                   std::vector<std::string>& gmOffsetExpr,
                                   std::vector<std::string>& staticOffsets) const;
    std::vector<std::string> BuildCopyInParamList(const std::string& dstTensor, const std::string& srcTensor,
                                                  const std::vector<std::string>& gmOffsetExpr,
                                                  const std::vector<int64_t>& staticOffsets,
                                                  const std::vector<std::string>& srcShape, bool isConv3D) const;
    std::vector<std::string> BuildCopyOutParamList(const std::string& dstTensor, const std::string& srcTensor,
                                                   const std::vector<std::string>& gmOffsetExpr,
                                                   const std::vector<int64_t>& staticOffsets, const std::string& realM,
                                                   const std::string& realN, const std::string& realCutW,
                                                   const std::string& cutW) const;

    std::string GenTemplateParamsForPutAndGet() const;
    std::string GenTemplateParamsForLoad() const;
    std::string GenTemplateParamsForStore() const;
    std::string GenTemplateParamsForSignal() const;
    std::string GenTemplateParamsForSet() const;
    std::string GenTemplateParamsDefault() const;

    std::string GenOffsetsAndRawShapesForShmemPut() const;
    std::string GenOffsetsAndRawShapesForShmemGet() const;
    std::string GenOffsetsAndRawShapesForShmemStore() const;
    std::string GenOffsetsAndRawShapesForShmemLoad() const;
    std::string GenOffsetsAndRawShapesForShmemSignal() const;
    std::string GenOffsetsAndRawShapesForShmemSet() const;
    std::string GenOffsetsAndRawShapesDefault() const;
    std::string GenTargetRankStr() const;

    void AddBinaryPrecisionTypeParm(std::vector<std::string>& templateParamList) const;
    void UpdateTileTensorInfo();
    void UpdateLoopInfo();
    std::vector<SymbolicScalar> GetLoopAxes();
    TileTensorShape BuildTileTensorShapeInLoop(int paramIdx);
    bool ShouldSkipProcInLoop(int paramIdx);

    template <typename T = int64_t>
    std::vector<T> GetShapeInLoop(const std::vector<T>& input)
    {
        ASSERT(OperErr::TENSOR_DIM_EXCEEDED, input.size() > SHAPE_DIM2)
            << "input size " << input.size() << " should be greater than 2";
        std::vector<T> reservedShapeExceptLoopAxes = {*(input.rbegin() + 1), input.back()};
        return reservedShapeExceptLoopAxes;
    }

    int GetCacheModeFlag(const std::string& cacheMode) const;

    template <typename T>
    bool GetAttrFromMap(const std::map<std::string, std::any>& attrMap, const std::string& key, T& value) const
    {
        auto it = attrMap.find(key);
        if (it == attrMap.end()) {
            CODEGEN_LOGI("can not find key: %s in attrMap", key.c_str());
            return false;
        }
        if (it->second.type() == typeid(T)) {
            value = AnyCast<T>(it->second);
            return true;
        }
        CODEGEN_LOGE(GenCodeErr::DATA_TYPE_MISMATCHED, "Type of attribute %s from PASS is mismatch: %s != %s",
                     key.c_str(), it->second.type().name(), typeid(T).name());
        return false;
    }

    template <typename T>
    bool GetOpAttr(const std::string& key, T& value) const
    {
        return GetAttrFromMap(opAttrs, key, value);
    }

    template <typename T>
    bool GetTensorAttr(int idx, const std::string& key, T& value) const
    {
        ASSERT(GenCodeErr::PARAM_IDX_INVALID, idx >= 0 && idx < MAX_OPERANDS)
            << "idx " << idx << " is out of range [0, " << MAX_OPERANDS << ")";
        return GetAttrFromMap(tensorAttrs[idx], key, value);
    }

    std::string GetLastUse() const;

    virtual TileTensor BuildTileTensor(int paramIdx, const std::string& usingType,
                                       const TileTensorShape& tileTensorShape = {});
    virtual void UpdateTileTensorShapeAndStride(int paramIdx, TileTensor& tileTensor, bool isSpillToGm,
                                                const TileTensorShape& tileTensorShape = {});
    std::vector<std::string> BuildStride(const std::vector<int64_t>& input);

    std::string GenMemCopyVar(bool isCopyLocalToGM, bool isSpillToGm = false, unsigned uf = 0) const;

    std::string GenGMAddrExprWithOffset(unsigned gmParamIdx, const std::string& addrExpr) const;

    // Add offset of local buffer variable when the variable is generated by spliting from "view" operation.
    template <typename T = std::string, typename... Args>
    void AppendLocalBufVarOffsetInOrder(Args&... args) const
    {
        tempVarsMap.clear();
        tempKey = 0;
        AppendLocalBufVarOffsetInOrderImpl<T>(args...);
    }

    void AppendLocalBufferVarOffset(const std::map<unsigned, std::reference_wrapper<std::string>>& vars) const;

    // get start offset in total block
    SymbolicScalar GetOperandStartOffset(int operandIdx) const;

    std::string GetGmTensorAddrByAttr(unsigned gmParamIdx) const;

    virtual std::string GenGmParamVar(unsigned gmParamIdx) const;

    std::vector<std::string> GenGetParamMacroPacked(unsigned gmParamIdx, int dim, const std::string& prefix) const;
    std::vector<std::string> GenDynRawShapePacked(unsigned gmParamIdx) const;
    std::vector<std::string> GenDynStridePacked(const std::vector<std::string>& dynRawShape) const;

    std::vector<std::string> GenParamIdxExprByIndex(unsigned gmParamIdx, int dim, const std::string& prefix) const;

    std::vector<std::string> GenSymbolicArgument(const std::vector<SymbolicScalar>& exprList) const;

    std::string GenMemUBTransfer(bool isCopyUBToGM) const;
    std::string GenVectorScalarOpByMode(VecScalMode mode) const;
    std::string GenVectorScalarOpScalarMode() const;
    std::string GenCubeOp(bool zeroC) const;
    virtual void UpdateGemvTileOpName();
    std::string GenRemainderSOp() const;
    std::string GenCmpOp() const;
    std::string GenHypotOp() const;
    std::string GenPreluOp() const;
    std::string GenPadOp() const;

    std::string PrintDupOp(const PrintDupOpParam& param) const;
    std::string PrintDupOpDynUnaligned(const PrintDupOpParam& param) const;
    std::string PrintDupOpStatic(const PrintDupOpParam& param) const;
    std::string PrintDupTileTensor(const PrintDupOpParam& param) const;

    std::string PrintRowMaxline(const PrintUnaryParam& param) const;
    std::string PrintRowMaxlineTileTensor() const;
    std::string PrintRowMaxlineDynamicUnaligned(const PrintUnaryParam& param) const;
    std::string PrintRowMaxlineStatic(const PrintUnaryParam& param) const;

    std::string PrintReduceEx(const PrintUnaryParam& param) const;
    std::string PrintReduceExStatic(const PrintUnaryParam& param) const;

    std::string PrintReduceSum(const PrintUnaryParam& param) const;
    std::string PrintReduceSumStatic(const PrintUnaryParam& param) const;

    std::string PrintVcopy(const PrintUnaryParam& param) const;
    std::string PrintVcopyStatic(const PrintUnaryParam& param) const;

    std::string PrintVnchwconv(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintVnchwconvStatic(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintPermuteLayout() const;
    std::string PrintUnaryWithTmpTileTensor() const;

    std::string PrintCompact(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintCompactStatic(const PrintUnaryTmpBuffParam& param) const;

    std::vector<std::string> GenTileOpParamForNormalCopyTileTensor(unsigned gmIdx, bool useOffsetFromAttr = true) const;
    void AppendGmValidShapeForReshapeCopy(std::vector<std::string>& tileOpParamList, unsigned gmIdx) const;
    std::string PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam& param) const;
    std::string PrintMemCopyWithL0CStatic(const PrintMemCopyWithL0CParam& param) const;
    std::string PrintMemCopyWithL0CDynamic(const PrintMemCopyWithL0CParam& param) const;
    std::string PrintL0CCopyOutDynamicUnalign(const PrintMemCopyWithL0CParam& param,
                                              std::vector<std::string>& gmShapeExpr,
                                              std::vector<std::string>& gmOffsetExpr) const;
    std::string PrintMemCopyWithL0CTileTensor(const PrintMemCopyWithL0CParam& param) const;

    std::pair<std::string, std::string> GetOuterInnerValueStr(unsigned gmIdx, const std::vector<int64_t>& gmShape,
                                                              bool isSpillingToGM = false) const;
    std::string PrintMemCopyWithL1(const PrintMemCopyWithL1Param& param) const;
    std::string PrintMemCopyWithL1Static(const PrintMemCopyWithL1Param& param) const;
    std::string PrintMemCopyWithL1Dynamic(const PrintMemCopyWithL1Param& param) const;
    std::string PrintMemCopyWithL1TileTensor(const PrintMemCopyWithL1Param& param) const;
    std::string PrintMemCopyInWithL1TileTensor(const PrintMemCopyWithL1Param& param) const;
    std::string PrintMemCopyOutWithL1TileTensor(const PrintMemCopyWithL1Param& param) const;

    std::string PrintMemCopyWithUB(PrintMemCopyWithUBParam& param) const;
    std::string PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam& param) const;
    std::string PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam& param) const;
    std::string PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam& param) const;
    std::string PrintMemCopyWithUBTileTensor(const PrintMemCopyWithUBParam& param) const;
    virtual std::vector<std::string> GetGmOffsetForTileTensor(unsigned gmIdx) const;

    std::string PrintGather(const PrintGatherParam& param) const;
    std::string PrintGatherDynamicUnaligned(const PrintGatherParam& param) const;
    std::string PrintGatherStatic(const PrintGatherParam& param) const;

    std::string PrintBinaryScalar(const PrintBinaryScalarParam& param) const;
    std::string PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam& param) const;
    std::string PrintBinaryScalarStatic(const PrintBinaryScalarParam& param) const;

    std::string PrintUnary(const PrintUnaryParam& param) const;
    std::string PrintUnaryTileTensor() const;
    void AddUnaryPrecisionTypeParm(std::vector<std::string>& templateParamList) const;
    std::string PrintUnaryDynamicUnaligned(const PrintUnaryParam& param) const;
    std::string PrintUnaryStatic(const PrintUnaryParam& param) const;

    std::string PrintBitwiseNot() const;

    SortParam PrepareSortParam() const;
    TiledSortParam PrepareTiledSortParam() const;
    std::string PrintTileSortTileTensor() const;
    std::string PrintTiledSortDynamicUnaligned(const TiledSortParam& param) const;
    std::string PrintTiledMrgSortDynamicUnaligned(const TiledSortParam& param) const;
    std::string PrintSortDynamicUnaligned(const SortParam& param) const;
    std::string PrintSortStatic(const SortParam& param) const;
    std::string PrintSortTileTensor() const;
    std::string PrintBitSortDynamicUnaligned(const SortParam& param) const;
    std::string PrintBitSortStatic(const SortParam& param) const;
    std::string PrintMrgSortDynamicUnaligned(const SortParam& param) const;
    std::string PrintMrgSortStatic(const SortParam& param) const;
    std::string PrintSortUBDynamicUnaligned(bool containDstType) const;

    std::string PrintBinaryStatic(const PrintBinaryParam& param) const;
    std::string PrintBinaryDynamicUnaligned(const PrintBinaryParam& param) const;
    std::string PrintBinaryTileTensor() const;
    std::string PrintBinary(const PrintBinaryParam& param) const;
    std::string PrintArgReduceTileTensor() const;

    std::string PrintBinaryBrcStatic(const PrintBinaryBrcParam& param) const;
    std::string PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam& param) const;
    std::string PrintBinaryBrc(const PrintBinaryBrcParam& param) const;

    std::string PrintTransposeDataMove(const PrintTransposeDataMoveParam& param) const;
    std::string PrintTransposeDataMoveLayout(const PrintTransposeDataMoveParam& param) const;
    std::string PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam& param) const;
    std::string PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam& param) const;
    std::string PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam& param) const;

    std::string PrintTransDataLayoutTmp() const;
    std::string PrintTransDataLayout(const std::vector<std::string>& param) const;

    std::string PrintGatherElementDynamicUnaligned(const PrintGatherEleParam& param) const;
    std::string PrintGatherElementStatic(const PrintGatherEleParam& param) const;
    std::string PrintGatherElementTileTensor(const PrintGatherEleParam& param) const;

    std::string PrintIndexOutCast(const PrintIndexOutCastParam& param) const;
    std::string PrintIndexOutCastStatic(const PrintIndexOutCastParam& param) const;
    std::string PrintIndexOutCastDynamic(const PrintIndexOutCastParam& param) const;
    std::string PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam& param) const;

    std::string PrintExpandDynamicUnaligned(const PrintUnaryParam& param, std::vector<int> expandAxes) const;
    std::string PrintExpandLayout(std::vector<int> expandAxes) const;
    std::string PrintExpand(const std::string& s0Var, const std::string& dVar, const std::string& srcDtypeStr,
                            const std::string& dstDtypeStr) const;
    std::string PrintOneHot(const PrintUnaryParam& param) const;
    std::string PrintOneHotLayout() const;
    std::string PrintUnaryOpWithTmpBuff() const;
    std::string PrintRound() const;
    std::string PrintRoundLayout() const;
    std::string PrintUnaryOpWithTmpTwoBuff() const;

    DynamicParamPackMTE PrepareDynamicShapeInfoForMTE(int dynShapeIdx, int ShapeDim = SHAPE_DIM4,
                                                      bool isGmSpill = false) const;

    std::string PrintReduceLastAxis(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintReduceLastAxisDynamicUnalign(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintReduceLastAxisTileTensor() const;

    std::string PrintRowSumline(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintRowSumlineTileTensor() const;
    std::string PrintRowSumlineDynamicUnaligned(const PrintUnaryTmpBuffParam& param) const;
    std::string PrintRowSumlineStatic(const PrintUnaryTmpBuffParam& param) const;

    std::string PrintExtractStatic() const;
    std::string PrintExtractDynamicUnaligned() const;
    std::string PrintExtractTileTensor() const;

    std::string PrintCastDynamicUnaligned(const PrintUnaryParam& param) const;
    std::string PrintCastTileTensor() const;
    std::string PrintVectorScalarTileTensor(const PrintUnaryParam& param) const;
    std::string PrintVectorScalarOpDynamicUnalign(const PrintUnaryParam& param) const;
    std::string PrintMemL1ToL0TileTensor() const;
    std::string PrintMatmulTileTensor(bool isAcc) const;
    std::string PrintMatmulTileTensor(bool isAcc,
                                      std::unordered_map<OperandType, std::string>& tensorWithMemType) const;
    std::string PrintTmove() const;
    std::string PrintL0CToL1TileTensor() const;

    std::string PrintScatterElementSOpStatic(const PrintScatterElemParam& param) const;
    std::string PrintScatterElementSOpDynamicUnaligned(const PrintScatterElemParam& param) const;
    std::string PrintScatterElementSTileTensor(const PrintScatterElemParam& param) const;
    std::string PrintScatterOpDynamicUnaligned(const PrintScatterParam& param) const;
    std::string PrintScatterTileTensor(const PrintScatterParam& param) const;

    std::string PrintIndexAddUBDynamicUnaligned(const PrintIndexAddParam& param) const;
    std::string PrintIndexAddUBTileTensor(const PrintIndexAddParam& param) const;

    std::string PrintIndexPut(const PrintIndexPutParam& param) const;
    std::string PrintIndexPutLayout(size_t indicesSize, bool accumulate) const;
    std::string PrintIndexPutDynamicUnaligned(const PrintIndexPutParam& param) const;

    std::string PrintTriULTileTensor(const std::string& diagonal, bool isUpper) const;

    std::string PrintCumSumDynamicUnaligned(const PrintCumSumParam& param) const;
    std::string PrintCumOperationTileTensor(int axis, bool is_sum) const;

    WhereParam PrepareWhereParam() const;
    void GetWhereVarAndType(std::vector<std::string>& varExpr, std::vector<std::string>& dataTypeExpr) const;
    std::string PrintWhereOp(const WhereParam& param) const;
    std::string PrintWhereOpTileTensor(const WhereParam& param) const;

    std::string PrintCmpTileTensor() const;
    std::string PrintHypotTileTensor() const;
    std::string PrintPreluTileTensor() const;
    std::string PrintPadTileTensor() const;
    std::string PrintLogicalAndTileTensor() const;
    std::string PrintLogicalNotTileTensor() const;

    void InitOpsGenMap();
    void InitScalaOpsMap();
    void InitMTEOpsMap();
    void InitVecOpsMap();
    void InitCubeOpsMap();
    void InitDistOpsMap();
    void InitPerfOpsMap();
    void InitAICPUOpsMap();

    std::string PrintCoord(size_t dim, const std::string& coord) const;
    std::pair<std::string, std::string> PrintDstSrcCoordFromAttr(int dstIdx = ToUnderlying(MISOIdx::DST_IDX),
                                                                 int srcIdx = ToUnderlying(MISOIdx::SRC0_IDX)) const;
    template <typename T>
    void FillParamWithFullInput(std::vector<std::string>& paramList, const std::vector<T>& input) const
    {
        FillParamWithInput(paramList, input, 0, input.size());
    }
    template <typename T>
    void FillParamWithInputExceptFirst(std::vector<std::string>& paramList, const std::vector<T>& input) const
    {
        FillParamWithInput(paramList, input, 1, input.size());
    }

    std::string PrintSyncInSingleKernel(bool isWait = false) const;

    const std::unordered_map<Opcode, std::function<std::string()>> mteFixPipeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> unaryOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> binaryOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> compositeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> sortOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> cubeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> syncOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> distributeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> gatherScatterOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> normalVecOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> quantOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> perfOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> aicpuOps_;

    std::unordered_map<Opcode, std::function<std::string()>> opsGenMap_;

    std::shared_ptr<ForBlockManager> forBlkMgr_;

    // <parameter index, tensor name>
    std::unordered_map<int, std::string> tensorNames_;

    mutable std::map<unsigned, std::reference_wrapper<std::string>> tempVarsMap;
    mutable unsigned tempKey = 0;
    template <typename T>
    void AppendLocalBufVarOffsetInOrderImpl() const
    {
        if (!tempVarsMap.empty()) {
            AppendLocalBufferVarOffset(tempVarsMap);
            tempVarsMap.clear();
            tempKey = 0;
        }
    }

    template <typename T, typename FirstArg, typename... RestArgs>
    void AppendLocalBufVarOffsetInOrderImpl(FirstArg& first_arg, RestArgs&... rest_args) const
    {
        bool isValidDType = std::is_same_v<std::remove_reference_t<FirstArg>, T>;
        ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, isValidDType) << "All arguments must be T (default: std::string)!";
        tempVarsMap.emplace(tempKey++, std::ref(first_arg));
        AppendLocalBufVarOffsetInOrderImpl<T>(rest_args...);
    }
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_NPU_H
