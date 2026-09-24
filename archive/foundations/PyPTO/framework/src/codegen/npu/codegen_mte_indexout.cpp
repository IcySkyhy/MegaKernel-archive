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
 * \file codegen_mte_indexout.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {
int CodeGenOpNPU::GetCacheModeFlag(const std::string& cacheMode) const
{
    const int PA_BNSD = 0;
    const int PA_NZ = 1;
    const int PA_BSND = 2;
    int cacheModeFlag = PA_BNSD;
    if (cacheMode == "PA_NZ") {
        cacheModeFlag = PA_NZ;
    } else if (cacheMode == "PA_BSND") {
        cacheModeFlag = PA_BSND;
    }
    return cacheModeFlag;
}

std::string CodeGenOpNPU::PrintIndexOutCastTileTensor() const
{
    auto cacheMode = AnyCast<std::string>(opAttrs.at(OpAttributeKey::cacheMode));
    auto blockSize = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::panzBlockSize));
    int cacheModeFlag = GetCacheModeFlag(cacheMode);

    int dim = rawShape[ID0].size();
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(ID0);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(dim, coordCp);
    const int usedOperandCnt = 3;
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder(usedOperandCnt);
    tileOpParamList.emplace_back(coord);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({std::to_string(cacheModeFlag), std::to_string(blockSize)});
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenIndexOutCastOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::cacheMode)) << "cannot get cacheMode attr";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::panzBlockSize)) << "cannot get panzBlockSize attr";
    auto cacheMode = AnyCast<std::string>(opAttrs.at(OpAttributeKey::cacheMode));
    auto blockSize = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::panzBlockSize));
    unsigned gmIdx = 0;
    unsigned localIdx = 1;

    std::vector<std::string> addrExpr(ID2);
    addrExpr[gmIdx] = GenGmParamVar(gmIdx);

    std::vector<int64_t> src0OriginShape = shape[ID1];
    std::vector<int64_t> src1OriginShape = shape[ID2];
    std::vector<int64_t> src0RawShape = rawShape[ID1];
    std::vector<int64_t> src1RawShape = rawShape[ID2];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::vector<std::string> dataTypeExpr = {dstDtypeStr, src0DtypeStr, src1DtypeStr};

    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    AppendLocalBufferVarOffset({{localIdx, std::ref(s0Var)}});

    std::vector gmShape = rawShape[gmIdx];
    CODEGEN_LOGI("genIndexOutCastOp gm shape: %s", IntVecToStr(gmShape).c_str());

    std::vector<int64_t> s0os = NormalizeShape(src0OriginShape, SHAPE_DIM4);
    std::vector<int64_t> gms = NormalizeShape(gmShape, SHAPE_DIM4);
    std::vector<int64_t> s0rs = NormalizeShape(src0RawShape, SHAPE_DIM4);
    std::vector<int64_t> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM4);
    std::string blockSizeStr = std::to_string(blockSize);

    return PrintIndexOutCast(
        {s0Var, s1Var, addrExpr, gms, s0os, s0rs, src1OriginShape, s1rs, dataTypeExpr, cacheMode, blockSizeStr});
}

std::string CodeGenOpNPU::PrintIndexOutCast(const PrintIndexOutCastParam& param) const
{
    if (isSupportTileTensor) {
        return PrintIndexOutCastTileTensor();
    }
    if (isSupportDynamicAligned) {
        return PrintIndexOutCastDynamic(param);
    } else if (isDynamicFunction) {
        return PrintIndexOutCastDynamicUnaligned(param);
    }
    return PrintIndexOutCastStatic(param);
}

// template <typename T,unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned src1Shape0,
// unsigned src1Shape1, unsigned GmShape0, unsigned GmShape1, unsigned GmShape2, unsigned GmShape3>
// TILEOP void TIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ int32_t* index, unsigned Offset0, unsigned
// Offset1) {
std::string CodeGenOpNPU::PrintIndexOutCastStatic(const PrintIndexOutCastParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<int64_t>& gmShape = param.gmShape;
    std::vector<int64_t>& src0OriginShape = param.src0OriginShape;
    std::vector<int64_t>& src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t>& src1OriginShape = param.src1OriginShape;
    std::vector<int64_t>& src1RawShape = param.src1RawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0OriginShape[ID0]), std::to_string(src0OriginShape[ID1]),
                                       std::to_string(src0OriginShape[ID3])});
    paramList.insert(paramList.end(), {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]),
                                       std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1OriginShape[ID0]));
    paramList.emplace_back(std::to_string(src1OriginShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.insert(paramList.end(), {std::to_string(gmShape[ID2]), std::to_string(gmShape[ID3])});
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";

    return oss.str();
}

std::string CodeGenOpNPU::PrintIndexOutCastDynamic(const PrintIndexOutCastParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    std::vector<int64_t>& src0OriginShape = param.src0OriginShape;
    std::vector<int64_t>& src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t>& src1OriginShape = param.src1OriginShape;
    std::vector<int64_t>& src1RawShape = param.src1RawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);

    auto paramPack = PrepareDynamicShapeInfoForMTE(ID0);

    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0OriginShape[ID0]), std::to_string(src0OriginShape[ID1]),
                                       std::to_string(src0OriginShape[ID3])});
    paramList.insert(paramList.end(), {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]),
                                       std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1OriginShape[ID0]));
    paramList.emplace_back(std::to_string(src1OriginShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);

    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    std::vector<int64_t>& src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t>& src1RawShape = param.src1RawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);

    auto paramPack = PrepareDynamicShapeInfoForMTE(ID0);

    auto src0ValidShape = dynamicValidShape[ID1];
    FillVecWithDummyInHead<SymbolicScalar>(src0ValidShape, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    auto src1ValidShape = dynamicValidShape[ID2];

    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]),
                                       std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    paramList.insert(paramList.end(), {SymbolicExpressionTable::BuildExpression(src0ValidShape[ID0]),
                                       SymbolicExpressionTable::BuildExpression(src0ValidShape[ID1]),
                                       SymbolicExpressionTable::BuildExpression(src0ValidShape[ID3])});
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(src1ValidShape[ID0]));
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(src1ValidShape[ID1]));
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

} // namespace npu::tile_fwk
