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
 * \file codegen_vector.cpp
 * \brief
 */

#include "interface/tensor/logical_tensor.h"
#include "codegen_op_npu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
std::string CodeGenOpNPU::PrintSortDynamicUnaligned(const SortParam& param) const
{
    auto dstShape = param.dstShape;
    auto src0Shape = param.srcShape;
    const std::string& s0Var = param.s0Var;
    const std::string& tVar = param.tVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;

    auto dynSrcShape = dynamicValidShape[ID2];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(src0Shape[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmpParam = "(__ubuf__ " + tmpDtypeStr + "*)" + tVar;
    paramList.insert(paramList.end(), {dstParam, srcParam, tmpParam});
    FillParamWithFullInput(paramList, dynSrcShape);
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintSortStatic(const SortParam& param) const
{
    auto dstShape = param.dstShape;
    auto src0Shape = param.srcShape;
    unsigned orisrcShape0 = 0;
    unsigned orisrcShape1 = 0;
    const std::string& s0Var = param.s0Var;
    const std::string& tVar = param.tVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::vector<int64_t>& oriSrc0Shape = shape[1];
    constexpr unsigned defaultDim = 1u;
    if (oriSrc0Shape.size() == 1) {
        orisrcShape0 = defaultDim;
        orisrcShape1 = oriSrc0Shape[0];
    } else {
        orisrcShape0 = oriSrc0Shape[0];
        orisrcShape1 = oriSrc0Shape[1];
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    // static only support 1~2 dim
    paramList.insert(paramList.end(), {std::to_string(dstShape[2]), std::to_string(dstShape[3])});
    paramList.insert(paramList.end(), {std::to_string(src0Shape[2]), std::to_string(src0Shape[3])});
    paramList.insert(paramList.end(), {std::to_string(orisrcShape0), std::to_string(orisrcShape1)});

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmpParam = "(__ubuf__ " + tmpDtypeStr + "*)" + tVar;
    paramList.insert(paramList.end(), {dstParam, srcParam, tmpParam});
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintTiledSortDynamicUnaligned(const TiledSortParam& param) const
{
    auto dstShape = param.dstShape;
    auto srcShape = param.srcShape;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    const std::string& s2Var = param.s2Var;
    const std::string& s3Var = param.s3Var;
    const std::string& tmpVar = param.tmpVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;

    auto dynSrc0Shape = dynamicValidShape[ID2];
    auto dynSrc1Shape = dynamicValidShape[ID3];
    auto dynSrc2Shape = dynamicValidShape[ID4];
    auto dynSrc3Shape = dynamicValidShape[ID5];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrc0Shape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    FillVecWithDummyInHead<SymbolicScalar>(dynSrc1Shape, SHAPE_DIM4 - dynamicValidShape[ID3].size(), 1);
    FillVecWithDummyInHead<SymbolicScalar>(dynSrc2Shape, SHAPE_DIM4 - dynamicValidShape[ID4].size(), 1);
    FillVecWithDummyInHead<SymbolicScalar>(dynSrc3Shape, SHAPE_DIM4 - dynamicValidShape[ID5].size(), 1);

    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0Param = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string src1Param = "(__ubuf__ " + srcDtypeStr + "*)" + s1Var;
    std::string src2Param = "(__ubuf__ " + srcDtypeStr + "*)" + s2Var;
    std::string src3Param = "(__ubuf__ " + srcDtypeStr + "*)" + s3Var;
    std::string tmpParam = "(__ubuf__ " + srcDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dstParam, src0Param, src1Param, src2Param, src3Param, tmpParam});
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(dynSrc0Shape[i].Dump());
    }
    paramList.emplace_back(dynSrc1Shape[ID3].Dump());
    paramList.emplace_back(dynSrc2Shape[ID3].Dump());
    paramList.emplace_back(dynSrc3Shape[ID3].Dump());
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintBitSortDynamicUnaligned(const SortParam& param) const
{
    return PrintSortDynamicUnaligned(param);
}

std::string CodeGenOpNPU::PrintBitSortStatic(const SortParam& param) const { return PrintSortStatic(param); }

std::string CodeGenOpNPU::PrintSortTileTensor() const
{
    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({GenOpAttr(false)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenBitSortOp() const
{
    SortParam sortParm = PrepareSortParam();
    if (isSupportTileTensor) {
        return PrintSortTileTensor();
    }
    if (isDynamicFunction) {
        return PrintBitSortDynamicUnaligned(sortParm);
    }
    return PrintBitSortStatic(sortParm);
}

std::string CodeGenOpNPU::PrintMrgSortDynamicUnaligned(const SortParam& param) const
{
    return PrintSortDynamicUnaligned(param);
}

std::string CodeGenOpNPU::PrintMrgSortStatic(const SortParam& param) const { return PrintSortStatic(param); }

std::string CodeGenOpNPU::PrintTiledMrgSortDynamicUnaligned(const TiledSortParam& param) const
{
    return PrintTiledSortDynamicUnaligned(param);
}

SortParam CodeGenOpNPU::PrepareSortParam() const
{
    const DataType dstDtype = operandDtype[ID0];
    const DataType tmpDtype = operandDtype[ID1];
    const DataType src0Dtype = operandDtype[ID2];

    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector dstShape = rawShape[0];
    std::vector tmpShape = rawShape[1];
    std::vector src0Shape = rawShape[2];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    std::vector<int64_t> ts = NormalizeShape(tmpShape, SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string tmpDtypeStr = DataType2CCEStr(tmpDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, src0Var);
    return {{ds[ID0], ds[ID1], ds[ID2], ds[ID3]},
            {ts[ID0], ts[ID1], ts[ID2], ts[ID3]},
            {ss[ID0], ss[ID1], ss[ID2], ss[ID3]},
            src0Var,
            dstVar,
            tmpVar,
            src0DtypeStr,
            dstDtypeStr,
            tmpDtypeStr};
}

std::string CodeGenOpNPU::GenMrgSortOp() const
{
    SortParam sortParm = PrepareSortParam();
    if (isSupportTileTensor) {
        return PrintSortTileTensor();
    }
    if (isDynamicFunction) {
        return PrintMrgSortDynamicUnaligned(sortParm);
    }
    return PrintMrgSortStatic(sortParm);
}

std::string CodeGenOpNPU::PrintExtractStatic() const
{
    SymbolManager::AllocRecord src0, dst;
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::vector src0RawShape = rawShape[1];
    unsigned tShape0 = 0;
    unsigned tShape1 = 0;

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::vector src0Shape = rawShape[0];
    AppendLocalBufVarOffsetInOrder(dVar, s0Var);
    constexpr unsigned defaultDim = 1u;
    if (rawShape[1].size() == 1) {
        tShape1 = std::min(src0RawShape[0], shape[0][0]);
        tShape0 = defaultDim;
    } else {
        tShape0 = std::min(src0RawShape[0], shape[0][0]);
        tShape1 = std::min(src0RawShape[1], shape[0][1]);
    }
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dstDtypeStr, src0DtypeStr});
    paramList.insert(paramList.end(), {std::to_string(tShape0), std::to_string(tShape1)});

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintExtractDynamicUnaligned() const
{
    SymbolManager::AllocRecord src0, dst;
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::vector src0RawShape = rawShape[1];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    AppendLocalBufVarOffsetInOrder(dVar, s0Var);

    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dstDtypeStr, src0DtypeStr});
    std::vector dstShape = rawShape[0];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    paramList.insert(paramList.end(), {std::to_string(ds[ID1]), std::to_string(ds[ID2]), std::to_string(ds[ID3])});

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();

    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    auto dynSrcShape = dynamicValidShape[1];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    auto dynDstShape = dynamicValidShape[0];
    FillVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynamicValidShape[0].size(), 1);
    for (int i = 0; i < SHAPE_DIM3; ++i) {
        auto tShape = dynSrcShape[i].Min(dynDstShape[i]);
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(tShape));
    }
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintExtractTileTensor() const
{
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({GenOpAttr(false)});
    oss << WrapParamByParentheses(GetTileOpParamsByOrder()) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenExtractOp() const
{
    if (isSupportTileTensor) {
        return PrintExtractTileTensor();
    }
    if (isDynamicFunction) {
        return PrintExtractDynamicUnaligned();
    }
    return PrintExtractStatic();
}

TiledSortParam CodeGenOpNPU::PrepareTiledSortParam() const
{
    const DataType dstDtype = operandDtype[ID0];
    const DataType srcDtype = operandDtype[ID2];

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string src2Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID4]);
    std::string src3Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID5]);

    std::vector dstShape = rawShape[ID0];
    std::vector src0Shape = rawShape[ID2];
    std::vector src3Shape = rawShape[ID5];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int64_t> s3 = NormalizeShape(src3Shape, SHAPE_DIM4);

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string srcDtypeStr = DataType2CCEStr(srcDtype);
    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, src0Var, src1Var, src2Var, src3Var);
    return {{ds[ID0], ds[ID1], ds[ID2], ds[ID3]},
            {s0[ID0], s0[ID1], s0[ID2], s0[ID3], s3[ID3]},
            src0Var,
            src1Var,
            src2Var,
            src3Var,
            tmpVar,
            dstVar,
            srcDtypeStr,
            dstDtypeStr};
}

std::string CodeGenOpNPU::PrintTileSortTileTensor() const
{
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({GenOpAttr(false)});
    auto tileOpParams = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    oss << WrapParamByParentheses(tileOpParams) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenTiledMrgSortOp() const
{
    TiledSortParam tiledSortParm = PrepareTiledSortParam();
    if (isSupportTileTensor) {
        return PrintTileSortTileTensor();
    }
    return PrintTiledMrgSortDynamicUnaligned(tiledSortParm);
}

std::string CodeGenOpNPU::GenRadixSelectOp() const
{
    int64_t isLargest = -1;
    bool hasLargest = GetOpAttr(OP_ATTR_PREFIX + "order", isLargest);
    ASSERT(OperErr::ATTRIBUTE_INVALID, hasLargest) << "Radix Select has no isLargest attribute.";
    int64_t k = -1;
    bool hasK = GetOpAttr(OP_ATTR_PREFIX + "kvalue", k);
    ASSERT(OperErr::ATTRIBUTE_INVALID, hasK) << "Radix Select has no k attribute.";
    std::ostringstream oss;
    auto tileOpParams = GetTileOpParamsByOrder();
    oss << tileOpName << WrapParamByAngleBrackets<int64_t>({k, isLargest}) << WrapParamByParentheses(tileOpParams)
        << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenSortOpWithParams(const std::set<int>& idx) const
{
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string idxDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::vector<std::string> varNames;
    for (int i = 0; i < operandCnt; ++i) {
        varNames.emplace_back(sm->QueryVarNameByTensorMagic(operandWithMagic[i]));
    }

    std::map<unsigned, std::reference_wrapper<std::string>> varsMap;
    for (int i = 0; i < operandCnt; ++i) {
        varsMap.emplace(i, std::ref(varNames[i]));
    }
    AppendLocalBufferVarOffset(varsMap);

    auto xShape = rawShape[ID0];
    auto idxShape = rawShape[ID1];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(idxDtypeStr);
    paramList.emplace_back(std::to_string(xShape[ID0]));
    paramList.emplace_back(std::to_string(xShape[ID1]));
    paramList.emplace_back(std::to_string(idxShape[ID0]));
    paramList.emplace_back(std::to_string(idxShape[ID1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    for (int i = 0; i < operandCnt; ++i) {
        std::string dtypeStr = idx.count(i) ? idxDtypeStr : xDtypeStr;
        std::string param = "(" + GetAddrTypeByOperandType(operandType[i]) + " " + dtypeStr + "*)" + varNames[i];
        paramList.emplace_back(param);
    }
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenSortOp() const { return GenSortOpWithParams({ID1}); }

std::string CodeGenOpNPU::GenMergeOp() const { return GenSortOpWithParams({ID1, ID4}); }

std::string CodeGenOpNPU::GenCompareAndSwapOp() const { return GenSortOpWithParams({ID1, ID3, ID5, ID7}); }

std::string CodeGenOpNPU::GenTopKSortOp() const
{
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    AppendLocalBufVarOffsetInOrder(yVar, tmpVar, xVar);

    std::string startIdx;
    if (extSymbolicScalar.IsValid()) {
        startIdx = SymbolicExpressionTable::BuildExpression(extSymbolicScalar);
    }

    auto xShape = rawShape[ID2];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    if (!isDynamicFunction) {
        templateParam += GenOpAttr();
    }

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + yVar;
    std::string tmp = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + xDtypeStr + "*)" + tmpVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID2]) + " " + xDtypeStr + "*)" + xVar;
    if (isDynamicFunction) {
        if (startIdx.empty()) {
            startIdx = GenOpAttr().substr(NUM2);
        }
        paramList.insert(paramList.end(), {y, tmp, x, startIdx});
    } else {
        paramList.insert(paramList.end(), {y, tmp, x});
    }
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenTopKMergeOp() const
{
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    AppendLocalBufVarOffsetInOrder(yVar, xVar);

    auto xShape = rawShape[ID0];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1] / NUM2));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + yVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + xDtypeStr + "*)" + xVar;
    paramList.insert(paramList.end(), {y, x});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenTopKExtractOp() const
{
    std::string yDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    AppendLocalBufVarOffsetInOrder(yVar, xVar);

    auto yShape = rawShape[ID0];
    auto xShape = rawShape[ID1];

    std::vector<std::string> paramList;
    paramList.emplace_back(yDtypeStr);
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(std::to_string(yShape[0]));
    paramList.emplace_back(std::to_string(yShape[1]));
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1] / NUM2));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + yDtypeStr + "*)" + yVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + xDtypeStr + "*)" + xVar;
    paramList.insert(paramList.end(), {y, x});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenTwoTileMrgSort() const
{
    if (isSupportTileTensor) {
        return PrintExtractTileTensor();
    }
    return PrintSortUBDynamicUnaligned(false);
}

std::string CodeGenOpNPU::PrintSortUBDynamicUnaligned(bool containDstType) const
{
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    AppendLocalBufVarOffsetInOrder(dstVar, srcVar);

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[ID1], SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[ID1];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);

    std::vector<std::string> paramList;
    if (containDstType) {
        paramList.insert(paramList.end(), {dstDtypeStr, srcDtypeStr});
    } else {
        paramList.emplace_back(srcDtypeStr);
    }
    FillParamWithFullInput(paramList, ds);
    FillParamWithFullInput(paramList, ss);
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();

    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + srcVar;
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dstVar;

    paramList.insert(paramList.end(), {dstParam, srcParam});
    FillParamWithInput(paramList, dynSrcShape, 0, SHAPE_DIM4);

    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName.c_str() << "<" << templateParam << ">"
        << "(" << tileOpParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenExtractSingleOp() const
{
    if (isSupportTileTensor) {
        return PrintExtractTileTensor();
    }
    return PrintSortUBDynamicUnaligned(true);
}
} // namespace npu::tile_fwk
