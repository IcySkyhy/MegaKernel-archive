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
 * \file codegen_vector_gather_scatter.cpp
 * \brief
 */
#include <string>

#include "interface/tensor/logical_tensor.h"
#include "codegen_op_npu.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "securec.h"

namespace npu::tile_fwk {
std::string CodeGenOpNPU::PrintGatherStatic(const PrintGatherParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& src1DtypeStr = param.src1DtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;

    std::vector dstShape = rawShape[ID0];
    std::vector src0Shape = rawShape[ID1];

    std::vector<int64_t> dos = NormalizeShape(shape[ID0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*DOS*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dos[i]));
    }
    paramList.emplace_back("/*SS*/");
    paramList.emplace_back(std::to_string(ss[ID3]));
    paramList.emplace_back("/*DS*/");
    paramList.emplace_back(std::to_string(ds[ID3]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintGatherDynamicUnaligned(const PrintGatherParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& src1DtypeStr = param.src1DtypeStr;
    const int64_t axis = param.axis;
    std::vector dstShape = rawShape[ID0];
    std::vector src0Shape = rawShape[ID1];
    std::vector src1Shape = rawShape[ID2];

    auto mul = [](uint32_t data, const int64_t in) { return data * in; };
    std::vector<int64_t> indexShape = NormalizeShape(src1Shape, SHAPE_DIM4);

    size_t inputRank = src0Shape.size();
    size_t outputRank = dstShape.size();
    int afterAxis = inputRank - axis - 1;
    int outputUBStride = dstShape[outputRank - afterAxis - 1];
    uint32_t before = std::accumulate(src0Shape.begin(), src0Shape.begin() + axis, 1, mul);
    uint32_t after = axis == (static_cast<int64_t>(src0Shape.size() - 1)) ?
                         1 :
                         std::accumulate(src0Shape.begin() + axis + 1, src0Shape.end(), 1, mul);
    auto dynIndexShape = dynamicValidShape[ID2];
    FillVecWithDummyInHead<SymbolicScalar>(dynIndexShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*before*/");
    paramList.emplace_back(std::to_string(before));

    paramList.emplace_back("/*after*/");
    paramList.emplace_back(std::to_string(after));
    paramList.emplace_back("/*axis_shape*/");
    paramList.emplace_back(std::to_string(src0Shape[axis]));

    for (int i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(indexShape[i]));
    }

    paramList.emplace_back(std::to_string(outputUBStride));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + param.dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + param.s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + param.s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    FillParamWithFullInput(paramList, dynIndexShape);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintGather(const PrintGatherParam& param) const
{
    if (isDynamicFunction) {
        return PrintGatherDynamicUnaligned(param);
    }
    return PrintGatherStatic(param);
}

std::string CodeGenOpNPU::GenGatherFromUBOp() const
{
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.find("op_attr_axis") != opAttrs.end())
        << "GenGatherOp: There is nop axis attribute here";
    const int64_t axis = AnyCast<int64_t>(opAttrs.at("op_attr_axis"));
    // shape: dst, src0, src1
    int dim = rawShape[ID1].size();
    ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, dim <= SHAPE_DIM4) << "GenGatherOp: dim is not supported: " << dim;

    std::vector dstShape = rawShape[ID0];
    std::vector src0Shape = rawShape[ID1];
    CODEGEN_LOGI("GenGatherOp, src0 Shape is [%ld,%ld]", static_cast<long>(src0Shape[0]),
                 static_cast<long>(src0Shape[1]));

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var, s1Var);
    return PrintGather({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr, axis});
}

std::string CodeGenOpNPU::PrintGatherElementStatic(const PrintGatherEleParam& param) const
{
    // Static only support 2Dim
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    std::vector<int64_t>& dstOriginShape = param.dstOriginShape;
    std::vector<int64_t>& dstRawShape = param.dstRawShape;
    std::vector<int64_t>& src0RawShape = param.src0RawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(dstOriginShape[ID0]), std::to_string(dstOriginShape[ID1])});
    paramList.emplace_back(std::to_string(src0RawShape[ID1]));
    paramList.emplace_back(std::to_string(dstRawShape[ID1]));
    paramList.emplace_back(std::to_string(param.axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    // func actual param
    paramList.clear();
    std::string dst = "(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";

    return oss.str();
}

std::string CodeGenOpNPU::PrintGatherElementDynamicUnaligned(const PrintGatherEleParam& param) const
{
    // support 1-4 dims
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> src0RawShape = NormalizeShape(param.src0RawShape, SHAPE_DIM4);
    std::vector<int64_t> src1RawShape = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
    for (size_t i = 1; i < src0RawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(src0RawShape[i]));
    }
    for (size_t i = 1; i < src1RawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(src1RawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.src1RawShape.size();
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    // func actual param
    paramList.clear();
    std::string dst = "(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    auto dstValidShape = dynamicValidShape[ID0];
    FillVecWithDummyInHead<SymbolicScalar>(dstValidShape, SHAPE_DIM4 - dstValidShape.size(), 1);
    FillParamWithInput(paramList, dstValidShape, 0, SHAPE_DIM4);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintGatherElementTileTensor(const PrintGatherEleParam& param) const
{
    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    std::vector<std::string> paramList;
    int axis = param.axis + SHAPE_DIM5 - param.src1RawShape.size();
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenGatherElementOp() const
{
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector dstShape = rawShape[ID0];
    std::vector src0Shape = rawShape[ID2];
    std::vector src1Shape = rawShape[ID3];
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID3]);
    AppendLocalBufVarOffsetInOrder(dVar, s0Var, s1Var);

    // [case1] src0: [S2,D], src1: [B,S], axis: 0, dst: [B,S]
    std::vector<int64_t> dos = shape[ID0];
    std::vector<int64_t> s0s = src0Shape;
    std::vector<int64_t> s1s = src1Shape;
    std::vector<int64_t> ds = dstShape;
    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, src0DtypeStr, src1DtypeStr};
    int gatherEleAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "axis");
    if (axis.has_value()) {
        gatherEleAxis = AnyCast<int64_t>(axis);
    }
    if (isSupportTileTensor) {
        return PrintGatherElementTileTensor({gatherEleAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
    }
    if (isDynamicFunction) {
        return PrintGatherElementDynamicUnaligned({gatherEleAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
    }
    return PrintGatherElementStatic({gatherEleAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
}

std::string CodeGenOpNPU::GenGatherMaskOp() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({GenOpAttr(false)});
    oss << WrapParamByParentheses({dstTensor, src0Tensor}) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintScatterElementSOpStatic(const PrintScatterElemParam& param) const
{
    const std::string& dstVar = param.dVar;
    const std::string& src0Var = param.s0Var;
    const std::string& src1Var = param.s1Var;
    std::vector<int64_t>& dstShape = param.dstRawShape;
    std::vector<int64_t>& src1RawShape = param.src1RawShape;
    // Static only support 2Dim
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dstShape.size() == SHAPE_DIM2)
        << "dst only support 2 Dim, current is : " << dstShape.size();
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, src1RawShape.size() == SHAPE_DIM2)
        << "src1 only support 2 Dim, current is : " << src1RawShape.size();
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    const Element& scala = extOperandVal;

    std::vector src1Shape = shape[ToUnderlying(MISOIdx::SRC1_IDX)];
    std::vector<int64_t> s1os = NormalizeShape(src1Shape, SHAPE_DIM2);
    std::vector<int64_t> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM2);
    std::vector<int64_t> drs = NormalizeShape(dstShape, SHAPE_DIM2);

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)]);
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)]);
    templateParams.emplace_back(std::to_string(s1rs[ToUnderlying(MISOIdx::SRC0_IDX)]));
    templateParams.emplace_back(std::to_string(drs[ToUnderlying(MISOIdx::SRC0_IDX)]));
    templateParams.emplace_back(std::to_string(s1os[ToUnderlying(MISOIdx::DST_IDX)]));
    templateParams.emplace_back(std::to_string(s1os[ToUnderlying(MISOIdx::SRC0_IDX)]));
    std::string templateParamStr = JoinString(templateParams, ", ");
    templateParamStr += ", " + std::to_string(param.axis);

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC0_IDX)] + "*)" + src0Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)] + "*)" + src1Var);
    std::string scalarTmpBuffer = FormatScalarLiteral(scala);
    callParams.emplace_back("(" + dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)] + ")" + scalarTmpBuffer);

    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintScatterElementSOpDynamicUnaligned(const PrintScatterElemParam& param) const
{
    const std::string& dstVar = param.dVar;
    const std::string& src0Var = param.s0Var;
    const std::string& src1Var = param.s1Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> src1RawShape = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    const Element& scala = extOperandVal;
    std::string scalarDtypeBuffer = DataType2CCEStr(scala.GetDataType());
    auto dynSrc1Shape = dynamicValidShape[ToUnderlying(MISOIdx::SRC1_IDX)];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrc1Shape,
                                           SHAPE_DIM4 - dynamicValidShape[ToUnderlying(MISOIdx::SRC1_IDX)].size(), 1);

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)]);
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)]);
    templateParams.emplace_back(scalarDtypeBuffer);
    for (size_t i = 1; i < src1RawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(src1RawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.src1RawShape.size();
    templateParams.emplace_back(std::to_string(axis));
    templateParams.emplace_back(std::to_string(param.scatterMode));
    std::string templateParamStr = JoinString(templateParams, ", ");

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC0_IDX)] + "*)" + src0Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)] + "*)" + src1Var);
    std::string scalarTmpBuffer = FormatScalarLiteral(scala);
    callParams.emplace_back("(" + scalarDtypeBuffer + ")" + scalarTmpBuffer);
    FillParamWithInput(callParams, dynSrc1Shape, 0, SHAPE_DIM4);
    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintScatterElementSTileTensor(const PrintScatterElemParam& param) const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::vector<std::string> paramList;
    std::string scalarDtypeBuffer = DataType2CCEStr(extOperandVal.GetDataType());
    int axis = param.axis + SHAPE_DIM5 - param.src1RawShape.size();
    paramList.emplace_back(std::to_string(axis));
    paramList.emplace_back(std::to_string(param.scatterMode));
    std::string scalarTmpBuffer = FormatScalarLiteral(extOperandVal);
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(paramList) << "(" << dstTensor << ", " << src1Tensor << ", ("
        << scalarDtypeBuffer << ")" << scalarTmpBuffer << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenScatterElementSOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "scatter_mode"))
        << "cannot get scatter mode attr";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    int scatterMode = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "scatter_mode"));
    const DataType dstDtype = operandDtype[ToUnderlying(MISOIdx::DST_IDX)];
    const DataType src0Dtype = operandDtype[ToUnderlying(MISOIdx::SRC0_IDX)];
    const DataType src1Dtype = operandDtype[ToUnderlying(MISOIdx::SRC1_IDX)];

    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC0_IDX)]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC1_IDX)]);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(MISOIdx::DST_IDX)]);

    std::vector dstRawShape = rawShape[ToUnderlying(MISOIdx::DST_IDX)];
    std::vector src1RawShape = rawShape[ToUnderlying(MISOIdx::SRC1_IDX)];

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    std::string src1DtypeStr = DataType2CCEStr(src1Dtype);
    CODEGEN_LOGI("GenScatterElementSOp, dstDtypeStr%s", dstDtypeStr.c_str());
    CODEGEN_LOGI("GenScatterElementSOp, src1DtypeStr%s", src1DtypeStr.c_str());

    AppendLocalBufVarOffsetInOrder(dstVar, src0Var, src1Var);

    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, src0DtypeStr, src1DtypeStr};
    if (isSupportTileTensor) {
        return PrintScatterElementSTileTensor(
            {axis, scatterMode, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, dataTypeExpr});
    }
    if (isDynamicFunction) {
        return PrintScatterElementSOpDynamicUnaligned(
            {axis, scatterMode, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, dataTypeExpr});
    }
    return PrintScatterElementSOpStatic(
        {axis, scatterMode, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, dataTypeExpr});
}

std::string CodeGenOpNPU::PrintScatterOpDynamicUnaligned(const PrintScatterParam& param) const
{
    const std::string& dstVar = param.dVar;
    const std::string& src1Var = param.s1Var;
    const std::string& src2Var = param.s2Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> src1RawShape = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    std::vector<int64_t> src2RawShape = NormalizeShape(param.src2RawShape, SHAPE_DIM4);
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    auto dynSrc1Shape = dynamicValidShape[ID3];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrc1Shape, SHAPE_DIM4 - dynamicValidShape[ID3].size(), 1);

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ID0]);
    templateParams.emplace_back(dataTypeExpr[ID1]);
    for (size_t i = 1; i < src1RawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(src1RawShape[i]));
    }
    for (size_t i = 1; i < src2RawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(src2RawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.src1RawShape.size();
    templateParams.emplace_back(std::to_string(axis));
    templateParams.emplace_back(std::to_string(param.scatterMode));
    std::string templateParamStr = JoinString(templateParams, ", ");

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ID1] + "*)" + src1Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ID2] + "*)" + src2Var);
    FillParamWithFullInput(callParams, dynSrc1Shape);
    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintScatterTileTensor(const PrintScatterParam& param) const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::string tmpTensor = QueryTileTensorNameByIdx(ID1);
    std::string src1Tensor = QueryTileTensorNameByIdx(ID3);
    std::string src2Tensor = QueryTileTensorNameByIdx(ID4);
    std::vector<std::string> paramList;
    int axis = param.axis + SHAPE_DIM5 - param.src1RawShape.size();
    paramList.emplace_back(std::to_string(axis));
    paramList.emplace_back(std::to_string(param.scatterMode));
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(paramList)
        << WrapParamByParentheses({dstTensor, src1Tensor, src2Tensor, tmpTensor}) << ";\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenScatterOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "scatter_mode"))
        << "cannot get scatter mode attr";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    int scatterMode = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "scatter_mode"));
    const DataType dstDtype = operandDtype[ID0];
    const DataType src1Dtype = operandDtype[ID3];
    const DataType src2Dtype = operandDtype[ID4];

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string src2Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID4]);

    std::vector dstRawShape = rawShape[ID0];
    std::vector src1RawShape = rawShape[ID3];
    std::vector src2RawShape = rawShape[ID4];

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src1DtypeStr = DataType2CCEStr(src1Dtype);
    std::string src2DtypeStr = DataType2CCEStr(src2Dtype);

    AppendLocalBufVarOffsetInOrder(dstVar, src1Var, src2Var);

    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, src1DtypeStr, src2DtypeStr};
    if (isSupportTileTensor) {
        return PrintScatterTileTensor(
            {axis, scatterMode, dstVar, src1Var, src2Var, dstRawShape, src1RawShape, src2RawShape, dataTypeExpr});
    }
    return PrintScatterOpDynamicUnaligned(
        {axis, scatterMode, dstVar, src1Var, src2Var, dstRawShape, src1RawShape, src2RawShape, dataTypeExpr});
}
} // namespace npu::tile_fwk
