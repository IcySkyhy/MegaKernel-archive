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
std::string GetBrcbOprandIdxStr(int64_t brcbOperandIdx)
{
    CODEGEN_LOGI("input brcbOperandIdx is %ld", static_cast<long>(brcbOperandIdx));
    std::string ret = "TileOp::";
    switch (brcbOperandIdx) {
        case ToUnderlying(BroadcastOperand::NONE):
            ret.append("BroadcastOperand::NONE");
            break;
        case ToUnderlying(BroadcastOperand::LEFT_OPERAND):
            ret.append("BroadcastOperand::LEFT_OPERAND");
            break;
        case ToUnderlying(BroadcastOperand::RIGHT_OPERAND):
            ret.append("BroadcastOperand::RIGHT_OPERAND");
            break;
        default:
            ret.append("BroadcastOperand::NONE");
    }
    return ret;
}

std::string CodeGenOpNPU::PrintBinaryStatic(const PrintBinaryParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& src1DtypeStr = param.src1DtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;

    std::vector<int64_t> os0 = NormalizeShape(shape[ID1], SHAPE_DIM4);
    std::vector<int64_t> os1 = NormalizeShape(shape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*OS1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os1[i]));
    }
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    int64_t brcOperandIdx = 0;
    if (GetOpAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        paramList.emplace_back(GetBrcbOprandIdxStr(brcOperandIdx));
    }
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

std::string CodeGenOpNPU::PrintBinaryDynamicUnaligned(const PrintBinaryParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& src1DtypeStr = param.src1DtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;

    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::vector<SymbolicScalar> dynSrcShape0 = dynamicValidShape[ID1];
    std::vector<SymbolicScalar> dynSrcShape1 = dynamicValidShape[ID2];

    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape0, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape1, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    int64_t brcOperandIdx = 0;
    if (GetOpAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        paramList.emplace_back(GetBrcbOprandIdxStr(brcOperandIdx));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    FillParamWithFullInput(paramList, dynSrcShape0);
    FillParamWithFullInput(paramList, dynSrcShape1);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

void CodeGenOpNPU::AddBinaryPrecisionTypeParm(std::vector<std::string>& templateParamList) const
{
    if (opCode == Opcode::OP_DIV || opCode == Opcode::OP_DIVS || opCode == Opcode::OP_POW ||
        opCode == Opcode::OP_POWS || opCode == Opcode::OP_MOD || opCode == Opcode::OP_MODS ||
        opCode == Opcode::OP_REM || opCode == Opcode::OP_REMS || opCode == Opcode::OP_REMRS) {
        int64_t precisionType = 0;
        (void)GetOpAttr(OpAttributeKey::precisionType, precisionType);
        std::string enumName = "";
        if (opCode == Opcode::OP_DIV || opCode == Opcode::OP_DIVS) {
            enumName = "DivAlgorithm";
        } else if (opCode == Opcode::OP_POW || opCode == Opcode::OP_POWS) {
            enumName = "PowAlgorithm";
        } else if (opCode == Opcode::OP_MOD) {
            enumName = "FmodAlgorithm";
        } else if (opCode == Opcode::OP_MODS) {
            enumName = "FmodSAlgorithm";
        } else if (opCode == Opcode::OP_REM || opCode == Opcode::OP_REMRS) {
            enumName = "RemAlgorithm";
        } else if (opCode == Opcode::OP_REMS) {
            enumName = "RemSAlgorithm";
        }
        std::string enumValue = "DEFAULT";
        if (precisionType == 1) {
            enumValue = "HIGH_PRECISION";
        }
        templateParamList.emplace_back("pto::" + enumName + "::" + enumValue);
    }
}

std::string CodeGenOpNPU::PrintBinaryTileTensor() const
{
    std::vector<std::string> tileOpCallParamList = GetTileOpParamsByOrder();

    std::vector<std::string> templateParamList;
    AddBinaryPrecisionTypeParm(templateParamList);

    std::vector<int64_t> brcOperand;
    std::string lastUse = GetLastUse();
    bool needBrcinline = GetOpAttr(OpAttributeKey::brcOperand, brcOperand);
    int64_t brcbIdxVal = 0;
    if (GetOpAttr(OpAttributeKey::brcbIdx, brcbIdxVal)) {
        ASSERT(OperErr::ATTRIBUTE_INVALID, needBrcinline && !brcOperand.empty())
            << "brcbIdx attribute is set but brcOperand is missing or empty";
        ASSERT(OperErr::ATTRIBUTE_INVALID, brcbIdxVal == brcOperand.back())
            << "brcbIdx (" << brcbIdxVal << ") and brcOperand.back() (" << brcOperand.back()
            << ") diverged — upstream pass must keep them synchronized";
    }
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    if (needBrcinline) {
        FillVecWithDummyInHead<int64_t>(brcOperand, MAX_DIM - brcOperand.size(), 0);
        FillParamWithFullInput(templateParamList, brcOperand);
    }

    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintBinary(const PrintBinaryParam& param) const
{
    if (isSupportTileTensor) {
        return PrintBinaryTileTensor();
    }
    if (isDynamicFunction) {
        return PrintBinaryDynamicUnaligned(param);
    }
    return PrintBinaryStatic(param);
}

std::string CodeGenOpNPU::GenBinaryOp() const
{
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector src0RawShape = rawShape[ID1];
    CODEGEN_LOGI("genBinaryOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);

    auto offset0 = GetOperandStartOffset(ID0);
    auto offset1 = GetOperandStartOffset(ID1);
    auto offset2 = GetOperandStartOffset(ID2);
    if (!offset0.ConcreteValid() || offset0.Concrete() != 0) {
        dVar += "+" + GetOperandStartOffset(ID0).Dump();
    }
    if (!offset1.ConcreteValid() || offset1.Concrete() != 0) {
        s0Var += "+" + GetOperandStartOffset(ID1).Dump();
    }
    if (!offset2.ConcreteValid() || offset2.Concrete() != 0) {
        s1Var += "+" + GetOperandStartOffset(ID2).Dump();
    }
    return PrintBinary({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
}

std::string CodeGenOpNPU::GenPairArgReduce() const
{
    std::vector<std::string> tileOpCallParamList = GetTileOpParamsByOrder();

    std::vector<std::string> templateParamList;
    AddBinaryPrecisionTypeParm(templateParamList);

    std::string lastUse = GetLastUse();
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }

    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenBinaryOpWithTmp() const
{
    std::vector<std::string> tileOpCallParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    std::vector<std::string> templateParamList;
    AddBinaryPrecisionTypeParm(templateParamList);
    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenVectorScalarOpWithTmp() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    std::string srcScalar = FormatScalarLiteral(extOperandVal);
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, srcScalar, tmpTensor};
    std::vector<std::string> templateParamList;
    AddBinaryPrecisionTypeParm(templateParamList);
    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenRemainderSOp() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    std::string srcScalar = FormatScalarLiteral(extOperandVal);
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, srcScalar, tmpTensor};
    std::string scalarDtypeStr = DataType2CCEStr(extOperandVal.GetDataType());
    std::vector<std::string> templateParamList = {scalarDtypeStr};
    AddBinaryPrecisionTypeParm(templateParamList);
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(templateParamList) << WrapParamByParentheses(tileOpParamList)
        << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenAxpyOp() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::string scalarAlpha = FormatFloat(extOperandVal.Cast<float>());

    std::vector<std::string> templateParamList;

    std::vector<int64_t> brcOperand;
    bool needBrcinline = GetOpAttr(OpAttributeKey::brcOperand, brcOperand);
    int64_t brcbIdxVal = 0;
    if (GetOpAttr(OpAttributeKey::brcbIdx, brcbIdxVal)) {
        ASSERT(OperErr::ATTRIBUTE_INVALID, needBrcinline && !brcOperand.empty())
            << "brcbIdx attribute is set but brcOperand is missing or empty";
        ASSERT(OperErr::ATTRIBUTE_INVALID, brcbIdxVal == brcOperand.back())
            << "brcbIdx (" << brcbIdxVal << ") and brcOperand.back() (" << brcOperand.back()
            << ") diverged — upstream pass must keep them synchronized";
    }
    if (needBrcinline) {
        FillVecWithDummyInHead<int64_t>(brcOperand, MAX_DIM - brcOperand.size(), 0);
        FillParamWithFullInput(templateParamList, brcOperand);
    }

    std::string dtypeStr = DataType2CCEStr(extOperandVal.GetDataType());
    std::string scalarParam = "(" + dtypeStr + ")" + scalarAlpha;
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, scalarParam};

    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintBinaryBrcStatic(const PrintBinaryBrcParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& src1DtypeStr = param.src1DtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    const std::string& tmpVar = param.tmpVar;

    std::vector<int64_t> os0 = NormalizeShape(shape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID3], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> brcParamList;
    brcParamList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(s0[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(s1[i]));
    }
    brcParamList.emplace_back(std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(brcParamList, ", ");

    brcParamList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    brcParamList.emplace_back(dst);
    brcParamList.emplace_back(src0);
    brcParamList.emplace_back(src1);
    brcParamList.emplace_back(tmp);

    std::string tiloOpCallParam = JoinString(brcParamList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& src1DtypeStr = param.src1DtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& s1Var = param.s1Var;
    const std::string& tmpVar = param.tmpVar;

    std::vector<int64_t> os0 = NormalizeShape(shape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID3], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[ID2];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    paramList.emplace_back("/*isCombineAxis*/");
    paramList.emplace_back(std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src0, src1, tmp});
    FillParamWithFullInput(paramList, dynSrcShape);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintBinaryBrc(const PrintBinaryBrcParam& param) const
{
    if (isDynamicFunction) {
        return PrintBinaryBrcDynamicUnaligned(param);
    }
    return PrintBinaryBrcStatic(param);
}

std::string CodeGenOpNPU::GenBinaryWithBrc() const
{
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector src0RawShape = rawShape[ID2];
    std::vector src1RawShape = rawShape[ID3];
    CODEGEN_LOGI("GenBinaryWithBrc %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID3]);

    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var, s1Var, tmpVar);
    int ret = 0;
    if (opCode == Opcode::OP_ADD_BRC || opCode == Opcode::OP_SUB_BRC || opCode == Opcode::OP_MUL_BRC ||
        opCode == Opcode::OP_DIV_BRC || opCode == Opcode::OP_MAX_BRC) {
        return PrintBinaryBrc({s0Var, s1Var, dVar, tmpVar, src0DtypeStr, src1DtypeStr, dstDtypeStr, tmpDtypeStr});
    }
    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "GenBinaryWithBrc sprintf_s failed ";
    return CG_ERROR;
}

std::string CodeGenOpNPU::GenVectorScalarOp() const { return GenVectorScalarOpByMode(VecScalMode::VEC_MODE); }

std::string CodeGenOpNPU::GenVectorScalarOpScalarMode() const
{
    return GenVectorScalarOpByMode(VecScalMode::SCALAR_MODE);
}

std::string CodeGenOpNPU::PrintBinaryScalarStatic(const PrintBinaryScalarParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;

    std::vector dstShape = rawShape[0];
    std::vector src0Shape = rawShape[1];

    std::vector<int64_t> os0 = NormalizeShape(shape[1], SHAPE_DIM3);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM3);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM3);

    std::ostringstream os;
    std::vector<std::string> binScalParmList;
    binScalParmList.emplace_back(dstDtypeStr);
    int dimScalar = static_cast<int>(param.dim);
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(binScalParmList, ", ");
    templateParam += GenOpAttr();
    binScalParmList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string scalarTmpBuffer = FormatScalarLiteral(extOperandVal);
    binScalParmList.emplace_back(dst);
    binScalParmList.emplace_back(src0);
    binScalParmList.emplace_back(scalarTmpBuffer);
    std::string tiloOpCallParam = JoinString(binScalParmList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& src0DtypeStr = param.src0DtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;

    std::vector dstShape = rawShape[0];
    std::vector src0Shape = rawShape[1];

    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM3);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM3);

    auto dynSrcShape = dynamicValidShape[1];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM3 - dynamicValidShape[1].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    int dimScalar = static_cast<int>(param.dim);
    paramList.emplace_back("/*DstRawShape*/");
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*Src0RawShape*/");
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string scalarTmpBuffer = FormatScalarLiteral(extOperandVal);
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(scalarTmpBuffer);
    FillParamWithInput(paramList, dynSrcShape, SHAPE_DIM3 - dimScalar, SHAPE_DIM3);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintVectorScalarTileTensor(const PrintUnaryParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    std::string scalarTmpBuffer = FormatScalarLiteral(extOperandVal);
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.emplace_back(scalarTmpBuffer);

    std::vector<std::string> templateParamList;
    AddBinaryPrecisionTypeParm(templateParamList);

    std::ostringstream oss;
    std::string lastUse = GetLastUse();
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    templateParamList.emplace_back(dstDtypeStr);
    oss << tileOpName;
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintBinaryScalar(const PrintBinaryScalarParam& param) const
{
    if (isDynamicFunction) {
        return PrintBinaryScalarDynamicUnaligned(param);
    }
    return PrintBinaryScalarStatic(param);
}

std::string CodeGenOpNPU::PrintVectorScalarOpDynamicUnalign(const PrintUnaryParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;

    auto newDynSrcValidShape = dynamicValidShape[1];
    FillVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    char scalarTmp[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmp, sizeof(scalarTmp), "%s", FormatScalarLiteral(extOperandVal).c_str());
    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "GenVectorScalarOpByMode sprintf_s failed ";

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0S*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(" + dstDtypeStr + ")" + scalarTmp;
    paramList.insert(paramList.end(), {dst, src, tmp});
    FillParamWithFullInput(paramList, newDynSrcValidShape);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenVectorScalarOpByMode(VecScalMode mode) const
{
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    char buffer[BUFFER_SIZE_512] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var);

    std::vector src0RawShape = rawShape[1];
    std::vector dstRawShape = rawShape[0];
    std::vector<int64_t> os0 = NormalizeShape(shape[1], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    if (mode == VecScalMode::SCALAR_MODE) {
        // Scalar op
        return PrintBinaryScalar({s0Var, dVar, dstDtypeStr, dstDtypeStr, rawShape[0].size()});
    }

    // Hack: should be optimized to memory copy in pass
    int64_t emuopc = -1;
    GetOpAttr(OP_EMUOP_PREFIX + "opc", emuopc);
    if (emuopc == EMUOP_TENSOR_EXTRACT) {
        int ret = sprintf_s(buffer, sizeof(buffer),
                            "RUNTIME_TensorExtract(/*type=*/%s, /*mem=*/__ubuf__, /*dst*/%s, /*src*/%s);\n",
                            dstDtypeStr.c_str(), dVar.c_str(), s0Var.c_str());
        ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "Gen " << opCodeStr << ":EMUOP_TENSOR_EXTRACT failed " << ret;
        return buffer;
    }

    if (isSupportTileTensor) {
        return PrintVectorScalarTileTensor({s0Var, dVar, dstDtypeStr, dstDtypeStr});
    }

    if (isDynamicFunction) {
        return PrintVectorScalarOpDynamicUnalign({s0Var, dVar, dstDtypeStr, dstDtypeStr});
    }

    std::string scalarTmpBuffer = FormatScalarLiteral(extOperandVal);
    int ret = sprintf_s(buffer, sizeof(buffer),
                        "%s_<%s, %d, %d, %d, %d, /*DS*/ %d, %d, %d, /*S0S*/ %d, %d, %d>"
                        "((__ubuf__ %s*)%s, (__ubuf__ %s*)%s, (%s)%s);\n",
                        tileOpName.c_str(), dstDtypeStr.c_str(), os0[ID0], os0[ID1], os0[ID2], os0[ID3], ds[ID1],
                        ds[ID2], ds[ID3], s0[ID1], s0[ID2], s0[ID3], dstDtypeStr.c_str(), dVar.c_str(),
                        dstDtypeStr.c_str(), s0Var.c_str(), dstDtypeStr.c_str(), scalarTmpBuffer.c_str());
    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "sprintf_s " << opCodeStr << "  failed " << ret;
    return buffer;
}

std::string CodeGenOpNPU::GenInterleaveLikeOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << opCodeStr << " only support tile tensor";
    return PrintTileOpWithFullParamsInOrder();
}

} // namespace npu::tile_fwk
