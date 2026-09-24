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
std::string CodeGenOpNPU::PrintVnchwconvStatic(const PrintUnaryTmpBuffParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& tmpVar = param.tmpVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> os0 = NormalizeShape(shape[ID2], SHAPE_DIM5);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    // func actual param
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& tmpVar = param.tmpVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    auto newDynSrcValidShape = dynamicValidShape[ID2];
    FillVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM5 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});
    FillParamWithFullInput(paramList, newDynSrcValidShape);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintUnaryWithTmpTileTensor() const
{
    return PrintTileOpWithFullParamsTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
}

std::string CodeGenOpNPU::PrintVnchwconv(const PrintUnaryTmpBuffParam& param) const
{
    if (isSupportTileTensor) {
        return PrintUnaryWithTmpTileTensor();
    }
    if (isDynamicFunction) {
        return PrintVnchwconvDynUnaligned(param);
    }
    return PrintVnchwconvStatic(param);
}

std::string CodeGenOpNPU::PrintReduceLastAxis(const PrintUnaryTmpBuffParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& tmpVar = param.tmpVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;

    std::vector<int64_t> dstOriginShape = NormalizeShape(shape[ID0], SHAPE_DIM4);
    std::vector<int64_t> srcOriginShape = NormalizeShape(shape[ID2], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    CODEGEN_LOGI("rawShape[2] is %s", IntVecToStr(rawShape[ID2]).c_str());

    if (isSupportTileTensor) {
        return PrintReduceLastAxisTileTensor();
    }

    if (isDynamicFunction) {
        return PrintReduceLastAxisDynamicUnalign({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    ret = sprintf_s(
        buffer, sizeof(buffer),
        "%s_<%s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ "
        "%s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), srcOriginShape[ID0], srcOriginShape[ID1], srcOriginShape[ID2],
        srcOriginShape[ID3], dstRawShape[ID1], dstRawShape[ID2], dstRawShape[ID3], srcRawShape[ID1], srcRawShape[ID2],
        srcRawShape[ID3], tmpRawShape[ID3], dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(),
        tmpDtypeStr.c_str(), tmpVar.c_str());
    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0)
        << "PrintReduceLastAxis" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed " << ret;
    return buffer;
}

std::string CodeGenOpNPU::PrintReduceLastAxisTileTensor() const
{
    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        oss << WrapParamByAngleBrackets({lastUse});
    }
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintArgReduceTileTensor() const
{
    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MILOIdx::TMP2_IDX)});
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.has_value()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    reduceAxis += SHAPE_DIM5 - rawShape[0].size();
    std::ostringstream oss;
    oss << tileOpName;
    if (opCode == Opcode::OP_ROWARGMAXWITHVALUE_LINE || opCode == Opcode::OP_ROWARGMINWITHVALUE_LINE) {
        oss << WrapParamByAngleBrackets({std::to_string(reduceAxis)});
    }
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintReduceLastAxisDynamicUnalign(const PrintUnaryTmpBuffParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& tmpVar = param.tmpVar;

    auto newDynSrcValidShape = dynamicValidShape[ID2];
    FillVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    // tmp only need the last axis shape
    paramList.emplace_back(std::to_string(tmpRawShape[ID3]));

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dstName = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcName = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmpName = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dstName, srcName, tmpName});
    FillParamWithFullInput(paramList, newDynSrcValidShape);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintCompactStatic(const PrintUnaryTmpBuffParam& param) const
{
    const std::string& s0Var = param.s0Var;
    const std::string& tmpVar = param.tmpVar;
    const std::string& dVar = param.dVar;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName.c_str() << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintCompact(const PrintUnaryTmpBuffParam& param) const { return PrintCompactStatic(param); }

std::string CodeGenOpNPU::PrintUnaryOpWithTmpTwoBuff() const { return PrintTileOpWithFullParamsInOrder(); }

std::string CodeGenOpNPU::PrintRoundLayout() const
{
    std::string scalarTmpBuffer = FormatFloat(extOperandVal.Cast<float>());
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.emplace_back(scalarTmpBuffer);

    std::ostringstream oss;
    oss << tileOpName << "<float>";
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintRound() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "Round only support tile tensor";
    return PrintRoundLayout();
}

std::string CodeGenOpNPU::PrintUnaryOpWithTmpBuff() const { return PrintTileOpWithFullParamsInOrder(); }

std::string CodeGenOpNPU::PrintRowSumlineStatic(const PrintUnaryTmpBuffParam& param) const
{
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.has_value()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    ASSERT(OperErr::ATTRIBUTE_INVALID, ((reduceAxis >= 0) && (reduceAxis < (int(rawShape[ID2].size()) - 1))))
        << "unsupported reduce axis" << reduceAxis;

    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> os = NormalizeShape(shape[ID2], SHAPE_DIM4);

    std::vector<std::string> paramList;
    paramList.emplace_back(param.dstDtypeStr);
    FillParamWithFullInput(paramList, os);
    FillParamWithInputExceptFirst(paramList, srcShape);
    FillParamWithInputExceptFirst(paramList, dstShape);
    FillParamWithInputExceptFirst(paramList, tmpShape);
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + param.dstDtypeStr + "*)" + param.dVar;
    std::string src = "(__ubuf__ " + param.srcDtypeStr + "*)" + param.s0Var;
    std::string tmp = "(__ubuf__ " + param.tmpDtypeStr + "*)" + param.tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintRowSumlineDynamicUnaligned(const PrintUnaryTmpBuffParam& param) const
{
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.has_value()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    ASSERT(OperErr::ATTRIBUTE_INVALID, ((reduceAxis >= 0) && (reduceAxis < (int(rawShape[ID2].size()) - 1))))
        << "unsupported reduce axis" << reduceAxis;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& srcDtypeStr = param.srcDtypeStr;
    const std::string& tmpDtypeStr = param.tmpDtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& s0Var = param.s0Var;
    const std::string& tmpVar = param.tmpVar;

    auto dynSrcShape = dynamicValidShape[ID2];
    // adjust reduceAxis for dim4
    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - rawShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(tmpShape[i]));
    }
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    paramList.emplace_back(tmp);
    FillParamWithFullInput(paramList, dynSrcShape);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintRowSumlineTileTensor() const
{
    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.has_value()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    ASSERT(OperErr::ATTRIBUTE_INVALID, ((reduceAxis >= 0) && (reduceAxis < (int(rawShape[ID2].size()) - 1))))
        << "unsupported reduce axis";
    reduceAxis += SHAPE_DIM5 - rawShape[0].size();
    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({std::to_string(reduceAxis)});
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintRowSumline(const PrintUnaryTmpBuffParam& param) const
{
    if (isSupportTileTensor) {
        return PrintRowSumlineTileTensor();
    }
    if (isDynamicFunction) {
        return PrintRowSumlineDynamicUnaligned(param);
    }
    return PrintRowSumlineStatic(param);
}

std::string CodeGenOpNPU::GenUnaryOpWithTmpBuff() const
{
    // In this scenario, frontend set tmp buffer in output to optimize ooo schedule result.
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector srcShape = rawShape[2];
    CODEGEN_LOGI("GenUnaryOpWithTmpBuff %s src raw shape: %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = rawShape[0];
    CODEGEN_LOGI("GenUnaryOpWithTmpBuff %s dst raw shape: %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, tmpVar, s0Var);

    if (opCode == Opcode::OP_TRANSPOSE_VNCHWCONV) {
        return PrintVnchwconv({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_SIGN || opCode == Opcode::OP_SIGNBIT || opCode == Opcode::OP_SINH ||
        opCode == Opcode::OP_COSH || opCode == Opcode::OP_TANH || opCode == Opcode::OP_ASIN ||
        opCode == Opcode::OP_ACOS || opCode == Opcode::OP_TAN || opCode == Opcode::OP_ASINH ||
        opCode == Opcode::OP_ACOSH || opCode == Opcode::OP_ATANH || opCode == Opcode::OP_ISFINITE ||
        opCode == Opcode::OP_LOG1P || opCode == Opcode::OP_ISNAN) {
        return PrintUnaryWithTmpTileTensor();
    }

    if (opCode == Opcode::OP_EXP2) {
        return PrintUnaryOpWithTmpTwoBuff();
    }

    if (opCode == Opcode::OP_ROUND) {
        return PrintRound();
    }

    if (opCode == Opcode::OP_EXPM1 || opCode == Opcode::OP_SIN || opCode == Opcode::OP_COS ||
        opCode == Opcode::OP_ERF || opCode == Opcode::OP_ERFC || opCode == Opcode::OP_ATAN) {
        return PrintUnaryOpWithTmpBuff();
    }

    if (opCode == Opcode::OP_ROWSUMLINE) {
        return PrintRowSumline({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }
    if (opCode == Opcode::OP_ROWSUM_SINGLE || opCode == Opcode::OP_ROWMAX_SINGLE ||
        opCode == Opcode::OP_ROWMIN_SINGLE || opCode == Opcode::OP_ROWPROD_SINGLE) {
        return PrintReduceLastAxis({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_COMPACT) {
        return PrintCompact({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    return CG_ERROR;
}

std::string CodeGenOpNPU::GenArgReduceWithValue() const { return PrintArgReduceTileTensor(); }

std::string CodeGenOpNPU::GenOnlineSoftmaxOp() const
{
    std::vector<std::string> params = GetTileOpParamsWithTmpBuf({ID3, ID4});
    auto scalarAttr = opAttrs.at(OpAttributeKey::scalar);
    ASSERT(OperErr::ATTRIBUTE_INVALID, (scalarAttr.has_value()) && (scalarAttr.type() == typeid(Element)))
        << "ONLINE_SOFTMAX requires scalar scale attribute.";
    auto scalar = AnyCast<Element>(scalarAttr);
    params.emplace_back("(" + std::string(DataType2CCEStr(scalar.GetDataType())) + ")" +
                        FormatFloat(scalar.Cast<float>(), scalar.GetDataType()));
    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(params) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenOnlineSoftmaxUpdateOp() const { return PrintTileOpWithFullParamsTmpBuf({ID3}); }

} // namespace npu::tile_fwk
