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
 * \file codegen_mte_gather.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {
std::string CodeGenOpNPU::PrintGatherInL1TileTensor() const
{
    std::string srcVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string offsetsVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::string blockTableVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC2_IDX));
    std::string dstVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    int64_t blockSize = AnyCast<int64_t>(opAttrs.at("op_attr_blocksize"));

    auto startOffset = opAttrs.at(OpAttributeKey::startOffset);
    ASSERT(OperErr::ATTRIBUTE_INVALID, startOffset.has_value() && (startOffset.type() == typeid(int64_t)))
        << "GenGatherInL1 startOffset must be int64_t!";
    auto srcColumnStartOffset = AnyCast<int64_t>(startOffset);
    std::string srcCoordCp = WrapParamByParentheses({std::to_string(srcColumnStartOffset)});
    std::string srcCoord = PrintCoord(SHAPE_DIM1, srcCoordCp);

    auto offsetsStartOffsets = GenParamIdxExprByIndex(ID2, SHAPE_DIM2, PREFIX_STR_OFFSET);
    std::string offsetCoordCp = WrapParamByParentheses(offsetsStartOffsets);
    std::string offsetCoord = PrintCoord(SHAPE_DIM2, offsetCoordCp);

    auto blockTableStartOffsets = GenParamIdxExprByIndex(ID3, SHAPE_DIM2, PREFIX_STR_OFFSET);
    std::string blockTableCoordCp = WrapParamByParentheses(blockTableStartOffsets);
    std::string blockTableCoord = PrintCoord(SHAPE_DIM2, blockTableCoordCp);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({std::to_string(blockSize)});
    oss << WrapParamByParentheses({dstVar, srcVar, blockTableVar, offsetsVar, srcCoord, offsetCoord, blockTableCoord});
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenGatherInL1() const
{
    if (isSupportTileTensor) {
        return PrintGatherInL1TileTensor();
    }
    const DataType dstDtype = operandDtype[ID0];
    const DataType srcDtype = operandDtype[ID1];
    const DataType offsetsDtype = operandDtype[ID2];
    ASSERT(GenCodeErr::DATA_TYPE_MISMATCHED, dstDtype == srcDtype) << "dstDtype and srcDtype must be same!";

    std::string srcVar = GenGmParamVar(ID1);
    std::string offsetsVar = GenGmParamVar(ID2);
    std::string blockTableVar = GenGmParamVar(ID3);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    auto dstRawShapes = rawShape[ID0];
    auto srcRawShapes = rawShape[ID1];
    auto offsetsRawShapes = rawShape[ID2];
    auto dstOriShapes = dynamicValidShape[ID0];
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, srcRawShapes.size() == SHAPE_DIM2)
        << "GenGatherInL1 only support 2-dim!";
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dstRawShapes.size() == SHAPE_DIM2)
        << "GenGatherInL1 only support 2-dim!";
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, offsetsRawShapes.size() == SHAPE_DIM2)
        << "GenGatherInL1 only support 2-dim!";
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dstOriShapes.size() == SHAPE_DIM2)
        << "GenGatherInL1 only support 2-dim!";

    auto offsetsStartOffsets = GenParamIdxExprByIndex(ID2, SHAPE_DIM2, PREFIX_STR_OFFSET);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string srcDtypeStr = DataType2CCEStr(srcDtype);
    std::string offsetsDtypeStr = DataType2CCEStr(offsetsDtype);
    std::string blockTableDtypeStr = DataType2CCEStr(operandDtype[ID3]);

    ASSERT(GenCodeErr::DATA_TYPE_MISMATCHED, dstDtypeStr == srcDtypeStr) << "dstDtypeStr and srcDtypeStr must be same!";
    ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, offsetsDtypeStr == "int64_t" || offsetsDtypeStr == "int32_t")
        << "offsetsDtypeStr must be int64_t or int32_t!";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.find("op_attr_blocksize") != opAttrs.end())
        << "GenGatherOp: There is nop blocksize attribute here";
    const int64_t blockSize = AnyCast<int64_t>(opAttrs.at("op_attr_blocksize"));

    auto startOffset = opAttrs.at(OpAttributeKey::startOffset);
    ASSERT(OperErr::ATTRIBUTE_INVALID, startOffset.has_value() && (startOffset.type() == typeid(int64_t)))
        << "GenGatherInL1 startOffset must be int64_t!";
    auto srcColumnStartOffset = AnyCast<int64_t>(startOffset);
    auto blockTableGMStride = GenParamIdxExprByIndex(ID3, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    auto blockTableStartOffsets = GenParamIdxExprByIndex(ID3, SHAPE_DIM2, PREFIX_STR_OFFSET);

    auto ret = sprintf_s(
        buffer, sizeof(buffer),
        "%s<%s, %s, %s, %lld, %lld, %lld, %lld>((__cbuf__ %s *)%s, %s, %s, (__gm__ %s *)%s, %lld, (__gm__ %s *)%s, "
        "(__gm__ %s *)%s, %s, %s, %s, %s, %s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), offsetsDtypeStr.c_str(), blockTableDtypeStr.c_str(), dstRawShapes[ID0],
        offsetsRawShapes[ID1], srcColumnStartOffset, blockSize, dstDtypeStr.c_str(), dstVar.c_str(),
        dstOriShapes[ID0].Dump().c_str(), dstOriShapes[ID1].Dump().c_str(), srcDtypeStr.c_str(), srcVar.c_str(),
        srcRawShapes[1], offsetsDtypeStr.c_str(), offsetsVar.c_str(), blockTableDtypeStr.c_str(), blockTableVar.c_str(),
        offsetsStartOffsets[ID0].c_str(), offsetsStartOffsets[ID1].c_str(), blockTableGMStride[ID1].c_str(),
        blockTableStartOffsets[ID0].c_str(), blockTableStartOffsets[ID1].c_str());

    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "GenGatherInL1 sprintf_s failed ";
    std::string ostring(buffer);
    return ostring;
}
/**
 * 辅助函数，对 axis 参数进行归一化
 * example:
 * parma [a,b]
 * axis 0
 * 归一化后
 * parma [1,1,a,b]
 * axis 2
 *
 */
inline int NormalizeAxis(int axis, int paramDim) { return axis + (SHAPE_DIM4 - paramDim); }
/**
 * 归一化的 gather 参数维度
 * example:
 * param [a,b]
 * indices [c]
 * axis 1
 * result [a,c]
 * 归一化后:
 * [1,1,a,b]
 * [1,c]
 * axis 3
 * result  [1,1,a,1,c]
 * 处理逻辑:
 * 1. 根据result的形状，还原出来的 param 和 indices 的维度
 * 2. param 归一化四维，indices 归一化到 两维
 * 3. 重新拼装处 result 形状
 */
template <typename T>
void NormalizeGatherShape(std::vector<T>& rawShape, const int paramDim, const int indicesDim, const int axis)
{
    bool isValidDType = (std::is_same_v<T, int64_t> || std::is_same_v<T, SymbolicScalar>);
    ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, isValidDType) << "T must be int64_t or SymbolicScalar";
    std::vector<T> paramShape{};
    std::vector<T> indicesShape{};
    indicesShape.assign(rawShape.begin() + axis, rawShape.begin() + axis + indicesDim);
    paramShape.assign(rawShape.begin(), rawShape.begin() + axis);
    paramShape.push_back(-1);
    paramShape.insert(paramShape.end(), rawShape.begin() + axis + indicesDim, rawShape.end());
    if constexpr (std::is_same_v<T, int64_t>) {
        paramShape = NormalizeShape(paramShape, SHAPE_DIM4);
        indicesShape = NormalizeShape(indicesShape, SHAPE_DIM2);
    } else if constexpr (std::is_same_v<T, SymbolicScalar>) {
        FillVecWithDummyInHead<SymbolicScalar>(paramShape, SHAPE_DIM4 - paramDim, 1);
        FillVecWithDummyInHead<SymbolicScalar>(indicesShape, SHAPE_DIM2 - indicesDim, 1);
    }
    rawShape = paramShape;
    int normalizedAxis = NormalizeAxis(axis, paramDim);
    rawShape.erase(rawShape.begin() + normalizedAxis);
    rawShape.insert(rawShape.begin() + normalizedAxis, indicesShape.begin(), indicesShape.end());
}
void HelpNormalize(std::vector<size_t>& index, int axis, int paramDim)
{
    size_t delNum = NUM4 - paramDim;
    index.erase(index.begin() + delNum);
    axis = NormalizeAxis(axis, paramDim);
    index.insert(index.begin() + axis, delNum);
}
std::string CodeGenOpNPU::PrintGatherDynamicUnaligned() const
{
    std::vector dstShape = rawShape[0];
    std::vector src0Shape = rawShape[1];

    std::string resultDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string paramDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string indicesDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    ASSERT(GenCodeErr::DATA_TYPE_MISMATCHED, resultDtypeStr == paramDtypeStr)
        << "resultDtypeStr: " << resultDtypeStr << ", paramDtypeStr: " << paramDtypeStr;
    const int64_t axis = AnyCast<int64_t>(opAttrs.at("op_attr_axis"));
    auto outputRawShapes = rawShape[ID0];
    auto paramRawShapes = rawShape[ID1];
    auto indicesRawShapes = rawShape[ID2];
    auto outputValidShapes = dynamicValidShape[ID0];
    auto paramValidShapes = dynamicValidShape[ID1];
    auto indicesValidShapes = dynamicValidShape[ID2];
    const int paramDim = paramRawShapes.size();
    const int indicesDim = indicesRawShapes.size();
    constexpr int paramIndex = 1;
    constexpr int indicesIndex = 2;
    auto normalizedOutputRawShapes = outputRawShapes;
    NormalizeGatherShape<int64_t>(normalizedOutputRawShapes, paramDim, indicesDim, axis);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(paramDtypeStr);
    paramList.emplace_back(indicesDtypeStr);
    paramList.emplace_back(std::to_string(NormalizeAxis(axis, paramDim)));
    std::transform(normalizedOutputRawShapes.begin() + 1, normalizedOutputRawShapes.end(), back_inserter(paramList),
                   [](int x) { return std::to_string(x); });

    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string paramVar = GenGmParamVar(paramIndex);
    std::string indicesVar = GenGmParamVar(indicesIndex);
    std::string outputVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string outputParamStr = "(__ubuf__ " + resultDtypeStr + "*)" + outputVar;
    std::string paramParamStr = "(__gm__ " + paramDtypeStr + "*)" + paramVar;
    std::string indicesParamStr = "(__gm__ " + indicesDtypeStr + "*)" + indicesVar;
    paramList.emplace_back(outputParamStr);
    paramList.emplace_back(paramParamStr);
    paramList.emplace_back(indicesParamStr);
    NormalizeGatherShape<SymbolicScalar>(outputValidShapes, paramDim, indicesDim, axis);

    std::transform(outputValidShapes.begin(), outputValidShapes.end(), back_inserter(paramList),
                   [](SymbolicScalar x) { return SymbolicExpressionTable::BuildExpression(x); });

    auto paramGMStride = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_RAW_SHAPE);
    auto paramStartOffsets = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_OFFSET);
    FillVecWithDummyInHead<std::string>(paramGMStride, SHAPE_DIM4 - paramDim, std::string("1"));
    FillVecWithDummyInHead<std::string>(paramStartOffsets, SHAPE_DIM4 - paramDim, std::string("0"));
    paramList.insert(paramList.end(), paramGMStride.begin() + 1, paramGMStride.end());
    paramList.insert(paramList.end(), paramStartOffsets.begin(), paramStartOffsets.end());

    auto indicesGMStride = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_RAW_SHAPE);
    auto indicesStartOffsets = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_OFFSET);
    FillVecWithDummyInHead<std::string>(indicesGMStride, SHAPE_DIM2 - indicesDim, std::string("1"));
    FillVecWithDummyInHead<std::string>(indicesStartOffsets, SHAPE_DIM2 - indicesDim, std::string("0"));
    paramList.insert(paramList.end(), indicesGMStride.begin() + 1, indicesGMStride.end());
    paramList.insert(paramList.end(), indicesStartOffsets.begin(), indicesStartOffsets.end());

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    paramList.clear();
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}
std::string CodeGenOpNPU::PrintGatherLayout() const
{
    auto outputRawShapes = rawShape[ID0];
    auto paramRawShapes = rawShape[ID1];
    auto indicesRawShapes = rawShape[ID2];
    auto outputValidShapes = dynamicValidShape[ID0];
    auto paramValidShapes = dynamicValidShape[ID1];
    auto indicesValidShapes = dynamicValidShape[ID2];
    size_t paramDim = paramValidShapes.size();
    size_t indicesDim = indicesValidShapes.size();
    const int64_t axis = AnyCast<int64_t>(opAttrs.at("op_attr_axis"));
    std::vector<size_t> helpIndex = {ID0, ID1, ID2, ID3, ID4};
    if (indicesDim == 1 && axis != 0) {
        HelpNormalize(helpIndex, axis, paramDim);
    }
    auto paramOffsetSymbol = GenGetParamMacroPacked(ID1, paramDim, PREFIX_STR_OFFSET);
    auto indicesOffsetSymbol = GenGetParamMacroPacked(ID2, indicesDim, PREFIX_STR_OFFSET);
    std::string coordCpparamOffset = WrapParamByParentheses(paramOffsetSymbol);
    std::string coordCpindicesOffset = WrapParamByParentheses(indicesOffsetSymbol);
    std::string coord4Param = PrintCoord(paramDim, coordCpparamOffset);
    std::string coord4Indices = PrintCoord(indicesDim, coordCpindicesOffset);

    auto tileOpParams = GetTileOpParamsByOrder();
    tileOpParams.insert(tileOpParams.end(), {coord4Param, coord4Indices});

    std::vector<std::string> paramList;
    paramList.emplace_back(std::to_string(NormalizeAxis(axis, paramDim)));
    std::transform(helpIndex.begin(), helpIndex.end(), back_inserter(paramList),
                   [](size_t x) { return std::to_string(x); });

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(paramList) << WrapParamByParentheses(tileOpParams) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenGatherOp() const
{
    if (isSupportTileTensor) {
        return PrintGatherLayout();
    }
    if (isDynamicFunction) {
        return PrintGatherDynamicUnaligned();
    }
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, false) << "Gather operator does not support static graph";
    return "";
}

std::string CodeGenOpNPU::PrintGatherInUBLayout() const
{
    constexpr int paramIndex = 1;
    constexpr int indicesIndex = 2;
    constexpr int blockTableIndex = 3;
    constexpr int paramDim = 2;
    constexpr int indicesDim = 2;
    constexpr int blockTableDim = 2;

    auto paramOffsetSymbol = GenGetParamMacroPacked(paramIndex, paramDim, PREFIX_STR_OFFSET);
    auto indicesOffsetSymbol = GenGetParamMacroPacked(indicesIndex, indicesDim, PREFIX_STR_OFFSET);
    auto blockTableOffsetSymbol = GenGetParamMacroPacked(blockTableIndex, blockTableDim, PREFIX_STR_OFFSET);

    std::string coordCpparamOffset = WrapParamByParentheses(paramOffsetSymbol);
    std::string coordCpindicesOffset = WrapParamByParentheses(indicesOffsetSymbol);
    std::string coordCpblockTableOffset = WrapParamByParentheses(blockTableOffsetSymbol);
    std::string coord4Param = PrintCoord(paramDim, coordCpparamOffset);
    std::string coord4Indices = PrintCoord(indicesDim, coordCpindicesOffset);
    std::string coord4BlockTable = PrintCoord(blockTableDim, coordCpblockTableOffset);
    std::vector<std::string> paramList;
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.find(OpAttributeKey::blockSize) != opAttrs.end())
        << "GenGatherOp: There is nop blockSize attribute here";
    const int64_t blockSize = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::blockSize));
    paramList.emplace_back(std::to_string(blockSize));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.insert(tileOpParamList.end(), {coord4Param, coord4Indices, coord4BlockTable});
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}
std::string CodeGenOpNPU::PrintGatherInUBDynamicUnaligned() const
{
    std::vector dstShape = rawShape[0];
    std::vector src0Shape = rawShape[1];

    std::string resultDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string paramDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string indicesDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string blockTableDtypeStr = DataType2CCEStr(operandDtype[ID3]);
    ASSERT(GenCodeErr::DATA_TYPE_MISMATCHED, resultDtypeStr == paramDtypeStr)
        << "resultDtypeStr and paramDtypeStr must be same!";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.find(OpAttributeKey::blockSize) != opAttrs.end())
        << "GenGatherOp: There is nop blockSize attribute here";
    const int64_t blockSize = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::blockSize));
    auto outputRawShapes = rawShape[ID0];
    auto paramRawShapes = rawShape[ID1];
    auto indicesRawShapes = rawShape[ID2];
    auto outputValidShapes = dynamicValidShape[ID0];
    auto paramValidShapes = dynamicValidShape[ID1];
    auto indicesValidShapes = dynamicValidShape[ID2];
    constexpr int paramDim = 2;
    constexpr int indicesDim = 2;
    constexpr int blockTableDim = 2;
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(paramDtypeStr);
    paramList.emplace_back(indicesDtypeStr);
    paramList.emplace_back(blockTableDtypeStr);
    paramList.emplace_back(std::to_string(outputRawShapes[0]));
    paramList.emplace_back(std::to_string(outputRawShapes[1]));
    paramList.emplace_back(std::to_string(blockSize));
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    constexpr int paramIndex = 1;
    constexpr int indicesIndex = 2;
    constexpr int blockTableIndex = 3;
    std::string paramVar = GenGmParamVar(paramIndex);
    std::string indicesVar = GenGmParamVar(indicesIndex);
    std::string blockTableVar = GenGmParamVar(blockTableIndex);
    std::string outputVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string outputParamStr = "(__ubuf__ " + resultDtypeStr + "*)" + outputVar;
    std::string paramParamStr = "(__gm__ " + paramDtypeStr + "*)" + paramVar;
    std::string indicesParamStr = "(__gm__ " + indicesDtypeStr + "*)" + indicesVar;
    std::string blockTableParamStr = "(__gm__ " + blockTableDtypeStr + "*)" + blockTableVar;
    paramList.emplace_back(outputParamStr);
    paramList.emplace_back(paramParamStr);
    paramList.emplace_back(indicesParamStr);
    paramList.emplace_back(blockTableParamStr);
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(outputValidShapes[1]));

    auto paramGMStride = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_RAW_SHAPE);
    auto paramStartOffsets = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_OFFSET);
    paramList.emplace_back(paramGMStride[1]);
    paramList.emplace_back(paramStartOffsets[0]);
    paramList.emplace_back(paramStartOffsets[1]);

    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(outputValidShapes[0]));
    auto indicesGMStride = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_RAW_SHAPE);
    auto indicesStartOffsets = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_OFFSET);
    paramList.emplace_back(indicesGMStride[1]);
    paramList.emplace_back(indicesStartOffsets[0]);
    paramList.emplace_back(indicesStartOffsets[1]);
    auto blockTableGMStride = GenParamIdxExprByIndex(blockTableIndex, blockTableDim, PREFIX_STR_RAW_SHAPE);
    auto blockTableStartOffsets = GenParamIdxExprByIndex(blockTableIndex, blockTableDim, PREFIX_STR_OFFSET);
    paramList.emplace_back(blockTableGMStride[1]);
    paramList.emplace_back(blockTableStartOffsets[0]);
    paramList.emplace_back(blockTableStartOffsets[1]);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    paramList.clear();
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::GenGatherInUB() const
{
    if (isSupportTileTensor) {
        return PrintGatherInUBLayout();
    }
    if (isDynamicFunction) {
        return PrintGatherInUBDynamicUnaligned();
    }
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, false) << "Gather operator does not support static graph";
    return "";
}
} // namespace npu::tile_fwk
