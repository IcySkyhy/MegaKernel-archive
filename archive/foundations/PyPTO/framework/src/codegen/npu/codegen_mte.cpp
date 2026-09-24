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
 * \file codegen_mte.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {
namespace {
const std::string TSTORE_CONF = "TStoreConfig";
const std::string TSTORE_CONF_VEC = "TStoreConfigVec";
const std::unordered_set<Opcode> RESHAPE_COPY_OP = {
    Opcode::OP_RESHAPE_COPY_IN,
    Opcode::OP_RESHAPE_COPY_OUT,
    Opcode::OP_L1_RESHAPE_COPY_IN,
    Opcode::OP_L0C_RESHAPE_COPY_OUT,
};
} // namespace

DynamicParamPackMTE CodeGenOpNPU::PrepareDynamicShapeInfoForMTE(int dynShapeIdx, int shapeDim, bool isGmSpill) const
{
    DynamicParamPackMTE pack;
    int dim = static_cast<int>(rawShape[dynShapeIdx].size());
    if (isGmSpill) {
        for (auto s : shapeFromAttr[dynShapeIdx]) {
            pack.gmShapeExpr.emplace_back(std::to_string(s));
        }
    } else {
        pack.gmShapeExpr = GenGetParamMacroPacked(dynShapeIdx, dim, PREFIX_STR_RAW_SHAPE);
    }
    FillVecWithDummyInHead<std::string>(pack.gmShapeExpr, shapeDim - dim, "1");
    CODEGEN_LOGI("dynamic gmShape param: %s", IntVecToStr(pack.gmShapeExpr).c_str());

    if (GetOffsetFromAttr(dynShapeIdx)[ID0].IsValid()) {
        pack.gmOffsetExpr = GenSymbolicArgument(GetOffsetFromAttr(dynShapeIdx));
    } else {
        pack.gmOffsetExpr = GenGetParamMacroPacked(dynShapeIdx, dim, PREFIX_STR_OFFSET);
    }
    FillVecWithDummyInHead<std::string>(pack.gmOffsetExpr, shapeDim - dim, "0");
    CODEGEN_LOGI("dynamic gmOffset param: %s", IntVecToStr(pack.gmOffsetExpr).c_str());

    for (const auto& gs : pack.gmShapeExpr) {
        pack.paramList.emplace_back(gs);
    }
    for (const auto& go : pack.gmOffsetExpr) {
        pack.paramList.emplace_back(go);
    }

    return pack;
}

std::string CodeGenOpNPU::GenMemL1CopyIn() const { return GenMemCopyCube(false, 0); }

std::string CodeGenOpNPU::GenMemL1CopyOut() const { return GenMemCopyCube(true, 0); }

std::string CodeGenOpNPU::GenMemL0CCopyOut() const { return GenMemCopyCube(true, 0); }

std::string CodeGenOpNPU::GenMemCopyCube(bool isLocalToGM, unsigned uf) const
{
    unsigned gmIdx = isLocalToGM ? 0 : 1;
    int64_t gmOffset{0};
    bool isSpillToGm = GetTensorAttr(gmIdx, OpAttributeKey::workspaceBaseOffset, gmOffset);
    return GenMemCopyVar(isLocalToGM, isSpillToGm, uf);
}

std::string CodeGenOpNPU::GenMemL1SpillToGM(bool isLocalToGM, unsigned uf) const
{
    unsigned gmIdx = isLocalToGM ? 0 : 1;
    unsigned l1Idx = isLocalToGM ? 1 : 0;
    DataType gmDtype = operandDtype[gmIdx];
    DataType l1Dtype = operandDtype[l1Idx];

    std::vector<std::string> addrTypeHead(ID2);
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[l1Idx] = GetAddrTypeByOperandType(BUF_L1);

    std::vector<std::string> addrExpr(ID2);
    addrExpr[gmIdx] = GenGMAddrExprWithOffset(gmIdx, GM_STACK_BASE);
    addrExpr[l1Idx] = sm->QueryVarNameByTensorMagic(operandWithMagic[l1Idx]);
    AppendLocalBufferVarOffset({{l1Idx, addrExpr[l1Idx]}});

    std::vector<int64_t> gmShape = rawShape[gmIdx];
    CODEGEN_LOGI("GenMemOpL1 op: %s, gmShape: %s", tileOpName.c_str(), IntVecToStr(gmShape).c_str());

    std::vector<int64_t> l1Shape = rawShape[l1Idx];
    CODEGEN_LOGI("GenMemOpL1 op: %s, l1Shape: %s", tileOpName.c_str(), IntVecToStr(l1Shape).c_str());

    // Spilling out scene only support 2-dim shape
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, l1Shape.size() == SHAPE_DIM2) << "L1 shape must be 2-dim!";
    int tileShape0 = l1Shape[ID0];
    int tileShape1 = l1Shape[ID1];

    std::vector<std::string> typeExpr(ID2);
    typeExpr[gmIdx] = DataType2CCEStr(gmDtype);
    typeExpr[l1Idx] = DataType2CCEStr(l1Dtype);

    int ret{0};
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    if (isDynamicFunction) {
        // The layout of gm stack data is continuous, so just use dynValidShape[ID1] as gm stride to match the
        // implementation of "Copy TileOp" in order to make gm gap value be zero in copy intrinsic.
        auto dynValidShape = dynamicValidShape[l1Idx];
        std::ostringstream oss;
        oss << tileOpName << "<" << typeExpr[gmIdx] << ", " << typeExpr[l1Idx] << ">"
            << "((" << addrTypeHead[ID0] << " " << typeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
            << "(" << addrTypeHead[ID1] << " " << typeExpr[ID1] << "*)" << addrExpr[ID1] << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", " << uf << ");\n";

        return oss.str();
    } else {
        ret = sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n",
                        tileOpName.c_str(), typeExpr[gmIdx].c_str(), typeExpr[l1Idx].c_str(), tileShape0, tileShape1,
                        gmShape[ID0], gmShape[ID1], addrTypeHead[ID0].c_str(), typeExpr[ID0].c_str(),
                        addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(), typeExpr[ID1].c_str(), addrExpr[ID1].c_str(),
                        uf);
    }
    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "sprintf_s failed in genMemOp_L1, return value:" << ret;
    return buffer;
}

std::string CodeGenOpNPU::GenL0CToUBTileTensor() const
{
    std::string nzVar = "CopyOutMode::NZ2ND"; // current only support NZ2ND in L0C -> UB
    int64_t isAcc = 0;
    int64_t reluMode = 0;
    GetOpAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    std::vector<std::string> storeConfigList = {nzVar, std::to_string(isAcc), std::to_string(reluMode)};
    std::string storeConfig = WrapParamByAngleBrackets(storeConfigList);
    std::vector<std::string> templateParam = {TSTORE_CONF + storeConfig};
    int64_t copyMode = 0;
    GetOpAttr(OpAttributeKey::localCopyLocalMode, copyMode);
    std::string cpModeStr = CopyModeToString(static_cast<Matrix::CopyMode>(copyMode));
    std::string dualDstMode = "DualDstMode::DUAL_DST_DISABLE";
    int64_t aivId = 0;
    GetOpAttr(OpAttributeKey::subBlockIdx, aivId);
    Element scaleValue = Element(DataType::DT_UINT64, 0);
    GetOpAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    std::vector<std::string> tileOpParamList;
    if (opCode == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
        std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
        // TMP_IDX is do not use in this case
        std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
        auto [coordDst, coordSrc] = PrintDstSrcCoordFromAttr(ToUnderlying(MIMOIdx::DST_IDX),
                                                             ToUnderlying(MIMOIdx::SRC0_IDX));
        cpModeStr = "CopyMode::EXTRACT";
        tileOpParamList = {dstTensor,
                           src0Tensor,
                           src0Tensor,
                           coordDst,
                           coordSrc,
                           std::to_string(aivId),
                           std::to_string(scaleValue.GetUnsignedData())};
        int64_t splitMN = 0;
        GetOpAttr(OpAttributeKey::splitMN, splitMN);
        dualDstMode = (splitMN == 0) ? "DualDstMode::DUAL_DST_SPLIT_M" : "DualDstMode::DUAL_DST_SPLIT_N";
    } else {
        std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
        std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
        std::string src1Tensor = src0Tensor;
        auto [coordDst, coordSrc] = PrintDstSrcCoordFromAttr();
        if ((!scaleValue.GetUnsignedData()) && ((operandDtype[ID1] == DT_INT32) && (operandDtype[ID0] == DT_FP16))) {
            src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
        }
        tileOpParamList = {dstTensor,
                           src0Tensor,
                           src1Tensor,
                           coordDst,
                           coordSrc,
                           std::to_string(aivId),
                           std::to_string(scaleValue.GetUnsignedData())};
    }
    templateParam.emplace_back(cpModeStr);
    templateParam.emplace_back(dualDstMode);
    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets(templateParam);
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintMemL1ToL0TileTensor() const
{
    bool isTrans = false;
    if ((opCode == Opcode::OP_L1_TO_L0_BT) || (opCode == Opcode::OP_L1_TO_L0_AT)) {
        isTrans = true;
    }
    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));
    size_t coordSize = rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size();
    std::vector<std::string> l0Offset;
    if (!dynOffset.empty()) {
        ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dynOffset.size() == SHAPE_DIM2 || dynOffset.size() == SHAPE_DIM3)
            << "GenMemL1ToL0 only support 2-dim or 3-dim!";
        FillParamWithFullInput(l0Offset, dynOffset);
    } else {
        for (size_t i = 0; i < coordSize; ++i) {
            l0Offset.push_back("0");
        }
    }
    // constructor call parameter ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coordCp = WrapParamByParentheses(l0Offset);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(coordSize, coordCp);
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.emplace_back(coord);
    std::ostringstream oss;
    oss << tileOpName;
    if (opCode != Opcode::OP_L1_TO_L0A_SCALE && opCode != Opcode::OP_L1_TO_L0B_SCALE) {
        bool isMX = false;
        GetOpAttr(OP_ATTR_PREFIX + "is_mx", isMX);
        oss << WrapParamByAngleBrackets({std::to_string(isTrans), std::to_string(isMX)});
    }
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToL0() const
{
    if (isSupportTileTensor) {
        return PrintMemL1ToL0TileTensor();
    }

    std::string paramStr = GenParamsStr();

    std::vector<int64_t> l1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0 %s, l1Shape is %s", tileOpName.c_str(), IntVecToStr(l1Shape).c_str());

    std::vector<int64_t> l0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0 %s, l0Shape is %s", tileOpName.c_str(), IntVecToStr(l0Shape).c_str());

    SymbolicScalar srcOffset0 = 0;
    SymbolicScalar srcOffset1 = 0;
    auto dynoffset = GetOffsetFromAttr(ID1);
    if (!dynoffset.empty()) {
        ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dynoffset.size() == SHAPE_DIM2)
            << "GenMemL1ToL0 only support 2-dim!";
        srcOffset0 = dynoffset[ID0];
        srcOffset1 = dynoffset[ID1];
    }

    unsigned srcShape0 = l1Shape[ID0];
    unsigned srcShape1 = l1Shape[ID1];
    unsigned tileShape0 = l0Shape[ID0];
    unsigned tileShape1 = l0Shape[ID1];

    std::string dtypeStr = DataType2CCEStr(operandDtype[ID0]);
    auto l1ShapeDyn = dynamicValidShape[ID1];
    auto l0ShapeDyn = dynamicValidShape[ID0];

    std::ostringstream oss;
    if (isDynamicFunction) {
        oss << tileOpName << "<" << dtypeStr << ", " << SymbolicExpressionTable::BuildExpression(srcOffset0) << ", "
            << SymbolicExpressionTable::BuildExpression(srcOffset1) << ">"
            << "(" << paramStr << ", " << SymbolicExpressionTable::BuildExpression(l0ShapeDyn[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(l0ShapeDyn[ID1]) << ", "
            << SymbolicExpressionTable::BuildExpression(l1ShapeDyn[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(l1ShapeDyn[ID1]) << ");\n";
    } else {
        oss << tileOpName << "<" << dtypeStr << ", " << tileShape0 << ", " << tileShape1 << ", " << srcOffset0 << ", "
            << srcOffset1 << ", " << srcShape0 << ", " << srcShape1 << ">"
            << "(" << paramStr << ");\n";
    }

    return oss.str();
}

std::string CodeGenOpNPU::PrintTmove() const
{
    std::ostringstream oss;
    std::vector<int64_t> tmpoffset(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), 0);
    std::string coordCp = WrapParamByParentheses(tmpoffset);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), coordCp);
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.emplace_back(coord);
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({"0"});
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToBt() const
{
    if (isSupportTileTensor) {
        return PrintTmove();
    }
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    auto dynValidShape = dynamicValidShape[ID0];
    auto dynoffset = GetOffsetFromAttr(ID1);
    // only support 2-dim shape
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dynoffset.size() == SHAPE_DIM2) << "GenMemL1ToBt only support 2-dim!";

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.emplace_back(dstDtypeStr);
    // only need the valid offset of tail axis
    ASSERT(GenCodeErr::TENSOR_OFFSET_INVALID, dynoffset[ID1].IsValid()) << "GenMemL1ToBt offset is invalid";
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynoffset[ID1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(uint64_t)" + dstVar;
    std::string src = "(";
    src.append(GetAddrTypeByOperandType(BUF_L1)).append(" ").append(srcDtypeStr).append("*)").append(srcVar);
    paramList.insert(paramList.end(), {dst, src});
    // only need the valid shape of tail axis
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]));
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenMemUBTransfer(bool isCopyUBToGM) const
{
    unsigned gmIdx = isCopyUBToGM ? 0 : 1;
    int64_t gmOffset{0};
    bool isSpillToGm = GetTensorAttr(gmIdx, OpAttributeKey::workspaceBaseOffset, gmOffset);
    return GenMemCopyVar(isCopyUBToGM, isSpillToGm);
}

std::string CodeGenOpNPU::GenUBCopyIn() const { return GenMemUBTransfer(false); }

std::string CodeGenOpNPU::GenUBCopyOut() const { return GenMemUBTransfer(true); }

std::string CodeGenOpNPU::GenReshapeCopyIn() const { return GenMemUBTransfer(false); }

std::string CodeGenOpNPU::GenReshapeCopyOut() const { return GenMemUBTransfer(true); }

std::string CodeGenOpNPU::PrintL0CToL1TileTensor() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = srcTensor;

    auto [coordDst, coordSrc] = PrintDstSrcCoordFromAttr();

    int64_t isAcc = 0;
    int64_t reluMode = 0;
    GetOpAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    std::string nzVar = "CopyOutMode::NZ2NZ";
    std::vector<std::string> storeConfigList = {nzVar, std::to_string(isAcc), std::to_string(reluMode)};
    std::string storeConfig = WrapParamByAngleBrackets(storeConfigList);
    std::vector<std::string> templateParam = {TSTORE_CONF + storeConfig};

    Element scaleValue = Element(DataType::DT_UINT64, 0);
    GetOpAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    if ((!scaleValue.GetUnsignedData()) && ((operandDtype[ID1] == DT_INT32) && (operandDtype[ID0] == DT_FP16))) {
        src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    }
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, src1Tensor, coordDst, coordSrc};

    tileOpParamList.emplace_back(std::to_string(scaleValue.GetUnsignedData()));

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(templateParam);
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL0CToL1() const
{
    if (isSupportTileTensor) {
        return PrintL0CToL1TileTensor();
    }
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    auto dstValidShape = dynamicValidShape[ID0];
    auto srcValidShape = dynamicValidShape[ID1];
    auto dynValidShapeFromAttr = dynValidShapeFromOpAttr[ID0];

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back(srcDtypeStr);
    int64_t reluMode = 0;
    GetOpAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    paramList.emplace_back(std::to_string(reluMode));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_L1) + " " + dstDtypeStr + "*)" + dstVar;
    std::string src = "(" + GetAddrTypeByOperandType(BUF_L0C) + " " + srcDtypeStr + "*)" + srcVar;
    paramList.insert(paramList.end(), {dst, src});

    FillParamWithFullInput(paramList, dynValidShapeFromAttr);
    FillParamWithFullInput(paramList, dstValidShape);
    auto l1Offset = GetOffsetFromAttr(ID0);
    FillParamWithFullInput(paramList, l1Offset);
    FillParamWithFullInput(paramList, srcValidShape);
    auto l0COffset = GetOffsetFromAttr(ID1);
    FillParamWithFullInput(paramList, l0COffset);

    Element scaleValue = Element(DataType::DT_UINT64, 0);
    GetOpAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    paramList.emplace_back(std::to_string(scaleValue.GetUnsignedData()));
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenUBToL1TileTensor() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "UB to L1 only support tile tensor";
    auto [coordDst, coordSrc] = PrintDstSrcCoordFromAttr();
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.insert(tileOpParamList.end(), {coordDst, coordSrc});

    int64_t copyMode = 0;
    if (opAttrs.count(OpAttributeKey::localCopyLocalMode)) {
        copyMode = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::localCopyLocalMode));
    }
    auto cpMode = static_cast<Matrix::CopyMode>(copyMode);
    std::string cpModeStr = CopyModeToString(cpMode);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({cpModeStr});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenUBToUBND2NZTileTensor() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "UB to UB ND2NZ only support tile tensor";

    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(GetTileOpParamsByOrder()) << STMT_END;
    return oss.str();
}

// In static shape scene, GM Offset is already calculated and added to GM Addr in host side, so TileOp do not need
// GM offset
std::string CodeGenOpNPU::GenMemCopyVar(bool isCopyLocalToGM, bool isSpillToGm, unsigned uf) const
{
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;
    OperandType localType = operandType[localIdx];
    std::vector<std::string> addrTypeHead(ID2);
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[localIdx] = GetAddrTypeByOperandType(localType);

    std::vector<int64_t> gmShape = rawShape[gmIdx];
    CODEGEN_LOGI("gmShape is %s", IntVecToStr(gmShape).c_str());
    std::vector<int64_t> localRawShape = rawShape[localIdx];
    CODEGEN_LOGI("localRawShape is %s", IntVecToStr(localRawShape).c_str());

    std::vector<std::string> addrExpr(ID2);
    addrExpr[localIdx] = sm->QueryVarNameByTensorMagic(operandWithMagic[localIdx]);
    addrExpr[gmIdx] = isSpillToGm ? GenGMAddrExprWithOffset(gmIdx, GM_STACK_BASE) : GenGmParamVar(gmIdx);

    std::vector<std::string> dataTypeExpr(ID2);
    dataTypeExpr[gmIdx] = DataType2CCEStr(operandDtype[gmIdx]);
    dataTypeExpr[localIdx] = DataType2CCEStr(operandDtype[localIdx]);

    if (localType == BUF_L0C) {
        return PrintMemCopyWithL0C({uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, localRawShape, dataTypeExpr});
    } else if (localType == BUF_L1) {
        return PrintMemCopyWithL1({isCopyLocalToGM, isSpillToGm, uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape,
                                   localRawShape, dataTypeExpr});
    } else if (localType == BUF_UB) {
        PrintMemCopyWithUBParam param = {gmIdx, localIdx, isSpillToGm, addrTypeHead, addrExpr, dataTypeExpr};
        return PrintMemCopyWithUB(param);
    }

    ASSERT(OperErr::OPERAND_TYPE_UNSUPPORTED, false)
        << "GenMemCopyVar: cannot support current localType!!!" << localType;
    return {};
}

std::string CodeGenOpNPU::PrintMemCopyWithL0CTileTensor(const PrintMemCopyWithL0CParam& param) const
{
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[param.gmIdx].size(), coordCp);
    std::string gmVarName = param.addrExpr[param.gmIdx];
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    int64_t reluMode = 0;
    GetOpAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    std::string src1Tensor = srcTensor;
    int64_t nzValue = 0;
    int64_t isAcc = 0;
    GetOpAttr(OP_ATTR_PREFIX + "atomic_add", isAcc);
    GetOpAttr(OpAttributeKey::copyIsNZ, nzValue);
    std::string nzVar = nzValue ? "CopyOutMode::NZ2NZ" : "CopyOutMode::NZ2ND";
    std::vector<std::string> storeConfigList = {nzVar, std::to_string(isAcc), std::to_string(reluMode)};
    std::string storeConfig = TSTORE_CONF + WrapParamByAngleBrackets(storeConfigList);

    Element scaleValue = Element(DataType::DT_UINT64, 0);
    if (!isAcc) {
        GetOpAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    }

    bool isFixBuf = (((operandDtype[param.localIdx] == DT_INT32) && (operandDtype[param.gmIdx] == DT_FP16)) ||
                     (operandDtype[param.gmIdx] == DT_INT8));
    if ((!scaleValue.GetUnsignedData()) && isFixBuf) {
        src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    }

    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, src1Tensor, coord};
    if (opCode == Opcode::OP_L0C_COPY_OUT) {
        auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(param.gmIdx, param.gmShape);
        tileOpParamList.insert(tileOpParamList.end(), {outerValueStr, innerValueStr});
    } else {
        AppendGmValidShapeForReshapeCopy(tileOpParamList, param.gmIdx);
    }
    tileOpParamList.emplace_back(std::to_string(scaleValue.GetUnsignedData()));

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({storeConfig});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::vector<std::string> CodeGenOpNPU::GenTileOpParamForNormalCopyTileTensor(unsigned gmIdx,
                                                                             bool useOffsetFromAttr) const
{
    std::vector<std::string> gmOffsetExpr;
    if (useOffsetFromAttr || functionType == FunctionType::STATIC) {
        gmOffsetExpr = GetGmOffsetForTileTensor(gmIdx);
    } else {
        gmOffsetExpr = GenGetParamMacroPacked(gmIdx, static_cast<int>(rawShape[gmIdx].size()), PREFIX_STR_OFFSET);
    }
    // e.g. ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[gmIdx].size(), coordCp);

    const int usedOperandCnt = 2; // Use default operandCnt after fix by PASS in some scene
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder(usedOperandCnt);
    tileOpParamList.emplace_back(coord);
    return tileOpParamList;
}

void CodeGenOpNPU::AppendGmValidShapeForReshapeCopy(std::vector<std::string>& tileOpParamList, unsigned gmIdx) const
{
    if (RESHAPE_COPY_OP.find(opCode) == RESHAPE_COPY_OP.end()) {
        return;
    }

    std::vector<SymbolicScalar> dynValidShape = dynValidShapeFromOpAttr[gmIdx];
    ASSERT(OperErr::ATTRIBUTE_INVALID, !dynValidShape.empty()) << "dynValidShape is invalid from attr";
    if (opCode == Opcode::OP_RESHAPE_COPY_IN || opCode == Opcode::OP_RESHAPE_COPY_OUT) {
        FillVecWithDummyInHead<SymbolicScalar>(dynValidShape, MAX_DIM - dynValidShape.size(), 1);
    }
    FillParamWithFullInput(tileOpParamList, dynValidShape);
}

std::string CodeGenOpNPU::PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam& param) const
{
    if (isSupportTileTensor) {
        return PrintMemCopyWithL0CTileTensor(param);
    }
    if (isDynamicFunction) {
        return PrintMemCopyWithL0CDynamic(param);
    }
    return PrintMemCopyWithL0CStatic(param);
}

std::string CodeGenOpNPU::PrintMemCopyWithL0CStatic(const PrintMemCopyWithL0CParam& param) const
{
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<int64_t>& gmShape = param.gmShape;
    const std::vector<int64_t>& localRawShape = param.localRawShape;
    const std::vector<SymbolicScalar>& outputOffset = GetOffsetFromAttr(gmIdx);
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    int oriTileShape0 = std::min(shape[localIdx][ID0], localRawShape[ID0]);
    int oriTileShape1 = std::min(shape[localIdx][ID1], localRawShape[ID1]);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
                             "%s<%s, %s, %u, %u, %d, %d, %s, %s, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n",
                             tileOpName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(),
                             localRawShape[ID0], localRawShape[ID1], gmShape[ID0], gmShape[ID1],
                             SymbolicExpressionTable::BuildExpression(outputOffset[ID0]).c_str(),
                             SymbolicExpressionTable::BuildExpression(outputOffset[ID1]).c_str(), oriTileShape0,
                             oriTileShape1, addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(),
                             addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrExpr[ID1].c_str(), uf);
    ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
        << "sprintf_s failed in genMemCopyVar(BUF_L0C), return value:" << printRet;
    return buffer;
}

std::string CodeGenOpNPU::PrintL0CCopyOutDynamicUnalign(const PrintMemCopyWithL0CParam& param,
                                                        std::vector<std::string>& gmShapeExpr,
                                                        std::vector<std::string>& gmOffsetExpr) const
{
    std::ostringstream os;
    std::vector<std::string> paramList;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    paramList.emplace_back(dataTypeExpr[gmIdx]);
    paramList.emplace_back(dataTypeExpr[localIdx]);
    int64_t nzValue = 0;
    int64_t isAcc = 0;
    auto ret = GetOpAttr(OP_ATTR_PREFIX + "atomic_add", isAcc);
    if (ret) {
        paramList.emplace_back(std::to_string(isAcc));
    }
    ret = GetOpAttr(OpAttributeKey::copyIsNZ, nzValue);
    if (ret && nzValue == 1) {
        paramList.emplace_back("false");
    } else {
        paramList.emplace_back("true");
    }
    int64_t reluMode = 0;
    ret = GetOpAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    paramList.emplace_back(std::to_string(reluMode));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(" + addrTypeHead[0] + " " + dataTypeExpr[0] + "*)" + addrExpr[0];
    std::string src = "(" + addrTypeHead[1] + " " + dataTypeExpr[1] + "*)" + addrExpr[1];
    paramList.insert(paramList.end(), {dst, src});
    auto dynValidShape = dynamicValidShape[localIdx];
    FillParamWithInput(paramList, dynValidShape, 0, SHAPE_DIM2);
    paramList.emplace_back(gmShapeExpr[0]);
    paramList.emplace_back(gmOffsetExpr[0]);
    int64_t outerValue = 0, innerValue = 0;
    ret = GetOpAttr("op_attr_curH", outerValue);
    ret = GetOpAttr("op_attr_curW", innerValue);
    auto gmShapeExprByIndex = GenParamIdxExprByIndex(gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    std::string outerValueStr = outerValue == 0 ? gmShapeExprByIndex[0] : std::to_string(outerValue);
    std::string innerValueStr = innerValue == 0 ? gmShapeExprByIndex[1] : std::to_string(innerValue);
    paramList.insert(paramList.end(), {outerValueStr, innerValueStr, std::to_string(param.uf)});
    Element scaleValue = Element(DataType::DT_UINT64, 0);
    ret = GetOpAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    if (!isAcc) {
        paramList.emplace_back(std::to_string(scaleValue.GetUnsignedData()));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintMemCopyWithL0CDynamic(const PrintMemCopyWithL0CParam& param) const
{
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<int64_t>& localRawShape = param.localRawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    int oriTileShape0 = std::min(shape[localIdx][ID0], localRawShape[ID0]);
    int oriTileShape1 = std::min(shape[localIdx][ID1], localRawShape[ID1]);

    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    CODEGEN_LOGI("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_OFFSET);
    CODEGEN_LOGI("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    int printRet{0};

    if (isSupportDynamicAligned) {
        printRet = sprintf_s(
            buffer, BUFFER_SIZE_1024, "%s<%s, %s, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %s, %s, %u);\n",
            tileOpName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0],
            localRawShape[ID1], oriTileShape0, oriTileShape1, addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(),
            addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrExpr[ID1].c_str(),
            gmShapeExpr[ID0].c_str(), gmOffsetExpr[ID0].c_str(), uf);
        ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
            << "sprintf_s failed in PrintMemCopyWithL0CDynamic, return value:" << printRet;
        return buffer;
    }

    return PrintL0CCopyOutDynamicUnalign(param, gmShapeExpr, gmOffsetExpr);
}

std::pair<std::string, std::string> CodeGenOpNPU::GetOuterInnerValueStr(unsigned gmIdx,
                                                                        const std::vector<int64_t>& gmShape,
                                                                        bool isSpillingToGM) const
{
    int64_t outerValue = 0;
    int64_t innerValue = 0;
    GetOpAttr("op_attr_outer_value", outerValue);
    GetOpAttr("op_attr_inner_value", innerValue);

    bool useStaticShape = functionType == FunctionType::STATIC || isSpillingToGM;
    auto gmShapeExprByIndex = dynamicRawShape[gmIdx].empty() ?
                                  GenParamIdxExprByIndex(gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE) :
                                  GenDynRawShapePacked(gmIdx);

    auto getValueStr = [useStaticShape, &gmShapeExprByIndex](int64_t value, size_t idx,
                                                             int64_t shapeValue) -> std::string {
        if (value != 0) {
            return std::to_string(value);
        }
        return useStaticShape ? std::to_string(shapeValue) : gmShapeExprByIndex[idx];
    };

    return {getValueStr(outerValue, 0, gmShape[0]), getValueStr(innerValue, 1, gmShape[1])};
}

std::string CodeGenOpNPU::PrintMemCopyWithL1TileTensor(const PrintMemCopyWithL1Param& param) const
{
    if (param.isCopyLocalToGM) {
        return PrintMemCopyOutWithL1TileTensor(param);
    }

    return PrintMemCopyInWithL1TileTensor(param);
}

std::string CodeGenOpNPU::PrintMemCopyInWithL1TileTensor(const PrintMemCopyWithL1Param& param) const
{
    std::vector<std::string> tileOpParamList = GenTileOpParamForNormalCopyTileTensor(param.gmIdx);
    std::string kIndex = "-1";
    if (opCode == Opcode::OP_L1_COPY_IN) {
        std::vector<int64_t> dstOffset = offset[ID0];
        std::string coordCp = WrapParamByParentheses(dstOffset);
        std::string coord = PrintCoord(rawShape[ID0].size(), coordCp);
        tileOpParamList.insert(tileOpParamList.end() - 1, coord);
        auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(param.gmIdx, param.gmShape, param.isSpillingToGM);
        // 获取 K 维度在 dynValidShape 中的索引（0 或 1）
        int64_t kIndexNum = -1;
        GetOpAttr(OP_ATTR_PREFIX + "copy_in_l1_k_index", kIndexNum);
        kIndex = std::to_string(kIndexNum);
        tileOpParamList.insert(tileOpParamList.end(), {outerValueStr, innerValueStr});
    }

    AppendGmValidShapeForReshapeCopy(tileOpParamList, param.gmIdx);

    int64_t copyInMode = -1;
    if (opAttrs.count(OpAttributeKey::copyInMode)) {
        copyInMode = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::copyInMode));
    }
    auto cpMode = static_cast<Matrix::CopyInMode>(copyInMode);
    int64_t nzValue = 0;
    auto ret = GetOpAttr(OpAttributeKey::copyIsNZ, nzValue);
    if (ret && nzValue) {
        cpMode = Matrix::CopyInMode::NZ2NZ;
    }
    std::string cpModeStr = CopyInModeToString(cpMode);

    int64_t paddingMode = 0;
    GetOpAttr(OP_ATTR_PREFIX + "copy_in_l1_padding_mode", paddingMode);
    std::string padModStr = PaddingModeToString(static_cast<Matrix::PaddingMode>(paddingMode));

    std::ostringstream oss;
    if (opCode == Opcode::OP_L1_COPY_IN_A_SCALE || opCode == Opcode::OP_L1_COPY_IN_B_SCALE) {
        oss << tileOpName << WrapParamByAngleBrackets({cpModeStr}) << WrapParamByParentheses(tileOpParamList)
            << STMT_END;
    } else {
        oss << tileOpName << WrapParamByAngleBrackets({cpModeStr, padModStr, kIndex})
            << WrapParamByParentheses(tileOpParamList) << STMT_END;
    }
    return oss.str();
}

// used in L1 spilling scene
std::string CodeGenOpNPU::PrintMemCopyOutWithL1TileTensor(const PrintMemCopyWithL1Param& param) const
{
    std::vector<std::string> tileOpParamList = GenTileOpParamForNormalCopyTileTensor(param.gmIdx);

    int64_t copyOutMode = -1;
    if (opAttrs.count(OpAttributeKey::copyOutMode)) {
        copyOutMode = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::copyOutMode));
    }
    auto cpMode = static_cast<Matrix::CopyOutMode>(copyOutMode);
    std::string cpModeStr = CopyOutModeToString(cpMode);
    std::vector<std::string> storeConfigList = {cpModeStr, "0", "0"};
    std::string storeConfig = WrapParamByAngleBrackets(storeConfigList);

    std::ostringstream oss;
    oss << tileOpName << "<" << TSTORE_CONF << storeConfig << ">";
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintMemCopyWithL1(const PrintMemCopyWithL1Param& param) const
{
    if (isSupportTileTensor) {
        return PrintMemCopyWithL1TileTensor(param);
    }

    if (param.isSpillingToGM) {
        return GenMemL1SpillToGM(param.isCopyLocalToGM, param.uf);
    }

    if (isDynamicFunction) {
        return PrintMemCopyWithL1Dynamic(param);
    }

    return PrintMemCopyWithL1Static(param);
}

std::string CodeGenOpNPU::PrintMemCopyWithL1Static(const PrintMemCopyWithL1Param& param) const
{
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<int64_t>& gmShape = param.gmShape;
    const std::vector<int64_t>& localRawShape = param.localRawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    std::string opName = tileOpName;
    char addrBuffer[BUFFER_SIZE_1024] = "";
    char oriAddrBuffer[BUFFER_SIZE_1024] = "";

    int printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
    ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
        << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
    int64_t nzValue = 0;
    auto ret = GetOpAttr(OpAttributeKey::copyIsNZ, nzValue);
    if (ret && nzValue == 1) {
        opName = "TileOp::L1CopyInNZ2NZ";
        std::string curAddrBuffer = GenGmParamVar(gmIdx);
        printRet = sprintf_s(oriAddrBuffer, BUFFER_SIZE_1024, "(__gm__ %s*)%s", dataTypeExpr[ID1].c_str(),
                             curAddrBuffer.c_str());
        ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
            << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
        printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
        auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(gmIdx, gmShape);
        printRet = sprintf_s(
            buffer, BUFFER_SIZE_1024, "%s<%s, %s, %u, %u, %d, %d, %s, %s>((%s %s*)%s, (%s %s*)%s, %s, %u);\n",
            opName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0],
            localRawShape[ID1], gmShape[ID0], gmShape[ID1], outerValueStr.c_str(), innerValueStr.c_str(),
            addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(),
            dataTypeExpr[ID1].c_str(), addrBuffer, oriAddrBuffer, uf);
        ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
            << "sprintf_s failed in genMemCopyVar, return value:" << printRet;
    } else {
        std::vector<SymbolicScalar> gmOffset = GetOffsetFromAttr(gmIdx);
        printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
        ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
            << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
        printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
                             "%s<%s, %s, %u, %u, %s, %s, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n", opName.c_str(),
                             dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0],
                             localRawShape[ID1], SymbolicExpressionTable::BuildExpression(gmOffset[ID0]).c_str(),
                             SymbolicExpressionTable::BuildExpression(gmOffset[ID1]).c_str(), gmShape[ID0],
                             gmShape[ID1], addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(),
                             addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrBuffer, uf);
    }
    ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
        << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
    return buffer;
}

std::string CodeGenOpNPU::PrintMemCopyWithL1Dynamic(const PrintMemCopyWithL1Param& param) const
{
    std::ostringstream oss;

    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<int64_t>& localRawShape = param.localRawShape;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    CODEGEN_LOGI("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_OFFSET);
    CODEGEN_LOGI("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::string opName = tileOpName;
    std::string addrBuffer = addrExpr[ID1];

    int64_t nzValue = 0;
    auto ret = GetOpAttr(OpAttributeKey::copyIsNZ, nzValue);
    if (ret && nzValue == 1) {
        opName = tileOpName + "NZ2NZ";
        auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(gmIdx, param.gmShape);
        if (isSupportDynamicAligned) {
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ", " << localRawShape[ID0]
                << ", " << localRawShape[ID1] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << gmShapeExpr[ID0] << ", " << gmOffsetExpr[ID0] << ", " << outerValueStr << ", " << innerValueStr
                << ", " << uf << ");\n";
        } else {
            auto dynValidShape = dynamicValidShape[localIdx];
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", " << gmShapeExpr[ID0] << ", "
                << gmOffsetExpr[ID0] << ", " << outerValueStr << ", " << innerValueStr << ", " << uf << ");\n";
        }
    } else {
        if (isSupportDynamicAligned) {
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ", " << localRawShape[ID0]
                << ", " << localRawShape[ID1] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << gmShapeExpr[ID0] << ", " << gmOffsetExpr[ID0] << ", " << uf << ");\n";
        } else {
            int64_t copyInMode = 1;
            std::string cpModeStr = "";
            const int64_t ND2ND = 0;
            if (opAttrs.count(OP_ATTR_PREFIX + "copy_in_mode")) {
                copyInMode = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "copy_in_mode"));
            }
            if (copyInMode == ND2ND) {
                cpModeStr = ", CopyInMode::ND2ND";
            }
            auto dynValidShape = dynamicValidShape[localIdx];
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << cpModeStr << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", " << gmShapeExpr[ID0] << ", "
                << gmOffsetExpr[ID0] << ", " << uf << ");\n";
        }
    }

    return oss.str();
}

// When ub tensor spilling to GM occurred, the spilling unit is entire raw shape of ub tensor.
// So ub offset is always zero under this scene, do not need to calculate anymore.
std::string CodeGenOpNPU::PrintMemCopyWithUB(PrintMemCopyWithUBParam& param) const
{
    unsigned localIdx = param.localIdx;
    std::vector<std::string>& addrExpr = param.addrExpr;
    if (isSupportTileTensor) {
        return PrintMemCopyWithUBTileTensor(param);
    }
    if (isSupportDynamicAligned) {
        AppendLocalBufferVarOffset({{localIdx, addrExpr[localIdx]}});
        return PrintMemCopyWithUBDynamic(param);
    }
    if (isDynamicFunction) {
        return PrintMemCopyWithUBDynamicSupportUnaligned(param);
    }
    AppendLocalBufferVarOffset({{localIdx, addrExpr[localIdx]}});
    return PrintMemCopyWithUBStatic(param);
}

std::string CodeGenOpNPU::PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam& param) const
{
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    std::vector<int64_t> os = NormalizeShape(shape[localIdx], SHAPE_DIM5);
    std::vector<int64_t> dstStride = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    std::vector<int64_t> srcStride = NormalizeShape(rawShape[ID1], SHAPE_DIM5);

    // Support int64
    if (dataTypeExpr[localIdx] == "int64_t") {
        constexpr int numAlign = 2;
        dataTypeExpr[localIdx] = "int32_t";
        os[SHAPE_DIM5 - 1] *= numAlign;
        srcStride[SHAPE_DIM5 - 1] *= numAlign;
        dstStride[SHAPE_DIM5 - 1] *= numAlign;
    }
    int printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
                             "%s<%s, %u, %u, %u, %u, %u, /*dst stride*/ %u, %u, %u, %u,"
                             "/*src stride*/ %u, %u, %u, %u %s>((%s %s*)%s, (%s %s*)%s);\n",
                             tileOpName.c_str(), dataTypeExpr[localIdx].c_str(), os[ID0], os[ID1], os[ID2], os[ID3],
                             os[4], dstStride[ID1], dstStride[ID2], dstStride[ID3], dstStride[4], srcStride[ID1],
                             srcStride[ID2], srcStride[ID3], srcStride[4], GenOpAttr().c_str(),
                             addrTypeHead[ID0].c_str(), dataTypeExpr[localIdx].c_str(), addrExpr[ID0].c_str(),
                             addrTypeHead[ID1].c_str(), dataTypeExpr[localIdx].c_str(), addrExpr[ID1].c_str());
    ASSERT(GenCodeErr::PRINT_FAILED, printRet >= 0)
        << "sprintf_s failed in PrintMemCopyWithUBStatic, return value:" << printRet;
    return buffer;
}

std::string CodeGenOpNPU::PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam& param) const
{
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    std::vector<int64_t> newOriginShape = shape[localIdx];
    FillVecWithDummyInHead<int64_t>(newOriginShape, MAX_DIM - shape[localIdx].size(), 1);
    const std::vector<int64_t>& localRawShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);

    auto paramPack = PrepareDynamicShapeInfoForMTE(gmIdx, MAX_DIM, param.isSpillingToGM);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dataTypeExpr[localIdx].c_str());
    for (auto ts : newOriginShape) {
        paramList.emplace_back(std::to_string(ts));
    }
    for (int i = 1; i < MAX_DIM; ++i) {
        paramList.emplace_back(std::to_string(localRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(" + addrTypeHead[ID0] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID0];
    std::string src = "(" + addrTypeHead[ID1] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID1];
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam& param) const
{
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string>& addrTypeHead = param.addrTypeHead;
    const std::vector<std::string>& addrExpr = param.addrExpr;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    std::vector<SymbolicScalar> newDynamicShape = dynamicValidShape[localIdx];
    FillVecWithDummyInHead<SymbolicScalar>(newDynamicShape, MAX_DIM - dynamicValidShape[localIdx].size(), 1);
    const std::vector<int64_t>& localRawShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);

    auto paramPack = PrepareDynamicShapeInfoForMTE(gmIdx, MAX_DIM, param.isSpillingToGM);
    std::vector<std::string> gmShapeExpr = paramPack.gmOffsetExpr;
    std::vector<std::string> gmOffsetExpr = paramPack.gmOffsetExpr;

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dataTypeExpr[localIdx].c_str());
    for (int i = 1; i < MAX_DIM; ++i) {
        paramList.emplace_back(std::to_string(localRawShape[i]));
    }
    if (localIdx == 0) { // means op is COPY_IN
        bool isPartialMem{false};
        GetTensorAttr(localIdx, "isPartialMem", isPartialMem);
        if (isPartialMem) {
            paramList.emplace_back("true");
        }
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(" + addrTypeHead[ID0] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID0];
    std::string src = "(" + addrTypeHead[ID1] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID1];
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    FillParamWithFullInput(paramList, newDynamicShape);
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());
    auto startOffset = GetOperandStartOffset(localIdx);
    if (!startOffset.ConcreteValid() || startOffset.Concrete() != 0) {
        paramList.emplace_back(startOffset.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::vector<std::string> CodeGenOpNPU::GetGmOffsetForTileTensor(unsigned gmIdx) const
{
    int dim = static_cast<int>(rawShape[gmIdx].size());
    std::vector<std::string> gmOffsetExpr;
    if (functionType == FunctionType::STATIC) {
        return std::vector<std::string>(dim, "0");
    }

    if (GetOffsetFromAttr(gmIdx)[ID0].IsValid()) {
        return GenSymbolicArgument(GetOffsetFromAttr(gmIdx));
    }

    return GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
}

std::string CodeGenOpNPU::PrintMemCopyWithUBTileTensor(const PrintMemCopyWithUBParam& param) const
{
    std::vector<std::string> tileOpParamList = GenTileOpParamForNormalCopyTileTensor(param.gmIdx);
    AppendGmValidShapeForReshapeCopy(tileOpParamList, param.gmIdx);
    std::ostringstream oss;
    oss << tileOpName;
    if (opCode == Opcode::OP_UB_COPY_OUT) {
        int64_t isAcc{0};
        GetOpAttr(OpAttributeKey::atomicAdd, isAcc);
        std::string att = isAcc == 0 ? "pto::AtomicType::AtomicNone" : "pto::AtomicType::AtomicAdd";
        std::string storeConfig = WrapParamByAngleBrackets({att});
        oss << "<" << TSTORE_CONF_VEC << storeConfig << ">";
    }
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToFB() const
{
    if (isSupportTileTensor) {
        return PrintTmove();
    }
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    auto dynValidShape = dynamicValidShape[ID0];
    auto dynoffset = GetOffsetFromAttr(ID1);
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, dynoffset.size() == SHAPE_DIM2) << "GenMemL1ToFB only support 2-dim!";

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    // only need the valid offset of tail axis
    ASSERT(GenCodeErr::TENSOR_OFFSET_INVALID, dynoffset[ID1].IsValid()) << "GenMemL1TFB offset is invalid";
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynoffset[ID1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_FIX) + " " + srcDtypeStr + "*)" + dstVar;
    std::string src = "(" + GetAddrTypeByOperandType(BUF_L1) + " " + srcDtypeStr + "*)" + srcVar;
    paramList.insert(paramList.end(), {dst, src});
    // only need the valid shape of tail axis
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]));
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::GenGMAddrExprWithOffset(unsigned gmParamIdx, const std::string& addrExpr) const
{
    // gm offset of spilling workspace is calculated by pass.
    int64_t gmOffset{0};
    GetTensorAttr(gmParamIdx, OpAttributeKey::workspaceBaseOffset, gmOffset);
    std::ostringstream oss;
    if (gmOffset == 0) {
        oss << addrExpr;
    } else {
        oss << "((__gm__ uint8_t*)" << addrExpr << " + " << gmOffset << ")";
    }

    return oss.str();
}

} // namespace npu::tile_fwk
