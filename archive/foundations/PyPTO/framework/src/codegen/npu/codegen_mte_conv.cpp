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
 * \file codegen_mte_conv.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "tilefwk/error_code.h"
#include "securec.h"

namespace npu::tile_fwk {

void CodeGenOpNPU::GetDynamicOffsetExpr(const std::vector<SymbolicScalar>& dynOffset, bool isConv3D,
                                        std::vector<std::string>& gmOffsetExpr,
                                        std::vector<int64_t>& staticOffsets) const
{
    // 统一输出为 SHAPE_DIM5 维，conv2d 的第 5 维(d)初始化为 0
    size_t inputDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;

    if (!(functionType == FunctionType::STATIC) && dynOffset[ID0].IsValid()) {
        // 动态场景：返回动态偏移表达式，conv2d 补充第 5 维为 "0"
        gmOffsetExpr = GenSymbolicArgument(dynOffset);
        if (!isConv3D) {
            gmOffsetExpr.emplace_back("0");
        }
    } else {
        // 静态场景：返回静态偏移值，初始化为 SHAPE_DIM5 个 0，conv2d 第 5 维保持为 0
        staticOffsets.resize(SHAPE_DIM5, 0);
        for (size_t i = 0; i < inputDim; i++) {
            staticOffsets[i] = dynOffset[i].Concrete();
        }
    }
}

void CodeGenOpNPU::GetNZ2NZDynamicOffsetExpr(const std::vector<SymbolicScalar>& dynOffset, bool isConv3D, bool isFmap,
                                             std::vector<std::string>& gmOffsetExpr,
                                             std::vector<std::string>& staticOffsets) const
{
    // 统一输出为 SHAPE_DIM5 维, conv3d fmap 删除第 6 维, weight 的第 5 维初始化为 0
    size_t inputDim = isFmap ? SHAPE_DIM5 : SHAPE_DIM4;

    if (!(functionType == FunctionType::STATIC) && dynOffset[ID0].IsValid()) {
        // 动态场景：返回动态偏移表达式，conv3d 删除第 6 维
        gmOffsetExpr = GenSymbolicArgument(dynOffset);
        if (isFmap && isConv3D) {
            gmOffsetExpr.pop_back();
        }
        if (!isFmap) {
            gmOffsetExpr.emplace_back("0");
        }
    } else {
        // 静态场景：返回静态偏移值，初始化为 SHAPE_DIM5 个 0，conv3d 第 6 维不设置, weight 第 5 维保持为 0
        staticOffsets.resize(SHAPE_DIM5, "0");
        for (size_t i = 0; i < inputDim; i++) {
            staticOffsets[i] = std::to_string(dynOffset[i].Concrete());
        }
    }
}

std::vector<std::string> CodeGenOpNPU::BuildCopyInParamList(const std::string& dstTensor, const std::string& srcTensor,
                                                            const std::vector<std::string>& gmOffsetExpr,
                                                            const std::vector<int64_t>& staticOffsets,
                                                            const std::vector<std::string>& srcShape,
                                                            bool isConv3D) const
{
    std::vector<std::string> tileOpCopyInParamList;
    tileOpCopyInParamList.emplace_back(dstTensor);
    tileOpCopyInParamList.emplace_back(srcTensor);

    // offset 参数顺序：conv2d 为 n,c,h,w,0 (d=0 放最后)；conv3d 为 n,c,d,h,w
    // staticOffsets/gmOffsetExpr 已在 GetDynamicOffsetExpr 中初始化为 SHAPE_DIM5 维
    if (functionType == FunctionType::STATIC) {
        for (size_t i = 0; i < SHAPE_DIM5; i++) {
            tileOpCopyInParamList.emplace_back(std::to_string(staticOffsets[i]));
        }
    } else {
        for (size_t i = 0; i < SHAPE_DIM5; i++) {
            tileOpCopyInParamList.emplace_back(gmOffsetExpr[i]);
        }
    }

    // shape 参数顺序：conv2d 为 n,c,h,w,0 (d=0 放最后)；conv3d 为 n,c,d,h,w
    size_t inputDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    for (size_t i = 0; i < inputDim; i++) {
        tileOpCopyInParamList.emplace_back(srcShape[i]);
    }
    if (!isConv3D) {
        tileOpCopyInParamList.emplace_back("0");
    }

    return tileOpCopyInParamList;
}

std::vector<std::string> CodeGenOpNPU::BuildCopyOutParamList(const std::string& dstTensor, const std::string& srcTensor,
                                                             const std::vector<std::string>& gmOffsetExpr,
                                                             const std::vector<int64_t>& staticOffsets,
                                                             const std::string& realM, const std::string& realN,
                                                             const std::string& realCutW, const std::string& cutW) const
{
    std::vector<std::string> tileOpCopyOutParamList;
    tileOpCopyOutParamList.emplace_back(dstTensor);
    tileOpCopyOutParamList.emplace_back(srcTensor);

    // offset 参数顺序：conv2d 为 n,c,h,w,0 (d=0 放最后)；conv3d 为 n,c,d,h,w
    // staticOffsets/gmOffsetExpr 已在 GetDynamicOffsetExpr 中初始化为 SHAPE_DIM5 维
    if (functionType == FunctionType::STATIC) {
        for (size_t i = 0; i < SHAPE_DIM5; i++) {
            tileOpCopyOutParamList.emplace_back(std::to_string(staticOffsets[i]));
        }
    } else {
        for (size_t i = 0; i < SHAPE_DIM5; i++) {
            tileOpCopyOutParamList.emplace_back(gmOffsetExpr[i]);
        }
    }

    tileOpCopyOutParamList.emplace_back(realM);
    tileOpCopyOutParamList.emplace_back(realN);
    tileOpCopyOutParamList.emplace_back(realCutW);
    tileOpCopyOutParamList.emplace_back(cutW);

    return tileOpCopyOutParamList;
}

int64_t CodeGenOpNPU::GetConvCopyInMode() const
{
    int64_t copyInMode = -1;
    auto ret = GetOpAttr(Conv::LoadStoreConvOpAttributeKey::copyInMode, copyInMode);
    ASSERT(ConvCodenGenError::CODEGEN_GET_ATTR_FAILED, ret) << "GenMemL1CopyInConv get CopyInMode failed.";
    bool isValidMode = copyInMode >= ToUnderlying(Matrix::CopyInMode::ND2NZ) &&
                       copyInMode <= ToUnderlying(Matrix::CopyInMode::DN2NZ);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, isValidMode)
        << "GenMemL1CopyInConv CopyInMode is invalid: " << copyInMode;
    return copyInMode;
}

std::string CodeGenOpNPU::GenMemL1CopyInConv() const
{
    std::vector<std::string> tileOpParams = GetTileOpParamsByOrder();
    int64_t copyInMode = GetConvCopyInMode();
    std::string copyInModeStr = CopyInModeToString(static_cast<Matrix::CopyInMode>(copyInMode));
    if (copyInMode == ToUnderlying(Matrix::CopyInMode::NZ2NZ)) {
        return GenMemL1CopyInConvNZ2NZ(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)],
                                       tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)], copyInModeStr);
    }

    bool isFmap = true, isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isFmap, isFmap);
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    std::vector<std::string> srcShape;
    if (isDynamicFunction) {
        std::vector<SymbolicScalar> srcShapeVec;
        GetOpAttr(OpAttributeKey::srcGmConvValidShape, srcShapeVec);
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == expectedDim)
            << "GenMemL1CopyInConv shape should be " << expectedDim << "-dim!";
        FillParamWithFullInput(srcShape, srcShapeVec);
    } else {
        auto srcShapeVec = shape[ToUnderlying(MISOIdx::SRC0_IDX)];
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == expectedDim)
            << "GenMemL1CopyInConv shape should be " << expectedDim << "-dim!";
        FillParamWithFullInput(srcShape, srcShapeVec);
    }

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == expectedDim)
        << "GenMemL1CopyInConv offset should be " << expectedDim << "-dim!";

    std::vector<int64_t> staticOffsets;
    std::vector<std::string> gmOffsetExpr;
    GetDynamicOffsetExpr(dynOffset, isConv3D, gmOffsetExpr, staticOffsets);

    std::vector<std::string> tileOpParamList = BuildCopyInParamList(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)],
                                                                    tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)],
                                                                    gmOffsetExpr, staticOffsets, srcShape, isConv3D);

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyInModeStr, std::to_string(isConv3D), std::to_string(isFmap)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1CopyInConvNZ2NZ(const std::string& dstTensor, const std::string& srcTensor,
                                                  const std::string& copyInModeStr) const
{
    bool isFmap = true, isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isFmap, isFmap);
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));

    // [n, c1, h, w, c0]/[n, d, c1, h, w], [c1hw, cout1, n0, c0, 0]/[dc1hw, cout1, n0, c0, 0]
    std::vector<std::string> srcShape;
    if (isDynamicFunction) {
        std::vector<SymbolicScalar> srcShapeVec;
        GetOpAttr(OpAttributeKey::srcGmConvValidShape, srcShapeVec);
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() >= SHAPE_DIM4)
            << "GenMemL1CopyInConv shape should be " << SHAPE_DIM4 << "-dim!";
        FillParamWithFullInput(srcShape, srcShapeVec);
    } else {
        auto srcShapeVec = shape[ToUnderlying(MISOIdx::SRC0_IDX)];
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() >= SHAPE_DIM4)
            << "GenMemL1CopyInConv shape should be " << SHAPE_DIM4 << "-dim!";
        FillParamWithFullInput(srcShape, srcShapeVec);
    }

    if (isFmap && isConv3D) {
        srcShape.pop_back();
    } else if (!isFmap) {
        srcShape.emplace_back("0");
    }

    std::vector<std::string> staticOffsets;
    std::vector<std::string> gmOffsetExpr;
    GetNZ2NZDynamicOffsetExpr(dynOffset, isConv3D, isFmap, gmOffsetExpr, staticOffsets);

    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor};
    if (functionType == FunctionType::STATIC) {
        tileOpParamList.insert(tileOpParamList.end(), staticOffsets.begin(), staticOffsets.end());
    } else {
        tileOpParamList.insert(tileOpParamList.end(), gmOffsetExpr.begin(), gmOffsetExpr.end());
    }
    tileOpParamList.insert(tileOpParamList.end(), srcShape.begin(), srcShape.end());

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyInModeStr, std::to_string(isConv3D), std::to_string(isFmap)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GetConvCopyOutMode() const
{
    int64_t copyOutMode = -1;
    auto ret = GetOpAttr(Conv::LoadStoreConvOpAttributeKey::copyOutMode, copyOutMode);
    ASSERT(ConvCodenGenError::CODEGEN_GET_ATTR_FAILED, ret) << "GenMemL0CCopyOutConv get CopyOutMode failed.";
    bool isValidMode = copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2ND) ||
                       copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2NZ) ||
                       copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2DN);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, isValidMode)
        << "GenMemL0CCopyOutConv CopyOutMode is invalid: " << copyOutMode;
    return CopyOutModeToString(static_cast<Matrix::CopyOutMode>(copyOutMode));
}

std::string CodeGenOpNPU::GenMemL0CCopyOutConv() const
{
    std::vector<std::string> tileOpParams = GetTileOpParamsByOrder();
    std::string copyOutModeStr = GetConvCopyOutMode();

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
    int64_t reluType = 0;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::reluType, reluType);

    // 获取cutW参数，默认值为0
    int64_t cutW = 0;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::cutW, cutW);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, cutW != 0) << "GenMemL0CCopyOutConv cutW should not be 0!";
    SymbolicScalar cutWValidShape = 0;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::realCutW, cutWValidShape);

    std::vector<SymbolicScalar> srcShapeVec;
    GetOpAttr(OpAttributeKey::l0cValidMN, srcShapeVec);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == SHAPE_DIM2)
        << "GenMemL0CCopyOutConv valid shape should be 2-dim!";
    std::string realM = SymbolicExpressionTable::BuildExpression(srcShapeVec[ID0]);
    std::string realN = SymbolicExpressionTable::BuildExpression(srcShapeVec[ID1]);
    std::string realCutW = SymbolicExpressionTable::BuildExpression(cutWValidShape);

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::DST_IDX));
    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() >= expectedDim)
        << "GenMemL0CCopyOutConv offset should be at least " << expectedDim << "-dim!";

    std::vector<int64_t> staticOffsets;
    std::vector<std::string> gmOffsetExpr;
    GetDynamicOffsetExpr(dynOffset, isConv3D, gmOffsetExpr, staticOffsets);

    std::vector<std::string> tileOpParamList = BuildCopyOutParamList(
        tileOpParams[ToUnderlying(MISOIdx::DST_IDX)], tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)], gmOffsetExpr,
        staticOffsets, realM, realN, realCutW, std::to_string(cutW));

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyOutModeStr, std::to_string(isConv3D), std::to_string(reluType)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToL0Load3D() const
{
    std::vector<std::variant<std::string, uint8_t, uint16_t, int, int64_t>> paramList;

    auto tileOpParams = GetTileOpParamsByOrder();
    paramList.insert(paramList.end(), tileOpParams.begin(), tileOpParams.end());

    auto loadParams = [this](auto& list, auto key, auto& value) {
        GetOpAttr(key, value);
        list.emplace_back(value);
    };

    int64_t val = 0;
    loadParams(paramList, OpAttributeKey::postM, val);
    loadParams(paramList, OpAttributeKey::postK, val);
    loadParams(paramList, OpAttributeKey::paddingLeft, val);
    loadParams(paramList, OpAttributeKey::paddingRight, val);
    loadParams(paramList, OpAttributeKey::paddingTop, val);
    loadParams(paramList, OpAttributeKey::paddingBottom, val);
    loadParams(paramList, OpAttributeKey::padValue, val);
    loadParams(paramList, OpAttributeKey::filterH, val);
    loadParams(paramList, OpAttributeKey::filterW, val);
    loadParams(paramList, OpAttributeKey::dilationH, val);
    loadParams(paramList, OpAttributeKey::dilationW, val);
    loadParams(paramList, OpAttributeKey::strideH, val);
    loadParams(paramList, OpAttributeKey::strideW, val);
    loadParams(paramList, OpAttributeKey::repeatStride, val);
    loadParams(paramList, OpAttributeKey::repeatTime, val);
    loadParams(paramList, OpAttributeKey::wStride, val);

    std::vector<int64_t> fmapL1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0Load3D %s, fmapL1Shape is %s", tileOpName.c_str(), IntVecToStr(fmapL1Shape).c_str());
    std::vector<int64_t> fmapL0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0Load3D %s, fmapL0Shape is %s", tileOpName.c_str(), IntVecToStr(fmapL0Shape).c_str());
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, fmapL0Shape.size() == SHAPE_DIM2)
        << "GenMemL1ToL0Load3D L0 fmap only support 2-dim!";

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    std::ostringstream oss;
    oss << tileOpName.c_str() << WrapParamByAngleBrackets({std::to_string(isConv3D)});
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToL0Load2D() const
{
    std::vector<std::variant<std::string, uint16_t, int, int64_t>> paramList;

    auto tileOpParams = GetTileOpParamsByOrder();
    paramList.insert(paramList.end(), tileOpParams.begin(), tileOpParams.end());

    int64_t kPos = 0, nPos = 0;
    GetOpAttr(OpAttributeKey::postK, kPos);
    GetOpAttr(OpAttributeKey::postN, nPos);
    paramList.emplace_back(kPos);
    paramList.emplace_back(nPos);

    std::vector<int64_t> weightL1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0Load2D %s, weightL1Shape is %s", tileOpName.c_str(), IntVecToStr(weightL1Shape).c_str());

    std::vector<int64_t> weightL0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0Load2D %s, weightL0Shape is %s", tileOpName.c_str(), IntVecToStr(weightL0Shape).c_str());
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, weightL0Shape.size() == SHAPE_DIM2)
        << "GenMemL1ToL0Load2D L0 weight only support 2-dim!";

    std::ostringstream oss;
    oss << tileOpName.c_str() << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
