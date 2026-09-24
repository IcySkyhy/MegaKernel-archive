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
 * \file test_codegen_dyn_conv.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

constexpr int64_t N0 = 16;
constexpr int64_t COPY_IN_MODE_NZ2NZ = 2;
constexpr int64_t COPY_IN_MODE_DN2NZ = 3;
constexpr int64_t COPY_OUT_MODE_NZ2NZ = 1;
constexpr int64_t COPY_OUT_MODE_NZ2DN = 3;

class TestCodegenDynConv : public CodegenTestBase {
public:
    TestCodegenDynConv()
        : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION,
                           .setTileTensor = true,
                           .tileTensorValue = true,
                           .setIdGen = true,
                           .resetTileTensorOnTearDown = true})
    {}
};

Function* GetFunctionConv(const std::string& funcName)
{
    const std::vector<int64_t> shape = {64, 64};
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);

    auto function = GenMockFuncDyn(funcName, shape);
    return function;
}

std::vector<int64_t> GetDstL1ShapeDN2NZ(const std::vector<int64_t>& gmShape, bool isFmap, bool isConv3D, DataType dtype)
{
    std::map<DataType, int64_t> k0Map = {{DataType::DT_FP16, 16}, {DataType::DT_BF16, 16}, {DataType::DT_FP32, 8}};
    int64_t k0 = k0Map[dtype];
    std::vector<int64_t> dstL1Shape;
    if (isConv3D) {
        if (isFmap) {
            dstL1Shape = {gmShape[0], gmShape[2], CeilDiv(gmShape[1], k0), gmShape[3], gmShape[4], k0};
        } else {
            dstL1Shape = {CeilDiv(gmShape[1], k0) * gmShape[2] * gmShape[3] * gmShape[4], CeilDiv(gmShape[0], N0), N0,
                          k0};
        }
    } else {
        if (isFmap) {
            dstL1Shape = {gmShape[0], CeilDiv(gmShape[1], k0), gmShape[2], gmShape[3], k0};
        } else {
            dstL1Shape = {CeilDiv(gmShape[1], k0) * gmShape[2] * gmShape[3], CeilDiv(gmShape[0], N0), N0, k0};
        }
    }
    return dstL1Shape;
}

std::vector<int64_t> GetOffsetForCopyInDN2NZ(bool isConv3D)
{
    if (isConv3D) {
        return {0, 0, 0, 0, 0};
    }
    return {0, 0, 0, 0};
}

std::vector<int64_t> GetOffsetForCopyInNZ2NZ(bool isFmap, bool isConv3D)
{
    if (isFmap) {
        if (isConv3D) {
            return {0, 0, 0, 0, 0, 0};
        }
        return {0, 0, 0, 0, 0};
    }
    return {0, 0, 0, 0};
}

void SetConvL1CopyInOpAttr(Operation& op, const std::vector<int64_t>& offset, const std::vector<int64_t>& gmShape,
                           const std::vector<int64_t>& dstL1Shape)
{
    auto copyAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MemoryType::MEM_L1,
                                                      OpImmediate::Specified(gmShape), OpImmediate::Specified(gmShape),
                                                      OpImmediate::Specified(dstL1Shape));
    op.SetOpAttribute(copyAttr);
    op.SetAttribute(OpAttributeKey::srcGmConvValidShape, SymbolicScalar::FromConcrete(gmShape));
}

std::string TestConvL1CopyInBody(const std::string& funcName, const std::vector<int64_t>& gmShape, bool isFmap = true,
                                 int64_t copyInMode = COPY_IN_MODE_DN2NZ, bool isConv3D = false,
                                 DataType dtype = DataType::DT_FP16)
{
    auto function = GetFunctionConv(funcName);
    auto gmTensor = CreateConvTensor(*function, dtype, gmShape, MemoryType::MEM_DEVICE_DDR);
    std::vector<int64_t> dstL1Shape;
    std::vector<int64_t> offset;

    if (copyInMode == COPY_IN_MODE_DN2NZ) {
        dstL1Shape = GetDstL1ShapeDN2NZ(gmShape, isFmap, isConv3D, dtype);
        offset = GetOffsetForCopyInDN2NZ(isConv3D);
    } else if (copyInMode == COPY_IN_MODE_NZ2NZ) {
        dstL1Shape = gmShape;
        offset = GetOffsetForCopyInNZ2NZ(isFmap, isConv3D);
    }

    auto localTensor = CreateConvTensor(*function, dtype, dstL1Shape, MemoryType::MEM_L1);

    auto& op = function->AddOperation(Opcode::OP_L1_COPY_IN_CONV, {gmTensor}, {localTensor});
    op.SetAttribute("IS_FMAP", isFmap);
    op.SetAttribute("IS_CONV3D", isConv3D);
    op.SetAttribute("COPY_IN_MODE", copyInMode);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    SetConvL1CopyInOpAttr(op, offset, gmShape, dstL1Shape);

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorFmapConv2D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorFmapConv2D", {1, 16, 1, 16});
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 0, 1>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 16, 1, 16, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorWeightConv2D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorWeightConv2D", {1, 16, 1, 1}, false);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 0, 0>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 16, 1, 1, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorFmapConv3D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorFmapConv3D", {1, 16, 1, 1, 16}, true, COPY_IN_MODE_DN2NZ,
                                           true);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 1, 1>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 16, 1, 1, 16);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorWeightConv3D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorWeightConv3D", {1, 16, 1, 1, 1}, false,
                                           COPY_IN_MODE_DN2NZ, true);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 1, 0>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 16, 1, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInNZ2NZFmapConv2D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInNZ2NZFmapConv2D", {1, 1, 16, 16, 16}, true, COPY_IN_MODE_NZ2NZ,
                                           false);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::NZ2NZ, 0, 1>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 1, 16, 16, 16);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInNZ2NZFmapConv3D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInNZ2NZFmapConv3D", {1, 1, 1, 16, 16, 16}, true, COPY_IN_MODE_NZ2NZ,
                                           true);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::NZ2NZ, 1, 1>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 1, 1, 16, 16);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInNZ2NZWeightConv2D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInNZ2NZWeightConv2D", {1, 1, 16, 16}, false, COPY_IN_MODE_NZ2NZ,
                                           false);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::NZ2NZ, 0, 0>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 1, 16, 16, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInNZ2NZWeightConv3D)
{
    std::string res = TestConvL1CopyInBody("L1CopyInNZ2NZWeightConv3D", {1, 1, 1, 16}, false, COPY_IN_MODE_NZ2NZ, true);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::NZ2NZ, 1, 0>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 1, 1, 16, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestConvL0COutBody(const std::string& funcName, const std::vector<int64_t>& l0cShape,
                               const std::vector<SymbolicScalar>& l0cValidShape, std::vector<int64_t>& gmShape,
                               int64_t copyOutMode = COPY_OUT_MODE_NZ2DN, bool isConv3D = false,
                               DataType dtype = DataType::DT_FP32, int64_t cutW = 16,
                               SymbolicScalar dynValidCutW = SymbolicScalar(16))
{
    std::vector<int64_t> offset = {0, 0, 0, 0};
    if (isConv3D) {
        offset = {0, 0, 0, 0, 0};
    }
    auto function = GetFunctionConv(funcName);
    auto gmTensor = CreateConvTensor(*function, dtype, gmShape, MemoryType::MEM_DEVICE_DDR, false);
    auto l0cTensor = CreateConvTensor(*function, DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, false);

    auto& op = function->rootFunc_->programs_[0]->AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {l0cTensor}, {gmTensor});
    auto shapeImme = OpImmediate::Specified(l0cShape);
    op.SetAttribute("COPY_OUT_MODE", copyOutMode);
    op.SetAttribute("IS_CONV3D", isConv3D);
    op.SetAttribute("CUT_W", cutW);
    op.SetAttribute("REAL_CUT_W", dynValidCutW);
    op.SetAttribute(OpAttributeKey::l0cValidMN, l0cValidShape);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified(offset),
                                                        OpImmediate::Specified(gmShape),
                                                        OpImmediate::Specified(gmShape), shapeImme));

    return GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynConv, L0COutTileTensorConv2D)
{
    std::vector<int64_t> l0cShape = {32, 16};
    std::vector<int64_t> gmShape = {1, 16, 2, 32};
    std::string res = TestConvL0COutBody("L0COutTileTensorConv2D", l0cShape, SymbolicScalar::FromConcrete(l0cShape),
                                         gmShape);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2DN, 0, 0>(gmTensor_9, l0cTensor_10, 0, 0, 0, 0, 0, 32, 16, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynConv, L0COutTileTensorConv2DTailTensor)
{
    std::vector<int64_t> l0cShape = {32, 32};
    std::vector<SymbolicScalar> l0cValidShape = {16, 16};
    std::vector<int64_t> gmShape = {1, 32, 2, 32};
    std::string res = TestConvL0COutBody("L0COutTileTensorConv2D", l0cShape, l0cValidShape, gmShape);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2DN, 0, 0>(gmTensor_9, l0cTensor_10, 0, 0, 0, 0, 0, 16, 16, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynConv, L0COutTileTensorConv3D)
{
    std::vector<int64_t> l0cShape = {32, 16};
    std::vector<int64_t> gmShape = {1, 16, 1, 2, 32};
    std::string res = TestConvL0COutBody("L0COutTileTensorConv3D", l0cShape, SymbolicScalar::FromConcrete(l0cShape),
                                         gmShape, COPY_OUT_MODE_NZ2DN, true);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2DN, 1, 0>(gmTensor_9, l0cTensor_10, 0, 0, 0, 0, 0, 32, 16, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynConv, L0COutNZ2NZConv2D)
{
    std::vector<int64_t> l0cShape = {16, 16};
    std::vector<int64_t> gmShape = {16, 16};
    std::string res = TestConvL0COutBody("L0COutNZ2NZConv2D", l0cShape, SymbolicScalar::FromConcrete(l0cShape), gmShape,
                                         COPY_OUT_MODE_NZ2NZ, false);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2NZ, 0, 0>(gmTensor_9, l0cTensor_10, 0, 0, 0, 0, 0, 16, 16, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynConv, L0COutNZ2NZConv3D)
{
    std::vector<int64_t> l0cShape = {16, 16};
    std::vector<int64_t> gmShape = {16, 16};
    std::string res = TestConvL0COutBody("L0COutNZ2NZConv3D", l0cShape, SymbolicScalar::FromConcrete(l0cShape), gmShape,
                                         COPY_OUT_MODE_NZ2NZ, true);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2NZ, 1, 0>(gmTensor_9, l0cTensor_10, 0, 0, 0, 0, 0, 16, 16, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

void SetConvLoad3DAttributes(Operation& op, const bool& isConv3D)
{
    op.SetAttribute(OpAttributeKey::postM, (int64_t)0);
    op.SetAttribute(OpAttributeKey::postK, (int64_t)0);
    op.SetAttribute(OpAttributeKey::paddingLeft, (int64_t)0);
    op.SetAttribute(OpAttributeKey::paddingRight, (int64_t)0);
    op.SetAttribute(OpAttributeKey::paddingTop, (int64_t)0);
    op.SetAttribute(OpAttributeKey::paddingBottom, (int64_t)0);
    op.SetAttribute(OpAttributeKey::padValue, (int64_t)0);
    op.SetAttribute(OpAttributeKey::filterH, (int64_t)1);
    op.SetAttribute(OpAttributeKey::filterW, (int64_t)1);
    op.SetAttribute(OpAttributeKey::dilationH, (int64_t)1);
    op.SetAttribute(OpAttributeKey::dilationW, (int64_t)1);
    op.SetAttribute(OpAttributeKey::strideH, (int64_t)1);
    op.SetAttribute(OpAttributeKey::strideW, (int64_t)1);
    op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
}

std::string TestConvLoad3DBody(const std::string& funcName, const bool& isConv3D)
{
    auto function = GetFunctionConv(funcName);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L1, {16, 16}, dynValidShape});
    auto l0Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0A, {16, 16}, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    l1Tensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto& op = function->AddOperation(Opcode::OP_LOAD3D_CONV, {l1Tensor}, {l0Tensor});
    SetConvLoad3DAttributes(op, isConv3D);

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConv, Load3DConv2D)
{
    std::string res = TestConvLoad3DBody("Load3DConv2D", false);
    std::string expect = R"!!!(TLoad3D<0>(l0aTensor_0, l1Tensor_1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, Load3DConv3D)
{
    std::string res = TestConvLoad3DBody("Load3DConv3D", true);
    std::string expect = R"!!!(TLoad3D<1>(l0aTensor_0, l1Tensor_1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

void SetConvLoad2DAttributes(Operation& op)
{
    op.SetAttribute(OpAttributeKey::postK, (int64_t)0);
    op.SetAttribute(OpAttributeKey::postN, (int64_t)0);
}

std::string TestConvLoad2DBody(const std::string& funcName)
{
    auto function = GetFunctionConv(funcName);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L1, {16, 16}, dynValidShape});
    auto l0Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0B, {16, 16}, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    l1Tensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto& op = function->AddOperation(Opcode::OP_LOAD2D_CONV, {l1Tensor}, {l0Tensor});
    SetConvLoad2DAttributes(op);

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConv, Load2DConv)
{
    std::string res = TestConvLoad2DBody("Load2DConv");
    std::string expect = R"!!!(TLoad2D(l0bTensor_0, l1Tensor_1, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
