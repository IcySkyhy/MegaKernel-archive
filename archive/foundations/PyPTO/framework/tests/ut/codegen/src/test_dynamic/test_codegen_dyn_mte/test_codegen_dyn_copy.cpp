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
 * \file test_codegen_dyn_copy.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

constexpr const int dummyRawMagic = 123;

class TestCodegenDynCopy : public CodegenTestBase {
public:
    TestCodegenDynCopy()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH,
                           .setTileTensor = true,
                           .setIdGen = true,
                           .resetTileTensorOnTearDown = true})
    {}
};

std::string TestL0COutBody(const std::string& funcName, bool isDynamicAligned = false)
{
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    auto function = GenMockFuncDyn(funcName);
    if (isDynamicAligned) {
        config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    }

    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L0CToOut"});
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_COPY_OUT, {localTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetOOpAtt(0, 0);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    if (!isDynamicAligned) {
        op.SetAttribute(OpAttributeKey::copyIsNZ, 1);
    }

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynCopy, L0CToOut)
{
    std::string res = TestL0COutBody("L0CToOut", true);
    std::string expect =
        R"!!!(TileOp::DynL0CCopyOut<float, float, 64, 64, 64, 64>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__cc__ float*)L0C_S0_E0, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CToOutUnalign)
{
    std::string res = TestL0COutBody("L0CToOutUnalign");
    std::string expect =
        R"!!!(TileOp::DynL0CCopyOut<float, float, false, 0>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__cc__ float*)L0C_S0_E0, 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 1), 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1ToFB)
{
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {64, 64});
    auto function = GenMockFuncDyn("L1ToFB");
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localInTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_FIX, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localInTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto& op = function->AddOperation(Opcode::OP_L1_TO_FIX_QUANT_PRE, {localInTensor}, {localOutTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_FIX, shapeImme, shapeImme));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynL1ToFB<float, 0>((__fbuf__ float*)FIXBUF_S0_E0, (__cbuf__ float*)L1_S0_E0, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestL1CopyInBody(const std::string& funcName, bool isNz = false, int outerValueForNz = 0,
                             int innerValueForNz = 0, bool isTileTensor = false)
{
    if (isTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    }
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);

    auto function = GenMockFuncDyn(funcName);

    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1CopyIn"});
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {localTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({0, 0}));
    copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    op.SetIOpAtt(0, 0);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    if (isNz) {
        op.SetAttribute(OP_ATTR_PREFIX + "is_nz", 1);
        op.SetAttribute(OP_ATTR_PREFIX + "outer_value", outerValueForNz);
        op.SetAttribute(OP_ATTR_PREFIX + "inner_value", innerValueForNz);
    }

    if (isTileTensor) {
        op.SetAttribute(OP_ATTR_PREFIX + "copy_in_mode", 2);
    }

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynCopy, L1CopyIn)
{
    std::string res = TestL1CopyInBody("L1CopyIn");
    std::string expect =
        R"!!!(TileOp::DynL1CopyIn<float, float>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInTileTensor)
{
    std::string res = TestL1CopyInBody("L1CopyInTileTensor", false, 0, 0, true);
    std::string expect =
        R"!!!(TLoad<CopyInMode::NZ2NZ, PaddingMode::NORMAL_PADDING_MODE, -1>(l1Tensor_0, gmTensor_1, Coord2Dim(0, 0), Coord2Dim(0, 0), 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInDynAligned)
{
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    std::string res = TestL1CopyInBody("L1CopyInDynAligned");
    std::string expect =
        R"!!!(TileOp::DynL1CopyIn<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInNZWithZeroAligned)
{
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    std::string res = TestL1CopyInBody("L1CopyInNZWithZeroAligned", true);
    std::string expect =
        R"!!!(TileOp::DynL1CopyInNZ2NZ<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 64, 64, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInNZWithValue)
{
    std::string res = TestL1CopyInBody("L1CopyInNZWithValue", true, 1, 1);
    std::string expect =
        R"!!!(TileOp::DynL1CopyInNZ2NZ<float, float>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 1, 1, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, TestGatherInL1TileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    std::vector<int64_t> gatherShape = {64, 64};
    auto shapeImme = OpImmediate::Specified(gatherShape);
    auto function = GenMockFuncDyn("GatherInL1TileTensor");
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto gatherTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, gatherShape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_L1, gatherShape, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    gatherTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    localOutTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    LogicalTensors inputs = {gatherTensor, gatherTensor, gatherTensor};
    LogicalTensors outputs = {localOutTensor};

    auto& gatherL1Op = function->AddOperation(Opcode::OP_GATHER_IN_L1, inputs, outputs);
    gatherL1Op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    int64_t blocksize{0};
    gatherL1Op.SetAttribute("op_attr_blocksize", blocksize);
    gatherL1Op.SetAttribute(OpAttributeKey::startOffset, blocksize);
    gatherL1Op.SetOOpAtt(0, 0);
    std::string res = GenOpCodeFromOp(*function, gatherL1Op, {.isMainBlk = true});
    std::string expect =
        R"!!!(TGatherInL1<0>(l1Tensor_0, gmTensor_1, gmTensor_1, gmTensor_1, Coord1Dim(0), Coord2Dim(GET_PARAM_OFFSET_BY_IDX(param, 0, -1, 2, 0), GET_PARAM_OFFSET_BY_IDX(param, 0, -1, 2, 1)), Coord2Dim(GET_PARAM_OFFSET_BY_IDX(param, 0, -1, 2, 0), GET_PARAM_OFFSET_BY_IDX(param, 0, -1, 2, 1)));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1ToBt)
{
    auto function = GenMockFuncDyn("L1ToBt");
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_BT, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto& op = function->AddOperation(Opcode::OP_L1_TO_BT, {localTensor}, {localOutTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_BT, shapeImme, shapeImme));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynL1ToBT<float, float, 0>((uint64_t)BIAS_S0_E0, (__cbuf__ float*)L1_S0_E0, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestMatmulMteBody(const std::string& funcName, Opcode opcode, MemoryType inType, MemoryType outType,
                              bool isSupportTileTensor = true)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, isSupportTileTensor);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);

    auto function = GenMockFuncDyn(funcName);
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_INT32, inType, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP16, outType, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    LogicalTensors inputs = {localTensor};
    LogicalTensors outputs = {localOutTensor};
    if (opcode == Opcode::OP_COPY_OUT || opcode == Opcode::OP_L0C_COPY_UB ||
        opcode == Opcode::OP_L0C_RESHAPE_COPY_OUT) {
        inputs.emplace_back(localTensor);
    }
    if (opcode == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
        outputs.emplace_back(localOutTensor);
    }

    auto& op = function->AddOperation(opcode, inputs, outputs);

    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    if (opcode == Opcode::OP_COPY_OUT || opcode == Opcode::OP_L0C_RESHAPE_COPY_OUT) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_TO_BT) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_BT, shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_COPY_IN || opcode == Opcode::OP_L1_RESHAPE_COPY_IN) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_TO_FIX_QUANT_PRE) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_FIX, shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_TO_L0A) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L0A, shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L0C_COPY_UB || opcode == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
        // Align with GenerateMoveOp::SetL0C2UBCopyAttr (CopyIn, toDynValidShape on UB destination).
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_UB,
                                                            shapeImme, shapeImme, OpImmediate::Specified(shape)));
    }

    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({0, 0}));
    copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    copyAttr->SetFromDynValidShape(OpImmediate::Specified({0, 0}));
    copyAttr->SetToDynValidShape(OpImmediate::Specified({0, 0}));

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynCopy, L1CopyInTensor)
{
    std::string res = TestMatmulMteBody("L1CopyInTensor", Opcode::OP_L1_COPY_IN, MemoryType::MEM_DEVICE_DDR,
                                        MemoryType::MEM_L1);
    std::string expect =
        R"!!!(TLoad<CopyInMode::ND2NZ, PaddingMode::NORMAL_PADDING_MODE, -1>(l1Tensor_0, gmTensor_1, Coord2Dim(0, 0), Coord2Dim(0, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, -1, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, -1, 2, 1));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1ReshapeCopyInTensor)
{
    std::string res = TestMatmulMteBody("L1ReshapeCopyInTensor", Opcode::OP_L1_RESHAPE_COPY_IN,
                                        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1);
    std::string expect =
        R"!!!(TReshapeLoad<CopyInMode::ND2NZ, PaddingMode::NORMAL_PADDING_MODE, -1>(l1Tensor_0, gmTensor_1, Coord2Dim(0, 0), 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyL0Tensor)
{
    std::string res = TestMatmulMteBody("L1CopyL0Tensor", Opcode::OP_L1_TO_L0A, MemoryType::MEM_L1,
                                        MemoryType::MEM_L0A);
    std::string expect = R"!!!(TExtractL1ToL0<0, 0>(l0aTensor_0, l1Tensor_1, Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyFBTensor)
{
    std::string res = TestMatmulMteBody("L1CopyFBTensor", Opcode::OP_L1_TO_FIX_QUANT_PRE, MemoryType::MEM_L1,
                                        MemoryType::MEM_FIX);
    std::string expect = R"!!!(TExtractL1ToFB<0>(fbufTensor_0, l1Tensor_1, Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyBTTensor)
{
    std::string res = TestMatmulMteBody("L1CopyBTTensor", Opcode::OP_L1_TO_BT, MemoryType::MEM_L1, MemoryType::MEM_BT);
    std::string expect = R"!!!(TExtractL1ToBT<0>(btTensor_0, l1Tensor_1, Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CCopyOut)
{
    std::string res = TestMatmulMteBody("L0CCopyOutTensor", Opcode::OP_COPY_OUT, MemoryType::MEM_L0C,
                                        MemoryType::MEM_DEVICE_DDR, false);
    std::string expect =
        R"!!!(TileOp::DynL0CCopyOut<half, int32_t, true, 0>((__gm__ half*)GET_PARAM_ADDR(param, 0, -1), (__cc__ int32_t*)L0C_S0_E0, 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, -1), GET_PARAM_OFFSET_2(param, 0, -1), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, -1, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, -1, 2, 1), 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CCopyOutTileTensor)
{
    std::string res = TestMatmulMteBody("L0CCopyOutTensorTileTensor", Opcode::OP_COPY_OUT, MemoryType::MEM_L0C,
                                        MemoryType::MEM_DEVICE_DDR);
    std::string expect =
        R"!!!(TStore<TStoreConfig<CopyOutMode::NZ2ND, 0, 0>>(gmTensor_0, l0cTensor_1, l0cTensor_1, Coord2Dim(0, 0), 64, 64, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CReshapeCopyOut)
{
    std::string res = TestMatmulMteBody("L0CReshapeCopyOut", Opcode::OP_L0C_RESHAPE_COPY_OUT, MemoryType::MEM_L0C,
                                        MemoryType::MEM_DEVICE_DDR);
    std::string expect =
        R"!!!(TReshapeStore<TStoreConfig<CopyOutMode::NZ2ND, 0, 0>>(gmTensor_0, l0cTensor_1, l0cTensor_1, Coord2Dim(0, 0), 0, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CCopyUBTensor)
{
    std::string res = TestMatmulMteBody("L0CCopyUBTensor", Opcode::OP_L0C_COPY_UB, MemoryType::MEM_L0C,
                                        MemoryType::MEM_UB);
    std::string expect =
        R"!!!(TCopyL0C2UB<TStoreConfig<CopyOutMode::NZ2ND, 0, 0>, CopyMode::UNKNOWN, DualDstMode::DUAL_DST_DISABLE>(ubTensor_0, l0cTensor_1, l0cTensor_1, Coord2Dim(0, 0), Coord2Dim(0, 0), 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CCopyUBTensorDualDst)
{
    std::string res = TestMatmulMteBody("L0CCopyUBTensorDualDst", Opcode::OP_L0C_COPY_UB_DUAL_DST, MemoryType::MEM_L0C,
                                        MemoryType::MEM_UB);
    std::string expect =
        R"!!!(TCopyL0C2UB<TStoreConfig<CopyOutMode::NZ2ND, 0, 0>, CopyMode::EXTRACT, DualDstMode::DUAL_DST_SPLIT_M>(ubTensor_0, l0cTensor_2, l0cTensor_2, Coord2Dim(0, 0), Coord2Dim(0, 0), 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestCopyL1Body(Opcode opcode, MemoryType inputType, MemoryType outputType)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    std::string funcName = OpcodeManager::Inst().GetOpcodeStr(opcode);
    auto function = GenMockFuncDyn(funcName);
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, inputType, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, outputType, shape, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    localOutTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    LogicalTensors inputs = {localTensor};
    LogicalTensors outputs = {localOutTensor};
    if (opcode == Opcode::OP_L0C_TO_L1) {
        auto localTensor1 = CreateLogicalTensor(
            {*function, DataType::DT_FP32, MemoryType::MEM_FIX, shape, dynValidShape});
        localTensor1->UpdateOffset(TensorOffset(offset, dynoffset));
        inputs.emplace_back(localTensor1);
    }
    auto& op = function->AddOperation(opcode, inputs, outputs);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    if (opcode == Opcode::OP_L0C_TO_L1) {
        auto shapeImme = OpImmediate::Specified(shape);
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                            shapeImme, shapeImme, OpImmediate::Specified(shape)));
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    }
    if (opcode == Opcode::OP_UB_COPY_L1) {
        auto shapeImme = OpImmediate::Specified(shape);
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L1, shapeImme, shapeImme));
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    }

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynCopy, UB2L1TileTensor)
{
    std::string res = TestCopyL1Body(Opcode::OP_UB_COPY_L1, MemoryType::MEM_UB, MemoryType::MEM_L1);
    std::string expect = R"!!!(TCopyUB2L1<CopyMode::UNKNOWN>(l1Tensor_0, ubTensor_1, Coord2Dim(0, 0), Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, UB2UBND2NZTileTensor)
{
    std::string res = TestCopyL1Body(Opcode::OP_UB_COPY_ND2NZ, MemoryType::MEM_UB, MemoryType::MEM_UB);
    std::string expect = R"!!!(TMoveND2NZ(ubTensor_0, ubTensor_0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CToL1TileTensor)
{
    std::string res = TestCopyL1Body(Opcode::OP_L0C_TO_L1, MemoryType::MEM_L0C, MemoryType::MEM_L1);
    std::string expect =
        R"!!!(TExtract<TStoreConfig<CopyOutMode::NZ2NZ, 0, 0>>(l1Tensor_0, l0cTensor_1, l0cTensor_1, Coord2Dim(0, 0), Coord2Dim(0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

void TestUBCopyInBody(const std::string funcName, const std::string& expect)
{
    auto function = GenMockFuncDyn(funcName);
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "UBCopyIn"});
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {localTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    GmOutOfRangeCheckInfo gmCheckInfo = {OpImmediate::Specified(SymbolicScalar(0)),
                                         OpImmediate::Specified(SymbolicScalar(64)),
                                         OpImmediate::Specified(SymbolicScalar(1024))};
    copyAttr->SetGmOutOfRangeCheck(gmCheckInfo);
    copyAttr->SetFromOffset(OpImmediate::Specified({0, 0}));
    copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    op.SetIOpAtt(0, 0);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynCopy, UBCopyIn)
{
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, CFG_RUNTIME_DEBUG_GM_OUT_OF_BOUNDS);
    std::string expect =
        R"!!!(TileOp::DynUBCopyIn<float, 1, 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 1, 1, 1, 64, 64, 1, 1, 1, GET_PARAM_RAWSHAPE_2(param, 0, 0), 0, 0, 0, 0, 0);
)!!!";
    TestUBCopyInBody("UBCopyIn", expect);
}

TEST_F(TestCodegenDynCopy, UBCopyInAligned)
{
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    std::string expect =
        R"!!!(TileOp::DynUBCopyIn<float, 1, 1, 1, 64, 64, 1, 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 1, 1, 1, GET_PARAM_RAWSHAPE_2(param, 0, 0), 0, 0, 0, 0, 0);
)!!!";
    TestUBCopyInBody("UBCopyInAligned", expect);
}

TEST_F(TestCodegenDynCopy, L0CToL1)
{
    auto function = GenMockFuncDyn("L0CToL1");
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_L0C_TO_L1, {localTensor}, {localOutTensor});
    auto shapeImmeL0C = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                        shapeImmeL0C, shapeImmeL0C, OpImmediate::Specified(shape)));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynL0CToL1<float, float, 0>((__cbuf__ float*)L1_S0_E0, (__cc__ float*)L0C_S0_E0, 64, 64, 64, 64, 0, 0, 64, 64, 0, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
