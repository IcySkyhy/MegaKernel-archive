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
 * \file test_main_block.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/pypto_fwk_log.h"
#include "interface/tensor/irbuilder.h"
#include "interface/operation/attribute.h"
#include "passes/pass_utils/pass_operation_utils.h"

#define private public
#include "machine/host/main_block.h"
#undef private

using namespace npu::tile_fwk;

static std::vector<SymbolicScalar> MakeConstIntVectors(const std::vector<int64_t>& values)
{
    IRBuilder builder;
    std::vector<SymbolicScalar> result;
    result.reserve(values.size());
    for (auto value : values) {
        result.emplace_back(builder.CreateConstInt(value));
    }
    return result;
}

class TestMainBlock : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE, 1);
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

static std::vector<SymbolicScalar> MakeCoaEntry(int64_t rawTensorIdx, const std::vector<int64_t>& offset,
                                                const std::vector<int64_t>& shape, const std::vector<int64_t>& rawshape,
                                                const std::vector<int64_t>& validshape)
{
    std::vector<SymbolicScalar> entry;
    entry.push_back(SymbolicScalar(rawTensorIdx));
    for (auto v : offset)
        entry.push_back(SymbolicScalar(v));
    for (auto v : shape)
        entry.push_back(SymbolicScalar(v));
    for (auto v : rawshape)
        entry.push_back(SymbolicScalar(v));
    for (auto v : validshape)
        entry.push_back(SymbolicScalar(v));
    return entry;
}

struct CopyInOutFixture {
    std::shared_ptr<Function> func;
    std::vector<SymbolicScalar> linearArgList;

    CopyInOutFixture(const std::string& name, const std::vector<int64_t>& validShape1,
                     const std::vector<int64_t>& validShape2)
    {
        func = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
        std::vector<int64_t> shape = {16, 16};

        auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
        auto ubTensor = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
        auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));

        func->inCasts_.push_back(input);
        func->outCasts_.push_back(output);

        auto& copyIn = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input}, {ubTensor});
        copyIn.SetIOpAtt(0, 1);

        auto& copyOut = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {ubTensor}, {output});
        copyOut.SetOOpAtt(0, 10);

        linearArgList.push_back(SymbolicScalar((int64_t)0));
        auto entry1 = MakeCoaEntry(0, {0, 0}, {16, 16}, {16, 16}, validShape1);
        linearArgList.insert(linearArgList.end(), entry1.begin(), entry1.end());
        auto entry2 = MakeCoaEntry(0, {0, 0}, {16, 16}, {16, 16}, validShape2);
        linearArgList.insert(linearArgList.end(), entry2.begin(), entry2.end());
    }
};

TEST_F(TestMainBlock, CollectLeafMainBlockConds_ShapeEqualsValidShape)
{
    CopyInOutFixture fix("testFunc", {16, 16}, {16, 16});

    MainBlockCondBulider builder;
    builder.CollectLeafMainBlockConds(fix.func.get(), fix.linearArgList);

    EXPECT_FALSE(builder.mainBlockDisabled_);
    EXPECT_FALSE(builder.mainBlockCondGroup_.empty());
}

TEST_F(TestMainBlock, CollectLeafMainBlockConds_ShapeNotEqualsValidShape)
{
    CopyInOutFixture fix("testFunc2", {8, 8}, {16, 16});

    MainBlockCondBulider builder;
    builder.CollectLeafMainBlockConds(fix.func.get(), fix.linearArgList);

    EXPECT_TRUE(builder.mainBlockDisabled_);
}

TEST_F(TestMainBlock, CollectLeafMainBlockConds_CoaIndexBaseNegative)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "testFunc3", "testFunc3", nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
    auto ubTensor = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));

    auto& addOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ADD, {input}, {ubTensor});
    (void)addOp;

    std::vector<SymbolicScalar> linearArgList;
    linearArgList.push_back(SymbolicScalar((int64_t)0));

    MainBlockCondBulider builder;
    builder.CollectLeafMainBlockConds(func.get(), linearArgList);

    EXPECT_FALSE(builder.mainBlockDisabled_);
}

TEST_F(TestMainBlock, CollectLeafMainBlockConds_DimZero)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "testFunc5", "testFunc5", nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
    auto ubTensor = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));

    auto& copyIn = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input}, {ubTensor});
    copyIn.SetIOpAtt(0, 1);
    copyIn.GetIOperands()[0]->shape.clear();

    std::vector<SymbolicScalar> linearArgList;
    for (int i = 0; i < 20; i++) {
        linearArgList.push_back(SymbolicScalar((int64_t)i));
    }

    MainBlockCondBulider builder;
    builder.CollectLeafMainBlockConds(func.get(), linearArgList);

    EXPECT_TRUE(builder.mainBlockDisabled_);
}

TEST_F(TestMainBlock, CollectLeafMainBlockConds_LinearArgListTooSmall)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "testFunc5", "testFunc5", nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
    auto ubTensor = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, MakeConstIntVectors(shape));

    auto& copyIn = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input}, {ubTensor});
    copyIn.SetIOpAtt(0, 1);

    auto& copyOut = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {ubTensor}, {output});
    copyOut.SetOOpAtt(0, 10);

    std::vector<SymbolicScalar> linearArgList;

    MainBlockCondBulider builder;
    builder.CollectLeafMainBlockConds(func.get(), linearArgList);

    EXPECT_TRUE(builder.mainBlockDisabled_);
}

TEST_F(TestMainBlock, CollectLeafMainBlockConds_DisabledWhenNotEnabled)
{
    config::SetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE, 0);
    config::SetPassGlobalConfig(KEY_ENABLE_VF, false);

    auto func = std::make_shared<Function>(Program::GetInstance(), "testFunc4", "testFunc4", nullptr);
    std::vector<SymbolicScalar> linearArgList;
    linearArgList.push_back(SymbolicScalar((int64_t)0));

    MainBlockCondBulider builder;
    builder.CollectLeafMainBlockConds(func.get(), linearArgList);

    EXPECT_TRUE(builder.mainBlockDisabled_);
}
