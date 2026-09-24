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
 * \file test_operation_token_dep.cpp
 * \brief Unit tests for Operation::ConsumerOpsByToken / ProducerOpsByToken
 */

#include "gtest/gtest.h"

#include "ir/span.h"
#include "ir/type.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "interface/function/function.h"
#undef private
#undef protected
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

class OperationTokenDepTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "OperationTokenDepTest SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "OperationTokenDepTest TearDownTestCase" << std::endl; }

    void SetUp() override
    {
        std::cout << "OperationTokenDepTest SetUp" << std::endl;
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        std::cout << "OperationTokenDepTest TearDown" << std::endl;
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

static ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetUnknownType(), ir::Span::Unknown());
}

TEST_F(OperationTokenDepTest, ConsumerOpsByTokenBasic)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepBasic")
    {
        Tensor mid = Exp(input);
        output = Add(mid, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepBasic");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 2);

    auto tokenA = MakeToken("tokenA");

    Function* opsFunc = ops[0]->BelongTo();
    auto& dep = opsFunc->GetVarDependency();
    dep.AddConsumer(tokenA, std::static_pointer_cast<const ir::Stmt>(ops[1]));

    ops[0]->result_token_ = {tokenA};

    auto consumers = ops[0]->ConsumerOpsByToken();
    EXPECT_EQ(consumers.size(), 1);
    EXPECT_TRUE(consumers.count(ops[1].get()) > 0);
}

TEST_F(OperationTokenDepTest, ConsumerOpsByTokenNilToken)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepNil") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepNil");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 1);

    auto consumers = ops[0]->ConsumerOpsByToken();
    EXPECT_TRUE(consumers.empty());
}

TEST_F(OperationTokenDepTest, ProducerOpsByTokenBasic)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepProd")
    {
        Tensor mid = Exp(input);
        output = Add(mid, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepProd");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 2);

    auto tokenA = MakeToken("tokenA");

    Function* opsFunc = ops[1]->BelongTo();
    auto& dep = opsFunc->GetVarDependency();
    dep.AddProducer(tokenA, std::static_pointer_cast<const ir::Stmt>(ops[0]));

    ops[1]->tokens_.push_back(tokenA);

    auto producers = ops[1]->ProducerOpsByToken();
    EXPECT_EQ(producers.size(), 1);
    EXPECT_TRUE(producers.count(ops[0].get()) > 0);
}

TEST_F(OperationTokenDepTest, ProducerOpsByTokenEmptyTokens)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepEmptyTok") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepEmptyTok");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 1);

    auto producers = ops[0]->ProducerOpsByToken();
    EXPECT_TRUE(producers.empty());
}

TEST_F(OperationTokenDepTest, MultiTokenDependency)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepMulti")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepMulti");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 3);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 3);

    auto tokenAB = MakeToken("tokenAB");
    auto tokenBC = MakeToken("tokenBC");

    Function* opsFunc = ops[0]->BelongTo();
    auto& dep = opsFunc->GetVarDependency();
    dep.AddProducer(tokenAB, std::static_pointer_cast<const ir::Stmt>(ops[0]));
    dep.AddConsumer(tokenAB, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddProducer(tokenBC, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddConsumer(tokenBC, std::static_pointer_cast<const ir::Stmt>(ops[2]));

    ops[0]->result_token_ = {tokenAB};

    ops[1]->tokens_.push_back(tokenAB);
    ops[1]->result_token_ = {tokenBC};

    ops[2]->tokens_.push_back(tokenBC);

    auto consumers0 = ops[0]->ConsumerOpsByToken();
    EXPECT_EQ(consumers0.size(), 1);
    EXPECT_TRUE(consumers0.count(ops[1].get()) > 0);

    auto producers1 = ops[1]->ProducerOpsByToken();
    EXPECT_EQ(producers1.size(), 1);
    EXPECT_TRUE(producers1.count(ops[0].get()) > 0);

    auto consumers1 = ops[1]->ConsumerOpsByToken();
    EXPECT_EQ(consumers1.size(), 1);
    EXPECT_TRUE(consumers1.count(ops[2].get()) > 0);

    auto producers2 = ops[2]->ProducerOpsByToken();
    EXPECT_EQ(producers2.size(), 1);
    EXPECT_TRUE(producers2.count(ops[1].get()) > 0);
}

TEST_F(OperationTokenDepTest, ConsumerOpsByTokenUnknownToken)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepUnk") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepUnk");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 1);

    auto unknownToken = MakeToken("unknown");
    ops[0]->result_token_ = {unknownToken};

    auto consumers = ops[0]->ConsumerOpsByToken();
    EXPECT_TRUE(consumers.empty());
}

TEST_F(OperationTokenDepTest, ProducerOpsByTokenUnknownToken)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDepProdUnk") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDepProdUnk");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 1);

    auto unknownToken = MakeToken("unknown");
    ops[0]->tokens_.push_back(unknownToken);

    auto producers = ops[0]->ProducerOpsByToken();
    EXPECT_TRUE(producers.empty());
}

TEST_F(OperationTokenDepTest, RefreshVarDependency)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("RefreshTokenDep")
    {
        Tensor mid = Exp(input);
        output = Add(mid, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RefreshTokenDep");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 2);

    Function* opsFunc = ops[0]->BelongTo();
    auto token = MakeToken("token");
    auto staleToken = MakeToken("staleToken");
    ops[0]->result_token_ = {token};
    ops[1]->tokens_ = {token};

    auto& dep = opsFunc->GetVarDependency();
    dep.AddProducer(staleToken, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddConsumer(staleToken, std::static_pointer_cast<const ir::Stmt>(ops[0]));

    opsFunc->RefreshVarDependency();

    auto producerStmt = std::static_pointer_cast<const ir::Stmt>(ops[0]);
    auto consumerStmt = std::static_pointer_cast<const ir::Stmt>(ops[1]);
    EXPECT_TRUE(dep.HasProducer(token, producerStmt));
    EXPECT_TRUE(dep.HasConsumer(token, consumerStmt));
    EXPECT_FALSE(dep.HasDependency(staleToken));
    EXPECT_TRUE(ops[0]->ConsumerOpsByToken().count(ops[1].get()) > 0);
    EXPECT_TRUE(ops[1]->ProducerOpsByToken().count(ops[0].get()) > 0);
}
