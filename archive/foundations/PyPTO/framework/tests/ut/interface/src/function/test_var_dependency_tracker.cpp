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
 * \file test_var_dependency_tracker.cpp
 * \brief Unit tests for VarDependency
 */

#include "gtest/gtest.h"

#include "ir/span.h"
#include "ir/type.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"

using namespace npu::tile_fwk;

static ir::VarPtr MakeVar(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetUnknownType(), ir::Span::Unknown());
}

static ir::StmtPtr MakeStmt() { return std::make_shared<ir::TensorOpStmt>(ir::Span::Unknown()); }

class VarDependencyTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "VarDependencyTest SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "VarDependencyTest TearDownTestCase" << std::endl; }

    void SetUp() override
    {
        dep.Clear();
        varA = MakeVar("varA");
        varB = MakeVar("varB");
        varC = MakeVar("varC");
        producer1 = MakeStmt();
        producer2 = MakeStmt();
        consumer1 = MakeStmt();
        consumer2 = MakeStmt();
    }

    void TearDown() override { dep.Clear(); }

    VarDependency dep;
    ir::VarPtr varA;
    ir::VarPtr varB;
    ir::VarPtr varC;
    ir::StmtPtr producer1;
    ir::StmtPtr producer2;
    ir::StmtPtr consumer1;
    ir::StmtPtr consumer2;
};

// ========== Add Tests ==========

TEST_F(VarDependencyTest, AddProducerSingle)
{
    dep.AddProducer(varA, producer1);
    EXPECT_TRUE(dep.HasDependency(varA));
    EXPECT_TRUE(dep.HasProducer(varA, producer1));
    EXPECT_EQ(dep.GetProducers(varA).size(), 1);
}

TEST_F(VarDependencyTest, AddProducerMultiple)
{
    dep.AddProducer(varA, producer1);
    dep.AddProducer(varA, producer2);
    EXPECT_EQ(dep.GetProducers(varA).size(), 2);
    EXPECT_TRUE(dep.HasProducer(varA, producer1));
    EXPECT_TRUE(dep.HasProducer(varA, producer2));
}

TEST_F(VarDependencyTest, AddProducerDuplicate)
{
    dep.AddProducer(varA, producer1);
    dep.AddProducer(varA, producer1);
    EXPECT_EQ(dep.GetProducers(varA).size(), 1);
}

TEST_F(VarDependencyTest, AddConsumerSingle)
{
    dep.AddConsumer(varA, consumer1);
    EXPECT_TRUE(dep.HasDependency(varA));
    EXPECT_TRUE(dep.HasConsumer(varA, consumer1));
    EXPECT_EQ(dep.GetConsumers(varA).size(), 1);
}

TEST_F(VarDependencyTest, AddConsumerMultiple)
{
    dep.AddConsumer(varA, consumer1);
    dep.AddConsumer(varA, consumer2);
    EXPECT_EQ(dep.GetConsumers(varA).size(), 2);
    EXPECT_TRUE(dep.HasConsumer(varA, consumer1));
    EXPECT_TRUE(dep.HasConsumer(varA, consumer2));
}

TEST_F(VarDependencyTest, AddConsumerDuplicate)
{
    dep.AddConsumer(varA, consumer1);
    dep.AddConsumer(varA, consumer1);
    EXPECT_EQ(dep.GetConsumers(varA).size(), 1);
}

// ========== Remove Tests ==========

TEST_F(VarDependencyTest, RemoveProducer)
{
    dep.AddProducer(varA, producer1);
    dep.AddProducer(varA, producer2);
    dep.RemoveProducer(varA, producer1);
    EXPECT_FALSE(dep.HasProducer(varA, producer1));
    EXPECT_TRUE(dep.HasProducer(varA, producer2));
    EXPECT_EQ(dep.GetProducers(varA).size(), 1);
}

TEST_F(VarDependencyTest, RemoveProducerNonExisting)
{
    dep.AddProducer(varA, producer1);
    dep.RemoveProducer(varA, producer2);
    EXPECT_EQ(dep.GetProducers(varA).size(), 1);
    EXPECT_TRUE(dep.HasProducer(varA, producer1));
}

TEST_F(VarDependencyTest, RemoveConsumer)
{
    dep.AddConsumer(varA, consumer1);
    dep.AddConsumer(varA, consumer2);
    dep.RemoveConsumer(varA, consumer1);
    EXPECT_FALSE(dep.HasConsumer(varA, consumer1));
    EXPECT_TRUE(dep.HasConsumer(varA, consumer2));
    EXPECT_EQ(dep.GetConsumers(varA).size(), 1);
}

TEST_F(VarDependencyTest, RemoveVar)
{
    dep.AddProducer(varA, producer1);
    dep.AddProducer(varB, producer2);
    dep.RemoveVar(varA);
    EXPECT_FALSE(dep.HasDependency(varA));
    EXPECT_TRUE(dep.HasDependency(varB));
    EXPECT_EQ(dep.Size(), 1);
}

TEST_F(VarDependencyTest, RemoveProducerNonExistingVar)
{
    dep.RemoveProducer(varA, producer1);
    EXPECT_FALSE(dep.HasDependency(varA));
}

TEST_F(VarDependencyTest, RemoveConsumerNonExistingVar)
{
    dep.RemoveConsumer(varA, consumer1);
    EXPECT_FALSE(dep.HasDependency(varA));
}

TEST_F(VarDependencyTest, RemoveVarNonExisting)
{
    dep.RemoveVar(varA);
    EXPECT_EQ(dep.Size(), 0);
}

// ========== Clear Tests ==========

TEST_F(VarDependencyTest, Clear)
{
    dep.AddProducer(varA, producer1);
    dep.AddProducer(varB, producer2);
    dep.AddConsumer(varC, consumer1);
    dep.Clear();
    EXPECT_EQ(dep.Size(), 0);
    EXPECT_TRUE(dep.Empty());
}

// ========== Query Tests ==========

TEST_F(VarDependencyTest, GetProducers)
{
    dep.AddProducer(varA, producer1);
    dep.AddProducer(varA, producer2);
    const auto& producers = dep.GetProducers(varA);
    EXPECT_EQ(producers.size(), 2);
    EXPECT_TRUE(producers.count(producer1) > 0);
    EXPECT_TRUE(producers.count(producer2) > 0);
}

TEST_F(VarDependencyTest, GetProducersNonExistingVar)
{
    const auto& producers = dep.GetProducers(varA);
    EXPECT_TRUE(producers.empty());
}

TEST_F(VarDependencyTest, GetConsumers)
{
    dep.AddConsumer(varA, consumer1);
    dep.AddConsumer(varA, consumer2);
    const auto& consumers = dep.GetConsumers(varA);
    EXPECT_EQ(consumers.size(), 2);
    EXPECT_TRUE(consumers.count(consumer1) > 0);
    EXPECT_TRUE(consumers.count(consumer2) > 0);
}

TEST_F(VarDependencyTest, GetConsumersNonExistingVar)
{
    const auto& consumers = dep.GetConsumers(varA);
    EXPECT_TRUE(consumers.empty());
}

TEST_F(VarDependencyTest, HasDependency)
{
    EXPECT_FALSE(dep.HasDependency(varA));
    dep.AddProducer(varA, producer1);
    EXPECT_TRUE(dep.HasDependency(varA));
}

TEST_F(VarDependencyTest, HasProducerFalse)
{
    dep.AddProducer(varA, producer1);
    EXPECT_FALSE(dep.HasProducer(varA, producer2));
}

TEST_F(VarDependencyTest, HasConsumerFalse)
{
    dep.AddConsumer(varA, consumer1);
    EXPECT_FALSE(dep.HasConsumer(varA, consumer2));
}

TEST_F(VarDependencyTest, HasProducerNonExistingVar) { EXPECT_FALSE(dep.HasProducer(varA, producer1)); }

TEST_F(VarDependencyTest, HasConsumerNonExistingVar) { EXPECT_FALSE(dep.HasConsumer(varA, consumer1)); }

// ========== Size/Empty Tests ==========

TEST_F(VarDependencyTest, Size)
{
    EXPECT_EQ(dep.Size(), 0);
    dep.AddProducer(varA, producer1);
    EXPECT_EQ(dep.Size(), 1);
    dep.AddConsumer(varA, consumer1);
    EXPECT_EQ(dep.Size(), 1);
    dep.AddProducer(varB, producer2);
    EXPECT_EQ(dep.Size(), 2);
}

TEST_F(VarDependencyTest, Empty)
{
    EXPECT_TRUE(dep.Empty());
    dep.AddProducer(varA, producer1);
    EXPECT_FALSE(dep.Empty());
    dep.RemoveVar(varA);
    EXPECT_TRUE(dep.Empty());
}

// ========== GetAllDependencies Tests ==========

TEST_F(VarDependencyTest, GetAllDependencies)
{
    dep.AddProducer(varA, producer1);
    dep.AddConsumer(varA, consumer1);
    dep.AddProducer(varB, producer2);
    const auto& deps = dep.GetAllDependencies();
    EXPECT_EQ(deps.size(), 2);
    EXPECT_TRUE(deps.find(varA) != deps.end());
    EXPECT_TRUE(deps.find(varB) != deps.end());
}

// ========== Multi-Var Tests ==========

TEST_F(VarDependencyTest, MultipleVarsIndependent)
{
    dep.AddProducer(varA, producer1);
    dep.AddConsumer(varA, consumer1);
    dep.AddProducer(varB, producer2);
    dep.AddConsumer(varB, consumer2);
    EXPECT_EQ(dep.Size(), 2);
    EXPECT_TRUE(dep.HasDependency(varA));
    EXPECT_TRUE(dep.HasDependency(varB));
    EXPECT_FALSE(dep.HasProducer(varA, producer2));
    EXPECT_FALSE(dep.HasConsumer(varB, consumer1));
}

TEST_F(VarDependencyTest, MultipleProducersSameConsumer)
{
    dep.AddConsumer(varA, consumer1);
    dep.AddConsumer(varB, consumer1);
    EXPECT_TRUE(dep.HasConsumer(varA, consumer1));
    EXPECT_TRUE(dep.HasConsumer(varB, consumer1));
}

TEST_F(VarDependencyTest, ChainDependency)
{
    auto midVar = MakeVar("midVar");
    auto prod = MakeStmt();
    auto cons = MakeStmt();
    dep.AddProducer(midVar, prod);
    dep.AddConsumer(midVar, cons);
    EXPECT_TRUE(dep.HasProducer(midVar, prod));
    EXPECT_TRUE(dep.HasConsumer(midVar, cons));
}

// ========== Function Integration Tests ==========

TEST_F(VarDependencyTest, FunctionGetVarDependency)
{
    auto& funcDep = Program::GetInstance().GetCurrentFunction()->GetVarDependency();
    auto var = MakeVar("funcVar");
    auto prod = MakeStmt();
    funcDep.AddProducer(var, prod);
    EXPECT_TRUE(funcDep.HasDependency(var));
    EXPECT_TRUE(funcDep.HasProducer(var, prod));
    funcDep.Clear();
}
