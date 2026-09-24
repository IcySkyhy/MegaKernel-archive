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
 * \file test_id_gen.cpp
 * \brief Unit tests for IdGen template (NewId, CurId, Reset, SetId)
 */

#include "gtest/gtest.h"
#include "interface/utils/id_gen.h"

using namespace npu::tile_fwk;

class TestIdGen : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestIdGen, NewIdMonotonicallyIncreasing)
{
    auto& gen = IdGen<IdType::RAW_TENSOR>::Inst();
    gen.Reset();
    int id1 = gen.NewId();
    int id2 = gen.NewId();
    int id3 = gen.NewId();
    EXPECT_EQ(id1, 0);
    EXPECT_EQ(id2, 1);
    EXPECT_EQ(id3, 2);
}

TEST_F(TestIdGen, CurIdReturnsCurrent)
{
    auto& gen = IdGen<IdType::FUNCTION>::Inst();
    gen.Reset();
    gen.NewId();
    gen.NewId();
    EXPECT_EQ(gen.CurId(), 2);
}

TEST_F(TestIdGen, ResetToZero)
{
    auto& gen = IdGen<IdType::TENSOR_INDEX>::Inst();
    gen.NewId();
    gen.NewId();
    gen.Reset();
    EXPECT_EQ(gen.CurId(), 0);
}

TEST_F(TestIdGen, SetIdSpecificValue)
{
    auto& gen = IdGen<IdType::LOGICAL_TENSOR>::Inst();
    gen.SetId(100);
    EXPECT_EQ(gen.CurId(), 100);
    int next = gen.NewId();
    EXPECT_EQ(next, 100);
    EXPECT_EQ(gen.CurId(), 101);
    gen.Reset();
}

TEST_F(TestIdGen, MultipleIdTypesIndependent)
{
    auto& raw = IdGen<IdType::RAW_TENSOR>::Inst();
    auto& func = IdGen<IdType::FUNCTION>::Inst();
    raw.Reset();
    func.Reset();
    raw.NewId();
    func.NewId();
    func.NewId();
    EXPECT_EQ(raw.CurId(), 1);
    EXPECT_EQ(func.CurId(), 2);
    raw.Reset();
    func.Reset();
}

TEST_F(TestIdGen, NewIdAfterReset)
{
    auto& gen = IdGen<IdType::RAW_TENSOR>::Inst();
    gen.Reset();
    int id = gen.NewId();
    EXPECT_EQ(id, 0);
    gen.Reset();
    id = gen.NewId();
    EXPECT_EQ(id, 0);
}

TEST_F(TestIdGen, CurIdInitiallyZero)
{
    auto& gen = IdGen<IdType::RAW_TENSOR>::Inst();
    gen.Reset();
    EXPECT_EQ(gen.CurId(), 0);
}
