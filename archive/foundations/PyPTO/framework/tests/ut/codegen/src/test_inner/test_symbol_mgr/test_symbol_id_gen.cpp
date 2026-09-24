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
 * \file test_symbol_id_gen.cpp
 * \brief Unit test for symbol_id_gen.h
 */

#include "gtest/gtest.h"

#include "codegen/symbol_mgr/symbol_id_gen.h"

namespace npu::tile_fwk {

class TestSymbolIdGen : public ::testing::Test {
protected:
    SymbolIdGen gen_;
};

TEST_F(TestSymbolIdGen, TestNewId)
{
    EXPECT_EQ(gen_.NewId(), 0);
    EXPECT_EQ(gen_.NewId(), 1);
    EXPECT_EQ(gen_.NewId(), 2);
}

TEST_F(TestSymbolIdGen, TestCurId)
{
    gen_.NewId();
    gen_.NewId();
    EXPECT_EQ(gen_.CurId(), 2);
}

TEST_F(TestSymbolIdGen, TestReset)
{
    gen_.NewId();
    gen_.NewId();
    gen_.Reset();
    EXPECT_EQ(gen_.CurId(), 0);
}

TEST_F(TestSymbolIdGen, TestSetId)
{
    gen_.SetId(100);
    EXPECT_EQ(gen_.CurId(), 100);
    EXPECT_EQ(gen_.NewId(), 100);
    EXPECT_EQ(gen_.NewId(), 101);
}

class TestSymbolIdGenMgr : public ::testing::Test {
protected:
    SymbolIdGenMgr mgr_;
};

TEST_F(TestSymbolIdGenMgr, TestNewIdUsingName)
{
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_USING_NAME>(), 0);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_USING_NAME>(), 1);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_USING_NAME>(), 2);
}

TEST_F(TestSymbolIdGenMgr, TestNewIdVarName)
{
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_VAR_NAME>(), 0);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_VAR_NAME>(), 1);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_VAR_NAME>(), 2);
}

TEST_F(TestSymbolIdGenMgr, TestDifferentTypesIndependent)
{
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_USING_NAME>(), 0);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_VAR_NAME>(), 0);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_USING_NAME>(), 1);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_VAR_NAME>(), 1);
}

TEST_F(TestSymbolIdGenMgr, TestCurId)
{
    mgr_.NewId<SymbolIdType::CG_USING_NAME>();
    mgr_.NewId<SymbolIdType::CG_USING_NAME>();
    EXPECT_EQ(mgr_.CurId<SymbolIdType::CG_USING_NAME>(), 2);
    EXPECT_EQ(mgr_.CurId<SymbolIdType::CG_VAR_NAME>(), 0);
}

TEST_F(TestSymbolIdGenMgr, TestReset)
{
    mgr_.NewId<SymbolIdType::CG_USING_NAME>();
    mgr_.NewId<SymbolIdType::CG_VAR_NAME>();
    mgr_.Reset<SymbolIdType::CG_USING_NAME>();
    EXPECT_EQ(mgr_.CurId<SymbolIdType::CG_USING_NAME>(), 0);
    EXPECT_EQ(mgr_.CurId<SymbolIdType::CG_VAR_NAME>(), 1);
}

TEST_F(TestSymbolIdGenMgr, TestSetId)
{
    mgr_.SetId<SymbolIdType::CG_USING_NAME>(50);
    mgr_.SetId<SymbolIdType::CG_VAR_NAME>(100);
    EXPECT_EQ(mgr_.CurId<SymbolIdType::CG_USING_NAME>(), 50);
    EXPECT_EQ(mgr_.CurId<SymbolIdType::CG_VAR_NAME>(), 100);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_USING_NAME>(), 50);
    EXPECT_EQ(mgr_.NewId<SymbolIdType::CG_VAR_NAME>(), 100);
}

} // namespace npu::tile_fwk
