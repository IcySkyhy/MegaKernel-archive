/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_assemble_new_logical_tensor.cpp
 * \brief Unit test for the AssembleNewLogicalTensor branch in Assemble().
 */

#include "gtest/gtest.h"
#include "interface/tensor/irbuilder.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

class TestAssembleNewLogicalTensor : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        savedFlag_ = IRContext::Get().AssembleNewLogicalTensor();
    }

    void TearDown() override { IRContext::Get().SetAssembleNewLogicalTensor(savedFlag_); }

private:
    bool savedFlag_{false};
};

TEST_F(TestAssembleNewLogicalTensor, NewLogicalTensorCreatesVersionedDest)
{
    IRContext::Get().SetAssembleNewLogicalTensor(true);
    TileShape::Current().SetVecTile(16, 16);

    Tensor src(DT_FP32, {16, 16}, "src");
    Tensor dst(DT_FP32, {32, 16}, "dst");
    auto origMagic = dst.GetStorage(false)->magic;

    FUNCTION("TestAssembleNewLT") { Assemble(src, {SymbolicScalar(0), SymbolicScalar(0)}, dst); }

    EXPECT_NE(dst.GetStorage(false)->magic, origMagic) << "Assemble should create a new LogicalTensor version";
}

TEST_F(TestAssembleNewLogicalTensor, LegacyWritesOriginalDest)
{
    IRContext::Get().SetAssembleNewLogicalTensor(false);
    TileShape::Current().SetVecTile(16, 16);

    Tensor src(DT_FP32, {16, 16}, "src");
    Tensor dst(DT_FP32, {32, 16}, "dst");
    auto origMagic = dst.GetStorage(false)->magic;

    FUNCTION("TestAssembleLegacy") { Assemble(src, {SymbolicScalar(0), SymbolicScalar(0)}, dst); }

    EXPECT_EQ(dst.GetStorage(false)->magic, origMagic) << "Legacy Assemble should not create a new version";
}
