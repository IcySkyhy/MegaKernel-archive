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
 * \file test_codegen_kirin9030_reciprocal.cpp
 * \brief
 */

#include "../include/test_codegen_reciprocal.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Reciprocal : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Reciprocal() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Reciprocal, test_Reciprocal_fp16_001)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_001();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_002)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_002();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_003)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_003();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_004)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_004();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_005)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_005();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_006)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_006();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_007)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_007();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_008)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_008();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_009)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_009();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_010)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_010();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_011)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_011();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_012)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_012();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_013)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_013();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp32_014)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp32_014();
}

TEST_F(TestCodeGenKirin9030Reciprocal, DISABLED_test_Reciprocal_fp16_015)
{
    TestCodeGenReciprocal::Instance().test_Reciprocal_fp16_015();
}
