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
 * \file test_codegen_kirin9030_unsqueeze.cpp
 * \brief
 */

#include "../include/test_codegen_unsqueeze.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Unsqueeze : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Unsqueeze() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Unsqueeze, test_unsqueeze_fp16_001)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_001();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_002)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_002();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_003)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_003();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_004)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_004();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_005)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_005();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_006)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_006();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_007)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_007();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_008)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_008();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp16_010)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp16_010();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_001)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_001();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_002)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_002();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_003)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_003();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_004)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_004();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_005)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_005();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_006)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_006();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_007)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_007();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_008)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_008();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_fp32_010)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_fp32_010();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int8_001)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int8_001();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int8_002)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int8_002();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int8_003)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int8_003();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int8_004)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int8_004();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int8_005)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int8_005();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int16_001)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int16_001();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int16_002)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int16_002();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int16_003)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int16_003();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int16_004)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int16_004();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int16_005)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int16_005();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int32_001)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int32_001();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int32_002)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int32_002();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int32_003)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int32_003();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int32_004)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int32_004();
}

TEST_F(TestCodeGenKirin9030Unsqueeze, DISABLED_test_unsqueeze_int32_005)
{
    TestCodeGenUnsqueeze::Instance().test_unsqueeze_int32_005();
}
