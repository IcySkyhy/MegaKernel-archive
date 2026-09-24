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
 * \file test_codegen_kirin9030_transpose.cpp
 * \brief
 */

#include "../include/test_codegen_transpose.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Transpose : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Transpose() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Transpose, test_transpose_fp16_001)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_001();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_002)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_002();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_003)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_003();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_004)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_004();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_005)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_005();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_006)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_006();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_007)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_007();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_008)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_008();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_009)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_009();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp16_010)
{
    TestCodeGenTranspose::Instance().test_transpose_fp16_010();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_001)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_001();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_002)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_002();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_003)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_003();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_004)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_004();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_005)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_005();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_006)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_006();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_007)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_007();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_008)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_008();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_009)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_009();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_fp32_010)
{
    TestCodeGenTranspose::Instance().test_transpose_fp32_010();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int32_001)
{
    TestCodeGenTranspose::Instance().test_transpose_int32_001();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int32_002)
{
    TestCodeGenTranspose::Instance().test_transpose_int32_002();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int32_003)
{
    TestCodeGenTranspose::Instance().test_transpose_int32_003();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int32_004)
{
    TestCodeGenTranspose::Instance().test_transpose_int32_004();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int32_005)
{
    TestCodeGenTranspose::Instance().test_transpose_int32_005();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_001)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_001();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_002)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_002();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_003)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_003();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_004)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_004();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_005)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_005();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_006)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_006();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_007)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_007();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_008)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_008();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_009)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_009();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_010)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_010();
}

TEST_F(TestCodeGenKirin9030Transpose, DISABLED_test_transpose_int16_011)
{
    TestCodeGenTranspose::Instance().test_transpose_int16_011();
}
