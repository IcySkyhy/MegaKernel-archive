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
 * \file test_codegen_kirinx90_sigmoid.cpp
 * \brief
 */

#include "../include/test_codegen_sigmoid.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Sigmoid : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Sigmoid() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Sigmoid, test_sigmoid_fp16_001) { TestCodeGenSigmoid::Instance().test_sigmoid_fp16_001(); }

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_002) { TestCodeGenSigmoid::Instance().test_sigmoid_002(); }

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_003)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_003();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_004)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_004();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_005)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_005();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_006)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_006();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_007)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_007();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_008)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_008();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_009)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_009();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_010)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_010();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_011)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_011();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_012)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_012();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_013)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_013();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_014)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_014();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp16_015)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp16_015();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_001)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_001();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_002)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_002();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_003)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_003();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_004)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_004();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_005)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_005();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_006)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_006();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_007)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_007();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_008)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_008();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_009)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_009();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_010)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_010();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_011)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_011();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_012)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_012();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_013)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_013();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_014)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_014();
}

TEST_F(TestCodeGenKirinX90Sigmoid, DISABLED_test_sigmoid_fp32_015)
{
    TestCodeGenSigmoid::Instance().test_sigmoid_fp32_015();
}
