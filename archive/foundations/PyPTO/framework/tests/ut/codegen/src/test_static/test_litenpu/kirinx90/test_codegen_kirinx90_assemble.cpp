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
 * \file test_codegen_kirinx90_assemble.cpp
 * \brief
 */

#include "../include/test_codegen_assemble.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Assemble : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Assemble() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Assemble, test_assemble_fp16_001)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_002)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_003)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_003();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_004)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_004();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_005)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_005();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_006)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_006();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_007)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_007();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_008)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_008();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_009)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_009();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp16_010)
{
    TestCodeGenAssemble::Instance().test_assemble_fp16_010();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_001)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_002)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_003)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_003();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_004)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_004();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_005)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_005();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_006)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_006();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_007)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_007();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_008)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_008();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_009)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_009();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_fp32_010)
{
    TestCodeGenAssemble::Instance().test_assemble_fp32_010();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int8_001)
{
    TestCodeGenAssemble::Instance().test_assemble_int8_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int8_002)
{
    TestCodeGenAssemble::Instance().test_assemble_int8_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int8_003)
{
    TestCodeGenAssemble::Instance().test_assemble_int8_003();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int8_004)
{
    TestCodeGenAssemble::Instance().test_assemble_int8_004();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int8_005)
{
    TestCodeGenAssemble::Instance().test_assemble_int8_005();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int16_001)
{
    TestCodeGenAssemble::Instance().test_assemble_int16_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int16_002)
{
    TestCodeGenAssemble::Instance().test_assemble_int16_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int16_003)
{
    TestCodeGenAssemble::Instance().test_assemble_int16_003();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int16_004)
{
    TestCodeGenAssemble::Instance().test_assemble_int16_004();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int16_005)
{
    TestCodeGenAssemble::Instance().test_assemble_int16_005();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int32_001)
{
    TestCodeGenAssemble::Instance().test_assemble_int32_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int32_002)
{
    TestCodeGenAssemble::Instance().test_assemble_int32_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int32_003)
{
    TestCodeGenAssemble::Instance().test_assemble_int32_003();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int32_004)
{
    TestCodeGenAssemble::Instance().test_assemble_int32_004();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_int32_005)
{
    TestCodeGenAssemble::Instance().test_assemble_int32_005();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_list_fp32_001)
{
    TestCodeGenAssemble::Instance().test_assemble_list_fp32_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_list_fp32_002)
{
    TestCodeGenAssemble::Instance().test_assemble_list_fp32_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_list_fp32_003)
{
    TestCodeGenAssemble::Instance().test_assemble_list_fp32_003();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_list_multi_shape_001)
{
    TestCodeGenAssemble::Instance().test_assemble_list_multi_shape_001();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_list_multi_shape_002)
{
    TestCodeGenAssemble::Instance().test_assemble_list_multi_shape_002();
}

TEST_F(TestCodeGenKirinX90Assemble, DISABLED_test_assemble_list_multi_shape_003)
{
    TestCodeGenAssemble::Instance().test_assemble_list_multi_shape_003();
}
