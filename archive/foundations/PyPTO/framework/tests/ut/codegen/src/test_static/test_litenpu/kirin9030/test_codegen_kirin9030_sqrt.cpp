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
 * \file test_codegen_kirin9030_sqrt.cpp
 * \brief
 */

#include "../include/test_codegen_sqrt.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Sqrt : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Sqrt() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Sqrt, test_Sqrt_fp16_001) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_001(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_002) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_002(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_003) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_003(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_004) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_004(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_005) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_005(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_006) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_006(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_007) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_007(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_008) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_008(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_009) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_009(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_010) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_010(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_011) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_011(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_012) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_012(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_013) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_013(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp32_014) { TestCodeGenSqrt::Instance().test_Sqrt_fp32_014(); }

TEST_F(TestCodeGenKirin9030Sqrt, DISABLED_test_Sqrt_fp16_015) { TestCodeGenSqrt::Instance().test_Sqrt_fp16_015(); }
