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
 * \file test_codegen_kirinx90_rsqrt.cpp
 * \brief
 */

#include "../include/test_codegen_rsqrt.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Rsqrt : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Rsqrt() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Rsqrt, test_Rsqrt_fp16_001) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_001(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_002) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_002(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_003) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_003(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_004) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_004(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_005) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_005(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_006) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_006(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_007) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_007(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_008) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_008(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_009) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_009(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_010) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_010(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_011) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_011(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_012) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_012(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_013) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_013(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp32_014) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp32_014(); }

TEST_F(TestCodeGenKirinX90Rsqrt, DISABLED_test_Rsqrt_fp16_015) { TestCodeGenRsqrt::Instance().test_Rsqrt_fp16_015(); }
