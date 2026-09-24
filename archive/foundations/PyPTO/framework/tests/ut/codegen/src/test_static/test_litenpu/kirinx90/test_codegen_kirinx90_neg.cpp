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
 * \file test_codegen_kirinx90_neg.cpp
 * \brief
 */

#include "../include/test_codegen_neg.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Neg : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Neg() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Neg, test_Neg_fp16_001) { TestCodeGenNeg::Instance().test_Neg_fp16_001(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_002) { TestCodeGenNeg::Instance().test_Neg_fp16_002(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_003) { TestCodeGenNeg::Instance().test_Neg_fp32_003(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_004) { TestCodeGenNeg::Instance().test_Neg_fp32_004(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_005) { TestCodeGenNeg::Instance().test_Neg_fp16_005(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_006) { TestCodeGenNeg::Instance().test_Neg_fp32_006(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_007) { TestCodeGenNeg::Instance().test_Neg_fp16_007(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_008) { TestCodeGenNeg::Instance().test_Neg_fp32_008(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_009) { TestCodeGenNeg::Instance().test_Neg_fp16_009(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_010) { TestCodeGenNeg::Instance().test_Neg_fp32_010(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_011) { TestCodeGenNeg::Instance().test_Neg_fp16_011(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_012) { TestCodeGenNeg::Instance().test_Neg_fp32_012(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_013) { TestCodeGenNeg::Instance().test_Neg_fp16_013(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp32_014) { TestCodeGenNeg::Instance().test_Neg_fp32_014(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_fp16_015) { TestCodeGenNeg::Instance().test_Neg_fp16_015(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_int16_016) { TestCodeGenNeg::Instance().test_Neg_int16_016(); }

TEST_F(TestCodeGenKirinX90Neg, DISABLED_test_Neg_int32_017) { TestCodeGenNeg::Instance().test_Neg_int32_017(); }
