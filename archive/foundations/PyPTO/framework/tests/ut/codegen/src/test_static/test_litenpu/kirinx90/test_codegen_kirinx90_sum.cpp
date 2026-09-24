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
 * \file test_codegen_kirinx90_sum.cpp
 * \brief
 */

#include "../include/test_codegen_sum.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Sum : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Sum() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Sum, test_sum_fp32_003) { TestCodeGenSum::Instance().test_sum_fp32_003(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_004) { TestCodeGenSum::Instance().test_sum_fp32_004(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_005) { TestCodeGenSum::Instance().test_sum_fp32_005(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_006) { TestCodeGenSum::Instance().test_sum_fp32_006(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_007) { TestCodeGenSum::Instance().test_sum_fp32_007(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_008) { TestCodeGenSum::Instance().test_sum_fp32_008(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_009) { TestCodeGenSum::Instance().test_sum_fp32_009(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_010) { TestCodeGenSum::Instance().test_sum_fp32_010(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_011) { TestCodeGenSum::Instance().test_sum_fp32_011(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_012) { TestCodeGenSum::Instance().test_sum_fp32_012(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_013) { TestCodeGenSum::Instance().test_sum_fp32_013(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_014) { TestCodeGenSum::Instance().test_sum_fp32_014(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_fp32_015) { TestCodeGenSum::Instance().test_sum_fp32_015(); }

TEST_F(TestCodeGenKirinX90Sum, test_sum_int32_003) { TestCodeGenSum::Instance().test_sum_int32_003(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_004) { TestCodeGenSum::Instance().test_sum_int32_004(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_005) { TestCodeGenSum::Instance().test_sum_int32_005(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_006) { TestCodeGenSum::Instance().test_sum_int32_006(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_007) { TestCodeGenSum::Instance().test_sum_int32_007(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_008) { TestCodeGenSum::Instance().test_sum_int32_008(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_009) { TestCodeGenSum::Instance().test_sum_int32_009(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_010) { TestCodeGenSum::Instance().test_sum_int32_010(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_011) { TestCodeGenSum::Instance().test_sum_int32_011(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_012) { TestCodeGenSum::Instance().test_sum_int32_012(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_013) { TestCodeGenSum::Instance().test_sum_int32_013(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_014) { TestCodeGenSum::Instance().test_sum_int32_014(); }

TEST_F(TestCodeGenKirinX90Sum, DISABLED_test_sum_int32_015) { TestCodeGenSum::Instance().test_sum_int32_015(); }
