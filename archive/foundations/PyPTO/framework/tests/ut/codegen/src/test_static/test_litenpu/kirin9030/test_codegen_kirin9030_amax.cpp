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
 * \file test_codegen_kirin9030_amax.cpp
 * \brief
 */

#include "../include/test_codegen_amax.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Amax : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Amax() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Amax, test_amax_fp16_003) { TestCodeGenAmax::Instance().test_amax_fp16_003(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_004) { TestCodeGenAmax::Instance().test_amax_fp16_004(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_005) { TestCodeGenAmax::Instance().test_amax_fp16_005(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_006) { TestCodeGenAmax::Instance().test_amax_fp16_006(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_007) { TestCodeGenAmax::Instance().test_amax_fp16_007(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_008) { TestCodeGenAmax::Instance().test_amax_fp16_008(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_009) { TestCodeGenAmax::Instance().test_amax_fp16_009(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_010) { TestCodeGenAmax::Instance().test_amax_fp16_010(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_011) { TestCodeGenAmax::Instance().test_amax_fp16_011(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_012) { TestCodeGenAmax::Instance().test_amax_fp16_012(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_013) { TestCodeGenAmax::Instance().test_amax_fp16_013(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_014) { TestCodeGenAmax::Instance().test_amax_fp16_014(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp16_015) { TestCodeGenAmax::Instance().test_amax_fp16_015(); }

TEST_F(TestCodeGenKirin9030Amax, test_amax_fp32_003) { TestCodeGenAmax::Instance().test_amax_fp32_003(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_004) { TestCodeGenAmax::Instance().test_amax_fp32_004(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_005) { TestCodeGenAmax::Instance().test_amax_fp32_005(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_006) { TestCodeGenAmax::Instance().test_amax_fp32_006(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_007) { TestCodeGenAmax::Instance().test_amax_fp32_007(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_008) { TestCodeGenAmax::Instance().test_amax_fp32_008(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_009) { TestCodeGenAmax::Instance().test_amax_fp32_009(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_010) { TestCodeGenAmax::Instance().test_amax_fp32_010(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_011) { TestCodeGenAmax::Instance().test_amax_fp32_011(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_012) { TestCodeGenAmax::Instance().test_amax_fp32_012(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_013) { TestCodeGenAmax::Instance().test_amax_fp32_013(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_014) { TestCodeGenAmax::Instance().test_amax_fp32_014(); }

TEST_F(TestCodeGenKirin9030Amax, DISABLED_test_amax_fp32_015) { TestCodeGenAmax::Instance().test_amax_fp32_015(); }
