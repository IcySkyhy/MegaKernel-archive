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
 * \file test_codegen_kirin9030_amin.cpp
 * \brief
 */

#include "../include/test_codegen_amin.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Amin : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Amin() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Amin, test_amin_fp16_003) { TestCodeGenAmin::Instance().test_amin_fp16_003(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_004) { TestCodeGenAmin::Instance().test_amin_fp16_004(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_005) { TestCodeGenAmin::Instance().test_amin_fp16_005(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_006) { TestCodeGenAmin::Instance().test_amin_fp16_006(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_007) { TestCodeGenAmin::Instance().test_amin_fp16_007(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_008) { TestCodeGenAmin::Instance().test_amin_fp16_008(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_009) { TestCodeGenAmin::Instance().test_amin_fp16_009(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_010) { TestCodeGenAmin::Instance().test_amin_fp16_010(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_011) { TestCodeGenAmin::Instance().test_amin_fp16_011(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_012) { TestCodeGenAmin::Instance().test_amin_fp16_012(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_013) { TestCodeGenAmin::Instance().test_amin_fp16_013(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_014) { TestCodeGenAmin::Instance().test_amin_fp16_014(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp16_015) { TestCodeGenAmin::Instance().test_amin_fp16_015(); }

TEST_F(TestCodeGenKirin9030Amin, test_amin_fp32_003) { TestCodeGenAmin::Instance().test_amin_fp32_003(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_004) { TestCodeGenAmin::Instance().test_amin_fp32_004(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_005) { TestCodeGenAmin::Instance().test_amin_fp32_005(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_006) { TestCodeGenAmin::Instance().test_amin_fp32_006(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_007) { TestCodeGenAmin::Instance().test_amin_fp32_007(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_008) { TestCodeGenAmin::Instance().test_amin_fp32_008(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_009) { TestCodeGenAmin::Instance().test_amin_fp32_009(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_010) { TestCodeGenAmin::Instance().test_amin_fp32_010(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_011) { TestCodeGenAmin::Instance().test_amin_fp32_011(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_012) { TestCodeGenAmin::Instance().test_amin_fp32_012(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_013) { TestCodeGenAmin::Instance().test_amin_fp32_013(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_014) { TestCodeGenAmin::Instance().test_amin_fp32_014(); }

TEST_F(TestCodeGenKirin9030Amin, DISABLED_test_amin_fp32_015) { TestCodeGenAmin::Instance().test_amin_fp32_015(); }
