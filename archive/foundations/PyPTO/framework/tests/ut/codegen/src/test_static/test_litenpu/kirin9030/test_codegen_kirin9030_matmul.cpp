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
 * \file test_codegen_kirin9030_matmul.cpp
 * \brief
 */

#include "../include/test_codegen_matmul.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Matmul : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Matmul() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Matmul, test_matmul_001) { TestCodeGenMatmul::Instance().test_matmul_001(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_002) { TestCodeGenMatmul::Instance().test_matmul_002(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_003) { TestCodeGenMatmul::Instance().test_matmul_003(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_004) { TestCodeGenMatmul::Instance().test_matmul_004(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_005) { TestCodeGenMatmul::Instance().test_matmul_005(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_006) { TestCodeGenMatmul::Instance().test_matmul_006(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_007) { TestCodeGenMatmul::Instance().test_matmul_007(); }

TEST_F(TestCodeGenKirin9030Matmul, test_matmul_008) { TestCodeGenMatmul::Instance().test_matmul_008(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_009) { TestCodeGenMatmul::Instance().test_matmul_009(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_010) { TestCodeGenMatmul::Instance().test_matmul_010(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_s8s8_001)
{
    TestCodeGenMatmul::Instance().test_matmul_s8s8_001();
}

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_s8s8_002)
{
    TestCodeGenMatmul::Instance().test_matmul_s8s8_002();
}

TEST_F(TestCodeGenKirin9030Matmul, test_matmul_s8s8_003) { TestCodeGenMatmul::Instance().test_matmul_s8s8_003(); }

TEST_F(TestCodeGenKirin9030Matmul, DISABLED_test_matmul_s8s8_004)
{
    TestCodeGenMatmul::Instance().test_matmul_s8s8_004();
}

TEST_F(TestCodeGenKirin9030Matmul, test_matmul_gemv_001) { TestCodeGenMatmul::Instance().test_matmul_gemv_001(); }
