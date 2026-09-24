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
 * \file test_codegen_kirinx90_full.cpp
 * \brief
 */

#include "../include/test_codegen_full.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Full : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Full() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Full, test_full_001) { TestCodeGenFull::Instance().test_full_001(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_002) { TestCodeGenFull::Instance().test_full_002(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_003) { TestCodeGenFull::Instance().test_full_003(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_004) { TestCodeGenFull::Instance().test_full_004(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_005) { TestCodeGenFull::Instance().test_full_005(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_006) { TestCodeGenFull::Instance().test_full_006(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_007) { TestCodeGenFull::Instance().test_full_007(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_008) { TestCodeGenFull::Instance().test_full_008(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_009) { TestCodeGenFull::Instance().test_full_009(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_010) { TestCodeGenFull::Instance().test_full_010(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_011) { TestCodeGenFull::Instance().test_full_011(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_012) { TestCodeGenFull::Instance().test_full_012(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_013) { TestCodeGenFull::Instance().test_full_013(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_014) { TestCodeGenFull::Instance().test_full_014(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_015) { TestCodeGenFull::Instance().test_full_015(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_016) { TestCodeGenFull::Instance().test_full_016(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_017) { TestCodeGenFull::Instance().test_full_017(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_018) { TestCodeGenFull::Instance().test_full_018(); }

TEST_F(TestCodeGenKirinX90Full, DISABLED_test_full_019) { TestCodeGenFull::Instance().test_full_019(); }
