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
 * \file test_codegen_kirinx90_cast.cpp
 * \brief
 */

#include "../include/test_codegen_cast.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90Cast : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90Cast() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90Cast, test_Cast_001) { TestCodeGenCast::Instance().test_Cast_001(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_002) { TestCodeGenCast::Instance().test_Cast_002(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_003) { TestCodeGenCast::Instance().test_Cast_003(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_004) { TestCodeGenCast::Instance().test_Cast_004(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_005) { TestCodeGenCast::Instance().test_Cast_005(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_006) { TestCodeGenCast::Instance().test_Cast_006(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_007) { TestCodeGenCast::Instance().test_Cast_007(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_008) { TestCodeGenCast::Instance().test_Cast_008(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_009) { TestCodeGenCast::Instance().test_Cast_009(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_010) { TestCodeGenCast::Instance().test_Cast_010(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_011) { TestCodeGenCast::Instance().test_Cast_011(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_012) { TestCodeGenCast::Instance().test_Cast_012(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_013) { TestCodeGenCast::Instance().test_Cast_013(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_014) { TestCodeGenCast::Instance().test_Cast_014(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_015) { TestCodeGenCast::Instance().test_Cast_015(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_016) { TestCodeGenCast::Instance().test_Cast_016(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_017) { TestCodeGenCast::Instance().test_Cast_017(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_018) { TestCodeGenCast::Instance().test_Cast_018(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_019) { TestCodeGenCast::Instance().test_Cast_019(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_020) { TestCodeGenCast::Instance().test_Cast_020(); }

TEST_F(TestCodeGenKirinX90Cast, DISABLED_test_Cast_021) { TestCodeGenCast::Instance().test_Cast_021(); }
