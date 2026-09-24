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
 * \file test_codegen_kirin9030_concat.cpp
 * \brief
 */

#include "../include/test_codegen_concat.h"

using namespace npu::tile_fwk;

class TestCodeGenKirin9030Concat : public CodegenTestLiteNPU {
public:
    TestCodeGenKirin9030Concat() : CodegenTestLiteNPU("Kirin9030") {};
};

TEST_F(TestCodeGenKirin9030Concat, test_concat_001) { TestCodeGenConcat::Instance().test_concat_001(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_002) { TestCodeGenConcat::Instance().test_concat_002(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_003) { TestCodeGenConcat::Instance().test_concat_003(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_004) { TestCodeGenConcat::Instance().test_concat_004(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_005) { TestCodeGenConcat::Instance().test_concat_005(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_006) { TestCodeGenConcat::Instance().test_concat_006(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_007) { TestCodeGenConcat::Instance().test_concat_007(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_008) { TestCodeGenConcat::Instance().test_concat_008(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_009) { TestCodeGenConcat::Instance().test_concat_009(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_010) { TestCodeGenConcat::Instance().test_concat_010(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_011) { TestCodeGenConcat::Instance().test_concat_011(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_012) { TestCodeGenConcat::Instance().test_concat_012(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_013) { TestCodeGenConcat::Instance().test_concat_013(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_014) { TestCodeGenConcat::Instance().test_concat_014(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_015) { TestCodeGenConcat::Instance().test_concat_015(); }

TEST_F(TestCodeGenKirin9030Concat, DISABLED_test_concat_016) { TestCodeGenConcat::Instance().test_concat_016(); }
