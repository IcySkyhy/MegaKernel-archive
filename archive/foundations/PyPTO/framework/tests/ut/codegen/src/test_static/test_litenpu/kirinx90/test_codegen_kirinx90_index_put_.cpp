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
 * \file test_codegen_kirinx90_index_put_.cpp
 * \brief
 */

#include "../include/test_codegen_index_put.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90IndexPut : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90IndexPut() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90IndexPut, test_index_put_001) { TestCodeGenIndexPut::Instance().test_index_put_001(); }

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_002)
{
    TestCodeGenIndexPut::Instance().test_index_put_002();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_003)
{
    TestCodeGenIndexPut::Instance().test_index_put_003();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_004)
{
    TestCodeGenIndexPut::Instance().test_index_put_004();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_005)
{
    TestCodeGenIndexPut::Instance().test_index_put_005();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_006)
{
    TestCodeGenIndexPut::Instance().test_index_put_006();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_007)
{
    TestCodeGenIndexPut::Instance().test_index_put_007();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_008)
{
    TestCodeGenIndexPut::Instance().test_index_put_008();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_009)
{
    TestCodeGenIndexPut::Instance().test_index_put_009();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_010)
{
    TestCodeGenIndexPut::Instance().test_index_put_010();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_011)
{
    TestCodeGenIndexPut::Instance().test_index_put_011();
}

TEST_F(TestCodeGenKirinX90IndexPut, DISABLED_test_index_put_012)
{
    TestCodeGenIndexPut::Instance().test_index_put_012();
}
