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
 * \file test_codegen_concat.h
 * \brief
 */

#pragma once

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "codegen/codegen.h"
#include "codegen/npu/litenpu/codegen_litenpu.h"
#include "test_codegen_common.h"

class TestCodeGenConcat {
public:
    void test_concat_001();
    void test_concat_002();
    void test_concat_003();
    void test_concat_004();
    void test_concat_005();
    void test_concat_006();
    void test_concat_007();
    void test_concat_008();
    void test_concat_009();
    void test_concat_010();
    void test_concat_011();
    void test_concat_012();
    void test_concat_013();
    void test_concat_014();
    void test_concat_015();
    void test_concat_016();
    static TestCodeGenConcat& Instance();

private:
    TestCodeGenConcat();
    ~TestCodeGenConcat();
};
