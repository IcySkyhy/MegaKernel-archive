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
 * \file test_codegen_mul.h
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

class TestCodeGenMul {
public:
    void test_mul_001();
    void test_mul_002();
    void test_mul_003();
    void test_mul_004();
    void test_mul_005();
    void test_mul_006();
    void test_mul_007();
    void test_mul_008();
    void test_mul_009();
    void test_mul_010();
    void test_mul_011();
    void test_mul_012();
    void test_mul_013();
    void test_mul_014();
    void test_mul_015();
    void test_mul_016();
    void test_mul_017();
    void test_mul_018();
    void test_mul_019();
    void test_mul_020();
    void test_mul_021();
    void test_mul_022();
    void test_mul_023();
    void test_mul_024();
    void test_mul_025();
    void test_mul_026();
    void test_mul_027();
    void test_mul_028();
    void test_mul_029();
    void test_mul_030();
    void test_mul_031();
    void test_mul_032();
    void test_mul_033();
    void test_mul_034();
    void test_mul_035();
    void test_mul_036();
    void test_mul_037();
    void test_mul_038();
    void test_mul_039();
    void test_mul_040();
    void test_mul_041();
    void test_mul_042();
    void test_mul_043();
    void test_mul_044();
    void test_mul_045();
    void test_mul_046();
    void test_mul_047();
    void test_mul_048();
    static TestCodeGenMul& Instance();

private:
    TestCodeGenMul();
    ~TestCodeGenMul();
};
